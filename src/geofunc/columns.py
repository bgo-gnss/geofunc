"""Position series as tables with chosen columns.

A data-agnostic column engine: given one point's positions over time (ECEF
X/Y/Z in a named geocentric frame, optionally the 3×3 covariance, the dates
and which epochs are kept), it builds the columns a request names. Which
station, which file, which epochs are good is the caller's business
(``gpscoord series`` in geo_dataread feeds it ``.pos`` series).

Column spec
-----------
``--columns`` is a comma list; each item is ``[NAME=]KEY[@FRAME][:REL]``:

=====================  ==========================================================
KEY                    column
=====================  ==========================================================
``date``               the day, YYYY-MM-DD (``datetime``: with the time)
``yearf`` / ``mjd``    decimal year / modified Julian day
``x y z``              ECEF in the series' frame [m]
``lat lon h``          geodetic on GRS80 at the observation epoch [deg, deg, m]
``east north``         a projected frame's axes, ``@FRAME`` required
                       (``east@isn93``) [m]; a frame fixed to an epoch (ISN93
                       1993.6, ISN2004 2004.6, ISN2016 2016.5) takes each
                       position back to that epoch on the point's velocity
                       (the plate) first, when one is given (``epoch_shift``)
``dn de du``           local N/E/U of the position less the reference position
                       (``ref_pos``) [units]
``sx sy sz sn se su``  formal σ from the covariance [units]
``kept``               1 when the caller's view keeps the epoch, else 0
=====================  ==========================================================

``:REL`` subtracts one value of that column, computed from the kept epochs:
``first5`` (the mean of the first 5; any N), ``first``, ``last``, ``mean``,
``median``, ``nearYYYY-MM-DD`` (the mean of the 5 kept epochs nearest the
date, either side; ``/N`` for another count), ``YYYY-MM-DD..YYYY-MM-DD``
(the mean over the dates, both included) or a number. A point with no kept day there (it started later)
gets an empty column and a note, not an error. A relative column is a difference, in ``units``,
named ``d<key>`` unless ``NAME=`` names it: ``dup=h:first5``.

A name of :data:`PRESETS` stands for its columns anywhere in the list:
``isn93`` = ``date,east@isn93,north@isn93,lat,lon,h``, so
``isn93,dup=h:first5`` is the borehole table.

Units: absolute positions are metres and degrees; differences and σ
(``d*``, ``s*``, ``:REL``) are in ``units`` (``mm`` default, or ``m``).

numpy only, except projected columns, which import pyproj through
:mod:`geofunc.coords` when asked for.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import datetime as dt
import re
from collections.abc import Mapping, Sequence
from typing import TextIO

import numpy as np
import numpy.typing as npt

from geofunc import local

FloatArray = npt.NDArray[np.float64]

__all__ = [
    "PRESETS",
    "Column",
    "ColumnSpec",
    "NoReference",
    "PositionSeries",
    "Table",
    "build_table",
    "parse_columns",
    "write_table",
]

#: names that stand for several columns, in --columns or --preset
PRESETS: dict[str, str] = {
    "isn93": "date,east@isn93,north@isn93,lat,lon,h",
    "isn2004": "date,east@isn2004,north@isn2004,lat,lon,h",
    "isn2016": "date,east@isn2016,north@isn2016,lat,lon,h",
    "llh": "date,lat,lon,h",
    "xyz": "date,x,y,z,sx,sy,sz",
    "neu": "date,yearf,dn,de,du,sn,se,su",
}
TIME_KEYS = ("date", "datetime", "yearf", "mjd")
XYZ_KEYS = ("x", "y", "z")
LLH_KEYS = ("lat", "lon", "h")
GRID_KEYS = ("east", "north")
NEU_KEYS = ("dn", "de", "du")
SIGMA_KEYS = ("sx", "sy", "sz", "sn", "se", "su")
KEYS = (*TIME_KEYS, *XYZ_KEYS, *LLH_KEYS, *GRID_KEYS, *NEU_KEYS, *SIGMA_KEYS, "kept")
UNITS = {"mm": 1000.0, "m": 1.0}
REF_HOWS = ("wmean", "mean", "median", "first")
_ITEM = re.compile(
    r"^(?:(?P<name>[A-Za-z_][\w]*)=)?(?P<key>[a-z]+)(?:@(?P<frame>[\w]+))?(?::(?P<rel>.+))?$"
)
_FIRST_N = re.compile(r"^first(\d+)$")
_NEAR = re.compile(r"^near(\d{4}-\d{2}-\d{2})(?:/(\d+))?$")
#: kept epochs averaged by :nearDATE
NEAR_DAYS = 5
_DATES = re.compile(r"^(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})$")


class NoReference(ValueError):
    """A ``:REL`` value with no kept epoch to take it from."""


@dataclasses.dataclass(frozen=True)
class PositionSeries:
    """One point's positions over time.

    Attributes:
        t: decimal years, (n,).
        xyz: ECEF [m], (n, 3), in ``frame``.
        frame: a geocentric :mod:`geofunc.coords` frame (``itrf2014`` …).
        cov: ECEF covariance [m²], (n, 3, 3), or None (no σ columns).
        dates: the epochs as datetimes (None: ``date``/``mjd`` from ``t``).
        kept: which epochs the caller keeps (None: all); the statistics of
            ``:REL`` and ``ref_pos`` use the kept epochs only.
        velocity: ECEF velocity [m/yr], (3,), that carries a position to a
            frame's reference epoch (the plate's), or None.
    """

    t: FloatArray
    xyz: FloatArray
    frame: str = "itrf2014"
    cov: FloatArray | None = None
    dates: Sequence[dt.datetime] | None = None
    kept: npt.NDArray[np.bool_] | None = None
    velocity: FloatArray | None = None

    @property
    def keep(self) -> npt.NDArray[np.bool_]:
        ok = np.isfinite(self.xyz).all(axis=1)
        return ok if self.kept is None else ok & np.asarray(self.kept, bool)


@dataclasses.dataclass(frozen=True)
class ColumnSpec:
    """One requested column: ``[name=]key[@frame][:rel]``."""

    key: str
    frame: str | None = None
    rel: str | None = None
    name: str | None = None

    @property
    def header(self) -> str:
        if self.name:
            return self.name
        base = self.key if self.frame is None else f"{self.key}_{self.frame}"
        return f"d{base}" if self.rel else base


@dataclasses.dataclass(frozen=True)
class Column:
    """A built column: its header, unit, values, decimals and meaning."""

    name: str
    unit: str
    values: FloatArray | list[str]
    decimals: int | None
    definition: str


@dataclasses.dataclass(frozen=True)
class Table:
    """The columns, the reference position used (ECEF, or None) and notes
    a reader must see (the datum of projected columns)."""

    columns: list[Column]
    ref_pos: FloatArray | None = None
    notes: list[str] = dataclasses.field(default_factory=list)

    def rows(self) -> list[list[str]]:
        n = len(self.columns[0].values) if self.columns else 0
        return [[_fmt(c, k) for c in self.columns] for k in range(n)]


def parse_columns(
    spec: str, *, presets: Mapping[str, str] = PRESETS
) -> list[ColumnSpec]:
    """``spec`` (comma list, presets expanded) → column specs.

    Raises:
        ValueError: an item that is not ``[NAME=]KEY[@FRAME][:REL]``, an
            unknown key, a grid key without a frame, or a repeated header.
    """
    out: list[ColumnSpec] = []
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        if item in presets:
            out += parse_columns(presets[item], presets={})
            continue
        m = _ITEM.match(item)
        if not m:
            raise ValueError(f"column {item!r}: use [NAME=]KEY[@FRAME][:REL]")
        key, frame = m["key"], m["frame"]
        if key not in KEYS:
            raise ValueError(
                f"column {item!r}: unknown key {key!r}; keys: {', '.join(KEYS)}"
            )
        if key in GRID_KEYS and frame is None:
            raise ValueError(f"column {item!r}: {key} needs a frame, e.g. {key}@isn93")
        if frame is not None and key not in (*GRID_KEYS, *XYZ_KEYS):
            raise ValueError(
                f"column {item!r}: only {', '.join(GRID_KEYS + XYZ_KEYS)} take @FRAME"
            )
        rel = m["rel"]
        if rel is not None and key in (*TIME_KEYS, "kept"):
            raise ValueError(f"column {item!r}: a time or kept column has no :REL")
        out.append(ColumnSpec(key, frame, rel, m["name"]))
    headers = [c.header for c in out]
    dup = sorted({h for h in headers if headers.count(h) > 1})
    if dup:
        raise ValueError(
            f"columns named twice: {', '.join(dup)} (rename one with NAME=)"
        )
    return out


def _fmt(col: Column, k: int) -> str:
    v = col.values[k]
    if isinstance(v, str):
        return v
    f = float(v)
    if not np.isfinite(f):
        return ""
    return f"{f:.{col.decimals}f}" if col.decimals is not None else f"{f:g}"


def _dates(s: PositionSeries) -> list[dt.datetime]:
    if s.dates is not None:
        return list(s.dates)
    out = []
    for t in np.asarray(s.t, float):
        year = int(t)
        start = dt.datetime(year, 1, 1)
        days = (dt.datetime(year + 1, 1, 1) - start).days
        out.append(start + dt.timedelta(days=(t - year) * days))
    return out


def _reference(s: PositionSeries, how: str | npt.ArrayLike) -> FloatArray:
    """The ECEF reference position of ``dn de du`` (kept epochs)."""
    if not isinstance(how, str):
        ref = np.asarray(how, float)
        if ref.shape != (3,):
            raise ValueError("ref_pos as a position is X,Y,Z")
        return ref
    keep = s.keep
    if not keep.any():
        raise ValueError("no kept epoch to take a reference position from")
    m = _FIRST_N.match(how)
    count = int(m[1]) if m else None
    mode = "mean" if m else how
    if mode not in REF_HOWS:
        raise ValueError(f"ref_pos {how!r}: {', '.join(REF_HOWS)}, firstN or X,Y,Z")
    cov = None if s.cov is None else s.cov[keep]
    if mode == "wmean" and cov is None:
        mode = "mean"
    return local.reference_position(
        s.xyz[keep],
        cov=cov,
        how=mode,  # type: ignore[arg-type]  # checked against REF_HOWS above
        count=count,
    )


def _rel_value(
    values: FloatArray, s: PositionSeries, rel: str, dates: list[dt.datetime]
) -> float:
    """The one value ``:REL`` subtracts, from the kept, finite epochs."""
    ok = s.keep & np.isfinite(values)
    try:
        return float(rel)
    except ValueError:
        pass
    m = _FIRST_N.match(rel)
    sel = values[ok]
    if m:
        n = int(m[1])
        if n < 1:
            raise ValueError(f":{rel}: N must be at least 1")
        sel = sel[:n]
    elif rel == "first":
        sel = sel[:1]
    elif rel == "last":
        sel = sel[-1:]
    elif (nm := _NEAR.match(rel)) is not None:
        # the N kept epochs nearest the date, either side: as close to it as
        # the point has (a later start: its first N)
        day0 = dt.date.fromisoformat(nm[1])
        n = int(nm[2]) if nm[2] else NEAR_DAYS
        if n < 1:
            raise ValueError(f":{rel}: N must be at least 1")
        idx = np.flatnonzero(ok)
        gap = np.array([abs((dates[i].date() - day0).days) for i in idx])
        sel = values[idx[np.argsort(gap, kind="stable")[:n]]]
    elif (w := _DATES.match(rel)) is not None:
        a, b = (dt.date.fromisoformat(x) for x in w.groups())
        day = np.array([d.date() for d in dates])
        sel = values[ok & (day >= a) & (day <= b)]
    elif rel not in ("mean", "median"):
        raise ValueError(
            f":{rel}: use firstN, first, last, mean, median, nearYYYY-MM-DD[/N], "
            "YYYY-MM-DD..YYYY-MM-DD or a number"
        )
    if sel.size == 0:
        raise NoReference(f":{rel}: no kept epoch to take it from")
    return float(np.median(sel) if rel == "median" else np.mean(sel))


def _grid(
    s: PositionSeries, frame: str, epoch_shift: bool
) -> tuple[FloatArray, FloatArray, str]:
    """East/North in projected ``frame``, and the note on its datum."""
    from geofunc.coords import get_frame, transform

    f = get_frame(frame)
    if not f.is_projected:
        raise ValueError(f"east/north@{frame}: not a projected frame")
    xyz = s.xyz
    shifted = epoch_shift and f.epoch is not None and s.velocity is not None
    if shifted:
        v = np.asarray(s.velocity, float).reshape(1, 3)
        xyz = xyz - v * (np.asarray(s.t, float) - float(f.epoch))[:, None]  # type: ignore[arg-type]
    g = local.ecef_to_geodetic(xyz)
    e, n, _ = transform(
        np.degrees(g.lon), np.degrees(g.lat), g.h, src="wgs84", dst=frame
    )
    if shifted:
        note = (
            f"{frame}: each position taken back to {f.epoch} on the point's velocity "
            f"({np.round(np.asarray(s.velocity) * 1000, 1).tolist()} mm/yr ECEF) then "
            f"projected; the {s.frame} to {frame} datum difference beyond that "
            "(frame realisations, deformation since the epoch) is not applied"
        )
    elif f.epoch is not None:
        note = (
            f"{frame}: {s.frame} at the observation epoch projected with no datum "
            f"shift; absolute east/north differ from {frame} (fixed at {f.epoch}) by "
            "the motion since then (~0.5-0.7 m for ISN93); changes are unaffected"
        )
    else:
        note = f"{frame}: {s.frame} at the observation epoch projected"
    return np.asarray(e, float), np.asarray(n, float), note


def build_table(
    s: PositionSeries,
    columns: Sequence[ColumnSpec],
    *,
    ref_pos: str | npt.ArrayLike = "wmean",
    units: str = "mm",
    decimals: int | None = None,
    epoch_shift: bool = True,
) -> Table:
    """The table of ``s`` with ``columns`` (see the module docstring).

    Args:
        ref_pos: the reference of ``dn de du``: ``wmean`` (1/σ² weighted, the
            products' default), ``mean``, ``median``, ``first``, ``firstN``
            (mean of the first N kept) or an ECEF X,Y,Z.
        units: ``mm`` or ``m`` for differences and σ.
        decimals: every number's decimals (default per unit).
        epoch_shift: take positions to an epoch-fixed frame's epoch on
            ``s.velocity`` before projecting.

    Raises:
        ValueError: an impossible request (σ without covariance, a frame the
            series is not in, no kept epoch for a reference).
    """
    if units not in UNITS:
        raise ValueError(f"units {units!r}: mm or m")
    scale = UNITS[units]
    diff_dec = 2 if units == "mm" else 5
    dates = _dates(s)
    xyz = np.asarray(s.xyz, float)
    geo = None
    neu = None
    ref = None
    grids: dict[str, tuple[FloatArray, FloatArray]] = {}
    notes: list[str] = []
    cols: list[Column] = []
    ref_label = ref_pos if isinstance(ref_pos, str) else "X,Y,Z"
    for c in columns:
        unit: str = "m"
        dec: int | None = 4
        what = ""
        vals: FloatArray | list[str]
        if c.key == "date":
            vals, unit, dec, what = (
                [d.strftime("%Y-%m-%d") for d in dates],
                "",
                None,
                "day",
            )
        elif c.key == "datetime":
            vals, unit, dec = [d.strftime("%Y-%m-%dT%H:%M:%S") for d in dates], "", None
            what = "epoch"
        elif c.key == "yearf":
            vals, unit, dec, what = np.asarray(s.t, float), "yr", 5, "decimal year"
        elif c.key == "mjd":
            mjd0 = dt.datetime(1858, 11, 17)
            vals = np.array([(d - mjd0).total_seconds() / 86400.0 for d in dates])
            unit, dec, what = "d", 4, "modified Julian day"
        elif c.key == "kept":
            vals = [str(int(k)) for k in s.keep]
            unit, dec, what = "", None, "1 = kept by the view"
        elif c.key in XYZ_KEYS:
            if c.frame not in (None, s.frame):
                raise ValueError(f"{c.key}@{c.frame}: the series is in {s.frame}")
            vals = xyz[:, XYZ_KEYS.index(c.key)]
            what = f"ECEF {c.key.upper()} {s.frame}"
        elif c.key in LLH_KEYS:
            if geo is None:
                geo = local.ecef_to_geodetic(xyz)
            vals = np.asarray(
                {"lat": np.degrees(geo.lat), "lon": np.degrees(geo.lon), "h": geo.h}[
                    c.key
                ],
                float,
            )
            unit, dec = ("deg", 9) if c.key != "h" else ("m", 4)
            what = f"{c.key} GRS80 {s.frame} at the observation epoch"
        elif c.key in GRID_KEYS:
            frame = str(c.frame)
            if frame not in grids:
                e, n, note = _grid(s, frame, epoch_shift)
                grids[frame] = (e, n)
                notes.append(note)
            vals = grids[frame][GRID_KEYS.index(c.key)]
            what = f"{c.key} {frame}"
        elif c.key in NEU_KEYS:
            if neu is None:
                ref = _reference(s, ref_pos)
                neu = local.displacements(xyz, ref, cov=s.cov)
            vals = np.asarray(neu.d, float)[:, NEU_KEYS.index(c.key)] * scale
            unit, dec = units, diff_dec
            what = f"local {c.key[1].upper()} from the reference position ({ref_label})"
        else:  # sigma
            if s.cov is None:
                raise ValueError(f"{c.key}: the series has no covariance")
            if c.key in ("sx", "sy", "sz"):
                var = np.diagonal(s.cov, axis1=1, axis2=2)[
                    :, ("sx", "sy", "sz").index(c.key)
                ]
            else:
                if neu is None:
                    ref = _reference(s, ref_pos)
                    neu = local.displacements(xyz, ref, cov=s.cov)
                var = np.diagonal(np.asarray(neu.cov), axis1=1, axis2=2)[
                    :, ("sn", "se", "su").index(c.key)
                ]
            vals = np.sqrt(np.clip(var, 0.0, None)) * scale
            unit, dec, what = units, diff_dec, f"formal σ {c.key[1].upper()}"
        if c.rel is not None:
            if unit == "deg":
                raise ValueError(f"{c.key}:{c.rel}: a difference of degrees; use dn/de")
            v = np.asarray(vals, float)
            try:
                ref_v = _rel_value(v, s, c.rel, dates)
            except NoReference:
                # the point has no kept day there (it started later): an
                # empty column, said in the header, not a lost table
                ref_v = math.nan
                notes.append(f"{c.header}: no kept day in {c.rel}; column left empty")
            # a difference already in `units`, else metres scaled
            vals = v - ref_v if unit == units else (v - ref_v) * scale
            unit, dec = units, diff_dec
            what = f"{what}, less its {c.rel} value"
        cols.append(
            Column(
                c.header,
                unit,
                vals,
                decimals if decimals is not None and dec is not None else dec,
                what,
            )
        )
    return Table(cols, ref, notes)


def write_table(
    table: Table, out: TextIO, *, header: Sequence[str] = (), delimiter: str = ","
) -> None:
    """``#`` header lines (the caller's, then each column's unit and
    meaning, then the notes), the column names, the rows."""
    for line in header:
        out.write(f"# {line}\n")
    for c in table.columns:
        out.write(f"# column {c.name} [{c.unit or '-'}] {c.definition}\n")
    for note in table.notes:
        out.write(f"# note {note}\n")
    w = csv.writer(out, delimiter=delimiter, lineterminator="\n")
    w.writerow([c.name for c in table.columns])
    w.writerows(table.rows())
