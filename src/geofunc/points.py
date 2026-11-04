"""Point tables: read a list of named points in any frame, convert, find the nearest.

Tables come from anywhere (a spreadsheet, a web page, a report), so the
reader is lenient on layout and strict on numbers:

- **Delimiter**: tab, then ``;``, then ``,``, else runs of whitespace — the
  first one the header line carries. A comma is a decimal mark
  (``329637,4``) whenever it is not the delimiter.
- **One cell per line** (a table copied from a web page): pass ``ncols``;
  the non-blank lines are regrouped into rows of that many cells.
- **Header**: the first row is a header when its coordinate cells are not
  numbers. The coordinate columns are found by name — ``x``/``east``/
  ``easting``/``e``/``lon``/``longitude`` and ``y``/``north``/``northing``/
  ``n``/``lat``/``latitude``, matched on the header's first word, so
  ``X (ISN93) [m]`` is ``x`` — else they are the last two numeric columns.
  The name is the first column that is not a coordinate.
- **Frame**: a registered frame name in the header (``X (ISN93) [m]``)
  is the table's frame (:func:`guess_frame`); x is always the first axis
  of the frame (longitude or Easting), y the second.

:func:`nearest` takes the candidates as plain ``(id, lon, lat)`` rows, so
which stations are searched (``stations.cfg``, TOS, a file) is the
caller's choice; distances are geodesic on GRS80.

Like :mod:`geofunc.coords` this module imports pyproj (lazily, through
:mod:`geofunc.coords` and :mod:`geofunc.frames`), never ``geofunc.local``'s
numpy-only path.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TextIO

__all__ = [
    "Neighbour",
    "Point",
    "guess_frame",
    "nearest",
    "read_points",
    "to_frame",
    "write_points",
]

X_KEYS = ("x", "east", "easting", "e", "lon", "longitude")
Y_KEYS = ("y", "north", "northing", "n", "lat", "latitude")
DELIMITERS = ("\t", ";", ",")
_NUMBER = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$")


@dataclass(frozen=True)
class Point:
    """One named point.

    Attributes:
        name: the point's name (first non-coordinate column).
        x: first axis of ``frame`` — longitude [deg] or Easting [m].
        y: second axis — latitude [deg] or Northing [m].
        frame: a :mod:`geofunc.coords` frame name, or None when unknown.
        fields: every other column of the row, header → cell, in order.
    """

    name: str
    x: float
    y: float
    frame: str | None = None
    fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Neighbour:
    """A candidate near a point: its id, geodesic distance [m] and the
    azimuth from the point to it [deg, clockwise from north]."""

    id: str
    distance: float
    azimuth: float


def _number(cell: str, decimal_comma: bool) -> float | None:
    s = cell.strip().replace("−", "-").replace(" ", "")
    if not s or not _NUMBER.match(s):
        return None
    if "," in s:
        if not decimal_comma:
            return None
        s = s.replace(",", ".")
    return float(s)


def _key(header: str) -> str:
    """``X (ISN93) [m]`` → ``x``: the first word, lower case, letters only."""
    words = re.findall(r"[^\W\d_]+", header.lower())
    return words[0] if words else ""


def guess_frame(headers: Iterable[str]) -> str | None:
    """The registered frame a header names, e.g. ``X (ISN93) [m]`` →
    ``isn93``; None when no header names one, or they name different ones."""
    from geofunc.coords import frame_names

    names = set(frame_names())
    found = {
        w for h in headers for w in re.findall(r"[a-z0-9]+", h.lower()) if w in names
    }
    return found.pop() if len(found) == 1 else None


def _cells(text: str, ncols: int | None) -> tuple[list[list[str]], str | None]:
    """Rows of cells and the delimiter (None: whitespace or one per line)."""
    lines = [
        ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if ncols:
        flat = [ln.strip() for ln in lines]
        if len(flat) % ncols:
            raise ValueError(
                f"{len(flat)} non-blank lines is not a whole number of {ncols}-cell rows"
            )
        return [flat[i : i + ncols] for i in range(0, len(flat), ncols)], None
    if not lines:
        return [], None
    first = lines[0]
    delim = next((d for d in DELIMITERS if d in first), None)
    if delim == ",":
        # a comma in a whitespace table of decimal-comma numbers is no delimiter
        tokens = first.split()
        if len(tokens) > 1 and all(_number(t, True) is not None for t in tokens[1:]):
            delim = None
    if delim is None:
        return [ln.split() for ln in lines], None
    rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delim))
    return [[c.strip() for c in r] for r in rows], delim


def _column(spec: str | int | None, headers: Sequence[str], what: str) -> int | None:
    """A column given by header (exact, or its first word) or 1-based index."""
    if spec is None:
        return None
    if isinstance(spec, int) or str(spec).isdigit():
        k = int(spec) - 1
        if not 0 <= k < len(headers):
            raise ValueError(
                f"{what} column {spec}: the table has {len(headers)} columns"
            )
        return k
    low = [h.lower() for h in headers]
    for cand in (str(spec).lower(), _key(str(spec))):
        if cand in low:
            return low.index(cand)
        keys = [_key(h) for h in headers]
        if cand in keys:
            return keys.index(cand)
    raise ValueError(f"{what} column {spec!r}: not among {', '.join(headers)}")


def read_points(
    source: str | Path | TextIO,
    *,
    frame: str | None = None,
    x: str | int | None = None,
    y: str | int | None = None,
    name: str | int | None = None,
    ncols: int | None = None,
) -> list[Point]:
    """The points of a table (see the module docstring for the layouts).

    Args:
        source: a file path, ``"-"`` for stdin, or an open text stream.
        frame: the table's frame; default the one its header names
            (:func:`guess_frame`), else None.
        x, y, name: the columns, by header or 1-based index (default found
            by name, see the module docstring).
        ncols: cells per row of a one-cell-per-line table.

    Raises:
        ValueError: no rows, no coordinate columns, or a row whose
            coordinates are not numbers.
    """
    if isinstance(source, (str, Path)):
        if str(source) == "-":
            import sys

            text = sys.stdin.read()
        else:
            text = Path(source).expanduser().read_text(encoding="utf-8-sig")
    else:
        text = source.read()
    rows, delim = _cells(text, ncols)
    if not rows:
        raise ValueError("no rows in the table")
    dec_comma = delim != ","
    width = max(len(r) for r in rows)
    first = rows[0]
    numeric = [_number(c, dec_comma) is not None for c in first]
    has_header = not (sum(numeric) >= 2)
    if has_header:
        headers = [h or f"col{k + 1}" for k, h in enumerate(first)]
        headers += [f"col{k + 1}" for k in range(len(headers), width)]
        body = rows[1:]
    else:
        headers = [f"col{k + 1}" for k in range(width)]
        body = rows
    if not body:
        raise ValueError("the table has a header and no rows")
    keys = [_key(h) for h in headers] if has_header else [""] * width
    ix = _column(x, headers, "x")
    iy = _column(y, headers, "y")
    if ix is None:
        ix = next((k for k, h in enumerate(keys) if h in X_KEYS), None)
    if iy is None:
        iy = next((k for k, h in enumerate(keys) if h in Y_KEYS), None)
    if ix is None or iy is None:
        # the last two columns that are numbers in every row
        num_cols = [
            k
            for k in range(width)
            if all(k < len(r) and _number(r[k], dec_comma) is not None for r in body)
        ]
        if len(num_cols) < 2:
            raise ValueError("cannot tell the coordinate columns; name them (x=…, y=…)")
        ix, iy = num_cols[-2], num_cols[-1]
    iname = _column(name, headers, "name")
    if iname is None:
        iname = next((k for k in range(width) if k not in (ix, iy)), None)
    if frame is None and has_header:
        frame = guess_frame([headers[ix], headers[iy]]) or guess_frame(headers)
    points = []
    for k, r in enumerate(body, 2 if has_header else 1):
        r = r + [""] * (width - len(r))
        xv, yv = _number(r[ix], dec_comma), _number(r[iy], dec_comma)
        if xv is None or yv is None:
            raise ValueError(f"row {k}: {r[ix]!r}, {r[iy]!r} are not coordinates")
        label = r[iname].strip() if iname is not None else f"P{k}"
        extra = {headers[j]: r[j] for j in range(width) if j not in (ix, iy, iname)}
        points.append(Point(label, xv, yv, frame, extra))
    return points


def to_frame(points: Sequence[Point], dst: str, src: str | None = None) -> list[Point]:
    """``points`` in frame ``dst`` (heights taken as 0; a 2-D table has none).

    Args:
        src: the points' frame when they carry none (overrides theirs).

    Raises:
        ValueError: a point without a frame and no ``src``.
    """
    from geofunc.coords import get_frame, transform

    get_frame(dst)  # unknown names fail before any work
    out = []
    for p in points:
        frm = src or p.frame
        if frm is None:
            raise ValueError(f"{p.name}: the table's frame is unknown; give it (src=…)")
        x, y, _ = transform(p.x, p.y, 0.0, src=frm, dst=dst)
        out.append(replace(p, x=float(x), y=float(y), frame=dst))
    return out


def nearest(
    point: tuple[float, float],
    candidates: Sequence[tuple[str, float, float]],
    k: int = 1,
    *,
    max_distance: float | None = None,
) -> list[Neighbour]:
    """The ``k`` candidates nearest ``point`` (lon, lat [deg]), nearest first.

    Args:
        candidates: ``(id, lon, lat)`` rows.
        max_distance: leave out candidates farther than this [m].
    """
    import numpy as np

    from geofunc.frames import geodesic

    if not candidates:
        return []
    ids = [c[0] for c in candidates]
    lon = np.array([c[1] for c in candidates], dtype=float)
    lat = np.array([c[2] for c in candidates], dtype=float)
    az, _, dist = geodesic().inv(
        np.full_like(lon, point[0]), np.full_like(lat, point[1]), lon, lat
    )
    order = np.argsort(dist, kind="stable")
    found = [
        Neighbour(ids[i], float(dist[i]), float(az[i]) % 360.0)
        for i in order
        if max_distance is None or dist[i] <= max_distance
    ]
    return found[:k]


def write_points(points: Sequence[Point], out: TextIO, *, delimiter: str = ",") -> None:
    """``points`` as a table: ``name``, their other columns, then the
    coordinates named by their frame's axes (``lon,lat`` or ``east,north``;
    ``x,y`` when the frame is unknown)."""
    from geofunc.coords import get_frame

    frames = {p.frame for p in points}
    axes = ("x", "y")
    if len(frames) == 1 and None not in frames:
        axes = get_frame(str(next(iter(frames)))).axis[:2]
    extra: list[str] = []
    for p in points:
        extra += [h for h in p.fields if h not in extra]
    w = csv.writer(out, delimiter=delimiter, lineterminator="\n")
    w.writerow(["name", *extra, *axes])
    for p in points:
        fmt = "{:.7f}" if axes[0] == "lon" else "{:.2f}"
        w.writerow([p.name, *(p.fields.get(h, "") for h in extra),
                    fmt.format(p.x), fmt.format(p.y)])  # fmt: skip
