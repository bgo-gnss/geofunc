"""Coordinate reference-frame registry and conversion facade.

Tier 0 geodesy: a data-driven registry of named coordinate frames
(ITRF realisations, WGS 84, and the Icelandic ISN national grids) plus a
thin :func:`transform` facade that pivots between any two via pyproj's
EPSG database.

Scope — static frame-to-frame only
----------------------------------
No epoch / tectonic-plate-motion handling lives here: that is the next
stage of work (it belongs with the data analyses, alongside
``geo.platePoles`` / ``geo.rotpole``). The registry is shaped so an
epoch-aware layer can be added without reshaping it — every
:class:`CoordFrame` carries its geographic-3D CRS
(:attr:`CoordFrame.geog3d`) as the pivot a future epoch transform needs.

Adding a frame
--------------
A new reference frame is a data entry, not code: append a row to
``_FRAME_SPECS`` (name, kind, axis, units, description, geog3d code, crs
code) and it is immediately usable as a ``src``/``dst`` name.

Axis-order contract
-------------------
:func:`transform` accepts and returns ``(x, y, z)`` where the meaning of
the three components follows the *target* frame's ``kind``::

    geographic  -> (longitude, latitude, ellipsoidal_height)   [deg, deg, m]
    geocentric  -> (X, Y, Z)  Earth-centred Earth-fixed           [m, m, m]
    projected   -> (Easting, Northing, ellipsoidal_height)        [m, m, m]

``always_xy=True`` is used everywhere, so longitude / Easting come first.
This is the OPPOSITE of the ``(lat, lon, h)`` order returned by
:func:`geofunc.geo.xyzell`, which honours pyproj's EPSG authority axis
order — do not mix the two.

Examples
--------
>>> from geofunc.coords import transform
>>> transform(-21.955490, 64.138788, 93.028, src="wgs84", dst="itrf2008")
(2587383.87..., -1043033.56..., 5716564.21...)
>>> transform(-21.955490, 64.138788, 93.028, src="wgs84", dst="isn2016")
(2556148.998, 207354.677, 93.028)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from pyproj import CRS, Transformer
from pyproj.crs import GeocentricCRS

__all__ = [
    "CoordFrame",
    "FRAMES",
    "frame_names",
    "get_frame",
    "transform",
]


@dataclass(frozen=True)
class CoordFrame:
    """A named coordinate frame and its canonical output representation."""

    name: str
    kind: str  # "geographic" | "geocentric" | "projected"
    crs: CRS  # CRS of THIS representation (the one transform() emits)
    geog3d: Optional[CRS]  # geographic-3D CRS of the same datum (pivot)
    axis: Tuple[str, str, str]
    units: Tuple[str, str, str]
    description: str = ""

    @property
    def is_geographic(self) -> bool:
        return self.kind == "geographic"

    @property
    def is_geocentric(self) -> bool:
        return self.kind == "geocentric"

    @property
    def is_projected(self) -> bool:
        return self.kind == "projected"


def _geocentric_of(geog3d: CRS) -> CRS:
    """Build a geocentric (ECEF) CRS sharing ``geog3d``'s datum.

    EPSG publishes only *geographic* codes for the newer ITRF realisations
    (ITRF2014 = EPSG:7912, ITRF2020 = EPSG:9989); their geocentric forms have
    no standalone EPSG code. ``GeocentricCRS(datum=...)`` rebuilds the
    Cartesian form while keeping the datum, so pyproj still finds the
    realisation-to-realisation Helmert (verified: the pipeline reports
    ``ITRF2008 to ITRF2014 (1)``).
    """
    return GeocentricCRS(datum=geog3d.datum)


#: name, kind, axis, units, description, geog3d_code, crs_code
#:
#: * ``geographic`` frames take ``geog3d_code`` (their crs *is* the geographic 3D CRS).
#: * ``geocentric`` frames take ``geog3d_code`` and, optionally, an explicit
#:   ``crs_code`` (EPSG:5332 for ITRF2008); without one the geocentric CRS is
#:   derived from ``geog3d_code`` via :func:`_geocentric_of`.
#: * ``projected`` frames take ``crs_code`` (the Lambert CRS).
_FRAME_SPECS = [
    # --- geographic -------------------------------------------------------
    (
        "wgs84",
        "geographic",
        ("lon", "lat", "h"),
        ("deg", "deg", "m"),
        "WGS 84 — geodetic longitude/latitude + ellipsoidal height",
        "EPSG:4979",
        None,
    ),
    # --- geocentric (ECEF) ------------------------------------------------
    (
        "itrf2008",
        "geocentric",
        ("x", "y", "z"),
        ("m", "m", "m"),
        "ITRF2008 — Earth-centred Earth-fixed (X, Y, Z)",
        "EPSG:7911",
        "EPSG:5332",
    ),
    (
        "itrf2014",
        "geocentric",
        ("x", "y", "z"),
        ("m", "m", "m"),
        "ITRF2014 — Earth-centred Earth-fixed (X, Y, Z)",
        "EPSG:7912",
        None,
    ),
    (
        "itrf2020",
        "geocentric",
        ("x", "y", "z"),
        ("m", "m", "m"),
        "ITRF2020 — Earth-centred Earth-fixed (X, Y, Z)",
        "EPSG:9989",
        None,
    ),
    # --- Icelandic national grids (projected) -----------------------------
    (
        "isn93",
        "projected",
        ("east", "north", "h"),
        ("m", "m", "m"),
        "ISN93 / Lambert 1993 — Icelandic national grid (E, N)",
        None,
        "EPSG:3057",
    ),
    (
        "isn2004",
        "projected",
        ("east", "north", "h"),
        ("m", "m", "m"),
        "ISN2004 / Lambert 2004 — Icelandic national grid (E, N)",
        None,
        "EPSG:5325",
    ),
    (
        "isn2016",
        "projected",
        ("east", "north", "h"),
        ("m", "m", "m"),
        "ISN2016 / Lambert 2016 — Icelandic national grid (E, N)",
        None,
        "EPSG:8088",
    ),
]


def _build_frames() -> Dict[str, CoordFrame]:
    frames: Dict[str, CoordFrame] = {}
    for name, kind, axis, units, description, geog3d_code, crs_code in _FRAME_SPECS:
        geog3d = CRS(geog3d_code) if geog3d_code else None

        if kind == "geographic":
            if geog3d is None:
                raise ValueError(f"geographic frame {name!r} needs a geog3d_code")
            crs = geog3d
        elif kind == "geocentric":
            if crs_code is not None:
                crs = CRS(crs_code)
            elif geog3d is not None:
                crs = _geocentric_of(geog3d)
            else:
                raise ValueError(
                    f"geocentric frame {name!r} needs geog3d_code or crs_code"
                )
        elif kind == "projected":
            if crs_code is None:
                raise ValueError(f"projected frame {name!r} needs a crs_code")
            crs = CRS(crs_code)
        else:
            raise ValueError(f"unknown frame kind {kind!r} for {name!r}")

        frames[name] = CoordFrame(name, kind, crs, geog3d, axis, units, description)
    return frames


#: The frame registry, keyed by frame name.
FRAMES: Dict[str, CoordFrame] = _build_frames()

#: Per (src, dst) transformer cache — pyproj Transformer construction is
#: cheap but not free, and a fleet-wide ``tos search`` projection will reuse
#: each pair hundreds of times.
_transformer_cache: Dict[Tuple[str, str], Transformer] = {}


def frame_names() -> Tuple[str, ...]:
    """Sorted names of every registered frame."""
    return tuple(sorted(FRAMES))


def get_frame(name: str) -> CoordFrame:
    """Return the :class:`CoordFrame` registered under ``name``.

    Raises:
        KeyError: if ``name`` is unknown, with the valid names listed.
    """
    try:
        return FRAMES[name]
    except KeyError:
        valid = ", ".join(frame_names())
        raise KeyError(
            f"unknown coordinate frame {name!r}; valid frames: {valid}"
        ) from None


def _transformer(src: CoordFrame, dst: CoordFrame) -> Transformer:
    key = (src.name, dst.name)
    if key not in _transformer_cache:
        _transformer_cache[key] = Transformer.from_crs(src.crs, dst.crs, always_xy=True)
    return _transformer_cache[key]


def transform(x, y, z, src: str = "wgs84", dst: str = "isn2004"):
    """Transform ``(x, y, z)`` from frame ``src`` to frame ``dst``.

    Component order follows each frame's ``kind`` (see the module
    docstring): geographic ``(lon, lat, h)``, geocentric ``(X, Y, Z)``,
    projected ``(E, N, h)``.

    Args:
        x, y, z: a single coordinate as three scalars, or three 1-D
            sequences (lists / numpy arrays) of equal length for a batch.
        src: source frame name (default ``"wgs84"``).
        dst: target frame name (default ``"isn2004"``).

    Returns:
        A 3-tuple ``(x', y', z')`` in the target frame, mirroring the input
        shape (scalars in → scalar tuple out; sequences in → array tuple out).
    """
    src_f = get_frame(src)
    dst_f = get_frame(dst)
    if src_f.name == dst_f.name:
        return (x, y, z)
    return _transformer(src_f, dst_f).transform(x, y, z)
