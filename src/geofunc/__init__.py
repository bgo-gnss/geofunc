"""geofunc — pure-Python geodesy primitives for the GPS library ecosystem.

Tier 0 foundation library. Coordinate reference-frame conversions live in
:mod:`geofunc.coords`; geodesy math (ellipsoidal/Cartesian conversions,
plate motion) in :mod:`geofunc.geo`; ECEF/geodetic/local transformations
with covariance propagation (numpy only) in :mod:`geofunc.local`.
"""

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

# Lazy re-export: ``geofunc.coords`` imports pyproj, which must not ride along
# with ``import geofunc.local`` (the numpy-only core used by leaf packages).
_COORDS = ("CoordFrame", "FRAMES", "frame_names", "get_frame", "transform")


def __getattr__(name: str):  # PEP 562
    if name in _COORDS:
        from . import coords

        return getattr(coords, name)
    raise AttributeError(f"module 'geofunc' has no attribute {name!r}")


__all__ = [
    "CoordFrame",
    "FRAMES",
    "frame_names",
    "get_frame",
    "transform",
]


def main() -> None:
    print("Hello from geofunc!")
