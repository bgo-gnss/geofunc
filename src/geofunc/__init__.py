"""geofunc — pure-Python geodesy primitives for the GPS library ecosystem.

Tier 0 foundation library. ECEF/geodetic/local transformations with
covariance propagation (numpy only) live in :mod:`geofunc.local`; plate-motion
models in :mod:`geofunc.plates` (numpy only); PROJ frames and Helmert
parameters in :mod:`geofunc.frames`; the named-frame facade in
:mod:`geofunc.coords`. :mod:`geofunc.geo` and :mod:`geofunc.geofunc` are
deprecated shims (removed after one release).
"""

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

# Lazy re-export: ``geofunc.coords`` imports pyproj, which must not ride along
# with ``import geofunc.local`` (the numpy-only core used by leaf packages).
# Submodules stay reachable as attributes after a bare ``import geofunc``
# (the old eager import made ``geofunc.coords`` work that way).
_COORDS = ("CoordFrame", "FRAMES", "frame_names", "get_frame", "transform")
_SUBMODULES = ("coords", "frames", "geo", "geofunc", "local", "plates")


def __getattr__(name: str):  # PEP 562
    import importlib

    if name in _COORDS:
        return getattr(importlib.import_module(".coords", __name__), name)
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
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
