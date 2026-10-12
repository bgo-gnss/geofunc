"""Reference frames from PROJ: geocentric CRSs, Helmert parameters, 4-D transforms.

PROJ/EPSG owns the frame definitions and their transformation parameters;
:mod:`geofunc.local` owns the covariance. This module is the bridge:

- :func:`helmert_params` pulls the 7 + 7 parameters of an EPSG
  time-dependent Helmert out of PROJ (via ``TransformerGroup(...).to_json()``)
  in the units :func:`geofunc.local.helmert_at` / ``helmert_apply`` take
  (m, unitless scale, rad; rates per year), so a frame change of a series
  carries its 3×3 covariances. Fed back through ``helmert_apply`` they
  reproduce PROJ's own 4-D transform to ~2e-9 m.
- :func:`transform_xyz` is the plain PROJ path for points ``(..., 3)`` with
  an optional epoch (faster than numpy when no covariance is needed).
- :func:`geodesic` — distances/azimuths on the ellipsoid (pyproj.Geod).

Frames are geocentric (ECEF X/Y/Z) and named by explicit EPSG codes so PROJ
finds the EPSG operations (a CRS derived on the fly carries no authority
code and gets a ballpark/no-op instead). The ITRF/ETRF frames are on GRS80;
EPSG:4978 is **WGS84**, not a substitute for them.

The named-frame facade for mixed geographic/projected output (WGS84,
ISN93/2004/2016 Lambert) stays in :mod:`geofunc.coords`.

Hazards
-------
- PROJ pipeline strings give rotations in arcsec(/yr); EPSG rows store
  mas(/yr). Never hand-build pipelines from table values — use
  :func:`helmert_params`, which converts through the unit factors.
- ``TransformerGroup`` can return several operations (e.g. ETRF2014 has
  two, reference epochs 1989 and 2010); ``pick`` chooses one and
  :attr:`HelmertParams.name` says which.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from pyproj import CRS, Geod, Transformer
from pyproj.transformer import TransformerGroup

from geofunc import local
from geofunc.local import FloatArray

__all__ = [
    "GEOCENTRIC",
    "SECONDS_PER_YEAR",
    "HelmertParams",
    "crs",
    "geodesic",
    "helmert_params",
    "transform_xyz",
    "transformer",
]

#: geocentric (ECEF) CRS per frame name
GEOCENTRIC: dict[str, str] = {
    "itrf2000": "EPSG:4919",
    "itrf2005": "EPSG:4896",
    "itrf2008": "EPSG:5332",
    "itrf2014": "EPSG:7789",
    "itrf2020": "EPSG:9988",
    "etrf2000": "EPSG:7930",
    "etrf2014": "EPSG:8401",
    "etrf2020": "EPSG:10569",
}

#: PROJ's year for time-dependent rates (Julian year in seconds, IERS)
SECONDS_PER_YEAR = 31556925.445

_ORDER = (
    "X-axis translation",
    "Y-axis translation",
    "Z-axis translation",
    "Scale difference",
    "X-axis rotation",
    "Y-axis rotation",
    "Z-axis rotation",
)


class HelmertParams(NamedTuple):
    """An EPSG time-dependent Helmert as :func:`geofunc.local.helmert_at` takes it.

    ``p0`` and ``rates`` are ``(tx, ty, tz, D, rx, ry, rz)`` in m, unitless,
    rad (rates per year), position-vector convention; ``t0`` the parameter
    reference epoch (None for a static transformation).
    """

    p0: FloatArray
    rates: FloatArray
    t0: float | None
    name: str
    accuracy: float | None

    def at(self, t: ArrayLike) -> FloatArray:
        """Parameters ``(..., 7)`` at epochs ``t`` (decimal years)."""
        if self.t0 is None:
            return np.broadcast_to(self.p0, np.shape(t) + (7,)).astype(np.float64)
        return local.helmert_at(self.p0, self.rates, t, t0=self.t0)


def crs(name: str) -> CRS:
    """Geocentric CRS of a frame name (``itrf2008``, ``etrf2014``, …) or any PROJ string."""
    return _crs(GEOCENTRIC.get(name.lower(), name))


@lru_cache(maxsize=None)
def _crs(code: str) -> CRS:
    return CRS(code)


@lru_cache(maxsize=None)
def transformer(src: str, dst: str) -> Transformer:
    """Cached 4-D-capable ``always_xy`` transformer between two frames."""
    return Transformer.from_crs(crs(src), crs(dst), always_xy=True)


def _si(param: dict[str, object]) -> float:
    value = float(param["value"])  # type: ignore[arg-type]
    unit = param.get("unit")
    if isinstance(unit, dict):
        return value * float(unit["conversion_factor"])
    return value


def helmert_params(src: str, dst: str, *, pick: int = 0) -> HelmertParams:
    """The EPSG Helmert ``src → dst`` from PROJ, in ``geofunc.local`` units.

    Raises:
        ValueError: when PROJ's operation is not a single position-vector
            Helmert (a concatenated or coordinate-frame one, a grid, a no-op).
    """
    ops = TransformerGroup(crs(src), crs(dst), always_xy=True).transformers
    if not ops:
        raise ValueError(f"PROJ has no operation {src} → {dst}")
    op = ops[pick]
    j = json.loads(op.to_json())
    method = j.get("method", {}).get("name", "")
    if j.get("type") != "Transformation" or "Position Vector" not in method:
        raise ValueError(
            f"{src} → {dst}: {j.get('name')!r} is {j.get('type')}/{method!r}, "
            "not a position-vector Helmert"
        )
    prm = {p["name"]: p for p in j["parameters"]}
    p0 = np.array([_si(prm[n]) if n in prm else 0.0 for n in _ORDER])
    rates = np.array(
        [
            _si(prm[k]) * SECONDS_PER_YEAR
            if (k := "Rate of change of " + n) in prm
            else 0.0
            for n in _ORDER
        ]
    )
    epoch = prm.get("Parameter reference epoch")
    t0 = None if epoch is None else float(epoch["value"])
    acc = j.get("accuracy")
    return HelmertParams(
        p0, rates, t0, str(j.get("name", "")), None if acc is None else float(acc)
    )


def transform_xyz(
    xyz: ArrayLike, src: str, dst: str, t: ArrayLike | None = None
) -> FloatArray:
    """PROJ transform of ECEF points ``(..., 3)`` at epochs ``t`` (decimal years).

    ``t`` broadcasts against the points' leading shape; without it a
    time-dependent transformation is evaluated at its reference epoch.
    """
    x = np.asarray(xyz, dtype=np.float64)
    if x.shape[-1] != 3:
        raise ValueError(f"xyz must have a trailing axis of 3, got {x.shape}")
    flat = x.reshape(-1, 3)
    tr = transformer(src, dst)
    if t is None:
        out = tr.transform(flat[:, 0], flat[:, 1], flat[:, 2])
    else:
        tt = np.broadcast_to(np.asarray(t, dtype=np.float64), x.shape[:-1]).reshape(-1)
        out = tr.transform(flat[:, 0], flat[:, 1], flat[:, 2], tt)[:3]
    res: FloatArray = np.stack(out, axis=-1).reshape(x.shape)
    return res


def geodesic(ellps: str = "GRS80") -> Geod:
    """``pyproj.Geod`` on ``ellps`` — ``.inv(lon1, lat1, lon2, lat2)`` → (az12, az21, dist)."""
    return Geod(ellps=ellps)
