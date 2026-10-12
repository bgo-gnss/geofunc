"""Plate-motion models: absolute Euler poles and the velocities they predict.

The one piece of frame bookkeeping PROJ/EPSG cannot supply — EPSG has no
plate-motion models — so the ITRF PMM tables live here as data, and the
velocity maths is :func:`geofunc.local.euler_velocity` (``v = ω × x``). Which
plate a station sits on is configuration (gps-config-data ``platefile``),
read by geo_dataread, not by this module.

Units and conventions
---------------------
- Tables are the published Cartesian rotation-vector components
  ``(ωx, ωy, ωz)`` in **mas/yr**; :func:`euler_pole` returns **rad/yr**, so
  with ECEF metres the velocities come out in **m/yr**.
- Velocities are in ECEF (``Local.d`` of :func:`plate_velocity`); rotate
  them to the station's local frame with :func:`plate_velocity_local`
  (``order="neu"`` default, like every ``geofunc.local`` result). The old
  ``geofunc.geofunc.plateVelo`` returned **E, N, U** under an N, E, U
  docstring and ``geo_dataread`` compensated for it; nothing here swaps.
- Like ``geofunc.local`` this module is pure numpy and never imports pyproj.

Models
------
``itrf2008`` — ITRF2008-PMM, Altamimi, Métivier & Collilieux 2012, JGR 117,
B07402, doi:10.1029/2011JB008930, Table 3 (the five plates the production
code has used since the legacy ``geo.platePoles``; values not re-checked
against the PDF, which the publisher blocks).

``itrf2014`` — ITRF2014-PMM, Altamimi, Métivier, Rebischung, Rouby &
Collilieux 2017, GJI 209, 1906–1912, doi:10.1093/gji/ggx136, Table 1
(checked 2026-10-12). No origin rate bias (not significant, §4).

``itrf2020`` — ITRF2020-PMM, Altamimi, Métivier, Rebischung, Collilieux,
Chanard & Barnéoud 2023, GRL 50, e2023GL106373,
doi:10.1029/2023GL106373, Table 1 (poles, 1σ) and Table 2 (origin rate
bias, ORB) (checked 2026-10-12; IGN ``ITRF2020-PMM.dat`` agrees). The
authors advise adding the ORB to the horizontal velocities for full
ITRF2020 consistency and ignoring its vertical part — ``orb=True``.

The EURA poles are pinned against PROJ in the tests: ITRF→ETRF is the
EURA plate rotation with the opposite sign, so a PROJ/EPSG update that
changes ETRF catches a stale table here.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from geofunc import local
from geofunc.local import FloatArray, Local, Order

__all__ = [
    "MAS_TO_RAD",
    "ORB",
    "PMM",
    "PMM_SIGMA",
    "PLATE_NAMES",
    "euler_pole",
    "frames",
    "plate_velocity",
    "plate_name",
    "plate_velocity_local",
    "plates",
    "pole_covariance",
]

Units = Literal["rad/yr", "mas/yr", "deg/yr", "deg/Myr"]

#: milliarcseconds → radians
MAS_TO_RAD = np.pi / (180.0 * 3600.0 * 1000.0)

#: ``{frame: {plate: (ωx, ωy, ωz)}}`` [mas/yr]
PMM: dict[str, dict[str, tuple[float, float, float]]] = {
    "itrf2008": {
        "EURA": (-0.083, -0.534, 0.750),
        "NOAM": (0.035, -0.662, -0.100),
        "NAZC": (-0.330, -1.551, 1.625),
        "INDI": (1.232, 0.303, 1.540),
        "NUBI": (0.095, -0.598, 0.723),
    },
    "itrf2014": {
        "ANTA": (-0.248, -0.324, 0.675),
        "ARAB": (1.154, -0.136, 1.444),
        "AUST": (1.510, 1.182, 1.215),
        "EURA": (-0.085, -0.531, 0.770),
        "INDI": (1.154, -0.005, 1.454),
        "NAZC": (-0.333, -1.544, 1.623),
        "NOAM": (0.024, -0.694, -0.063),
        "NUBI": (0.099, -0.614, 0.733),
        "PCFC": (-0.409, 1.047, -2.169),
        "SOAM": (-0.270, -0.301, -0.140),
        "SOMA": (-0.121, -0.794, 0.884),
    },
    "itrf2020": {
        "AMUR": (-0.131, -0.551, 0.837),
        "ANTA": (-0.269, -0.312, 0.678),
        "ARAB": (1.129, -0.146, 1.438),
        "AUST": (1.487, 1.175, 1.223),
        "CARB": (0.207, -1.422, 0.726),
        "EURA": (-0.085, -0.519, 0.753),
        "INDI": (1.137, 0.013, 1.444),
        "NAZC": (-0.327, -1.561, 1.605),
        "NOAM": (0.045, -0.666, -0.098),
        "NUBI": (0.090, -0.585, 0.717),
        "PCFC": (-0.404, 1.021, -2.154),
        "SOAM": (-0.261, -0.282, -0.157),
        "SOMA": (-0.081, -0.719, 0.864),
    },
}

#: published 1σ of the pole components [mas/yr] (ITRF2020-PMM only)
PMM_SIGMA: dict[str, dict[str, tuple[float, float, float]]] = {
    "itrf2020": {
        "AMUR": (0.009, 0.014, 0.015),
        "ANTA": (0.003, 0.003, 0.004),
        "ARAB": (0.025, 0.027, 0.016),
        "AUST": (0.003, 0.003, 0.003),
        "CARB": (0.076, 0.174, 0.061),
        "EURA": (0.003, 0.003, 0.003),
        "INDI": (0.008, 0.040, 0.016),
        "NAZC": (0.006, 0.018, 0.008),
        "NOAM": (0.003, 0.003, 0.003),
        "NUBI": (0.003, 0.003, 0.004),
        "PCFC": (0.003, 0.003, 0.004),
        "SOAM": (0.004, 0.004, 0.003),
        "SOMA": (0.014, 0.015, 0.005),
    },
}

#: origin rate bias ``(Ṫx, Ṫy, Ṫz)`` [m/yr] and its 1σ (ITRF2020-PMM Table 2)
ORB: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "itrf2020": ((0.37e-3, 0.35e-3, 0.74e-3), (0.08e-3, 0.10e-3, 0.09e-3)),
}

#: plate code → name, for titles and labels (the legacy ``plateFullname``
#: strings for the plates IMO plots, kept verbatim so figure titles do not move)
PLATE_NAMES: dict[str, str] = {
    "AMUR": "Amurian Plate",
    "ANTA": "Antarctic Plate",
    "ARAB": "Arabia",
    "AUST": "Australian Plate",
    "CARB": "Caribbean Plate",
    "EURA": "Eurasian Plate",
    "INDI": "Indian Plate",
    "NAZC": "Nazca",
    "NOAM": "North American plate",
    "NUBI": "Nubian Plate",
    "PCFC": "Pacific Plate",
    "SOAM": "South American Plate",
    "SOMA": "Somalian Plate",
}

_TO_MAS = {
    "mas/yr": 1.0,
    "rad/yr": MAS_TO_RAD,
    "deg/yr": 1.0 / 3.6e6,
    "deg/Myr": 1.0 / 3.6,
}


def _frame(frame: str) -> str:
    key = frame.lower()
    if key not in PMM:
        raise KeyError(f"no plate-motion model for {frame!r}; have {frames()}")
    return key


def frames() -> tuple[str, ...]:
    """Frames with a plate-motion model."""
    return tuple(PMM)


def plates(frame: str = "itrf2008") -> tuple[str, ...]:
    """Plate codes in a frame's model."""
    return tuple(PMM[_frame(frame)])


def plate_name(plate: str) -> str:
    """Name of a plate code (``"EURA"`` → ``"Eurasian Plate"``)."""
    try:
        return PLATE_NAMES[plate.upper()]
    except KeyError:
        raise KeyError(f"unknown plate code {plate!r}") from None


def euler_pole(
    plate: str, frame: str = "itrf2008", *, units: Units = "rad/yr"
) -> FloatArray:
    """Cartesian rotation vector ``(3,)`` of ``plate`` in ``frame``'s PMM."""
    table = PMM[_frame(frame)]
    try:
        w = np.asarray(table[plate.upper()], dtype=np.float64)
    except KeyError:
        raise KeyError(
            f"plate {plate!r} not in the {frame} model; have {tuple(table)}"
        ) from None
    out: FloatArray = w * _TO_MAS[units]
    return out


def pole_covariance(plate: str, frame: str = "itrf2020") -> FloatArray | None:
    """Diagonal pole covariance ``(3, 3)`` [(rad/yr)²], or None if unpublished.

    The papers give per-component σ only; the off-diagonal terms are not
    published, so this is the diagonal.
    """
    sig = PMM_SIGMA.get(_frame(frame), {}).get(plate.upper())
    if sig is None:
        return None
    s = np.asarray(sig, dtype=np.float64) * MAS_TO_RAD
    out: FloatArray = np.diag(s**2)
    return out


def plate_velocity(
    plate: str,
    xyz: ArrayLike,
    *,
    frame: str = "itrf2008",
    orb: bool = False,
    with_cov: bool = False,
) -> Local:
    """ECEF velocity ``(..., 3)`` [m/yr] of points ``xyz`` riding on ``plate``.

    ``orb=True`` adds the model's origin rate bias (ITRF2020-PMM only; a
    model without one raises). ``with_cov=True`` propagates the published
    pole σ (and the ORB σ when added); ``Local.cov`` is None without it or
    when the model publishes none.
    """
    key = _frame(frame)
    w = euler_pole(plate, key)
    cw = pole_covariance(plate, key) if with_cov else None
    v = local.euler_velocity(w, xyz, cov_omega=cw)
    if not orb:
        return v
    if key not in ORB:
        raise ValueError(f"the {frame} plate-motion model has no origin rate bias")
    t, st = (np.asarray(a, dtype=np.float64) for a in ORB[key])
    d: FloatArray = v.d + t
    if v.cov is None:
        return Local(d, None)
    cov: FloatArray = v.cov + np.diag(st**2)
    return Local(d, cov)


def plate_velocity_local(
    plate: str,
    xyz: ArrayLike,
    *,
    frame: str = "itrf2008",
    order: Order = "neu",
    orb: bool = False,
    with_cov: bool = False,
    ellipsoid: local.Ellipsoid = local.GRS80,
) -> Local:
    """Plate velocity rotated to each point's local frame [m/yr].

    Components in ``order`` (default N, E, U). With ``orb=True`` the
    vertical part of the ORB is kept; the ITRF2020-PMM authors advise
    ignoring it — use the horizontal components.
    """
    v = plate_velocity(plate, xyz, frame=frame, orb=orb, with_cov=with_cov)
    g = local.ecef_to_geodetic(xyz, ellipsoid=ellipsoid)
    return local.ecef_to_local(v.d, g.lat, g.lon, cov=v.cov, order=order)
