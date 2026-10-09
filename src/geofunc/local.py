"""ECEF ↔ geodetic ↔ local (N/E/U) transformations, each with its covariance.

The working representation of a GNSS position is geocentric ECEF ``(X, Y, Z)``
with a 3×3 ECEF covariance (one station) or a 3n×3n covariance (a network
day). Everything else — geodetic coordinates, local north/east/up
displacements, baselines, frame (Helmert) changes — is a transformation of
that, and every function here transforms the covariance with it by the
linear (or linearised) law ``C′ = J·C·Jᵀ`` with the exact Jacobian ``J``.

Conventions (binding for this module)
-------------------------------------
- Angles in **radians**, lengths in **metres**. Geodetic results are the
  named tuple :class:`Geodetic` ``(lat, lon, h)`` — never rely on position
  (``geo.xyzell`` silently swapped lat/lon order once; names cannot).
- Local component order is ``order="neu"`` (IMO default, the order of the
  GAMIT/GLOBK ``.dat1/2/3`` files) or ``"enu"``.
- Arrays broadcast over leading axes: points ``(..., 3)``, covariances
  ``(..., 3, 3)``. Inputs are never mutated; outputs are float64.
- Pure numpy, no I/O, no pyproj import (this module must stay cheap to
  import for the leaf analysis packages).

Local displacement conventions
------------------------------
``convention="geodetic"`` (topocentric): ``d = R(φ₀, λ₀)·(x − x₀)``, the
standard rotation at the reference point (Hofmann-Wellenhof et al. 2008,
GNSS, §5.6.2 eq. 5.111). Exactly linear in x.

``convention="globk"``: the ``Loc.`` coordinates glorg prints and the
production TOT ``mb_*_TOT.dat`` series are built from:
``N = a·φ``, ``E = a·cos φ₀·λ``, ``U = h`` (a = semi-major axis), so
``dN = a·Δφ``, ``dE = a·cos φ₀·Δλ``, ``dU = Δh``. Compared with the
geodetic local frame this scales N by ``a/(M+h)`` and E by
``a/(N+h)`` — about −0.3 % at Iceland's latitudes. Verified 2026-10-09 on
217 stations against glorg 10.71 ``Loc.`` lines across the 2026-09-11
frame jump (median ratio 1.000008 in E, 0.99999 in N). NB the GLOBK
Reference Manual (10.6, §print files) describes E as Δλ times the radius
"at the nearest 1 degree latitude line"; the 10.71 output does not do
that. The absolute ``Loc.`` E value carries a per-station constant that
differs from ``a·cos φ·λ`` by ~1e-5 relative — irrelevant for
displacements (only differences are used), so it is not reproduced.

References
----------
- Hofmann-Wellenhof, Lichtenegger & Wasle 2008, *GNSS*, Springer, §5.6
  (ECEF↔ellipsoidal, local-level frame).
- Bowring 1976, Survey Review 23(181), 323–327 (latitude iteration start).
- Petit & Luzum (eds.) 2010, IERS Conventions, TN 36, §4.2.5 eq. 4.13
  (7-parameter similarity transformation, small-angle form).
- Herring, King, Floyd & McClusky 2015, GLOBK Reference Manual 10.6 (Loc.).
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "GRS80",
    "WGS84",
    "Ellipsoid",
    "Geodetic",
    "Local",
    "baseline",
    "displacements",
    "ecef_to_geodetic",
    "ecef_to_local",
    "geodetic_jacobian",
    "geodetic_to_ecef",
    "helmert_apply",
    "helmert_design",
    "local_rotation",
    "local_to_ecef",
    "propagate",
    "radii",
]

FloatArray = NDArray[np.float64]
Order = Literal["neu", "enu"]
Convention = Literal["geodetic", "globk"]


class Ellipsoid(NamedTuple):
    """Reference ellipsoid: semi-major axis ``a`` [m] and flattening ``f``."""

    a: float
    f: float

    @property
    def e2(self) -> float:
        """First eccentricity squared."""
        return self.f * (2.0 - self.f)


#: ITRF/ISN ellipsoid (EPSG:7019).
GRS80 = Ellipsoid(6378137.0, 1.0 / 298.257222101)
#: WGS84 (EPSG:7030) — differs from GRS80 by 0.1 mm in the semi-minor axis.
WGS84 = Ellipsoid(6378137.0, 1.0 / 298.257223563)


class Geodetic(NamedTuple):
    """Geodetic latitude, longitude [rad] and ellipsoidal height [m]."""

    lat: FloatArray
    lon: FloatArray
    h: FloatArray


class Local(NamedTuple):
    """Local components ``d`` (..., 3) and their covariance (..., 3, 3) or None."""

    d: FloatArray
    cov: FloatArray | None


_GLOBK_POLE_COS = 1e-6  # |cos φ₀| below this: globk east is undefined


def _points(x: ArrayLike, name: str, width: int = 3) -> FloatArray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim < 1 or a.shape[-1] != width:
        raise ValueError(f"{name}: expected shape (..., {width}), got {a.shape}")
    return a


def _covs(c: ArrayLike, name: str, width: int = 3) -> FloatArray:
    a = np.asarray(c, dtype=np.float64)
    if a.ndim < 2 or a.shape[-2:] != (width, width):
        raise ValueError(
            f"{name}: expected shape (..., {width}, {width}), got {a.shape}"
        )
    return a


def _order_perm(order: str) -> list[int]:
    """Row permutation from the internal NEU order."""
    if order == "neu":
        return [0, 1, 2]
    if order == "enu":
        return [1, 0, 2]
    raise ValueError(f"order must be 'neu' or 'enu', got {order!r}")


def _wrap(dlon: FloatArray) -> FloatArray:
    return (dlon + np.pi) % (2.0 * np.pi) - np.pi


def propagate(jac: ArrayLike, cov: ArrayLike) -> FloatArray:
    """``J·C·Jᵀ`` batched over leading axes (the linear covariance law)."""
    j = np.asarray(jac, dtype=np.float64)
    c = np.asarray(cov, dtype=np.float64)
    out: FloatArray = np.matmul(np.matmul(j, c), np.swapaxes(j, -1, -2))
    return out


def radii(
    lat: ArrayLike, *, ellipsoid: Ellipsoid = GRS80
) -> tuple[FloatArray, FloatArray]:
    """Meridian ``M`` and prime-vertical ``N`` radii of curvature [m] at ``lat``."""
    s2 = np.sin(np.asarray(lat, dtype=np.float64)) ** 2
    w = np.sqrt(1.0 - ellipsoid.e2 * s2)
    return ellipsoid.a * (1.0 - ellipsoid.e2) / w**3, ellipsoid.a / w


def ecef_to_geodetic(xyz: ArrayLike, *, ellipsoid: Ellipsoid = GRS80) -> Geodetic:
    """ECEF ``(..., 3)`` → :class:`Geodetic` ``(lat, lon, h)``.

    Fixed-point latitude iteration from Bowring's start value; the height
    uses the pole-safe form ``h = p·cos φ + z·sin φ − a·√(1 − e² sin² φ)``.
    Converges below 1e-15 rad within 3–4 steps for terrestrial points.
    """
    x = _points(xyz, "xyz")
    X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
    a, e2 = ellipsoid.a, ellipsoid.e2
    b = a * (1.0 - ellipsoid.f)
    ep2 = e2 / (1.0 - e2)
    p = np.hypot(X, Y)
    lon = np.arctan2(Y, X)
    u = np.arctan2(Z * a, p * b)  # Bowring 1976 start
    lat = np.arctan2(Z + ep2 * b * np.sin(u) ** 3, p - e2 * a * np.cos(u) ** 3)
    for _ in range(8):
        _, n = radii(lat, ellipsoid=ellipsoid)
        new = np.arctan2(Z + e2 * n * np.sin(lat), p)
        done = np.max(np.abs(new - lat), initial=0.0) < 1e-15
        lat = new
        if done:
            break
    s = np.sin(lat)
    h = p * np.cos(lat) + Z * s - a * np.sqrt(1.0 - e2 * s**2)
    return Geodetic(lat, lon, h)


def geodetic_to_ecef(
    lat: ArrayLike, lon: ArrayLike, h: ArrayLike, *, ellipsoid: Ellipsoid = GRS80
) -> FloatArray:
    """Geodetic ``lat, lon`` [rad], ``h`` [m] → ECEF ``(..., 3)``."""
    lat_, lon_, h_ = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (lat, lon, h))
    )
    _, n = radii(lat_, ellipsoid=ellipsoid)
    c = np.cos(lat_)
    return np.stack(
        [
            (n + h_) * c * np.cos(lon_),
            (n + h_) * c * np.sin(lon_),
            (n * (1.0 - ellipsoid.e2) + h_) * np.sin(lat_),
        ],
        axis=-1,
    )


def local_rotation(
    lat: ArrayLike, lon: ArrayLike, *, order: Order = "neu"
) -> FloatArray:
    """Rotation ``(..., 3, 3)`` taking an ECEF vector to local components.

    Rows (NEU): ``N = [−sφ cλ, −sφ sλ, cφ]``, ``E = [−sλ, cλ, 0]``,
    ``U = [cφ cλ, cφ sλ, sφ]`` (Hofmann-Wellenhof et al. 2008 eq. 5.111).
    Orthonormal, so its transpose is the inverse (:func:`local_to_ecef`).
    """
    lat_, lon_ = np.broadcast_arrays(
        np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
    )
    sp, cp, sl, cl = np.sin(lat_), np.cos(lat_), np.sin(lon_), np.cos(lon_)
    z = np.zeros_like(sp)
    r = np.stack(
        [
            np.stack([-sp * cl, -sp * sl, cp], axis=-1),
            np.stack([-sl, cl, z], axis=-1),
            np.stack([cp * cl, cp * sl, sp], axis=-1),
        ],
        axis=-2,
    )
    return r[..., _order_perm(order), :]


def geodetic_jacobian(xyz: ArrayLike, *, ellipsoid: Ellipsoid = GRS80) -> FloatArray:
    """``∂(lat, lon, h)/∂(X, Y, Z)`` ``(..., 3, 3)`` at ``xyz``.

    ``dφ = dN/(M+h)``, ``dλ = dE/((N+h)·cos φ)``, ``dh = dU``, so
    ``J = diag(1/(M+h), 1/((N+h) cos φ), 1)·R_neu``. Propagate an ECEF
    covariance to geodetic with ``propagate(geodetic_jacobian(x), C)``.
    """
    g = ecef_to_geodetic(xyz, ellipsoid=ellipsoid)
    m, n = radii(g.lat, ellipsoid=ellipsoid)
    scale = np.stack(
        [1.0 / (m + g.h), 1.0 / ((n + g.h) * np.cos(g.lat)), np.ones_like(g.h)], -1
    )
    return scale[..., :, None] * local_rotation(g.lat, g.lon)


def ecef_to_local(
    dxyz: ArrayLike,
    lat: ArrayLike,
    lon: ArrayLike,
    *,
    cov: ArrayLike | None = None,
    order: Order = "neu",
) -> Local:
    """Rotate ECEF vectors (and their covariance) into the local frame at ``lat, lon``."""
    v = _points(dxyz, "dxyz")
    r = local_rotation(lat, lon, order=order)
    d = np.matmul(r, v[..., None])[..., 0]
    return Local(d, None if cov is None else propagate(r, _covs(cov, "cov")))


def local_to_ecef(
    dlocal: ArrayLike,
    lat: ArrayLike,
    lon: ArrayLike,
    *,
    cov: ArrayLike | None = None,
    order: Order = "neu",
) -> Local:
    """Inverse of :func:`ecef_to_local` (``Local.d`` is then ECEF)."""
    v = _points(dlocal, "dlocal")
    rt = np.swapaxes(local_rotation(lat, lon, order=order), -1, -2)
    d = np.matmul(rt, v[..., None])[..., 0]
    return Local(d, None if cov is None else propagate(rt, _covs(cov, "cov")))


def displacements(
    xyz: ArrayLike,
    ref: ArrayLike,
    *,
    cov: ArrayLike | None = None,
    order: Order = "neu",
    convention: Convention = "geodetic",
    ellipsoid: Ellipsoid = GRS80,
) -> Local:
    """Local displacements of ECEF positions ``xyz (..., 3)`` from ``ref (3,)``.

    ``convention="geodetic"``: ``R(ref)·(x − ref)``, linear; covariance
    ``R C Rᵀ`` with the fixed reference rotation.
    ``convention="globk"``: ``(a·Δφ, a·cos φ₀·Δλ, Δh)`` from exact geodetic
    differences, reproducing glorg ``Loc.``/TOT; covariance through the
    Jacobian at each epoch, ``diag(a/(M+h), a cos φ₀/((N+h) cos φ), 1)·R_neu``.
    """
    x = _points(xyz, "xyz")
    r0 = _points(ref, "ref")
    if r0.shape != (3,):
        raise ValueError(f"ref: expected one point (3,), got {r0.shape}")
    c = None if cov is None else _covs(cov, "cov")
    g0 = ecef_to_geodetic(r0, ellipsoid=ellipsoid)
    perm = _order_perm(order)
    if convention == "geodetic":
        return ecef_to_local(x - r0, g0.lat, g0.lon, cov=c, order=order)
    if convention != "globk":
        raise ValueError(
            f"convention must be 'geodetic' or 'globk', got {convention!r}"
        )
    cos0 = float(np.cos(g0.lat))
    if abs(cos0) < _GLOBK_POLE_COS:
        raise ValueError("convention='globk' is undefined at the poles (cos φ₀ ≈ 0)")
    a = ellipsoid.a
    g = ecef_to_geodetic(x, ellipsoid=ellipsoid)
    d = np.stack(
        [a * (g.lat - g0.lat), a * cos0 * _wrap(g.lon - g0.lon), g.h - g0.h], -1
    )
    out_cov = None
    if c is not None:
        m, n = radii(g.lat, ellipsoid=ellipsoid)
        scale = np.stack(
            [a / (m + g.h), a * cos0 / ((n + g.h) * np.cos(g.lat)), np.ones_like(g.h)],
            -1,
        )
        jac = scale[..., :, None] * local_rotation(g.lat, g.lon)
        out_cov = propagate(jac[..., perm, :], c)
    return Local(d[..., perm], out_cov)


def baseline(
    xyz_a: ArrayLike,
    xyz_b: ArrayLike,
    *,
    cov: ArrayLike | None = None,
    order: Order = "neu",
    at: Literal["a", "b"] = "a",
    ellipsoid: Ellipsoid = GRS80,
) -> Local:
    """Baseline ``a − b`` in the local frame at station ``at``.

    ``cov`` is the JOINT ``(..., 6, 6)`` covariance of ``[x_a; x_b]``, so the
    cross term counts once: ``J = [R, −R]`` gives
    ``R (C_aa + C_bb − C_ab − C_ba) Rᵀ``. With two independent 3×3 blocks,
    build it as ``block_diag(C_aa, C_bb)`` (the quadrature result).
    """
    xa, xb = _points(xyz_a, "xyz_a"), _points(xyz_b, "xyz_b")
    if at not in ("a", "b"):
        raise ValueError(f"at must be 'a' or 'b', got {at!r}")
    g = ecef_to_geodetic(xa if at == "a" else xb, ellipsoid=ellipsoid)
    r = local_rotation(g.lat, g.lon, order=order)
    d = np.matmul(r, (xa - xb)[..., None])[..., 0]
    if cov is None:
        return Local(d, None)
    c = _covs(cov, "cov", width=6)
    return Local(d, propagate(np.concatenate([r, -r], axis=-1), c))


def helmert_design(xyz: ArrayLike) -> FloatArray:
    """Small-angle 7-parameter design ``(..., 3, 7)`` at ``xyz``.

    Parameters ``p = (tx, ty, tz, D, rx, ry, rz)`` [m, unitless, rad], IERS
    TN 36 eq. 4.13: ``x′ = x + T + D·x + R·x`` with
    ``R = [[0, −rz, ry], [rz, 0, −rx], [−ry, rx, 0]]``, i.e. ``x′ = x + A·p``.
    Drop columns for a 6-parameter (no scale) or 3-parameter fit.
    """
    x = _points(xyz, "xyz")
    X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
    o, z = np.ones_like(X), np.zeros_like(X)
    rows = [
        [o, z, z, X, z, Z, -Y],
        [z, o, z, Y, -Z, z, X],
        [z, z, o, Z, Y, -X, z],
    ]
    return np.stack([np.stack(r, axis=-1) for r in rows], axis=-2)


def helmert_apply(
    xyz: ArrayLike,
    params: ArrayLike,
    *,
    cov: ArrayLike | None = None,
    params_cov: ArrayLike | None = None,
) -> Local:
    """Apply a small-angle 7-parameter transformation (and its covariance).

    ``x′ = x + A(x)·p`` (:func:`helmert_design`). The point covariance maps
    through ``∂x′/∂x = (1+D)·I + R``; parameter uncertainty adds
    ``A·C_p·Aᵀ``. With per-point ``cov (..., 3, 3)`` the result is per-point
    blocks: the cross-point correlation ``A_i C_p A_jᵀ`` that a common
    parameter set induces is NOT represented — build the full network
    matrix from :func:`helmert_design` when it matters.
    ``Local.d`` holds the transformed ECEF positions.
    """
    x = _points(xyz, "xyz")
    p = np.asarray(params, dtype=np.float64)
    if p.shape != (7,):
        raise ValueError(f"params: expected (7,), got {p.shape}")
    a = helmert_design(x)
    out = x + np.matmul(a, p)
    if cov is None and params_cov is None:
        return Local(out, None)
    _, _, _, dsc, rx, ry, rz = p
    jx = np.array([[1 + dsc, -rz, ry], [rz, 1 + dsc, -rx], [-ry, rx, 1 + dsc]])
    c = np.zeros(x.shape + (3,)) if cov is None else propagate(jx, _covs(cov, "cov"))
    if params_cov is not None:
        c = c + propagate(a, _covs(params_cov, "params_cov", width=7))
    return Local(out, c)
