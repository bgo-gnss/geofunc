"""Local N/E/U → any other set of axes, with the covariance.

A baseline or displacement in local north/east/up can be shown along other
axes: re-signed so that + is lengthening, along/across the baseline, along
the 3-D (slant) line between the stations, along the motion, or in a slope's
own frame. Each is a fixed 3×3 matrix ``R`` whose rows are the new axes
written in N/E/U, applied as ``d′ = R·d`` and ``C′ = R·C·Rᵀ``
(:func:`rotate`). The builders here only make ``R``; estimating a direction
from data belongs to the analysis packages.

Conventions (binding for this module, as in :mod:`geofunc.local`)
------------------------------------------------------------------
- Input and output component order is N/E/U; arrays ``(..., 3)`` and
  covariances ``(..., 3, 3)`` broadcast over leading axes; float64.
- Angles in **radians**. Azimuth clockwise from north. Plunge positive
  DOWNWARD from the horizontal (the structural-geology convention), so a
  slope's downslope direction has plunge = dip > 0.
- The second axis is always horizontal and 90° CLOCKWISE of the first's
  azimuth ("across", + to the right looking along the first axis); the third
  completes the set and points up. With azimuth 0 and plunge 0 the axes are
  exactly N/E/U.
- Pure numpy, no I/O.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from geofunc.local import propagate

__all__ = [
    "Axes",
    "Rotated",
    "along_axes",
    "baseline_azimuth",
    "baseline_elevation",
    "rotate",
    "sign_axes",
    "slant_axes",
    "tilted_axes",
]

FloatArray = NDArray[np.float64]
Keys = tuple[str, str, str]

#: a baseline component shorter than this [m] has no meaningful sign
SIGN_MIN_LENGTH = 1.0
_ORTHO_TOL = 1e-9


class Axes(NamedTuple):
    """A set of axes: ``matrix`` rows are the new axes in N/E/U.

    ``keys`` name the components (``("north", "east", "up")``,
    ``("along", "across", "up")`` …); ``raw[i]`` marks a component
    :func:`sign_axes` left as it was (too short to carry a sign).
    """

    matrix: FloatArray
    keys: Keys
    raw: tuple[bool, bool, bool] = (False, False, False)


class Rotated(NamedTuple):
    """:func:`rotate` result; ``cov``/``sigma`` are None without input ones."""

    values: FloatArray
    cov: FloatArray | None
    sigma: FloatArray | None


def _vector(b0: ArrayLike) -> FloatArray:
    v = np.asarray(b0, dtype=np.float64)
    if v.shape != (3,):
        raise ValueError(f"expected one N/E/U vector, shape (3,), got {v.shape}")
    return v


def _checked(matrix: ArrayLike, keys: Keys) -> Axes:
    r = np.asarray(matrix, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError(f"axes matrix: expected (3, 3), got {r.shape}")
    if not np.allclose(r @ r.T, np.eye(3), atol=_ORTHO_TOL):
        raise ValueError("axes matrix is not orthonormal")
    return Axes(r, keys)


def baseline_azimuth(b0: ArrayLike) -> float:
    """Azimuth [rad, 0..2π, clockwise from N] of the horizontal part of ``b0``."""
    v = _vector(b0)
    if np.hypot(v[0], v[1]) == 0.0:
        raise ValueError("a vertical baseline has no azimuth")
    return float(np.arctan2(v[1], v[0]) % (2.0 * np.pi))


def baseline_elevation(b0: ArrayLike) -> float:
    """Elevation angle [rad] of ``b0`` above the horizontal (+ when it rises)."""
    v = _vector(b0)
    return float(np.arctan2(v[2], np.hypot(v[0], v[1])))


def sign_axes(b0: ArrayLike, *, min_length: float = SIGN_MIN_LENGTH) -> Axes:
    """N/E/U re-signed so that + is the baseline LENGTHENING in that component.

    Each component takes the sign of the a-priori baseline ``b0`` (rover −
    reference, N/E/U metres) in that component: a growing |component| plots
    positive whichever way round the pair is written. A component shorter
    than ``min_length`` keeps its rover − reference sign and is flagged in
    ``raw``. A reflection (determinant −1 for an odd number of flips), not a
    rotation — the covariance law is the same.
    """
    v = _vector(b0)
    short = np.abs(v) < min_length
    signs = np.where(short | (v >= 0.0), 1.0, -1.0)
    raw = (bool(short[0]), bool(short[1]), bool(short[2]))
    return Axes(np.diag(signs), ("north", "east", "up"), raw)


def tilted_axes(
    azimuth: float,
    plunge: float = 0.0,
    *,
    keys: Keys = ("along", "across", "up"),
) -> Axes:
    """Axes whose first one points to ``azimuth``, ``plunge`` below horizontal.

    First axis ``(cos p cos a, cos p sin a, −sin p)``; second the horizontal
    ``(−sin a, cos a, 0)`` (90° clockwise); third ``(sin p cos a,
    sin p sin a, cos p)``, perpendicular to both and pointing up. ``plunge=0``
    gives along/across/up; a slope's frame is ``tilted_axes(downslope_az,
    dip, keys=("downslope", "across", "normal"))``, the third axis then being
    the slope normal.
    """
    ca, sa = np.cos(azimuth), np.sin(azimuth)
    cp, sp = np.cos(plunge), np.sin(plunge)
    rows = [[cp * ca, cp * sa, -sp], [-sa, ca, 0.0], [sp * ca, sp * sa, cp]]
    return _checked(rows, keys)


def along_axes(azimuth: float) -> Axes:
    """Along/across/up for a horizontal direction (a baseline, a motion)."""
    return tilted_axes(azimuth, 0.0)


def slant_axes(b0: ArrayLike) -> Axes:
    """Along the 3-D line between the stations: (along, across, normal).

    The first axis is ``b0``'s unit vector (its length change is what a tape
    between the antennas measures), the second horizontal across it, the
    third perpendicular to both, upward.
    """
    return tilted_axes(
        baseline_azimuth(b0),
        -baseline_elevation(b0),
        keys=("along", "across", "normal"),
    )


def rotate(
    values: ArrayLike,
    axes: Axes | ArrayLike,
    *,
    cov: ArrayLike | None = None,
    sigma: ArrayLike | None = None,
) -> Rotated:
    """``values (..., 3)`` along ``axes``, with ``cov`` or ``sigma`` carried.

    ``cov (..., 3, 3)`` is rotated in full (``R·C·Rᵀ``). ``sigma (..., 3)``
    alone is taken as uncorrelated, which is exact only when ``R`` is
    diagonal; the output ``sigma`` is always the root of the rotated
    covariance's diagonal. NaN in a component spreads only to the output
    components that mix it (a re-signing keeps a missing Up out of N and E;
    plain ``R·d`` would spread it everywhere, NaN·0 being NaN).
    """
    r = axes.matrix if isinstance(axes, Axes) else np.asarray(axes, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if v.ndim < 1 or v.shape[-1] != 3:
        raise ValueError(f"values: expected shape (..., 3), got {v.shape}")
    uses = r != 0.0  # output i mixes input j
    out: FloatArray = np.matmul(np.nan_to_num(v), r.T)
    out[_mixed(np.isnan(v), uses)] = np.nan
    if cov is None and sigma is None:
        return Rotated(out, None, None)
    if cov is not None:
        c = np.asarray(cov, dtype=np.float64)
        bad = np.isnan(c).any(axis=-1)
    else:
        s = np.asarray(sigma, dtype=np.float64)
        c = s[..., :, None] * np.eye(3) * s[..., None, :]
        bad = np.isnan(s)
    c2 = propagate(r, np.nan_to_num(c))
    lost = _mixed(bad, uses)
    c2[lost[..., :, None] | lost[..., None, :]] = np.nan
    return Rotated(out, c2, np.sqrt(np.diagonal(c2, axis1=-2, axis2=-1)))


def _mixed(bad: NDArray[np.bool_], uses: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Output components that take a bad (NaN) input component."""
    out: NDArray[np.bool_] = (bad[..., None, :] & uses).any(axis=-1)
    return out
