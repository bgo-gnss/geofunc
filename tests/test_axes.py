"""geofunc.axes: re-signed, along/across, slant and tilted axes, with covariance."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from geofunc import axes
from geofunc.local import propagate

DEG = np.pi / 180.0


def _spd(seed: int = 0) -> np.ndarray:
    a = np.random.default_rng(seed).normal(size=(3, 3)) * 1e-3
    return a @ a.T + np.eye(3) * 1e-7


def test_import_is_numpy_only() -> None:
    code = (
        "import sys, geofunc.axes; "
        "print(','.join(m for m in ('pyproj', 'gps_parser', 'pandas', 'scipy') "
        "if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""


def test_sign_axes_plus_is_lengthening_in_every_component() -> None:
    # SVIE-SKFC-like: rover east, a little south and up of the reference
    a = axes.sign_axes([-1900.0, 9100.0, 600.0])
    assert np.allclose(np.diag(a.matrix), [-1.0, 1.0, 1.0])
    assert a.keys == ("north", "east", "up") and a.raw == (False, False, False)
    # moving further south and further east: both components lengthen
    assert np.allclose(axes.rotate([-0.01, 0.02, 0.0], a).values, [0.01, 0.02, 0.0])
    # the pair written the other way round shows the same plotted change
    b = axes.sign_axes([1900.0, -9100.0, -600.0])
    assert np.allclose(axes.rotate([0.01, -0.02, 0.0], b).values, [0.01, 0.02, 0.0])


def test_sign_axes_keeps_a_component_too_short_to_have_a_sign() -> None:
    a = axes.sign_axes([2582.0, -351.0, -0.4])
    assert np.allclose(np.diag(a.matrix), [1.0, -1.0, 1.0])
    assert a.raw == (False, False, True)
    assert axes.sign_axes([0.0, -5.0, 0.0], min_length=0.0).raw == (False,) * 3


def test_tilted_axes_reduce_to_neu_and_turn_clockwise() -> None:
    assert np.allclose(axes.along_axes(0.0).matrix, np.eye(3))
    east = axes.along_axes(90.0 * DEG)
    # along = east, across = 90° clockwise of east = south
    assert np.allclose(east.matrix[0], [0.0, 1.0, 0.0], atol=1e-15)
    assert np.allclose(east.matrix[1], [-1.0, 0.0, 0.0], atol=1e-15)
    assert east.keys == ("along", "across", "up")


@pytest.mark.parametrize("az,pl", [(0.3, 0.0), (4.0, 0.5), (2.0, -0.3)])
def test_tilted_axes_are_orthonormal_with_an_upward_third(az: float, pl: float) -> None:
    r = axes.tilted_axes(az, pl).matrix
    assert np.allclose(r @ r.T, np.eye(3))
    assert r[2, 2] > 0.0  # third axis points up
    assert r[1, 2] == 0.0  # second axis horizontal
    # positive plunge = first axis points down
    assert np.sign(r[0, 2]) == -np.sign(pl) or pl == 0.0


def test_slope_frame_normal_is_perpendicular_to_the_slope() -> None:
    dip, az = 28.0 * DEG, 210.0 * DEG
    r = axes.tilted_axes(az, dip, keys=("downslope", "across", "normal")).matrix
    downslope = r[0]
    assert np.isclose(np.degrees(np.arcsin(-downslope[2])), 28.0)
    assert np.isclose(np.degrees(np.arccos(r[2, 2])), 28.0)  # normal tilted by dip
    assert np.isclose(r[2] @ downslope, 0.0)


def test_slant_axes_first_axis_is_the_baseline_direction() -> None:
    b0 = np.array([-120.0, -690.0, -172.0])  # SVIN-SVIE-like, steep down
    a = axes.slant_axes(b0)
    assert np.allclose(a.matrix[0], b0 / np.linalg.norm(b0))
    assert a.keys == ("along", "across", "normal")
    # a 1 cm stretch along the line shows only on the first axis
    d = 0.01 * b0 / np.linalg.norm(b0)
    assert np.allclose(axes.rotate(d, a).values, [0.01, 0.0, 0.0], atol=1e-15)


def test_baseline_azimuth_and_elevation() -> None:
    assert np.isclose(np.degrees(axes.baseline_azimuth([1.0, 1.0, 0.0])), 45.0)
    assert np.isclose(np.degrees(axes.baseline_azimuth([1.0, -1.0, 0.0])), 315.0)
    assert np.isclose(np.degrees(axes.baseline_elevation([3.0, 4.0, 5.0])), 45.0)
    with pytest.raises(ValueError):
        axes.baseline_azimuth([0.0, 0.0, 10.0])


def test_along_keeps_the_horizontal_length_and_up() -> None:
    a = axes.along_axes(102.0 * DEG)
    d = np.random.default_rng(1).normal(size=(50, 3))
    out = axes.rotate(d, a).values
    assert np.allclose(out[:, 0] ** 2 + out[:, 1] ** 2, d[:, 0] ** 2 + d[:, 1] ** 2)
    assert np.allclose(out[:, 2], d[:, 2])


def test_rotate_carries_the_full_covariance_and_broadcasts() -> None:
    a = axes.tilted_axes(1.1, 0.4)
    c = np.stack([_spd(k) for k in range(4)])
    v = np.random.default_rng(2).normal(size=(4, 3))
    out = axes.rotate(v, a, cov=c)
    assert out.cov is not None and out.sigma is not None
    assert np.allclose(out.cov, propagate(a.matrix, c))
    assert np.allclose(out.sigma, np.sqrt(np.diagonal(out.cov, axis1=1, axis2=2)))
    # trace (total variance) is invariant under a rotation
    assert np.allclose(
        np.trace(out.cov, axis1=1, axis2=2), np.trace(c, axis1=1, axis2=2)
    )


def test_rotate_sigma_only_treats_components_as_uncorrelated() -> None:
    s = np.array([[1.0, 2.0, 3.0]])
    flipped = axes.rotate(np.zeros((1, 3)), axes.sign_axes([-5.0, 5.0, 5.0]), sigma=s)
    assert flipped.sigma is not None
    assert np.allclose(flipped.sigma, s)  # re-signing leaves sigmas alone
    along = axes.rotate(np.zeros((1, 3)), axes.along_axes(np.pi / 4), sigma=s)
    assert along.sigma is not None
    assert np.allclose(along.sigma[0, :2], np.sqrt((1.0 + 4.0) / 2.0))


def test_rotate_spreads_nan_only_into_mixed_components() -> None:
    v = np.array([[np.nan, 1.0, 2.0]])
    assert np.isnan(axes.rotate(v, axes.sign_axes([-5, 5, 5])).values[0]).tolist() == [
        True,
        False,
        False,
    ]
    out = axes.rotate(v, axes.along_axes(0.5)).values[0]
    assert np.isnan(out[:2]).all() and out[2] == 2.0


def test_bad_inputs_are_refused() -> None:
    with pytest.raises(ValueError):
        axes.sign_axes([1.0, 2.0])
    with pytest.raises(ValueError):
        axes.rotate([1.0, 2.0], axes.along_axes(0.0))
    with pytest.raises(ValueError):
        axes._checked(np.ones((3, 3)), ("a", "b", "c"))


def test_rotate_keeps_a_missing_sigma_out_of_unmixed_components() -> None:
    s = np.array([[1.0, 2.0, np.nan]])
    out = axes.rotate(np.zeros((1, 3)), axes.along_axes(0.7), sigma=s)
    assert out.sigma is not None and out.cov is not None
    assert np.isfinite(out.sigma[0, :2]).all() and np.isnan(out.sigma[0, 2])
    assert np.isfinite(out.cov[0, :2, :2]).all() and np.isnan(out.cov[0, 2]).all()
