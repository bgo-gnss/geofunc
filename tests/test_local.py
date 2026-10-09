"""geofunc.local: ECEF ↔ geodetic ↔ local transformations with covariance.

Reference values marked "glorg" are copied from production glorg 10.71
output on okada (``icel.org`` of 2026-09-10/11 and the OUT_SOL stabilised
solution of 2026-10-07), so the GLOBK convention is pinned to the program
that writes the TOT series, not to its manual.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from geofunc import local

# --- glorg reference data -------------------------------------------------
# "N. SITE X/Y/Z coordinate (m)" and "Loc. SITE N/E/U coordinate (m)" values
# on 2026-09-10 (doy 253) and 2026-09-11 (doy 254, the +84 mm east jump).
GLORG = {
    "SEY1": {
        253: (
            (2597505.40166, -647383.40162, 5769976.96468),
            (7264464.04639, 16121403.33908, 196.82578),
        ),
        254: (
            (2597505.45100, -647383.33546, 5769976.96806),
            (7264464.01890, 16121403.41500, 196.84219),
        ),
    },
    "AKUR": {
        253: (
            (2502918.04579, -819167.02846, 5789715.19625),
            (7312068.53368, 15669412.02103, 134.23722),
        ),
        254: (
            (2502918.09575, -819166.96640, 5789715.20406),
            (7312068.51126, 15669412.09534, 134.25594),
        ),
    },
}
# VFLN 2026-10-07: glorg OUT_SOL stabilised XYZ + its 3x3 covariance (SINEX),
# and the Loc. N/E/U sigmas glorg printed for the same solution.
VFLN_XYZ = np.array([2631982.41029146, -907844.716854841, 5719840.06336417])
VFLN_COV = np.array(
    [
        [5.9010787347022e-06, -1.4501266194746e-06, 9.2245508778502e-06],
        [-1.4501266194746e-06, 1.6496941619491e-06, -2.7722103615125e-06],
        [9.2245508778502e-06, -2.7722103615125e-06, 2.0956829457082e-05],
    ]
)
VFLN_LOC_SIGMA = np.array([0.00125, 0.00110, 0.00507])  # glorg prints 0.01 mm

REYK = np.array([2587383.91122, -1043033.58722, 5716564.1974])


def _spd(n: int, seed: int = 0) -> np.ndarray:
    a = np.random.default_rng(seed).normal(size=(n, n)) * 1e-3
    return a @ a.T + np.eye(n) * 1e-7


def test_import_is_numpy_only() -> None:
    code = (
        "import sys, geofunc.local; "
        "print(','.join(m for m in ('pyproj', 'gps_parser', 'pandas', 'scipy') "
        "if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""


def test_package_reexports_still_work_lazily() -> None:
    import geofunc

    assert callable(geofunc.transform)
    assert callable(geofunc.coords.get_frame)  # submodule as attribute, as before
    with pytest.raises(AttributeError):
        _ = geofunc.no_such_name


# --- geodetic -------------------------------------------------------------


def test_geodetic_matches_pyproj_and_round_trips() -> None:
    pyproj = pytest.importorskip("pyproj")
    rng = np.random.default_rng(3)
    lat = rng.uniform(-89.9, 89.9, 200)
    lon = rng.uniform(-180, 180, 200)
    h = rng.uniform(-500, 9000, 200)
    t = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    xyz = np.stack(t.transform(lon, lat, h), -1)
    g = local.ecef_to_geodetic(xyz, ellipsoid=local.WGS84)
    np.testing.assert_allclose(np.degrees(g.lat), lat, atol=1e-10)
    np.testing.assert_allclose(np.degrees(g.lon), lon, atol=1e-10)
    np.testing.assert_allclose(g.h, h, atol=1e-6)
    back = local.geodetic_to_ecef(g.lat, g.lon, g.h, ellipsoid=local.WGS84)
    np.testing.assert_allclose(back, xyz, atol=1e-6)


def test_geodetic_names_are_lat_lon_h_reykjavik() -> None:
    g = local.ecef_to_geodetic(REYK)
    # the same REYK check that exposed the xyzell axis flip (geo.py docstring)
    assert np.degrees(g.lat) == pytest.approx(64.138788, abs=1e-6)
    assert np.degrees(g.lon) == pytest.approx(-21.955490, abs=1e-6)
    assert g.h == pytest.approx(93.028, abs=1e-3)


def test_geodetic_at_the_poles_and_equator() -> None:
    b = local.GRS80.a * (1 - local.GRS80.f)
    g = local.ecef_to_geodetic([[0.0, 0.0, b + 10.0], [local.GRS80.a, 0.0, 0.0]])
    np.testing.assert_allclose(g.lat, [np.pi / 2, 0.0], atol=1e-15)
    np.testing.assert_allclose(g.h, [10.0, 0.0], atol=1e-8)


def test_geodetic_jacobian_matches_finite_differences() -> None:
    j = local.geodetic_jacobian(REYK)
    num = np.empty((3, 3))
    for k in range(3):
        dx = np.zeros(3)
        dx[k] = 1.0  # large step: ECEF ~6e6 m leaves h with ~1e-9 m float noise
        p = local.ecef_to_geodetic(REYK + dx)
        m = local.ecef_to_geodetic(REYK - dx)
        num[:, k] = (np.array(p) - np.array(m)) / 2.0
    np.testing.assert_allclose(j[:2], num[:2], rtol=1e-6, atol=1e-15)  # rad/m
    np.testing.assert_allclose(j[2], num[2], atol=1e-9)  # m/m


# --- local frame ----------------------------------------------------------


def test_rotation_is_orthonormal_and_orders_are_a_permutation() -> None:
    g = local.ecef_to_geodetic(REYK)
    r = local.local_rotation(g.lat, g.lon)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-15)
    re = local.local_rotation(g.lat, g.lon, order="enu")
    np.testing.assert_array_equal(re, r[[1, 0, 2]])
    up = r[2]
    np.testing.assert_allclose(up, REYK / np.linalg.norm(REYK), atol=4e-3)  # ≈ radial


def test_local_round_trip_with_covariance_and_broadcasting() -> None:
    g = local.ecef_to_geodetic(np.stack([REYK, VFLN_XYZ]))
    v = np.array([[0.01, -0.02, 0.03], [0.5, 0.1, -0.2]])
    c = np.stack([_spd(3, 1), _spd(3, 2)])
    loc = local.ecef_to_local(v, g.lat, g.lon, cov=c)
    back = local.local_to_ecef(loc.d, g.lat, g.lon, cov=loc.cov)
    np.testing.assert_allclose(back.d, v, atol=1e-15)
    np.testing.assert_allclose(back.cov, c, atol=1e-18)
    r = local.local_rotation(g.lat[1], g.lon[1])
    np.testing.assert_allclose(loc.cov[1], r @ c[1] @ r.T, atol=1e-18)


def test_geodetic_displacements_are_the_reference_rotation() -> None:
    xs = REYK + np.array([[0, 0, 0], [0.1, -0.2, 0.3]])
    out = local.displacements(xs, REYK, cov=np.stack([_spd(3)] * 2))
    g = local.ecef_to_geodetic(REYK)
    r = local.local_rotation(g.lat, g.lon)
    np.testing.assert_allclose(out.d[1], r @ [0.1, -0.2, 0.3], atol=1e-15)
    np.testing.assert_allclose(out.d[0], 0.0, atol=0)
    np.testing.assert_allclose(out.cov[1], r @ _spd(3) @ r.T, atol=1e-18)


@pytest.mark.parametrize("site", sorted(GLORG))
def test_globk_convention_reproduces_glorg_loc(site: str) -> None:
    """dN/dE/dU between two days == differences of glorg's own Loc. lines."""
    (x0, l0), (x1, l1) = GLORG[site][253], GLORG[site][254]
    out = local.displacements(np.array(x1), np.array(x0), convention="globk")
    np.testing.assert_allclose(out.d, np.subtract(l1, l0), atol=2e-5)  # print res. 1e-5
    geo = local.displacements(np.array(x1), np.array(x0))
    # the geodetic frame differs by ~0.3 % — measurable on this ~8 cm jump
    assert abs(geo.d[1] - out.d[1]) > 1e-4


def test_globk_sigma_reproduces_glorg_vfln() -> None:
    out = local.displacements(
        VFLN_XYZ[None], VFLN_XYZ, cov=VFLN_COV[None], convention="globk"
    )
    sig = np.sqrt(np.diag(out.cov[0]))
    np.testing.assert_allclose(sig, VFLN_LOC_SIGMA, atol=6e-6)  # 0.01 mm printing


def test_globk_jacobian_matches_finite_differences() -> None:
    ref = VFLN_XYZ
    x = VFLN_XYZ + np.array([0.3, -0.1, 0.2])
    out = local.displacements(x, ref, cov=np.eye(3), convention="globk", order="enu")
    num = np.empty((3, 3))
    for k in range(3):
        dx = np.zeros(3)
        dx[k] = 1.0
        p = local.displacements(x + dx, ref, convention="globk", order="enu").d
        m = local.displacements(x - dx, ref, convention="globk", order="enu").d
        num[:, k] = (p - m) / 2.0
    np.testing.assert_allclose(out.cov, num @ num.T, atol=1e-8)


def test_globk_wraps_longitude_across_the_antimeridian() -> None:
    ref = local.geodetic_to_ecef(0.5, np.pi - 1e-7, 0.0)
    x = local.geodetic_to_ecef(0.5, -np.pi + 1e-7, 0.0)
    out = local.displacements(x, ref, convention="globk")
    assert out.d[1] == pytest.approx(local.GRS80.a * np.cos(0.5) * 2e-7, rel=1e-6)


# --- baselines --------------------------------------------------------------


def test_baseline_counts_the_cross_covariance_once() -> None:
    xa, xb = VFLN_XYZ, VFLN_XYZ + np.array([80.0, -50.0, -30.0])
    c = _spd(6, 4)
    out = local.baseline(xa, xb, cov=c)
    g = local.ecef_to_geodetic(xa)
    r = local.local_rotation(g.lat, g.lon)
    np.testing.assert_allclose(out.d, r @ (xa - xb), atol=1e-12)
    caa, cbb, cab = c[:3, :3], c[3:, 3:], c[:3, 3:]
    np.testing.assert_allclose(out.cov, r @ (caa + cbb - cab - cab.T) @ r.T, atol=1e-18)


def test_identical_errors_cancel_in_a_baseline() -> None:
    blk = _spd(3, 5)
    c = np.block([[blk, blk], [blk, blk]])
    out = local.baseline(REYK, REYK + 100.0, cov=c)
    np.testing.assert_allclose(out.cov, 0.0, atol=1e-18)


def test_baseline_at_b_uses_b_frame() -> None:
    xb = REYK + np.array([5e4, 2e4, -1e4])  # ~55 km: the two frames differ
    a = local.baseline(REYK, xb, at="a").d
    b = local.baseline(REYK, xb, at="b").d
    np.testing.assert_allclose(np.linalg.norm(a), np.linalg.norm(b), rtol=1e-12)
    assert not np.allclose(a, b)


# --- Helmert ----------------------------------------------------------------


def test_helmert_design_matches_the_iers_formula() -> None:
    p = np.array([0.01, -0.02, 0.005, 2e-9, 1e-9, -2e-9, 3e-9])
    t, d, (rx, ry, rz) = p[:3], p[3], p[4:]
    rmat = np.array([[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]])
    expect = REYK + t + d * REYK + rmat @ REYK
    out = local.helmert_apply(REYK, p)
    np.testing.assert_allclose(out.d, expect, atol=1e-9)
    assert out.cov is None


def test_helmert_covariance_adds_parameter_uncertainty() -> None:
    p = np.zeros(7)
    cp = np.diag([1e-6, 1e-6, 1e-6, 0, 0, 0, 0])  # 1 mm translation σ only
    c = _spd(3, 6)
    out = local.helmert_apply(REYK[None], p, cov=c[None], params_cov=cp)
    np.testing.assert_allclose(out.cov[0], c + np.eye(3) * 1e-6, atol=1e-18)
    only_p = local.helmert_apply(REYK, p, params_cov=cp)
    np.testing.assert_allclose(only_p.cov, np.eye(3) * 1e-6, atol=1e-18)


def test_helmert_inverse_is_the_negated_parameters_to_first_order() -> None:
    p = np.array([0.01, -0.02, 0.005, 2e-9, 1e-9, -2e-9, 3e-9])
    there = local.helmert_apply(REYK, p).d
    back = local.helmert_apply(there, -p).d
    np.testing.assert_allclose(back, REYK, atol=1e-7)  # second-order residue


# --- refusals ---------------------------------------------------------------


def test_refusals() -> None:
    with pytest.raises(ValueError, match="expected shape"):
        local.ecef_to_geodetic([1.0, 2.0])
    with pytest.raises(ValueError, match="order must be"):
        local.local_rotation(0.1, 0.2, order="une")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="convention must be"):
        local.displacements(REYK, REYK, convention="loc")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one point"):
        local.displacements(REYK, np.stack([REYK, REYK]))
    with pytest.raises(ValueError, match=r"\(\.\.\., 6, 6\)"):
        local.baseline(REYK, REYK, cov=np.eye(3))
    with pytest.raises(ValueError, match="at must be"):
        local.baseline(REYK, REYK, at="c")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="params"):
        local.helmert_apply(REYK, np.zeros(6))
    pole = np.array([0.0, 0.0, local.GRS80.a * (1 - local.GRS80.f)])
    with pytest.raises(ValueError, match="poles"):
        local.displacements(pole, pole, convention="globk")
