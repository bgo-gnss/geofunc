"""geofunc.plates: PMM tables, velocities, and their consistency with PROJ/legacy code."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from geofunc import local, plates

REYK = np.array([2587384.18278, -1043033.53969, 5716564.08400])  # EURA
SKSH = np.array([2532461.0, -982040.0, 5747930.0])  # NOAM side, approx.


def test_import_is_numpy_only() -> None:
    code = (
        "import sys, geofunc.plates; "
        "print(','.join(m for m in ('pyproj', 'gps_parser', 'pandas', 'scipy') "
        "if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""


def test_euler_pole_units_and_lookup() -> None:
    mas = plates.euler_pole("eura", "ITRF2014", units="mas/yr")
    np.testing.assert_array_equal(mas, [-0.085, -0.531, 0.770])
    rad = plates.euler_pole("EURA", "itrf2014")
    np.testing.assert_allclose(rad, mas * np.pi / 180 / 3.6e6, rtol=1e-15)
    np.testing.assert_allclose(
        plates.euler_pole("EURA", "itrf2014", units="deg/Myr"), mas / 3.6
    )
    assert "NOAM" in plates.plates("itrf2008") and "itrf2020" in plates.frames()
    with pytest.raises(KeyError, match="not in the itrf2008 model"):
        plates.euler_pole("PCFC", "itrf2008")
    with pytest.raises(KeyError, match="no plate-motion model"):
        plates.euler_pole("EURA", "itrf1997")


@pytest.mark.parametrize(
    ("frame", "atol_mas"), [("itrf2014", 1e-12), ("itrf2020", 1.5e-3)]
)
def test_eura_pole_is_minus_the_proj_etrf_rotation_rates(
    frame: str, atol_mas: float
) -> None:
    # ITRF→ETRF removes the EURA rotation: rates = −ω. EUREF's ETRF2020 uses
    # 0.086 where the published PMM table rounds to 0.085 (σ 0.003).
    frames = pytest.importorskip("geofunc.frames")
    h = frames.helmert_params(frame, "e" + frame[1:])
    rates_mas = h.rates[4:] / plates.MAS_TO_RAD
    np.testing.assert_allclose(
        -rates_mas, plates.euler_pole("EURA", frame, units="mas/yr"), atol=atol_mas
    )
    assert not h.rates[:4].any()  # rotation only, no translation/scale rate


@pytest.mark.parametrize("frame", ["itrf2008", "itrf2014"])
@pytest.mark.parametrize("plate", ["EURA", "NOAM"])
@pytest.mark.parametrize("x", [REYK, SKSH])
def test_parity_with_legacy_geo(frame: str, plate: str, x: np.ndarray) -> None:
    # legacy geo.platePoles → rotpole → eccell, which returns rows E, N, U
    geo = pytest.importorskip("geofunc.geo")
    w = geo.platePoles(plate, reference=frame.upper())
    llh = np.array(geo.itrf08towgs84.transform(*x, radians=True))
    enu = geo.eccell(llh, geo.rotpole(w, x))
    neu = plates.plate_velocity_local(plate, x, frame=frame).d
    np.testing.assert_allclose(neu, enu[[1, 0, 2]], atol=1e-12)


def test_plate_velocity_is_horizontal_and_iceland_sized() -> None:
    v = plates.plate_velocity_local("EURA", REYK, frame="itrf2014").d
    # Eurasia at Reykjavik: ~2 cm/yr, mostly north-east; |U| tiny (sphere vs ellipsoid)
    assert 0.015 < np.hypot(v[0], v[1]) < 0.025
    assert abs(v[2]) < 1e-4
    ecef = plates.plate_velocity("EURA", REYK, frame="itrf2014").d
    np.testing.assert_allclose(
        ecef, np.cross(plates.euler_pole("EURA", "itrf2014"), REYK), rtol=1e-15
    )


def test_broadcasts_over_points() -> None:
    x = np.stack([REYK, SKSH])
    v = plates.plate_velocity_local("NOAM", x, frame="itrf2020", order="enu")
    assert v.d.shape == (2, 3) and v.cov is None
    one = plates.plate_velocity_local("NOAM", SKSH, frame="itrf2020", order="enu")
    np.testing.assert_allclose(v.d[1], one.d, atol=1e-18)


def test_covariance_and_orb() -> None:
    v = plates.plate_velocity("EURA", REYK, frame="itrf2020", with_cov=True)
    assert v.cov is not None
    e = local.euler_matrix(REYK)
    cw = plates.pole_covariance("EURA", "itrf2020")
    assert cw is not None
    np.testing.assert_allclose(v.cov, e @ cw @ e.T, rtol=1e-12)
    # 0.003 mas/yr on each axis at 6400 km ≈ 0.1 mm/yr
    assert 5e-5 < np.sqrt(np.trace(v.cov)) < 3e-4
    assert plates.pole_covariance("EURA", "itrf2014") is None
    assert (
        plates.plate_velocity("EURA", REYK, frame="itrf2014", with_cov=True).cov is None
    )

    w = plates.plate_velocity("EURA", REYK, frame="itrf2020", orb=True, with_cov=True)
    np.testing.assert_allclose(w.d - v.d, [0.37e-3, 0.35e-3, 0.74e-3], rtol=1e-12)
    assert w.cov is not None
    np.testing.assert_allclose(
        np.diag(w.cov - v.cov), np.square([0.08e-3, 0.10e-3, 0.09e-3]), rtol=1e-12
    )
    with pytest.raises(ValueError, match="no origin rate bias"):
        plates.plate_velocity("EURA", REYK, frame="itrf2014", orb=True)
