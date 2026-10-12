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


# N, E, U [m/yr] from the legacy geo.platePoles → rotpole → eccell chain
# (eccell rows are E, N, U; reordered), frozen 2026-10-12 before geo.py retired
LEGACY = {
    ("REYK", "EURA", "itrf2008"): (0.016230482733457816, 0.006743774093761006, 4.287914896126171e-05),
    ("REYK", "EURA", "itrf2014"): (0.016167736912028996, 0.00709676764751457, 4.271338140721361e-05),
    ("REYK", "NOAM", "itrf2008"): (0.01853098690874565, -0.009111886997883446, 4.89568277856137e-05),
    ("REYK", "NOAM", "itrf2014"): (0.019573083166746746, -0.00866030309968424, 5.170993140012832e-05),
    ("SKSH", "EURA", "itrf2008"): (0.016270166576665336, 0.006652720742979958, 4.214728604382013e-05),
    ("SKSH", "EURA", "itrf2014"): (0.01620624413424372, 0.006998280140388368, 4.198169723729812e-05),
    ("SKSH", "NOAM", "itrf2008"): (0.018633504956046633, -0.008895985841326602, 4.8269429798450995e-05),
    ("SKSH", "NOAM", "itrf2014"): (0.019675649024553665, -0.008445358236996606, 5.0969066827196425e-05),
}  # fmt: skip


@pytest.mark.parametrize(("sta", "plate", "frame"), list(LEGACY))
def test_parity_with_legacy_geo(sta: str, plate: str, frame: str) -> None:
    x = {"REYK": REYK, "SKSH": SKSH}[sta]
    neu = plates.plate_velocity_local(plate, x, frame=frame).d
    np.testing.assert_allclose(neu, LEGACY[sta, plate, frame], atol=1e-12)


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


def test_plate_names_cover_every_model_and_keep_legacy_titles() -> None:
    codes = {c for f in plates.frames() for c in plates.plates(f)}
    assert codes <= set(plates.PLATE_NAMES)
    assert plates.plate_name("noam") == "North American plate"
    assert plates.plate_name("EURA") == "Eurasian Plate"
    with pytest.raises(KeyError, match="unknown plate code"):
        plates.plate_name("XXXX")
