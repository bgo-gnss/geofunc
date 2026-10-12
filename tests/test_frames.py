"""geofunc.frames: Helmert parameters pulled from PROJ reproduce PROJ's 4-D transform."""

from __future__ import annotations

import numpy as np
import pytest

from geofunc import frames, local

REYK = np.array([2587384.18278, -1043033.53969, 5716564.08400])
PAIRS = [
    ("itrf2008", "itrf2014"),
    ("itrf2014", "itrf2020"),
    ("itrf2008", "itrf2020"),
    ("itrf2014", "etrf2014"),
    ("itrf2020", "etrf2020"),
]


def test_named_frames_are_epsg_geocentric() -> None:
    for name, code in frames.GEOCENTRIC.items():
        c = frames.crs(name)
        assert c.to_string() == code
        assert c.is_geocentric and c.ellipsoid is not None
        assert c.ellipsoid.name == "GRS 1980"
    assert frames.crs("ITRF2014") is frames.crs("itrf2014")  # cached


@pytest.mark.parametrize(("src", "dst"), PAIRS)
def test_helmert_params_reproduce_proj(src: str, dst: str) -> None:
    h = frames.helmert_params(src, dst)
    assert h.t0 is not None and h.p0.shape == h.rates.shape == (7,)
    t = np.array([1996.3, 2010.0, 2026.8])
    x = REYK + np.array([[0.0, 0, 0], [10.0, -5, 2], [-3.0, 8, 1]])
    ours = local.helmert_apply(x, h.at(t)).d
    np.testing.assert_allclose(ours, frames.transform_xyz(x, src, dst, t), atol=5e-9)


def test_series_frame_change_carries_the_covariance() -> None:
    h = frames.helmert_params("itrf2008", "itrf2020")
    c = np.diag([1e-6, 2e-6, 4e-6])
    out = local.helmert_apply(REYK, h.at(2020.0), cov=c)
    assert out.cov is not None
    # a cm-level shift barely moves a mm-level covariance (scale+rotation ~1e-9)
    np.testing.assert_allclose(out.cov, c, rtol=1e-8, atol=1e-20)


def test_rates_are_per_year() -> None:
    # ITRF2014→ETRF2014: EURA rotation 0.085/0.531/−0.770 mas/yr
    h = frames.helmert_params("itrf2014", "etrf2014")
    mas = h.rates[4:] / (np.pi / 180 / 3.6e6)
    np.testing.assert_allclose(mas, [0.085, 0.531, -0.770], atol=1e-12)
    assert "ETRF2014" in h.name


def test_transform_without_epoch_and_shapes() -> None:
    x = np.stack([REYK, REYK + 1.0])
    out = frames.transform_xyz(x, "itrf2014", "itrf2020")
    assert out.shape == (2, 3)
    np.testing.assert_allclose(np.diff(out, axis=0), [[1.0, 1.0, 1.0]], atol=1e-6)
    with pytest.raises(ValueError, match="trailing axis of 3"):
        frames.transform_xyz(np.zeros((2, 2)), "itrf2014", "itrf2020")


def test_not_a_helmert_raises() -> None:
    # ITRF2014 → WGS 84 (EPSG:4978) is a PROJ ballpark/no-op, not an EPSG Helmert
    with pytest.raises(ValueError, match="not a position-vector Helmert"):
        frames.helmert_params("itrf2014", "EPSG:4978")


def test_geodesic_grs80() -> None:
    lon1, lat1, lon2, lat2 = -21.9555, 64.1388, -15.1981, 64.2672  # REYK, HOFN
    _, _, dist = frames.geodesic().inv(lon1, lat1, lon2, lat2)
    p1, l1, p2, l2 = np.radians([lat1, lon1, lat2, lon2])
    hav = (
        np.sin((p2 - p1) / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
    )
    sphere = 2 * 6371e3 * np.arcsin(np.sqrt(hav))
    assert abs(dist - sphere) / sphere < 5e-3  # ellipsoid vs sphere, ~328 km
