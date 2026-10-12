"""Tests for :mod:`geofunc.coords` — the reference-frame registry + facade.

Known coordinates (from ``geofunc.geo.xyzell``'s docstring, REYK):

    WGS84 (lon, lat, h) = (-21.955490, 64.138788, 93.028)
    ITRF2008 ECEF        ≈ (2587383.87, -1043033.57, 5716564.22)

ISN values were derived via pyproj in the session that built this module
(ISN2016 shares ISN2004's Lambert parameters but with the ITRF2014 datum,
hence the 1,000,000 m Easting offset between the two false origins).
"""

import numpy as np
import pytest
from pyproj import CRS

from geofunc.coords import FRAMES, CoordFrame, frame_names, get_frame, transform

REYK_WGS84 = (-21.955490, 64.138788, 93.028)  # lon, lat, h
REYK_ITRF2008 = (2587383.87, -1043033.57, 5716564.22)  # X, Y, Z


def test_registry_has_expected_frames():
    expected = {
        "wgs84",
        "itrf2008",
        "itrf2014",
        "itrf2020",
        "isn93",
        "isn2004",
        "isn2016",
    }
    assert expected <= set(FRAMES)
    assert frame_names() == tuple(sorted(FRAMES))


def test_kinds():
    assert FRAMES["wgs84"].is_geographic
    assert FRAMES["itrf2008"].is_geocentric
    assert FRAMES["itrf2014"].is_geocentric
    assert FRAMES["itrf2020"].is_geocentric
    assert FRAMES["isn93"].is_projected
    assert FRAMES["isn2004"].is_projected
    assert FRAMES["isn2016"].is_projected


def test_wgs84_to_itrf2008_ecef():
    x, y, z = transform(*REYK_WGS84, src="wgs84", dst="itrf2008")
    assert x == pytest.approx(REYK_ITRF2008[0], abs=0.05)
    assert y == pytest.approx(REYK_ITRF2008[1], abs=0.05)
    assert z == pytest.approx(REYK_ITRF2008[2], abs=0.05)


def test_wgs84_to_isn2016():
    e, n, h = transform(*REYK_WGS84, src="wgs84", dst="isn2016")
    assert e == pytest.approx(2556148.998, abs=0.01)
    assert n == pytest.approx(207354.677, abs=0.01)
    assert h == pytest.approx(93.028, abs=0.001)


def test_wgs84_to_isn93_and_isn2004():
    e93, n93, _ = transform(*REYK_WGS84, src="wgs84", dst="isn93")
    e04, n04, _ = transform(*REYK_WGS84, src="wgs84", dst="isn2004")
    # ISN2004 vs ISN93 use different false origins.
    assert e93 == pytest.approx(356148.998, abs=0.01)
    assert e04 == pytest.approx(1556148.998, abs=0.01)


def test_roundtrip_preserves_coordinates():
    llh = (-19.080568, 63.674361, 1438.359)  # AUST
    for frame in ("itrf2008", "itrf2014", "itrf2020", "isn93", "isn2004", "isn2016"):
        there = transform(*llh, src="wgs84", dst=frame)
        back = transform(*there, src=frame, dst="wgs84")
        assert back[0] == pytest.approx(llh[0], abs=1e-6)
        assert back[1] == pytest.approx(llh[1], abs=1e-6)
        assert back[2] == pytest.approx(llh[2], abs=1e-3)


def test_identity_when_same_frame():
    assert transform(*REYK_WGS84, src="wgs84", dst="wgs84") == REYK_WGS84


def test_itrf_realisations_agree_at_reference_epoch():
    """Static (epoch-free) ITRF08/14/20 ECEF are identical at 2010.0."""
    x08, y08, z08 = transform(*REYK_WGS84, src="wgs84", dst="itrf2008")
    x14, y14, z14 = transform(*REYK_WGS84, src="wgs84", dst="itrf2014")
    x20, y20, z20 = transform(*REYK_WGS84, src="wgs84", dst="itrf2020")
    assert x08 == pytest.approx(x14, abs=1e-6)
    assert x14 == pytest.approx(x20, abs=1e-6)
    assert y08 == pytest.approx(y14, abs=1e-6)
    assert z14 == pytest.approx(z20, abs=1e-6)


def test_vectorized_batch():
    lons = np.array([-21.955490, -19.080568])
    lats = np.array([64.138788, 63.674361])
    hs = np.array([93.028, 1438.359])
    e, n, h = transform(lons, lats, hs, src="wgs84", dst="isn2016")
    assert e.shape == (2,)
    assert n.shape == (2,)
    assert h.shape == (2,)


def test_unknown_frame_raises():
    with pytest.raises(KeyError) as exc:
        transform(*REYK_WGS84, src="wgs84", dst="bogus")
    assert "valid frames" in str(exc.value)
    assert "isn2004" in str(exc.value)


def test_get_frame_returns_coordframe():
    f = get_frame("itrf2014")
    assert isinstance(f, CoordFrame)
    assert f.kind == "geocentric"
    assert isinstance(f.crs, CRS)
