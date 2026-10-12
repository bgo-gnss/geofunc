"""The deprecated geofunc.geo / geofunc.geofunc shims: same answers, loud import."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from geofunc import plates

REYK = [2587383.91122, -1043033.58722, 5716564.19740]
# frozen from the pre-shim plateVelo (E, N, U) — see test_plates.LEGACY
REYK_NOAM_2008_ENU = (
    -0.009111887907600187,
    0.018530986455884538,
    4.895682345376907e-05,
)


def _fresh(name: str):  # type: ignore[no-untyped-def]
    sys.modules.pop(name, None)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        return importlib.import_module(name)


def test_geo_shim_warns_and_keeps_pole_units() -> None:
    geo = _fresh("geofunc.geo")
    np.testing.assert_array_equal(
        geo.platePoles("EURA", units="masy"), [-0.083, -0.534, 0.750]
    )
    ry = geo.platePoles("NOAM", reference="ITRF2014")
    np.testing.assert_allclose(ry, plates.euler_pole("NOAM", "itrf2014"), rtol=1e-15)
    np.testing.assert_allclose(
        geo.platePoles("EURA", units="degMy"), np.array([-0.083, -0.534, 0.75]) / 3.6
    )
    np.testing.assert_array_equal(geo.platePoles("EURA", reference="X"), np.zeros(3))
    np.testing.assert_allclose(geo.rotpole(ry, REYK), np.cross(ry, REYK))


def test_geofunc_shim_plateVelo_keeps_E_N_U(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    gf = _fresh("geofunc.geofunc")
    coord = tmp_path / "coord.xyz"
    coord.write_text(" ".join(map(str, REYK)) + " REYK\n")
    plate = tmp_path / "plate"
    plate.write_text("REYK\tNOAM\n")
    files = {"coordFile": str(coord), "platefile": str(plate)}
    monkeypatch.setattr(gf, "_postprocess_file", files.__getitem__)
    assert gf.plateDict() == {"REYK": "NOAM"}
    assert gf.plateFullname("NOAM") == "North American plate"
    v = gf.plateVelo(["REYK", "NONE"])
    np.testing.assert_allclose(v[0], REYK_NOAM_2008_ENU, atol=1e-12)
    np.testing.assert_array_equal(v[1], np.zeros(3))
    np.testing.assert_allclose(
        gf.plateVelo(["REYK"], plate="EURA")[0][1], 0.0162, atol=1e-3
    )
