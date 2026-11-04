"""Tests for :mod:`geofunc.columns` — the position-table column engine."""

import datetime as dt
import io
import subprocess
import sys

import numpy as np
import pytest

from geofunc import columns as C
from geofunc import local
from geofunc.coords import transform

# a point near Svartsengi, ten days, rising 1 mm/day; day 2 not kept
LAT, LON, H = 63.8822, -22.4337, 97.0
DATES = [dt.datetime(2024, 1, 1, 11, 59) + dt.timedelta(days=k) for k in range(10)]
T = np.array(
    [
        2024.0 + (d - dt.datetime(2024, 1, 1)).total_seconds() / 86400 / 366
        for d in DATES
    ]
)


def series(**kw):
    h = H + 0.001 * np.arange(10)
    h[2] += 0.050  # a blunder the view drops
    xyz = local.geodetic_to_ecef(
        np.full(10, np.radians(LAT)), np.full(10, np.radians(LON)), h
    )
    cov = np.broadcast_to(np.diag([4e-6, 1e-6, 9e-6]), (10, 3, 3)).copy()
    kept = np.ones(10, bool)
    kept[2] = False
    return C.PositionSeries(T, np.asarray(xyz), cov=cov, dates=DATES, kept=kept, **kw)


def table(spec, s=None, **kw):
    t = C.build_table(s or series(), C.parse_columns(spec), **kw)
    return {c.name: c for c in t.columns}, t


def test_parse_presets_frames_rel_names():
    cols = C.parse_columns("isn93,dup=h:first5,du:2024-01-01..2024-01-03")
    assert [c.header for c in cols] == [
        "date", "east_isn93", "north_isn93", "lat", "lon", "h", "dup", "ddu",
    ]  # fmt: skip
    assert cols[6] == C.ColumnSpec("h", None, "first5", "dup")
    for bad, msg in [("east", "needs a frame"), ("foo", "unknown key"),
                     ("lat@isn93", "take @FRAME"), ("date:first5", "no :REL"),
                     ("h,h", "named twice"), ("1h", "use \\[NAME")]:  # fmt: skip
        with pytest.raises(ValueError, match=msg):
            C.parse_columns(bad)


def test_dup_relative_to_first_five_kept():
    cols, _ = table("date,h,dup=h:first5")
    # the first five KEPT days are 0,1,3,4,5 mm (day 2 is the blunder)
    first5 = np.mean([0, 1, 3, 4, 5])
    dup = np.asarray(cols["dup"].values)
    assert dup[0] == pytest.approx(0 - first5, abs=1e-6)
    assert dup[9] == pytest.approx(9 - first5, abs=1e-6)
    assert cols["dup"].unit == "mm"
    assert cols["h"].unit == "m" and np.asarray(cols["h"].values)[0] == pytest.approx(H)
    assert cols["date"].values[0] == "2024-01-01"
    _, m = table("dup=h:first5", units="m")
    assert np.asarray(m.columns[0].values)[9] == pytest.approx(
        (9 - first5) / 1000, abs=1e-9
    )


def test_rel_variants():
    cols, _ = table("a=h:first,b=h:last,c=h:median,d=h:2024-01-04..2024-01-05,e=h:97")
    v = {k: np.asarray(cols[k].values) for k in "abcde"}
    assert v["a"][0] == pytest.approx(0, abs=1e-6)
    assert v["b"][9] == pytest.approx(0, abs=1e-6)
    assert v["d"][3] == pytest.approx(-0.5, abs=1e-6)  # days 3 and 4: 3.5 mm
    assert v["e"][0] == pytest.approx(0, abs=1e-6)  # 97 m less 97 m
    cols, t = table("h:2023-01-01..2023-01-05")  # before the series: empty + note
    assert np.isnan(np.asarray(cols["dh"].values)).all()
    assert t.rows()[0] == [""]
    assert t.notes == ["dh: no kept day in 2023-01-01..2023-01-05; column left empty"]


def test_neu_and_sigma():
    cols, t = table("dn,de,du,sn,se,su,sx", ref_pos="first")
    du = np.asarray(cols["du"].values)
    assert du[0] == pytest.approx(0, abs=1e-6) and du[9] == pytest.approx(9, abs=1e-3)
    assert np.asarray(cols["sx"].values)[0] == pytest.approx(2.0)  # 2 mm
    assert t.ref_pos is not None
    with pytest.raises(ValueError, match="covariance"):
        C.build_table(C.PositionSeries(T, series().xyz), C.parse_columns("sx"))


def test_isn93_no_shift_matches_coords_transform():
    cols, t = table("east@isn93,north@isn93")
    e, n, _ = transform(LON, LAT, H, src="wgs84", dst="isn93")
    assert np.asarray(cols["east_isn93"].values)[0] == pytest.approx(e, abs=1e-3)
    assert np.asarray(cols["north_isn93"].values)[0] == pytest.approx(n, abs=1e-3)
    assert "no datum shift" in t.notes[0]


def test_epoch_shift_moves_back_on_the_velocity():
    # 20 mm/yr due east at the point, as ECEF
    lon = np.radians(LON)
    v = 0.020 * np.array([-np.sin(lon), np.cos(lon), 0.0])
    _, plain = table("east@isn93", s=series())
    cols, shifted = table("east@isn93", s=series(velocity=v))
    years = T[0] - 1993.6
    d = (
        np.asarray(cols["east_isn93"].values)[0]
        - np.asarray(plain.columns[0].values)[0]
    )
    assert d == pytest.approx(-0.020 * years, rel=2e-3)  # Lambert scale ~1
    assert "taken back to 1993.6" in shifted.notes[0]
    _, off = table("east@isn93", s=series(velocity=v), epoch_shift=False)
    assert (
        np.asarray(off.columns[0].values)[0] == np.asarray(plain.columns[0].values)[0]
    )


def test_write_table_header_and_rows():
    _, t = table("date,dup=h:first5")
    buf = io.StringIO()
    C.write_table(t, buf, header=["station SENG"])
    lines = buf.getvalue().splitlines()
    assert lines[0] == "# station SENG"
    assert lines[1].startswith("# column date")
    assert lines[3] == "date,dup"
    assert lines[4] == "2024-01-01,-2.60"


def test_import_is_numpy_only():
    code = "import sys, geofunc.columns; print('pyproj' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr
