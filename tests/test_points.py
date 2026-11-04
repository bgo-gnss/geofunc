"""Tests for :mod:`geofunc.points` — point tables, frames, nearest."""

import io

import pytest

from geofunc import points
from geofunc.cli import main

# Reykjanes boreholes as ÍSOR lists them (ISN93, decimal commas, Icelandic headers)
TSV = (
    "Auðkenni\tStaðarnr.\tX (ISN93) [m]\tY (ISN93) [m]\n"
    "HSK-06\tB16606\t329637,4\t383731,4\n"
    "TD-04\tB22264\t347410,9\t387059,4\n"
)
# the same, copied from a web page: one cell per line, blank lines between
PASTED = "\n\n".join(
    ["Auðkenni", "Staðarnr.", "X (ISN93) [m]", "Y (ISN93) [m]",
     "HSK-06", "B16606", "329637,4", "383731,4",
     "TD-04", "B22264", "347410,9", "387059,4"]
)  # fmt: skip


def test_reads_decimal_comma_icelandic_headers():
    pts = points.read_points(io.StringIO(TSV))
    assert [p.name for p in pts] == ["HSK-06", "TD-04"]
    assert pts[0].x == pytest.approx(329637.4)  # X = Easting
    assert pts[0].y == pytest.approx(383731.4)
    assert pts[0].frame == "isn93"  # from the header
    assert pts[0].fields == {"Staðarnr.": "B16606"}


def test_one_cell_per_line():
    assert points.read_points(io.StringIO(PASTED), ncols=4) == points.read_points(
        io.StringIO(TSV)
    )
    with pytest.raises(ValueError, match="whole number"):
        points.read_points(io.StringIO(PASTED), ncols=5)


def test_isn93_lands_on_reykjanes():
    (p, q) = points.to_frame(points.read_points(io.StringIO(TSV)), "wgs84")
    assert (p.x, p.y) == pytest.approx((-22.47253, 63.91508), abs=1e-5)
    assert (q.x, q.y) == pytest.approx((-22.11421, 63.95319), abs=1e-5)
    back = points.to_frame([p], "isn93")[0]
    assert (back.x, back.y) == pytest.approx((329637.4, 383731.4), abs=1e-3)


def test_csv_whitespace_and_no_header():
    csv_pts = points.read_points(io.StringIO("marker,lon,lat\nSENG,-22.4,63.9\n"))
    assert csv_pts[0].name == "SENG" and csv_pts[0].frame is None
    ws = points.read_points(io.StringIO("A 1,5 2,5\nB 3 4\n"), frame="isn93")
    assert [(p.name, p.x, p.y) for p in ws] == [("A", 1.5, 2.5), ("B", 3.0, 4.0)]


def test_columns_by_name_or_index():
    t = "id;north;east;note\nP;10;20;x\n"
    (p,) = points.read_points(io.StringIO(t))
    assert (p.x, p.y) == (20.0, 10.0)  # east is x, whatever the column order
    (q,) = points.read_points(io.StringIO(t), x=2, y="east", name="note")
    assert (q.name, q.x, q.y) == ("x", 10.0, 20.0)


def test_bad_rows():
    with pytest.raises(ValueError, match="not coordinates"):
        points.read_points(io.StringIO("name\tx\ty\nA\t1\tfoo\n"))
    with pytest.raises(ValueError, match="frame"):
        points.to_frame([points.Point("A", 1.0, 2.0)], "wgs84")


def test_nearest_known_distance():
    # one degree of latitude, 64-65° N on GRS80 (meridian arc)
    cands = [("FAR", -22.0, 66.0), ("NEAR", -22.0, 65.0)]
    (nb,) = points.nearest((-22.0, 64.0), cands)
    assert nb.id == "NEAR"
    assert nb.distance == pytest.approx(111_485, abs=5)
    assert nb.azimuth == pytest.approx(0.0, abs=1e-6)
    assert [n.id for n in points.nearest((-22.0, 64.0), cands, k=5)] == ["NEAR", "FAR"]
    assert points.nearest((-22.0, 64.0), cands, max_distance=1000) == []


def test_cli_points_and_nearest(tmp_path, capsys):
    table = tmp_path / "b.tsv"
    table.write_text(TSV, encoding="utf-8")
    cands = tmp_path / "s.csv"
    cands.write_text("marker,lon,lat\nSENG,-22.4250,63.8794\nVOGC,-22.1010,63.9520\n")
    assert main(["points", str(table)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "name,Staðarnr.,lon,lat"
    assert out[1].startswith("HSK-06,B16606,-22.47253")
    assert main(["nearest", str(table), "--candidates", str(cands)]) == 0
    rows = capsys.readouterr().out.splitlines()
    assert rows[0] == "name,lon,lat,rank,nearest,distance_km,azimuth_deg"
    assert rows[1].split(",")[4] == "SENG" and rows[2].split(",")[4] == "VOGC"
