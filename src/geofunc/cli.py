"""``geofunc``: point tables from the command line (:mod:`geofunc.points`).

    geofunc points FILE [--from FRAME] [--to FRAME]      # convert a table
    geofunc nearest FILE --candidates FILE [-k N]        # nearest candidates

FILE ``-`` reads stdin (paste a table). The output is CSV on stdout (or
``-o FILE``): the name, the table's other columns, then the coordinates.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import TextIO


def _table_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "file", help="the table (- = stdin): tab, ;, comma or space separated"
    )
    p.add_argument("--from", dest="src", default=None, metavar="FRAME",
                   help="the table's frame (default: the one its header names, "
                   "e.g. 'X (ISN93) [m]', else wgs84)")  # fmt: skip
    p.add_argument("--x", default=None, help="x column (header or 1-based index)")
    p.add_argument("--y", default=None, help="y column (header or 1-based index)")
    p.add_argument("--name", default=None, help="name column (header or 1-based index)")
    p.add_argument("--ncols", type=int, default=None,
                   help="cells per row of a table pasted one cell per line")  # fmt: skip
    p.add_argument("-o", "--output", default=None, help="output file (default stdout)")


def _read(args: argparse.Namespace) -> list:  # type: ignore[type-arg]
    from geofunc import points

    pts = points.read_points(args.file, frame=args.src, x=args.x, y=args.y,
                             name=args.name, ncols=args.ncols)  # fmt: skip
    if pts[0].frame is None:
        pts = [points.Point(p.name, p.x, p.y, "wgs84", p.fields) for p in pts]
        if any(abs(p.x) > 360 or abs(p.y) > 90 for p in pts):
            raise ValueError(
                "the coordinates are not lon/lat; give the table's frame (--from)"
            )
    return pts


def _out(path: str | None) -> AbstractContextManager[TextIO]:
    """``path`` opened for writing, or stdout (left open)."""
    if path:
        return open(path, "w", encoding="utf-8")
    return nullcontext(sys.stdout)


def build_parser() -> argparse.ArgumentParser:
    from geofunc.coords import frame_names

    frames = ", ".join(frame_names())
    p = argparse.ArgumentParser(prog="geofunc", description=__doc__.split("\n")[0],
                                epilog=f"frames: {frames}")  # fmt: skip
    sub = p.add_subparsers(dest="command", required=True)
    pt = sub.add_parser("points", help="a point table in another frame",
                        epilog=f"frames: {frames}")  # fmt: skip
    _table_args(pt)
    pt.add_argument("--to", dest="dst", default="wgs84", metavar="FRAME",
                    help="output frame (default wgs84: lon, lat)")  # fmt: skip
    nb = sub.add_parser("nearest", help="the candidates nearest each point",
                        epilog=f"frames: {frames}")  # fmt: skip
    _table_args(nb)
    nb.add_argument("--candidates", required=True, metavar="FILE",
                    help="the points to search (a table read the same way)")  # fmt: skip
    nb.add_argument("--candidates-from", default=None, metavar="FRAME",
                    help="the candidates' frame (default: from their header, else wgs84)")  # fmt: skip
    nb.add_argument("-k", type=int, default=1, help="how many per point (default 1)")
    nb.add_argument("--max-km", type=float, default=None,
                    help="leave out candidates farther than this")  # fmt: skip
    return p


def main(argv: Sequence[str] | None = None) -> int:
    from geofunc import points

    p = build_parser()
    args = p.parse_args(argv)
    try:
        pts = points.to_frame(_read(args), "wgs84")
        if args.command == "points":
            if args.dst != "wgs84":
                pts = points.to_frame(pts, args.dst)
            with _out(args.output) as fh:
                points.write_points(pts, fh)
            return 0
        cand_args = argparse.Namespace(**{**vars(args), "file": args.candidates,
                                          "src": args.candidates_from, "x": None,
                                          "y": None, "name": None, "ncols": None})  # fmt: skip
        cands = [(c.name, c.x, c.y) for c in points.to_frame(_read(cand_args), "wgs84")]
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    max_m = args.max_km * 1000.0 if args.max_km is not None else None
    with _out(args.output) as fh:
        import csv

        w = csv.writer(fh, lineterminator="\n")
        w.writerow(
            ["name", "lon", "lat", "rank", "nearest", "distance_km", "azimuth_deg"]
        )
        for pt in pts:
            for rank, nb in enumerate(points.nearest((pt.x, pt.y), cands, args.k,
                                                     max_distance=max_m), 1):  # fmt: skip
                w.writerow([pt.name, f"{pt.x:.7f}", f"{pt.y:.7f}", rank, nb.id,
                            f"{nb.distance / 1000:.3f}", f"{nb.azimuth:.1f}"])  # fmt: skip
    return 0
