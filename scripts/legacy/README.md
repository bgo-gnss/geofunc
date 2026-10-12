# Legacy research scripts (not maintained)

Ad-hoc scripts and their data files that used to sit in `tests/` (pytest
collected `test_geo.py`, whose two functions failed: a missing data file and
the long-gone `cparser` module). They call `geofunc.geo`/`geofunc.geofunc`
functions removed on 2026-10-12, so they no longer run as-is; kept for
reference. Excluded from pytest, ruff and mypy.

Current API: `geofunc.local` (ECEF↔geodetic↔local with covariance),
`geofunc.plates` (plate-motion models), `geofunc.frames` (PROJ frames and
Helmert parameters), `geofunc.coords` (named frame facade).
