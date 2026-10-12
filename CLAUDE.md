# geofunc

Pure-Python geodesy primitives for the GPS library ecosystem.
**Tier 0** — foundation library, zero internal dependencies.

## Status

Stable core + `local` (2026-10-09, branch feat/local-frames): ECEF-first transformations with
covariance propagation for the daily SINEX/layer-2 store. Consumed by `geo_dataread`.

## Layout

```
geofunc/
├── src/geofunc/
│   ├── __init__.py       # public API: LAZY re-export of coords (keeps pyproj out of `import geofunc.local`)
│   ├── local.py          # numpy-only ECEF↔geodetic↔local (NEU/ENU, geodetic|globk), baselines, Helmert — all with C′=JCJᵀ
│   ├── geo.py            # core geodesy: ellipsoidal/Cartesian conversions, plate motion
│   ├── geofunc.py        # higher-level helpers: station coords, plate-velocity wrappers
│   ├── coords.py         # reference-frame registry + transform() facade (ITRF/WGS84/ISN)
│   ├── frames.py         # PROJ bridge: EPSG geocentric CRSs, helmert_params extraction, 4-D transform_xyz, geodesic
│   └── plates.py         # ITRF2008/2014/2020 PMM tables (mas/yr, DOIs) + plate_velocity[_local] (numpy only)
├── tests/                # test_geo.py, test_coords.py + ad-hoc scripts
├── perlscripts/          # legacy Perl utilities (reference only)
└── pyproject.toml
```

## Key Functions

| Module | Function | Purpose |
|--------|----------|---------|
| `geo.py` | `xyzell` | ECEF (X,Y,Z) → ellipsoidal (lat, lon, h) |
| `geo.py` | `platePoles` | Tectonic plate Euler poles (legacy → `plates.euler_pole`) |
| `geo.py` | `rotpole` | Rotate pole between reference frames |
| `geo.py` | `greatcircd` | Great-circle distance |
| `geofunc.py` | `convllh` | Coordinate conversion driver |
| `geofunc.py` | `plateVelo` | Plate velocity at a point — returns **E,N,U** despite its docstring (legacy → `plates.plate_velocity_local`) |
| `geofunc.py` | `getStationCoordinates` | Look up station coordinates |
| `coords.py` | `transform` | Convert (x,y,z) between named frames (wgs84, itrf08/14/20, isn93/2004/2016) |
| `local.py` | `ecef_to_geodetic` / `geodetic_to_ecef` | Named `Geodetic(lat, lon, h)`, radians, GRS80 default |
| `local.py` | `displacements` | N/E/U (or E/N/U) from ECEF + covariance; `convention="globk"` reproduces glorg `Loc.`/TOT (N=a·Δφ, E=a·cosφ₀·Δλ, U=Δh) |
| `local.py` | `baseline` / `ecef_to_local` / `geodetic_jacobian` / `propagate` | Local vectors, joint 6×6 baseline covariance, Jacobians |
| `local.py` | `helmert_design` / `helmert_apply` / `helmert_at` | IERS 7-param small-angle transformation (+ point and parameter covariance); one set or one per epoch (`helmert_at`: p + ṗ·(t−t0)) |
| `local.py` | `reference_position` | Reference point of a series: wmean (full 3×3 cov) / mean / median / first, optional first-`count` epochs |
| `local.py` | `apply_local_offsets` | N/E/U offsets → ECEF along each point's own axes (junctions, antenna heights) before any displacement |
| `local.py` | `euler_matrix` / `euler_velocity` / `linear_motion` | ω × x with pole covariance; v·(t−t0) to remove/add secular motion in ECEF |
| `local.py` | `chain` / `ecef_jacobian` / `globk_scale` | Compose Jacobians (application order); ∂XYZ/∂(φ,λ,h); geodetic→GLOBK Loc. factor |
| `frames.py` | `helmert_params` / `HelmertParams.at` | EPSG time-dependent Helmert from PROJ in `local.helmert_at` units (≤5e-9 m vs PROJ) → frame change of a series WITH covariance |
| `frames.py` | `transform_xyz` / `crs` / `geodesic` | PROJ 4-D ECEF transform (no cov), named EPSG geocentric frames (itrf2000…2020, etrf2000/2014/2020), pyproj.Geod |
| `plates.py` | `euler_pole` / `plate_velocity` / `plate_velocity_local` | PMM pole (rad/yr) → ECEF or N/E/U velocity, optional pole σ and ITRF2020 ORB; EURA pinned to −(PROJ ITRF→ETRF rates) |
| `coords.py` | `FRAMES` / `get_frame` | Data-driven frame registry (add a frame = one data row) |

## Dependencies

- **In** (this package needs): `pyproj`, `numpy`, `scipy` (external only).
  `geofunc.py` additionally imports `gps_parser` (config-driven helpers) — the
  `coords.py` conversion core is deliberately pure (pyproj + numpy only).
- **Out** (used by): `geo_dataread` (coordinate transforms, plate-velocity calcs).

### Local frames + covariance (`local.py`)

numpy only — `tests/test_local.py::test_import_is_numpy_only` guards that `import
geofunc.local` pulls no pyproj/scipy/gps_parser (leaf packages depend on it).
GLOBK `Loc.` convention is pinned to glorg 10.71 output (SEY1/AKUR across the
2026-09-11 frame jump, VFLN σ from OUT_SOL), NOT to the manual's "nearest 1°
small circle", which 10.71 does not do. Geodetic vs globk differ by ~0.3 % in N and E.

### Coordinate frames (`coords.py`)

Static frame-to-frame only (no epoch/plate-motion — that lands with the data
analyses). Registered frames: `wgs84` (lon/lat/h), `itrf2008`/`itrf2014`/
`itrf2020` (ECEF X/Y/Z), `isn93`/`isn2004`/`isn2016` (Lambert E/N). Axis order
is always x-first (`always_xy=True`), unlike `geo.xyzell`'s (lat, lon, h).
Adding a frame is a data row in `coords._FRAME_SPECS`. `tos search` will consume
this as computed coordinate columns (separate work).

## Console Scripts

```bash
geofunc       # entry: geofunc:main
```

## Cross-References

- `../CLAUDE.md` — ecosystem overview + dependency graph
- Vault hub: `/home/bgo/notes/bgovault/2.Areas/VI_GPS_Library/1776347706-gps-library-ecosystem-hub.md`

---

*Last reviewed: 2026-10-12*
