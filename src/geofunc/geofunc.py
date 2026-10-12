"""DEPRECATED config-driven helpers — kept as thin shims for one release.

These read station configuration (gps_parser), which a Tier-0 package must
not do; the config lookups moved to Tier 1 and the maths to ``geofunc.plates``:

=========================  ===================================================
legacy                     replacement
=========================  ===================================================
``plateVelo``              ``geo_dataread.station_catalog.plate_velocity_neu``
                           (returns N, E, U — ``plateVelo`` returns E, N, U)
``plateDict``              ``geo_dataread.station_catalog.plate_table``
``plateFullname``          ``geofunc.plates.plate_name``
=========================  ===================================================

Removed 2026-10-12 (no callers left in the ecosystem): ``convllh``,
``extractfromGamitBakf``, ``savedisp`` and ``getStationCoordinates`` (whose
columns were labelled lon, lat but held lat, lon since the pyproj 2 move).
``gps_parser`` is imported inside the functions — it was never a declared
dependency of geofunc.
"""

from __future__ import annotations

import warnings

import numpy as np

from geofunc import plates

warnings.warn(
    "geofunc.geofunc is deprecated: use geo_dataread.station_catalog and "
    "geofunc.plates",
    DeprecationWarning,
    stacklevel=2,
)


def _postprocess_file(key: str) -> str:
    import gps_parser as cp

    return cp.ConfigParser().getPostProcessConfig(key)


def plateFullname(plate):
    """Name of a plate code (see :func:`geofunc.plates.plate_name`)."""
    return plates.plate_name(plate)


def plateDict():
    """``{station: plate code}`` from postprocess.cfg ``platefile``."""
    out = {}
    with open(_postprocess_file("platefile")) as f:
        for line in f:
            tok = line.split()
            if len(tok) >= 2:
                out[tok[0]] = tok[1]
    return out


def plateVelo(locList, plate=None, reference="ITRF2008"):
    """Plate velocity ``(len(locList), 3)`` [m/yr] in **E, N, U** order (legacy).

    A station without an a-priori coordinate gets zeros and a warning.
    """
    coords = {}
    with open(_postprocess_file("coordFile")) as f:
        for line in f:
            tok = line.split()
            if len(tok) >= 4:
                coords[tok[3]] = [float(v) for v in tok[:3]]
    table = None if plate else plateDict()
    out = np.zeros([len(locList), 3], dtype="float64")
    for i, sta in enumerate(locList):
        if sta not in coords:
            print(f"WARNING: Cannot determine plate velocity at station {sta}")
            continue
        code = plate if plate else table[sta]
        out[i] = plates.plate_velocity_local(
            code, coords[sta], frame=reference, order="enu"
        ).d
    return out


__all__ = ["plateDict", "plateFullname", "plateVelo"]
