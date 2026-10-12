"""DEPRECATED legacy geodesy helpers — kept as thin shims for one release.

Use instead:

==========================  ==================================================
legacy                      replacement
==========================  ==================================================
``xyzell``                  ``geofunc.local.ecef_to_geodetic`` (named
                            ``Geodetic(lat, lon, h)``, radians, GRS80) or
                            ``geofunc.coords.transform`` (lon, lat, h)
``platePoles``              ``geofunc.plates.euler_pole``
``rotpole``                 ``geofunc.local.euler_velocity`` /
                            ``geofunc.plates.plate_velocity``
``eccell``/``xyz2enumatr``  ``geofunc.local.ecef_to_local`` (N, E, U by
                            default; the legacy rows were E, N, U)
``itrf08to*`` transformers  ``geofunc.frames`` / ``geofunc.coords``
==========================  ==================================================

Removed 2026-10-12 (no callers left in the ecosystem): ``loc_to_geod`` and
``geod_to_loc`` (the GLOBK "nearest 10 arcsec small circle" convention, which
glorg 10.71 does not use — see ``geofunc.local`` ``convention="globk"``),
``gdatum`` and ``ellecc`` (``raise NotImplemented``), ``greatcircd`` (a stub;
use ``geofunc.frames.geodesic``), ``M``, ``eccell``, ``xyz2enumatr`` and
``appendSpherical_np``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pyproj as proj
from pyproj import CRS, Transformer

from geofunc import plates as _plates

warnings.warn(
    "geofunc.geo is deprecated: use geofunc.local, geofunc.plates or geofunc.frames",
    DeprecationWarning,
    stacklevel=2,
)

EARTH_ERAD = 6378137.0

itrf2008 = CRS("EPSG:5332")
wgs84 = CRS("EPSG:4326")
lonlat = CRS("EPSG:4326")
isn2004 = CRS("EPSG:5322")
isn93 = CRS("EPSG:3057")

itrf08towgs84 = Transformer.from_crs(itrf2008, wgs84, always_xy=True)
itrf08toisn04 = Transformer.from_crs(itrf2008, isn2004)
itrf08toisn03 = Transformer.from_crs(itrf2008, isn93)


def xyzell(xyzcoor, datum=itrf2008, radians=True):
    """Geocentric ``xyz`` → ``np.array([LATITUDE, LONGITUDE, height])``.

    .. warning:: (latitude, longitude, h) — NOT (lon, lat, h). pyproj ≥ 2
       honours EPSG:4326's authority axis order here (no ``always_xy``); the
       pyproj 1.x code returned (lon, lat). Pinned by
       ``tests/test_xyzell_axis_order.py``. Height is ellipsoidal.
    """
    return np.array(proj.transform(datum, lonlat, *xyzcoor, radians=radians))


def platePoles(plate, units="ry", reference="ITRF2008"):
    """Euler pole of ``plate`` (``units``: ``ry`` rad/yr, ``masy``, ``degy``, ``degMy``).

    An unknown ``reference`` gives a zero pole, as the legacy function did.
    """
    if reference.lower() not in _plates.PMM:
        return np.zeros(3)
    pole = _plates.euler_pole(plate, reference, units="mas/yr")
    if units == "masy":
        return pole
    deg = pole / 3.6e6
    if units == "degy":
        return deg
    if units == "degMy":
        return deg * 1e6
    if units == "ry":
        return np.radians(deg)
    return None


def rotpole(wxwywz, xyz, plate=None):
    """Velocity ``ω × x`` (ECEF) of a point ``xyz`` on a pole ``wxwywz``."""
    if xyz is not None and wxwywz is not None:
        return np.cross(wxwywz, xyz)
    print("you need to pass a location in rad and angular velocity")
    return None
