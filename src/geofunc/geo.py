#!/usr/bin/python3
"""
loc_to_geod( loc_coord )
geod_to_loc(pos_geod )
gdatum(datum="IGS08")
xyzell(datum,xyzcoor): 				XYZ -> llh using the datum designed for using
                                                    datum() as input
ellecc(llh,vector,optional)
eccell(llh,vector, swich ):			        converting a vector in xyz coord. system to local                                                        enu system in m
xyz2enumatr(refarray,swich):			rotation matrix to convert xyz coord. system to local enu system or vice versa:w
greatcircd(coor1,coor2):				distance along great circle
M:
platePoles(plate,units="ry")
rotpole(plate,xyz,swich):
appendSpherical_np(xyz)
"""

import pyproj as proj
from pyproj import CRS, Transformer

import numpy as np

EARTH_ERAD = 6378137.0

# itrf2008=proj.Proj("+proj=geocent +ellps=GRS80 +units=m +no_defs") # itrf2008
# lonlat=proj.Proj(init="EPSG:4326")
# isn2004=proj.Proj("+proj=lcc +lat_1=64.25 +lat_2=65.75 +lat_0=65 +lon_0=-19 +x_0=1700000 +y_0=300000 +no_defs +a=6378137 +rf=298.257222101 +to_meter=1") #isnet 2004
itrf2008 = CRS("EPSG:5332")
# itrf2008 = CRS.from_epsg(5332)
wgs84 = CRS("EPSG:4326")
lonlat = CRS("EPSG:4326")
isn2004 = CRS("EPSG:5322")
isn93 = CRS("EPSG:3057")

itrf08towgs84 = Transformer.from_crs(itrf2008, wgs84, always_xy=True)
itrf08toisn04 = Transformer.from_crs(itrf2008, isn2004)
itrf08toisn03 = Transformer.from_crs(itrf2008, isn93)


### core functions for manipulating and working with vectors on earth
def loc_to_geod(loc_coord):
    """

    Function to convert a location to geodetic coordinates. It calculates the following:
    colat           - Colatitude (used so that loc_coord and pos_geod - can be the same arrays)
    colat_trun      - Coltitude truncated to the nearest 10 asecs - Used to compute small circle radius
    loc_coord(3)    - longitude (rad), latitude (rad) and height (m)
    pos_geod(3)     - Distance from equator, distance from Greenwich - meridian, and height above ellipsiod (all m)


    Args:
        loc_coord: numpy array longitude (rad), latitude (rad) and height (m)

    Returns:
        pos_geod: numpy array distance from equator, distance from Greenwich - meridian, and height above ellipsiod (all m)

    """
    pos_geod = np.zeros(3)

    #  Distance from equator
    colat = np.pi / 2 - loc_coord[1]
    # ! 20000 = 1/10 in radians
    colat_trun = np.round(colat * 20000) / 20000

    pos_geod[0] = loc_coord[1] * EARTH_ERAD

    # Distance from Greenwich meridian along small circle
    if loc_coord[0] < 0:
        pos_geod[1] = (2 * np.pi + loc_coord[0]) * EARTH_ERAD * np.sin(colat_trun)
    else:
        pos_geod[1] = loc_coord[0] * EARTH_ERAD * np.sin(colat_trun)

    # Height above ellipssiod

    pos_geod[2] = loc_coord[2]

    return pos_geod


def geod_to_loc(pos_geod):
    """

    Function to convert geodetic coordinates to a location. It calculates the following:
    colat           - Colatitude (used so that loc_coord and pos_geod
                    - can be the same arrays)
    colat_trun      - Coltitude truncated to the nearest 10 asecs
                    - Used to compute small circle radius
    loc_coord(3)    - longitude (rad), latitude (rad) and
                    - height (m)
    pos_geod(3)     - Distance from equator, distance from Greenwich
                    - meridian, and height above ellipsiod (all m)

    Returns:
        loc_coord: numpy array longitude (rad), latitude (rad) and height (m)
    """

    loc_coord = np.zeros(3)

    #  Get the colat and loc_coord(1)
    loc_coord[1] = pos_geod[0] / EARTH_ERAD

    colat = np.pi / 2 - loc_coord[1]
    #   ! 20000 = 1/10" in radians
    colat_trun = np.round(colat * 20000) / 20000

    # Now get longitude
    loc_coord[0] = pos_geod[1] / (EARTH_ERAD * np.sin(colat_trun))
    if loc_coord[0] > np.pi:
        loc_coord[0] = loc_coord[0] - 2 * np.pi

    # Height above ellipssiod

    loc_coord[2] = pos_geod[2]

    return loc_coord


def gdatum(datum="IGS08"):
    """
    Function that extracts Datum information from bernese DATUM file. It returns the major axes ($ae), minor axes ($be), scale ($sc), shift (@dx), rotation (@rx)
    (Planned function - yet to be implemented)
    Args:
        datum

    Returns:
        To be Implemented
    """
    raise NotImplementedError


def xyzell(xyzcoor, datum=itrf2008, radians=True):
    """Convert geocentric ECEF ``xyz`` to geodetic ``lat, lon, height``.

    .. warning::

       **Returns (LATITUDE, LONGITUDE, HEIGHT) — not (lon, lat, height).**

       ``lonlat`` is ``CRS("EPSG:4326")``, whose *authority* axis order is
       (latitude, longitude), and :func:`pyproj.transform` honours that order
       here because it is called without ``always_xy=True``. Contrast
       ``itrf08towgs84`` in this same module, which *does* pass
       ``always_xy=True`` and so yields x/y (lon/lat) order — this module is
       not internally consistent, so check each helper rather than assuming.

       This docstring previously claimed ``(lon, lat, height)``. That was true
       under pyproj 1.x, where ``lonlat`` was ``proj.Proj(init="EPSG:4326")``
       (see the commented-out definitions at the top of this module) and the
       traditional x/y order applied. The pyproj 1→2 migration silently flipped
       the axis order; the docstring was never updated, so the two disagreed
       for years and callers were written against the wrong contract.

       Verified 2026-08-10 against REYK (ECEF 2587383.91122, -1043033.58722,
       5716564.1974) → ``[64.138788, -21.955490, 93.028]``: 64.14 is the
       latitude of Reykjavík, -21.96 its longitude. The same two numbers appear
       *correctly labelled but in the opposite order* in the pre-migration
       example still shown in :func:`geofunc.geofunc.getStationCoordinates`,
       which is the clearest evidence of when the flip happened.

    Args:
        xyzcoor: sequence of geocentric ``(X, Y, Z)`` in metres.
        datum: source CRS for the ECEF coordinates (default ITRF2008,
            EPSG:5332, GRS80 ellipsoid).
        radians: when True (the default) latitude and longitude are returned in
            radians; pass ``radians=False`` for degrees. Height is always in
            metres and is **ellipsoidal**, not orthometric — geoid separation
            is location dependent (~60-70 m in Iceland) and must be applied
            separately if an orthometric height is wanted.

    Returns:
        ``np.array([latitude, longitude, height_over_ellipsoid])``
    """

    return np.array(proj.transform(datum, lonlat, *xyzcoor, radians=radians))


def ellecc(llh, vector):
    """

    Function that converts a list of vectors in ECEF coordinates to the local enu coordinate system at llh.

    Args:
        enu displacement -> xyz displacement
        usage: &ellecc(llh,vector,optional)
        llh: reference to coordinates in ll or llh
        vector: reference to vector array in neu coor

    Returns:
        displacement in enu coordinate system.

    (Planned function, not yet implemented)

    """
    raise NotImplementedError


def eccell(ll, vector):
    """
    Function that converts xyz displacement to enu displacement for a list of vectors in ECEF coordinates to the local enu coordinate system at llh.
    xyz displacement -> enu displacement

    Args:
        llh: list of coordinates in ll or llh line vector (long,lat,[height]): (rad,rad,m)
        vector: A list of vector in ECEF coordinates (m)

    Returns:
        List of the vectors in vectors in a local enu reference system.
    """

    if len(ll) != len(vector):
        raise ValueError

    rmat = xyz2enumatr(ll)

    if len(vector.shape) == 1:  # has to be numpy array
        vector = np.expand_dims(vector, axis=0)

    return np.squeeze(
        [np.transpose(x * y.reshape(3, 1)) for (x, y) in zip(rmat, vector)]
    )


def xyz2enumatr(ll):
    """
    Function that calculates a tranformation matrix to convert xyz to local enu coordinates system enu or viceversa.

    Args:
        ll list of lon,lat (radians) locations to compute the transformation matrix

    Returns:
        list of transformation matrixes for the corresponding ll values

    """

    def f(x):
        return [
            np.sin(x),
            np.cos(x),
        ]  # For calculating sin(lambda),shi(phi),cos(lambda),cos(phi)

    # Creating the rotation matrix [ [-slmb, clmb, 0], [-sphi*clmb, -sphi*slmb, cphi], [cphi*clmb, cphi*slmb, sphi]]
    # a = (slmb, sphi, clmb, cphi)
    def m(x):
        return [
            np.matrix(
                [
                    [-a[0], a[2], 0],
                    [-a[1] * a[2], -a[1] * a[0], a[3]],
                    [a[3] * a[2], a[3] * a[0], a[1]],
                ]
            )
            for a in x
        ]

    if len(ll.shape) == 1:  # In case the array is only 1 demintion
        ll = np.expand_dims(ll, axis=0)

    # calculating
    [slmb, sphi], [clmb, cphi] = f(ll[:, 0:2].transpose())

    rmat = m(zip(slmb, sphi, clmb, cphi))
    return rmat


def greatcircd(coor1, coor2, ellps="GRS80"):
    """
    Function that calculates the  distance along great circle between two points on the Earth's surface.

    Args:
        coor1: reference to one point on earth's surface given in long, lat radians.
        coor2: reference to another point on earth's surface given in long, lat radians.

    Returns:
        great circle distance between coor1 and coor2

    """

    geod = proj.Geod(ellps=ellps)

    print(geod)


def M():
    """
    Function that calculates the Matrix of convenience in the local enu coordinate system.

    Args:

    Returns:

    """

    M = np.matrix("1, 0, 0, -1, 0, 0; 0, 1, 0, 0, -1, 0; 0, 0, 1, 0, 0, -1")
    return M


def platePoles(plate, units="ry", reference="ITRF2008"):
    """
    Function that calculates the angular velocities of tectonic plates as published in altamimi et al (2012).

    Args:
        plate: plate name (string)
        units: angular velocity units (string)
        reference: reference frame to use. Default="ITRF2008"

    Returns:
        pole: angular velocity of the plate (mas/Y)

    """
    mas2dy = 1.0 / (60 * 60) / 1000  # converting mas/Y to deg/Y

    # The plate name
    plates = {
        "EURA": 0,
        "NOAM": 1,
        "NAZC": 2,
        "INDI": 3,
        "NUBI": 4,
    }

    if reference == "ITRF2008":
        # ITRIF2008 Absolute Plate rotation poles From Table 3 in altamimi etal2012
        poles = np.array(
            [
                [-0.083, -0.534, 0.750],
                [0.035, -0.662, -0.100],
                [-0.330, -1.551, 1.625],
                [1.232, 0.303, 1.540],
                [0.095, -0.598, 0.723],
            ]
        )  # [mas/y]
    elif reference == "ITRF2014":
        # ITRIF2014 Absolute Plate rotation poles From Table 1
        # from altamimi etal2017 10.1093/gji/ggx136
        poles = np.array(
            [
                [-0.085, -0.531, 0.770],
                [0.024, -0.694, -0.063],
                [-0.333, -1.544, 1.623],
                [1.154, -0.005, 1.454],
                [0.099, -0.614, 0.733],
            ]
        )  # [mas/y]
    else:
        poles = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )  # [mas/y]

    pole = poles[plates[plate], :]

    if units == "masy":
        return pole

    # converting to deg/Y
    pole = pole * mas2dy
    if units == "degy":
        return pole
    if units == "degMy":
        return pole * 1000000

    if units == "ry":
        return np.radians(pole)  # to radians/year


def rotpole(wxwywz, xyz, plate=None):
    """
    Function that calculates the nnr-nuvela velocity of a given plate in geocentric coordinates.

    If plate is passed:
        Calculate the nnr-nuvela velocity of a given plate in geocentric coordinates.
    If plate is not passed:
        Calculate the velocity  (cross product) in a given point in ECEF coordinates with for a given rotation pole wxwywz

    Args:
        plate: plate name (string) or None
        wxwywz: angular velocity of the plate (mas/Y)
        xyz: reference to a point in ECEF coordinates

    Returns:
        nnr-nuvela velocity of a given plate in geocentric coordinates

    """
    if plate is not None:
        pass
    else:
        if xyz is not None and wxwywz is not None:
            return np.cross(wxwywz, xyz)
        else:
            print("you need to pass a location in rad and angular velocity")


def appendSpherical_np(xyz):
    """
    Function that calculates the spherical coordinates of a point.

    Args:
        xyz: reference to a point in ECEF coordinates

    Returns:
        spherical coordinates
    """

    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    ptsnew[:, 3] = np.sqrt(xy + xyz[:, 2] ** 2)
    ptsnew[:, 4] = np.arctan2(
        np.sqrt(xy), xyz[:, 2]
    )  # for elevation angle defined from Z-axis down
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:, 5] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return ptsnew
