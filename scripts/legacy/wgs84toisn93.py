#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
"""
bgo@vedur.is
Iceland Met Office
Feb, 2021
"""


def wgs84toisn93_custdef(llh):
    """
    transforming geodetic ellipssoid wgs84 coordinates to isnet 93

    """
    import pyproj as proj

    wgs84 = proj.Proj("EPSG:4326")  # geodetic on ellipsiod wgs84
    isn93 = proj.Proj(
        "+proj=lcc +lat_1=64.25 +lat_2=65.75 +lat_0=65 \
                     +lon_0=-19 +x_0=500000 +y_0=500000 +no_defs +a=6378137 \
                     +rf=298.257222101 +to_meter=1"
    )  # isnet 1993

    return proj.transform(wgs84, isn93, *llh)  # transfroming to isn93


def wgs84toisn93_code(llh):
    """
    transforming geodetic ellipssoid wgs84 coordinates to isnet 93

    """
    import pyproj as proj

    wgs84 = proj.Proj("EPSG:4326")  # geodetic on ellipsiod wgs84
    isn93 = proj.Proj("EPSG:3057")

    return proj.transform(wgs84, isn93, *llh)  # transfroming to isn93


### main ###
def main():

    import numpy as np

    # example array of coordinates
    llh_coord = np.array(
        [
            [65.257793168, -13.994878478, 196.7861],  # SEY1
            [65.256080055, -13.998536838, 167.3533],  # SEY2
            [65.262081486, -13.988818388, 218.5411],
        ]
    )  # SEY3

    print("-" * 40)
    print("Geodetic coords\nLatitude [dec]\tLongditude [dec]\tHeight [m]")
    [print("{0:.9f}\t{1:.9f}\t{2:.9f}".format(*i)) for i in llh_coord]
    print("-" * 40)

    # converting the array using standard EPSG code.
    isn93_coord = np.apply_along_axis(wgs84toisn93_code, 1, llh_coord)
    print("-" * 40)
    print(
        "ISNET93 coords using standard EPSG:3075 code for isn93\
              \nEast [m]\tNorth [m]\tHeight [m]"
    )
    [print("{0:.4f}\t{1:.4f}\t{2:.4f}".format(*i)) for i in isn93_coord]
    print("-" * 40)

    # converting the array using manually defined tranformation .
    isn93_coord = np.apply_along_axis(wgs84toisn93_custdef, 1, llh_coord)
    print(
        "ISNET93 coords using manually defined isn93\nEast [m]\tNorth [m]\tHeight [m]"
    )
    [print("{0:.4f}\t{1:.4f}\t{2:.4f}".format(*i)) for i in isn93_coord]
    print("-" * 40)


if __name__ == "__main__":
    main()
