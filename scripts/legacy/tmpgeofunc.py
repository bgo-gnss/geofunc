#!/usr/bin/python
"""
convllh(llh,radians=True)
extractfromGamitBakf(cfile, stations)
savedisp(dataDict,fname=None, header="")
"""

import numpy as np
import geofunc.geo as geo


def convllh(llh, radians=True):
    """
    input:
        llh: array_like longitude (deg/rad), latitude (deg/rad) and height (m)

    out:
        llh: numpy array longitude (rad/deg), latitude (rad/deg) and height (m)
    """

    if type(llh) is not np.ndarray:
        llh = np.array(llh)

    ll = llh.copy()

    if radians:
        ll[:, 0:2] = np.degrees(ll[:, 0:2])
    else:
        ll[:, 0:2] = np.radians(ll[:, 0:2])

    return ll


def plateVelo(locList, plate=None):
    """ """
    import pyproj as proj
    import cparser as cp
    import numpy as np

    staDict = {}

    NEU_vel = np.zeros([len(locList), 3], dtype="float64")

    f = open(cp.Parser().getPostprocessConfig()["coordfile"], "r")
    staDict.update(
        dict([[line.split()[3], list(map(float, line.split()[0:3]))] for line in f])
    )

    index = 0
    for sta in locList:
        if plate:
            rotp = geo.platePoles(plate)
        else:
            rotp = geo.platePoles(plateDict()[sta])
        if sta in staDict:
            disp = geo.rotpole(rotp, staDict[sta])
            # llh  = np.array(proj.transform(geo.itrf2008, geo.lonlat, *staDict[sta] , radians=True) ) #obsolete
            llh = np.array(proj.transform(geo.itrf2008, geo.lonlat, *staDict[sta]))

            llh[0:2] = np.deg2rad(llh[0:2])  # inplace of radians=True
            NEU_vel[index, :] = geo.eccell(llh, disp)
        else:
            NEU_vel[index, :] = np.zeros([1, 3], dtype="float64")
            print("WARNING: Cannot determine plate velocity at station {}".format(sta))
        index += 1

    return NEU_vel


def plateFullname(plate):
    """
    Maps tectonik plate short name to its full name
    (unfinished)
    """

    plateFullname = {
        "NOAM": "North American plate",
        "EURA": "Eurasian Plate",
        "NAZC": "Nazca",
        "INDI": "Indian Plate",
        "NUBI": "Nubian Plate",
        "ARAB": "Arabia",
    }

    return plateFullname[plate]


def plateDict():
    """
    returns station: plate shortname dictionary
    """
    import cparser as cp

    plateDict = {}

    f = open(cp.Parser().getPostprocessConfig()["pfile"], "r")
    for line in f:
        sta, plate = line.rstrip().split()
        plateDict[sta] = plate
    f.close()

    return plateDict


### routines to extract and save coordiantes and time series from gamit
def extractfromGamitBakf(cfile, stations):
    """ """
    import re

    slines = []

    site = re.compile(stations)
    f = open(cfile, "r")

    for line in f:
        if site.search(line):  # or tim.search(line):
            slines.append(line.rstrip())

    return slines


def savedisp(dataDict, fname=None, header=""):
    """ """
    from collections import OrderedDict

    valtype = type(list(dataDict.values())[0])

    dataDict = OrderedDict(sorted(dataDict.items()))

    datavalues = list(dataDict.values())
    if (valtype is list) or (valtype is np.ndarray):
        fmt = "% 3.8f\t% 2.8f\t% 2.8f\t%s"
        ab = np.zeros(
            len(dataDict.keys()),
            dtype=[
                ("var1", "float"),
                ("var2", "float"),
                ("var3", "float"),
                ("var4", "a4"),
            ],
        )
        ab["var1"] = np.squeeze(datavalues)[:, 0]
        ab["var2"] = np.squeeze(datavalues)[:, 1]
        ab["var3"] = np.squeeze(datavalues)[:, 2]
        ab["var4"] = [str(item) for item in dataDict.keys()]
    if valtype is tuple:
        fmt = "% 3.8f\t% 2.8f\t% 2.8f\t%2.8f\t%2.8f\t%s"
        ab = np.zeros(
            len(dataDict.keys()),
            dtype=[
                ("var1", "float"),
                ("var2", "float"),
                ("var3", "float"),
                ("var4", "float"),
                ("var5", "float"),
                ("var6", "a4"),
            ],
        )
        ab["var1"] = np.squeeze(list(zip(*datavalues[:]))[0])[:, 0]
        ab["var2"] = np.squeeze(list(zip(*datavalues[:]))[0])[:, 1]
        ab["var3"] = np.squeeze(list(zip(*datavalues[:]))[1])[:, 0]
        ab["var4"] = np.squeeze(list(zip(*datavalues[:]))[1])[:, 1]
        ab["var5"] = np.squeeze(list(zip(*datavalues[:]))[1])[:, 2]
        ab["var6"] = [str(item) for item in dataDict.keys()]

    if fname:
        np.savetxt(fname, ab, fmt=fmt, header=header)
    return ab


def getStationCoordinates(station_list=None):
    """
    Return pandas dataframe with stations and longitude, latitude, height.
    INPUT: list of station codes (4ch names)
        if None, return all available stations
    OUTPUT: pandas dataframe (index:station IDs, colums: coordinates)
    EXAMPLE:
        station_list=['REYK','AUST','SVIN']
        sLLH = getStationCoordinates(station_list)
        sLLH
        [Out]
                    lon        lat           elev
        REYK -21.955488  64.138786      93.012961
        AUST -19.080568  63.674361    1438.359718
        SVIN -16.816561  64.009312     727.272716
    """
    import cparser as cp
    import geofunc.geo as geo
    import pandas as pd

    sta_llh = {}
    sta_locs = cp.Parser().getStationCoordinates()
    if station_list is None or not station_list:
        station_list = sta_locs.copy()
    for k in station_list:
        if k in sta_locs:
            sta_llh[k] = geo.xyzell(sta_locs[k], radians=False)  # lon,lat,elev.
        else:
            sta_llh[k] = np.array([None, None, None], float)
    df_sloc = pd.DataFrame.from_dict(sta_llh, orient="index")
    df_sloc.columns = ["lon", "lat", "elev"]
    return df_sloc
