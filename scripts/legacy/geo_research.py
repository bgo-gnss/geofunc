#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
""" """

import re

import chardet
import numpy as np
import pyproj as proj
from pyproj import CRS, Transformer

import geofunc.geofunc as gf

isn2004 = CRS("EPSG:5322")
isn93 = CRS("EPSG:3057")
wgs84 = CRS("EPSG:4326")
itrf2008 = CRS("EPSG:5332")
# isnet16 = CRS("EPSG:8086")
isnet16 = CRS("EPSG:8088")

itrf08towgs84 = Transformer.from_crs(itrf2008, wgs84)
itrf08toisn16 = Transformer.from_crs(itrf2008, isnet16)


def wanda_test():
    """
    plate velocity according to UNAVCO plate motion calculator (They don't have ITRF2008)

    Lon(°E)    Lat(°N)   Evel(mm/yr) Nvel(mm/yr) Plate(Reference) Model Site
    66.451443  30.913863   28.02        30.57        IN(NNR)      GEODVEL 2010

    Lon(°E)     Lat(°N)     Xvel(mm/yr) Yvel(mm/yr)) Zvel(mm/yr) Plate(Reference) Model Site
    66.451443  30.913863    -31.93       -3.13          26.27       IN(NNR) GEODVEL 2010


    ########### Ooutput from this program #############
         ##############################
         geocentric cartesian coordiantes
         chmn_xyz in m: 2188616.19513 5021830.02567 3258378.79998
         ##############################
             chmn_llh in deg: 66.45144336 30.91386331 1313.5007
          chmn_llhnew in deg: [   66.45144336    30.91386331  1313.5007    ]
         ##############################
         modeled plate velocity of CHMN in ECEF in (x,y,z) mm/yr: -32.70712821 -3.12149102306 26.7798660615
                                     and in ENU in mm/year: 28.7362208025, 31.1589843048, 0.0920831739681
    ###########      ############         #########          #####################

              chmn_ECEF = [2188616.19513, 5021830.02567, 3258378.79998]
    """

    import numpy as np

    import geofunc.geo as geo

    ## On the border of Afganistan and Pakistan. Indian plate
    indirotp = geo.platePoles("INDI")

    # Coordinates in degrees
    #  This was a source of error
    # chmn_llh = [30.91386331, 66.45144336, 1313.5007]   # (  2192857.94363 1313119.14335 5825445.72125 )
    # lat,long -> long,lat can also be acheved in pyproj using inverse=True (I think !!)
    chmn_llh = [
        66.45144336,
        30.91386331,
        1313.5007,
    ]  # ( 2188616.19513 5021830.02567 3258378.79998)

    ## transforming to earth centered cartasian coordinates according to ITRF2008
    chmn_ECEF = proj.transform(
        wgs84, itrf2008, *chmn_llh
    )  # coordinates in m earth centered earth fixed
    print("#" * 30)
    print("geocentric cartesian coordiantes")
    print("chmn_xyz in m: %s %s %s" % chmn_ECEF)

    chmn_llhdeg = np.array(
        proj.transform(itrf2008, wgs84, *chmn_ECEF, radians=False)
    )  # here radians=False as we want to compare with the original
    print("#" * 30)
    print("   chmn_llh in deg: %s %s %s" % tuple(chmn_llh))
    print("chmn_llhnew in deg: %s " % chmn_llhdeg)
    print("#" * 30)

    # Here radians=True to use as input in geo.eccell(indirotp,chmn_ECEF)
    chmn_llhrad = np.array(proj.transform(itrf2008, wgs84, *chmn_ECEF, radians=True))

    # calculate the modeled plate velocity according ot atlamini 2012
    chmn_disp = geo.rotpole(indirotp, chmn_ECEF)
    print(
        "modeled plate velocity of CHMN in ECEF in (x,y,z) mm/yr: %s %s %s"
        % tuple(chmn_disp * 1000)
    )  # *1000 for m/year -> mm/year

    chmn_enu = geo.eccell(chmn_llhrad, chmn_disp)  ## East North Up in meters/year
    chmn_enu *= 1000  # converting from m/year -> mm/year
    print(" " * 29, "and in ENU in mm/year: %s, %s, %s" % tuple(chmn_enu))


def general_test():

    import numpy as np

    import geofunc.geo as geo

    # warnings.filterwarnings( "ignore")

    eurrotp = geo.platePoles("EURA")

    # itrf2008=proj.Proj("+proj=geocent +ellps=GRS80 +units=m +no_defs") # itrf2008
    # isn2004=proj.Proj("+proj=lcc +lat_1=64.25 +lat_2=65.75 +lat_0=65 +lon_0=-19 +x_0=1700000 +y_0=300000 +no_defs +a=6378137 +rf=298.257222101 +to_meter=1") #isnet 2004
    # isn93=proj.Proj("+proj=lcc +lat_1=64.25 +lat_2=65.75 +lat_0=65 +lon_0=-19 +x_0=500000 +y_0=500000 +no_defs +a=6378137 +rf=298.257222101 +to_meter=1") #isnet 2004
    # wgs84=proj.Proj("+init=EPSG:4326")

    rhof_ECEF = [2456169.74968, -701823.61767, 5824743.24245]
    vonc_ECEF = [2606031.31317, -834418.62174, 5743215.41035]
    coords = np.array(vonc_ECEF)
    # print( coords)
    print("VONC ECEF: %s" % vonc_ECEF)

    itrf08towgs84 = Transformer.from_crs(itrf2008, wgs84)
    itrf08toisn04 = Transformer.from_crs(itrf2008, isn2004)
    itrf08toisn03 = Transformer.from_crs(itrf2008, isn93)

    print("######################################")
    print("rhof_llh in deg: %s %s %s" % itrf08towgs84.transform(*rhof_ECEF))
    print("rhof isnet 2004: %s %s %s" % itrf08toisn04.transform(*rhof_ECEF))
    print("rhof isn93: %s %s %s" % itrf08toisn03.transform(*rhof_ECEF))
    print("######################################")

    vonc_llh = itrf08towgs84.transform(*vonc_ECEF, radians=True)
    print("vonc_llh: {}".format(np.asarray(vonc_llh)))
    vonc_disp = geo.rotpole(eurrotp, np.asarray(vonc_ECEF))
    print("modeled plate velocity of VONC in ECEF: %s" % vonc_disp)
    vonc_enu = geo.eccell(np.asarray(vonc_llh), vonc_disp) * 1000
    print("and in ENU: %s, %s, %s" % tuple([x for x in np.squeeze(vonc_enu)]))
    print("######################################")

    # print "vonc_llh in radians: %s" % vonc_llh
    # vonc_xyz = proj.transform(wgs84, itrf2008, *vonc_llh, radians=True)
    # print vonc_xyz
    # print "vonc_xyz in m: %s %s %s" % vonc_xyz

    llh = np.array(itrf08towgs84.transform(*coords.transpose(), radians=True))
    # transformer = Transformer.from_crs(itrf2008, wgs84)
    # print(transformer)
    disp = geo.rotpole(eurrotp, coords)
    print(geo.eccell(llh, disp) * 1000)

    ##print("DISP: {}".format(disp))
    ##print("len DISP: {}".format(len(disp[0])))


def extract_loftn_haed():
    """
    extract antenna info
    """

    slines = {}
    last_st_line = re.compile("9999 999 00 00 00")
    start_height = 64
    end_height = 70
    start_marker = 1
    end_marker = 5
    start_antenna = 171
    end_antenna = 191

    f = "./station.info.sopac.apr05"
    # Detect file encoding
    with open(f, "rb") as raw_file:
        result = chardet.detect(raw_file.read())
        encoding = result["encoding"]

    # Fallback to latin-1 if detection fails
    if encoding is None:
        encoding = "latin-1"

    print(f"Detected encoding: {encoding}")
    with open(f, "r", encoding=encoding, errors="replace") as f:
        for line in f:
            try:
                # Process your line here
                if line.strip().startswith("#"):
                    continue

                if last_st_line.search(line):  # or tim.search(line):
                    height = line[start_height:end_height]
                    marker = line[start_marker:end_marker]
                    antenna = line[start_antenna:end_antenna]

                    slines[marker] = [height, antenna]
                    # print(f"{marker} {height} {antenna}")

            except UnicodeDecodeError:
                pass

    return slines


def test_bulkdata():
    """ """

    from geofunc.geo import eccell, platePoles, rotpole

    eurrotp = platePoles("EURA")
    noamrotp = platePoles("NOAM")

    # extracting ECEF coordinates from icel.org.bak file
    tmp = gf.extractfromGamitBakf("./icel.org.bak", "Unc")
    tmp = [re.sub(r"_.PS", "", line) for line in tmp]

    staDict = {}
    staDict.update(
        dict([[line.split()[1], list(map(float, line.split()[2:5]))] for line in tmp])
    )

    xyz = np.array(list(staDict.values()))
    llh = np.array(itrf08towgs84.transform(*xyz.transpose(), radians=True)).transpose()
    isn16 = np.array(itrf08toisn16.transform(*xyz.transpose())).transpose()
    disp = rotpole(eurrotp, xyz)
    enu_eura_vel = eccell(llh, disp)
    enu_eura_vel = dict(zip(staDict.keys(), enu_eura_vel))

    disp = rotpole(noamrotp, xyz)
    enu_noam_vel = eccell(llh, disp)
    enu_noam_vel = dict(zip(staDict.keys(), enu_noam_vel))

    with open("station_coord.xyz", "w", encoding="utf-8") as f:
        for key, value in sorted(staDict.items()):
            decoded_key = key.decode("utf-8") if isinstance(key, bytes) else key
            f.write(
                "{: >14.5f}\t{:>14.5f}\t{:>14.5f}\t{:s}\n".format(
                    value[0], value[1], value[2], decoded_key
                )
            )

    # gf.savedisp(staDict, "station_coord.xyz", "")
    header = "East [m]\t  North [m]\t   Up [m]\tStation"

    isn16 = dict(zip(staDict.keys(), isn16))
    gf.savedisp(isn16, "station_coord_isn16.enu", header)
    # gf.savedisp(enu_eura_vel,"eura_disp.txt",header)


def extr_xyz(site):
    """ """

    import re

    import gtimes.timefunc as tf

    import geofunc.geofunc as gf

    slines = []
    smatch = re.compile(site)

    # extracting ECEF coordinates from icel.org.bak file
    tmp = gf.extractfromGamitBakf("icel.org.bak", "Unc")
    tmp = [re.sub(r"_.PS", "", line) for line in tmp]

    for line in tmp:
        if smatch.search(line):
            slines.append(line)

    # staDict = {}
    # staDict.update( [ [ line.split()[8], map(float,line.split()[2:5]) ] for line in slines ]  )

    # for line in slines:
    #    print line.split()[8],line.split()[2],line.split()[3],line.split()[4]

    txyz = [
        map(float, [line.split()[8], line.split()[2], line.split()[3], line.split()[4]])
        for line in slines
    ]

    f = open("%s.txyz" % site, "w")
    for t in txyz:
        t = list(t)
        line = "%s\t%s\t%s\t%s\t%s\n" % (
            tf.TimefromYearf(t[0], "%Y/%m/%d %H:%M:%S"),
            t[0],
            t[1],
            t[2],
            t[3],
        )
        f.write(line)
        print(line)

    f.close()

    # print( tf.TimefromYearf(txyz[:,0],"%Y/%m/%d %H:%M:%S"))
    # print( tf.TimefromYearf(2015.084,"%Y/%m/%d %H:%M:%S"))


def test_platevel():
    """ """

    import re

    import cparser as cp
    import geo_dataread.gps_read as gdr
    import numpy as np
    from pyproj import Transformer

    import geofunc.geo as geo
    import geofunc.geofunc as gf
    from geofunc.geo import itrf2008, wgs84

    itrf08towgs84 = Transformer.from_crs(itrf2008, wgs84)
    # itrf2008 = proj.Proj("+proj=geocent +ellps=GRS80 +units=m +no_defs")  # itrf2008
    # isn2004 = proj.Proj(
    #     "+proj=lcc +lat_1=64.25 +lat_2=65.75 +lat_0=65 +lon_0=-19 +x_0=1700000 +y_0=300000 +no_defs +a=6378137 +rf=298.257222101 +to_meter=1"
    # )  # isnet 2004
    # wgs84 = proj.Proj("+init=EPSG:4326")

    eurrotp = geo.platePoles("EURA")
    noamrotp = geo.platePoles("NOAM")

    # extracting ECEF coordinates from icel.org.bak file
    tmp = gf.extractfromGamitBakf("icel.org.bak", "Unc")

    tmp = [re.sub(r"_.PS", "", line) for line in tmp]

    staDict = {}
    staDict.update(
        dict([[line.split()[1], list(map(float, line.split()[2:5]))] for line in tmp])
    )

    # ----------------------------------

    # f = open('station-plate','w')
    # for station in sorted(StationList):
    #    line = "%s\t%s\n" % (station,"NOAM")
    #    f.write(line)

    noamList = []
    euraList = []

    f = open(cp.Parser().getPostprocessConfig()["pfile"], "r")

    for line in f:
        plateList = line.strip().split()
        if plateList[1] == "EURA":
            euraList.append(plateList[0])
        if plateList[1] == "NOAM":
            noamList.append(plateList[0])

    eura_staDict = {k: staDict[k] for k in euraList if k in staDict}
    noam_staDict = {k: staDict[k] for k in noamList if k in staDict}

    # for,"NOAM" key in staDict.keys():
    #   print staDict[key]

    # transfroming to llh
    # EURA
    eura_xyz = np.array(list(eura_staDict.values()))
    eura_llh = np.array(
        itrf08towgs84.transform(*eura_xyz.transpose(), radians=True)
    ).transpose()
    eura_disp = geo.rotpole(eurrotp, eura_xyz)
    enu_eura_vel = geo.eccell(eura_llh, eura_disp)
    # Add lon lat to the list
    enu_eura_vel = zip(gf.convllh(eura_llh[:, 0:2]), enu_eura_vel)
    enu_eura_vel = dict(zip(eura_staDict.keys(), enu_eura_vel))

    # NOAM
    noam_xyz = np.array(list(noam_staDict.values()))
    noam_llh = np.array(
        itrf08towgs84.transform(*noam_xyz.transpose(), radians=True)
    ).transpose()
    noam_disp = geo.rotpole(noamrotp, noam_xyz)
    enu_noam_vel = geo.eccell(noam_llh, noam_disp)
    # Add lon lat to the list
    enu_noam_vel = zip(gf.convllh(noam_llh[:, 0:2]), enu_noam_vel)
    enu_noam_vel = dict(zip(noam_staDict.keys(), enu_noam_vel))

    # Join NOAM and EURA in  one dict
    enu_vel = enu_eura_vel.copy()
    enu_vel.update(enu_noam_vel)

    staDict = eura_staDict.copy()
    staDict.update(noam_staDict)
    xyz = np.array(list(staDict.values()))
    llh = np.array(itrf08towgs84.transform(*xyz.transpose(), radians=True)).transpose()
    llhDict = dict(zip(staDict.keys(), llh))

    header = "lon [°]\t  lat [°]\t  East [m]\t  North [m]\t Up [m]\t\tStation"

    gdr.savedisp(enu_vel, "plate_ISGPSvel.enu", header)
    # ab=geo.savedisp(enu_vel,header=header)
    # geo.savedisp(enu_noam_vel,"noam_disp.txt",header)

    # geo.savedisp(enu_eura_vel,"eura_disp.txt",header)
    gdr.savedisp(staDict, "station_coord.xyz", "")
    gdr.savedisp(llhDict, "station_coord.llh", "")


def plate_remove():
    """ """
    from geofunc.geofunc import plateVelo

    staList = ["HAFS", "HOFN", "SAUD", "VONC"]

    test = plateVelo(staList)
    print(test)


def phase_center_coord():
    """"""

    loftnet = extract_loftn_haed()
    # extracting ECEF coordinates from icel.org.bak file
    tmp = gf.extractfromGamitBakf("./icel.org.bak", "Unc")
    tmp = [re.sub(r"_.PS", "", line) for line in tmp]

    staDict = {}
    staDict.update(
        dict([[line.split()[1], list(map(float, line.split()[2:5]))] for line in tmp])
    )

    xyz = np.array(list(staDict.values()))
    isn16 = np.array(itrf08toisn16.transform(*xyz.transpose())).transpose()
    isn16 = dict(zip(staDict.keys(), isn16))

    with open("station_coord_isn16.enu", "w", encoding="utf-8") as f:
        for key, value in sorted(isn16.items()):
            decoded_key = key.decode("utf-8") if isinstance(key, bytes) else key
            try:
                f.write(
                    "{: >14.5f}, {:>14.5f}, {:>14.5f}, {:.5f}, {:s}, {:s}\n".format(
                        value[0],
                        value[1],
                        value[2],
                        float(loftnet[decoded_key][0]),
                        loftnet[decoded_key][1],
                        decoded_key,
                    )
                )
            except KeyError:
                continue

    # gf.savedisp(staDict, "station_coord.xyz", "")
    # header = "East [m]\t  North [m]\t   Up [m]\tStation"
    # isn16 = dict(zip(staDict.keys(), isn16))
    # gf.savedisp(isn16, "station_coord_isn16.enu", header)


### main for testing stuff
def main():
    # wanda_test()
    # general_test()
    # extract_loftn_haed()
    # phase_center_coord()

    # test_bulkdata()
    # test_platevel()
    plate_remove()
    # extr_xyz("HAFC")


if __name__ == "__main__":
    main()
