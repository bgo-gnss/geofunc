def xyztollh(xyz):
    """
    transforming geodetic ellipssoid wgs84 coordinates to isnet 93

    """
    from pyproj import CRS
    from pyproj import Transformer

    wgs84 = CRS("EPSG:4326")  # geodetic on ellipsiod wgs84
    itrf2008 = CRS("EPSG:5332")

    itrf08towgs84 = Transformer.from_crs(itrf2008, wgs84)

    # transfroming to llh
    return itrf08towgs84.transform(*xyz.transpose(), radians=False)


def llhtoisn94(llh):
    """
    transforming geodetic ellipssoid wgs84 coordinates to isnet 93
    """

    from pyproj import CRS
    from pyproj import Transformer

    wgs84 = CRS("EPSG:4326")  # geodetic on ellipsiod wgs84
    isnet93 = CRS("EPSG:15514")

    llhtoisn93 = Transformer.from_crs(wgs84, isnet93)

    # transfroming to llh
    return llhtoisn93.transform(*llh.transpose(), radians=False)


def ddd2dm(lon, lat, height):
    """
    fractional deegres to d.m
    """

    lon = "{0} {1}".format(int(lon), int(lon % 1 * 60))
    lat = "{0} {1}".format(int(lat), int(lat % 1 * 60))
    height = "{0:.5f}".format(height)

    return lon, lat, height


def ddd2dm_gp(coord):
    """
    fractional degrees to d.m
    """

    return ddd2dm(*coord[0])


def main():
    import pandas as pd
    import geopandas as gpd
    from pyproj import CRS
    # from dataprep.clean import clean_lat_long

    wgs84 = CRS("EPSG:4326")  # geodetic on ellipsiod wgs84
    itrf2008 = CRS("EPSG:5332")

    colNames = ["x", "y", "z"]
    xyz_coord = pd.read_csv(
        "station_coord.xyz",
        delim_whitespace=True,
        index_col=3,
        names=colNames,
        header=0,
    )
    llh_coord = gpd.GeoDataFrame(
        xyz_coord,
        geometry=gpd.points_from_xy(xyz_coord["x"], xyz_coord["y"], xyz_coord["z"]),
        crs=itrf2008,
    ).to_crs(wgs84)
    [
        print(
            "{0:.6f}\t{1:.6f}\t{2:>8.4f}\t{3:s}".format(
                *list(i.geometry.coords)[0], i.Index
            )
        )
        for i in llh_coord.itertuples()
    ]

    # tmp = llh_coord.apply(
    #     lambda x: ddd2dm(*x.geometry.coords[0]),
    #     axis=1,
    # )

    # for sta, i in zip(tmp.index,tmp):
    #     print("{0} {1} {2} {3}".format(sta, *i))


if __name__ == "__main__":
    main()
