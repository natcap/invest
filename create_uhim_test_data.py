"""Create a clip of SF data for UHIM."""
import os
import math

import pygeoprocessing
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

WORKSPACE_DIR = 'uhim_test_data'


def main():
    """Entry point."""
    base_data_path_list = [
        r"C:\Users\jdouglass\Downloads\Urban heat data SF\Buildings.shp",
        r"C:\Users\jdouglass\Downloads\Urban heat data SF\ETo_SFBA.tif",
        r"C:\Users\jdouglass\Downloads\Urban heat data SF\Tair_Sept.tif",
        r"C:\Users\jdouglass\Downloads\Urban heat data SF\LULC_SFBA.tif",
        r"C:\Users\jdouglass\Downloads\Urban heat data SF\Draft_Watersheds_SFEI\Draft_Watersheds_SFEI.shp",]

    clip_polygon_path = r"C:\Users\rpsharp\Documents\bitbucket_repos\invest\uhim_clip.gpkg"
    clip_polygon_vector = gdal.OpenEx(clip_polygon_path, gdal.OF_VECTOR)
    clip_polygon_layer = clip_polygon_vector.GetLayer()
    clip_polygon_feature = next(clip_polygon_layer)
    clip_polygon = clip_polygon_feature.GetGeometryRef()
    clip_centroid = clip_polygon.Centroid()
    print(clip_centroid)
    target_epsg = get_epsg_utm_code(clip_centroid.GetX(), clip_centroid.GetY())
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(target_epsg)
    print(clip_polygon.GetEnvelope())
    clip_bb_wgs84 = convert_to_from_interleaved(clip_polygon.GetEnvelope())
    clip_bb = pygeoprocessing.transform_bounding_box(
        convert_to_from_interleaved(clip_polygon.GetEnvelope()),
        clip_polygon_layer.GetSpatialRef().ExportToWkt(),
        target_srs.ExportToWkt())
    print(clip_bb)

    try:
        os.makedirs(WORKSPACE_DIR)
    except:
        pass

    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4632)

    for path in base_data_path_list:
        target_path = os.path.join(WORKSPACE_DIR, os.path.basename(path))
        if path.endswith('.tif'):
            raster_info = pygeoprocessing.get_raster_info(path)
            target_pixel_size = raster_info['pixel_size']
            target_bb = pygeoprocessing.transform_bounding_box(
                clip_bb_wgs84, wgs84_srs.ExportToWkt(),
                raster_info['projection'])
            pygeoprocessing.warp_raster(
                path, target_pixel_size, target_path,
                'near', target_bb=target_bb)


def get_epsg_utm_code(lng, lat):
    """Return EPSG UTM code for the point (lng, lat)."""
    utm_code = (math.floor((
        lng + 180)/6) % 60) + 1
    lat_code = 6 if lat > 0 else 7
    epsg_code = int('32%d%02d' % (lat_code, utm_code))
    return epsg_code


def convert_to_from_interleaved(bounding_box):
    """Convert `bounding_box` to or from interleaved coordinates.

    Parameters:
        bounding_box (list): a list of [
            xmin, (xmax|ymin), (ymin|xmax), (ymax)] tuples. The order is
            considered "interleaved" if the list is [xmin, ymin, xmax, ymax]
            and "sequential" if [xmin, xmax, ymin, ymax]. This function will
            transpose the inner arguments to convert to/from interleaved
            mode.

    Returns:
        bounding box in the order [0, 2, 1, 3].

    """
    return [bounding_box[i] for i in (0, 2, 1, 3)]


def convert_degree_pixel_size_to_meters(pixel_size, center_lat):
    """Calculate meter size of a wgs84 square pixel.

    Adapted from: https://gis.stackexchange.com/a/127327/2397

    Parameters:
        pixel_size (float): [xsize, ysize] in degrees.
        center_lat (float): latitude of the center of the pixel. Note this
            value +/- half the `pixel-size` must not exceed 90/-90 degrees
            latitude or an invalid area will be calculated.

    Returns:
        `pixel_size` in meters.

    """
    m1 = 111132.92
    m2 = -559.82
    m3 = 1.175
    m4 = -0.0023
    p1 = 111412.84
    p2 = -93.5
    p3 = 0.118
    lat = center_lat * math.pi / 180
    latlen = (
        m1 + m2 * math.cos(2 * lat) + m3 * math.cos(4 * lat) +
        m4 * math.cos(6 * lat))
    longlen = abs(
        p1 * math.cos(lat) + p2 * math.cos(3 * lat) + p3 * math.cos(5 * lat))
    return (longlen * pixel_size[0], latlen * pixel_size[1])

if __name__ == '__main__':
    main()
