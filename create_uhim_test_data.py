"""Create a clip of SF data for UHIM."""
import math

import pygeoprocessing
from osgeo import gdal
from osgeo import ogr
from osgeo import osr


def main():
    """Entry point."""
    base_data_path_list = [
        r"C:\Users\rpsharp\Dropbox\Urban InVEST\Urban heat data SF\Buildings.shp",
        r"C:\Users\rpsharp\Dropbox\Urban InVEST\Urban heat data SF\ETo_SFBA.tif",
        r"C:\Users\rpsharp\Dropbox\Urban InVEST\Urban heat data SF\Tair_Sept.tif",
        r"C:\Users\rpsharp\Dropbox\Urban InVEST\Urban heat data SF\LULC_SFBA.tif",
        r"C:\Users\rpsharp\Dropbox\Urban InVEST\Urban heat data SF\Draft_Watersheds_SFEI\Draft_Watersheds_SFEI.shp",]

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
    clip_bb = pygeoprocessing.transform_bounding_box(
        convert_to_from_interleaved(clip_polygon.GetEnvelope()),
        clip_polygon_layer.GetSpatialRef().ExportToWkt(),
        target_srs.ExportToWkt())
    print(clip_bb)


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

if __name__ == '__main__':
    main()
