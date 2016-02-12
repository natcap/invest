"""Script to generate shapefiles from InVEST logging database."""

import os
import sqlite3
import urllib

import Pyro4
from osgeo import osr
from osgeo import ogr

Pyro4.config.SERIALIZER = 'marshal'

INVEST_USAGE_LOGGER_URL = ('http://data.naturalcapitalproject.org/'
                           'server_registry/invest_usage_logger/')


def get_run_summary_as_shapefile(database_path, run_summary_shapefile_path):
    """Construct a shapefile of polygons of model run bounding boxes."""
    driver = ogr.GetDriverByName('ESRI Shapefile')

    if os.path.isfile(run_summary_shapefile_path):
        os.remove(run_summary_shapefile_path)
    datasource = driver.CreateDataSource(run_summary_shapefile_path)

    lat_lng_ref = osr.SpatialReference()
    lat_lng_ref.ImportFromEPSG(4326)  # EPSG 4326 is lat/lng

    polygon_layer = datasource.CreateLayer(
        'model_run_summary', lat_lng_ref, ogr.wkbPolygon)

    polygon_layer.CreateField(ogr.FieldDefn('n_runs', ogr.OFTInteger))
    polygon_layer.CreateField(ogr.FieldDefn('model', ogr.OFTString))

    db_connection = sqlite3.connect(database_path)
    db_cursor = db_connection.cursor()
    db_cursor.execute(
        """SELECT model_name, bounding_box_intersection,
count(model_name) FROM model_log_table WHERE bounding_box_intersection
not LIKE 'None' GROUP BY model_name, bounding_box_intersection;""")

    for line in db_cursor:
        try:
            model_name, bounding_box_string, n_runs = line
            n_runs = int(n_runs)
            bounding_box = list(
                [float(x) for x in
                 bounding_box_string[1:-1].split(',')])
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(bounding_box[0], bounding_box[3])
            ring.AddPoint(bounding_box[0], bounding_box[1])
            ring.AddPoint(bounding_box[2], bounding_box[1])
            ring.AddPoint(bounding_box[2], bounding_box[3])
            ring.AddPoint(bounding_box[0], bounding_box[3])
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            feature = ogr.Feature(polygon_layer.GetLayerDefn())
            feature.SetGeometry(poly)
            feature.SetField('n_runs', n_runs)
            feature.SetField('model', str(model_name))
            polygon_layer.CreateFeature(feature)
        except Exception:
            print 'unable to create a bounding box for ' + line

    datasource.SyncToDisk()

if __name__ == '__main__':
    PATH = urllib.urlopen(INVEST_USAGE_LOGGER_URL).read().rstrip()
    LOGGING_SERVER = Pyro4.Proxy(PATH)
    print 'download db'
    DB_STRING = LOGGING_SERVER.get_run_summary_db()
    print 'writing string'
    DB_PATH = 'run_summary.db'
    open(DB_PATH, 'wb').write(DB_STRING)
    RUN_SUMMARY_SHAPEFILE_PATH = 'run_summary.shp'
    print 'generating summary shapefile'
    get_run_summary_as_shapefile(DB_PATH, RUN_SUMMARY_SHAPEFILE_PATH)
