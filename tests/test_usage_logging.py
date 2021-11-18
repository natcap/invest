"""InVEST Model Logging tests."""

import unittest
import tempfile
import shutil
import os

from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import pygeoprocessing
import shapely.geometry
import numpy
import numpy.testing


class UsageLoggingTests(unittest.TestCase):
    """Tests for the InVEST usage logging framework."""

    def setUp(self):
        """Initalize a workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_bounding_boxes(self):
        """Usage logger test that we can extract bounding boxes."""
        from natcap.invest import utils
        from natcap.invest import usage

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31s
        srs_wkt = srs.ExportToWkt()

        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        driver = gdal.GetDriverByName('GTiff')
        raster_array = numpy.ones((20, 20))
        raster = driver.Create(
            raster_path, raster_array.shape[1], raster_array.shape[0],
            1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
        raster.SetProjection(srs_wkt)
        raster_band = raster.GetRasterBand(1)
        raster_band.WriteArray(raster_array)
        raster_band.SetNoDataValue(255)
        raster_geotransform = [2, 2, 0, -2, 0, -2]
        raster.SetGeoTransform(raster_geotransform)
        raster = None

        vector_path = os.path.join(self.workspace_dir, 'vector.gpkg')
        pygeoprocessing.shapely_geometry_to_vector(
            [shapely.geometry.LineString([(4, -4), (10, -10)])], vector_path,
            srs_wkt, "GPKG", ogr_geom_type=ogr.wkbLineString)

        model_args = {
            'raster': raster_path,
            'vector': vector_path,
            'not_a_gis_input': 'foobar',
            'blank_raster_path': '',
            'blank_vector_path': '',
        }

        args_spec = {
            'args': {
                'raster': {'type': 'raster'},
                'vector': {'type': 'vector'},
                'not_a_gis_input': {'type': 'freestyle_string'},
                'blank_raster_path': {'type': 'raster'},
                'blank_vector_path': {'type': 'vector'},
            }
        }

        output_logfile = os.path.join(self.workspace_dir, 'logfile.txt')
        with utils.log_to_file(output_logfile):
            bb_inter, bb_union = usage._calculate_args_bounding_box(
                model_args, args_spec)

        numpy.testing.assert_allclose(
            bb_inter, [-87.234108, -85.526151, -87.233424, -85.526205])
        numpy.testing.assert_allclose(
            bb_union, [-87.237771, -85.526132, -87.23321, -85.526491])

        # Verify that no errors were raised in calculating the bounding boxes.
        self.assertTrue('ERROR' not in open(output_logfile).read(),
                        'Exception logged when there should not have been.')
