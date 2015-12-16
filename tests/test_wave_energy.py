"""Module for Testing the InVEST Wave Energy module."""
import unittest
import tempfile
import shutil
import os
import collections
import csv
import struct

import pygeoprocessing.testing
from pygeoprocessing.testing import scm
from pygeoprocessing.testing import sampledata
import numpy
import numpy.testing
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry.polygon import LinearRing

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), 'data', 'wave_energy')

ReferenceData = collections.namedtuple(
    'ReferenceData', 'projection origin pixel_size')

def projection_wkt(epsg_id):
    """
    Import a projection from an EPSG code.

    Parameters:
        proj_id(int): If an int, it's an EPSG code

    Returns:
        A WKT projection string.
    """
    reference = osr.SpatialReference()
    result = reference.ImportFromEPSG(epsg_id)
    if result != 0:
        raise RuntimeError('EPSG code %s not recognixed' % epsg_id)

    return reference.ExportToWkt()

SRS_LATLONG = ReferenceData(
    projection=projection_wkt(4326),
    origin=(-70.5, 42.5),
    pixel_size=lambda x: (x, -1. * x))

class WindEnergyUnitTests(unittest.TestCase):
    """Unit tests for the Wave Energy module."""

    def setUp(self):
        """Overriding setUp function to create temporary workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_pixel_size_transform(self):
        """WaveEnergy: testing pixel size transform helper function.

        Function name is : 'pixel_size_based_on_coordinate_transform_uri'.
        """
        from natcap.invest.wind_energy import wind_energy

        temp_dir = self.workspace_dir

        srs = sampledata.SRS_WILLAMETTE
        srs_wkt = srs.projection
        spat_ref = osr.SpatialReference()
        spat_ref.ImportFromWkt(srs_wkt)

        srs_latlong = SRS_LATLONG

        # Get a point from the clipped data object to use later in helping
        # determine proper pixel size
        matrix = numpy.array([[1, 1, 1, 1], [1, 1, 1, 1]])
        input_path = os.path.join(temp_dir, 'input_raster.tif')
        # Create raster to use as testing input
        raster_uri = pygeoprocessing.testing.create_raster_on_disk(
            [matrix], srs_latlong.origin, srs_latlong.projection, -1.0,
            srs_latlong.pixel_size(0.033333), filename=input_path)

        raster_gt = pygeoprocessing.geoprocessing.get_geotransform_uri(
            raster_uri)
        point = (raster_gt[0], raster_gt[3])
        raster_wkt = srs_latlong.projection

        # Create a Spatial Reference from the rasters WKT
        raster_sr = osr.SpatialReference()
        raster_sr.ImportFromWkt(raster_wkt)

        # A coordinate transformation to help get the proper pixel size of
        # the reprojected raster
        coord_trans = osr.CoordinateTransformation(raster_sr, spat_ref)
        # Call the function to test
        result = wind_energy.pixel_size_based_on_coordinate_transform_uri(
            raster_uri, coord_trans, point)

        expected_res = (5576.152641937137, 1166.6139341676608)

        # Compare
        for res, exp in zip(result, expected_res):
            pygeoprocessing.testing.assert_close(res, exp, 1e-9)

    def test_count_pixels_groups(self):
        """WaveEnergy: testing 'count_pixels_groups' function."""
        from natcap.invest.wave_energy import wave_energy

        temp_dir = self.workspace_dir
        raster_uri = os.path.join(temp_dir, 'pixel_groups.tif')
        srs = sampledata.SRS_WILLAMETTE

        group_values = [1,3,5,7]
        matrix = numpy.array([[1,3,5,9], [3,7,1,5], [2,4,5,7]])

        # Create raster to use for testing input
        raster_uri = pygeoprocessing.testing.create_raster_on_disk(
            [matrix], srs.origin, srs.projection, -1, srs.pixel_size(100),
            datatype=gdal.GDT_Int32, filename=raster_uri)

        results = wave_energy.count_pixels_groups(raster_uri, group_values)

        expected_results = [2, 2, 3, 2]

        for res, exp_res in zip(results, expected_results):
            pygeoprocessing.testing.assert_close(res, exp_res, 1e-9)

    def test_calculate_percentiles_from_raster(self):
        """WaveEnergy: testing 'calculate_percentiles_from_raster' function."""
        from natcap.invest.wave_energy import wave_energy

        temp_dir = self.workspace_dir
        raster_uri = os.path.join(temp_dir, 'percentile.tif')
        srs = sampledata.SRS_WILLAMETTE

        matrix = numpy.arange(1,101)
        matrix = matrix.reshape(10,10)
        raster_uri = pygeoprocessing.testing.create_raster_on_disk(
            [matrix], srs.origin, srs.projection, -1, srs.pixel_size(100),
            datatype=gdal.GDT_Int32, filename=raster_uri)

        percentiles = [0, 25, 50, 75]

        results = wave_energy.calculate_percentiles_from_raster(
            raster_uri, percentiles)

        expected_results = [1, 26, 51, 76]

        for res, exp_res in zip(results, expected_results):
            self.assertEqual(res, exp_res)


    def test_create_percentile_ranges(self):
        """WaveEnergy: testing 'create_percentile_ranges' function."""
        from natcap.invest.wave_energy import wave_energy

        percentiles = [20, 40, 60, 80]
        units_short = " m/s"
        units_long = " speed of a bullet in m/s"
        start_value = "5"

        result = wave_energy.create_percentile_ranges(
            percentiles, units_short, units_long, start_value)

        exp_result = ["5 - 20 speed of a bullet in m/s",
                      "20 - 40 m/s", "40 - 60 m/s", "60 - 80 m/s",
                      "Greater than 80 m/s"]

        for res, exp_res in zip(result, exp_result):
            self.assertEqual(res, exp_res)

class WaveEnergyRegressionTests(unittest.TestCase):
    """Regression tests for the Wave Energy module."""

    def setUp(self):
        """Overriding setUp function to create temporary workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def generate_base_args(workspace_dir):
        """Generate an args list that is consistent across regression tests."""
        args = {
            'workspace_dir': workspace_dir,
            'wave_base_data_uri': os.path.join(
                SAMPLE_DATA, 'WaveEnergy', 'input', 'WaveData'),
            'analysis_area_uri': 'West Coast of North America and Hawaii',
            'machine_perf_uri': os.path.join(
                SAMPLE_DATA, 'WaveEnergy', 'input',
                'Machine_Pelamis_Performance.csv'),
            'machine_param_uri': os.path.join(
                SAMPLE_DATA, 'WaveEnergy', 'input',
                'Machine_Pelamis_Parameter.csv'),
            'dem_uri': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Marine', 'DEMs', 'global_dem')
        }
        return args

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_valuation(self):
        """WaveEnergy: testing valuation component."""
        from natcap.invest.wave_energy import wave_energy

        args = WaveEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'AOI_WCVI.shp')
        args['valuation_container'] = True
        args['land_gridPts_uri'] = os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'LandGridPts_WCVI.csv')
        args['machine_econ_uri'] =  os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'Machine_Pelamis_Economic.csv')
        args['number_of_machines'] = 28

        wave_energy.execute(args)

        raster_results = [
            'wp_rc.tif', 'wp_kw.tif', 'capwe_rc.tif', 'capwe_mwh.tif',
            'npv_rc.tif', 'npv_usd.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'valuation', raster_path),
                1e-9)

        vector_results = ['GridPts_prj.shp', 'LandPts_prj.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'valuation', vector_path),
                1e-9)

        table_results = ['capwe_rc.csv', 'wp_rc.csv', 'npv_rc.csv']

        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, 'valuation', table_path),
                1e-9)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_aoi(self):
        """WaveEnergy: testing Biophysical component with an AOI."""
        from natcap.invest.wave_energy import wave_energy

        args = WaveEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'AOI_WCVI.shp')

        wave_energy.execute(args)

        raster_results = [
            'wp_rc.tif', 'wp_kw.tif', 'capwe_rc.tif', 'capwe_mwh.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'aoi', raster_path),
                1e-9)

        table_results = ['capwe_rc.csv', 'wp_rc.csv']

        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, 'aoi', table_path),
                1e-9)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_no_aoi(self):
        """WaveEnergy: testing Biophysical component with no AOI."""
        from natcap.invest.wave_energy import wave_energy

        args = WaveEnergyRegressionTests.generate_base_args(self.workspace_dir)

        wave_energy.execute(args)

        raster_results = [
            'wp_rc.tif', 'wp_kw.tif', 'capwe_rc.tif', 'capwe_mwh.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'noaoi', raster_path),
                1e-9)

        table_results = ['capwe_rc.csv', 'wp_rc.csv']

        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, 'noaoi', table_path),
                1e-9)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_valuation_suffix(self):
        """WaveEnergy: testing suffix through Valuation."""
        from natcap.invest.wave_energy import wave_energy

        args = WaveEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'AOI_WCVI.shp')
        args['valuation_container'] = True
        args['land_gridPts_uri'] = os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'LandGridPts_WCVI.csv')
        args['machine_econ_uri'] =  os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'Machine_Pelamis_Economic.csv')
        args['number_of_machines'] = 28
        args['suffix'] = 'val'

        wave_energy.execute(args)

        raster_results = [
            'wp_rc_val.tif', 'wp_kw_val.tif', 'capwe_rc_val.tif',
            'capwe_mwh_val.tif', 'npv_rc_val.tif', 'npv_usd_val.tif']

        for raster_path in raster_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', raster_path)))

        vector_results = ['GridPts_prj_val.shp', 'LandPts_prj_val.shp']

        for vector_path in vector_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', vector_path)))

        table_results = ['capwe_rc_val.csv', 'wp_rc_val.csv', 'npv_rc_val.csv']

        for table_path in table_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', table_path)))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_valuation_suffix_underscore(self):
        """WaveEnergy: testing suffix with an underscore through Valuation."""
        from natcap.invest.wave_energy import wave_energy

        args = WaveEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'AOI_WCVI.shp')
        args['valuation_container'] = True
        args['land_gridPts_uri'] = os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'LandGridPts_WCVI.csv')
        args['machine_econ_uri'] =  os.path.join(
            SAMPLE_DATA, 'WaveEnergy', 'input', 'Machine_Pelamis_Economic.csv')
        args['number_of_machines'] = 28
        args['suffix'] = '_val'

        wave_energy.execute(args)

        raster_results = [
            'wp_rc_val.tif', 'wp_kw_val.tif', 'capwe_rc_val.tif',
            'capwe_mwh_val.tif', 'npv_rc_val.tif', 'npv_usd_val.tif']

        for raster_path in raster_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', raster_path)))

        vector_results = ['GridPts_prj_val.shp', 'LandPts_prj_val.shp']

        for vector_path in vector_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', vector_path)))

        table_results = ['capwe_rc_val.csv', 'wp_rc_val.csv', 'npv_rc_val.csv']

        for table_path in table_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', table_path)))
