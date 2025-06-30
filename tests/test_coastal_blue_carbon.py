# -*- coding: utf-8 -*-
"""Tests for Coastal Blue Carbon Functions."""
import glob
import logging
import os
import pprint
import shutil
import tempfile
import textwrap
import unittest

import numpy
import pandas
from scipy.sparse import dok_matrix
import pygeoprocessing
from natcap.invest import utils
from natcap.invest import validation
from osgeo import gdal
from osgeo import osr

gdal.UseExceptions()
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'coastal_blue_carbon')
LOGGER = logging.getLogger(__name__)


def make_raster_from_array(base_raster_path, array):
    """Create a raster on designated path with arbitrary values.
    Args:
        base_raster_path (str): the raster path for making the new raster.
    Returns:
        None.
    """
    # UTM Zone 10N
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)
    projection_wkt = srs.ExportToWkt()

    origin = (461261, 4923265)
    pixel_size = (1, 1)
    no_data = -1

    pygeoprocessing.numpy_array_to_raster(
        array, no_data, pixel_size, origin, projection_wkt,
        base_raster_path)

class TestPreprocessor(unittest.TestCase):
    """Test Coastal Blue Carbon preprocessor functions."""

    def setUp(self):
        """Create a temp directory for the workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_constructed_data(self):
        """CBC Preprocessor: Test on constructed data."""
        from natcap.invest.coastal_blue_carbon import preprocessor

        pixelsize = (10, -10)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        wkt = srs.ExportToWkt()
        nodata = -1
        origin_2000 = (30, -50)
        lulc_2000_array = numpy.array([
            [nodata, nodata, 1, 2],
            [nodata, nodata, 2, 2],
            [nodata, nodata, 2, 1]], dtype=numpy.int32)
        origin_2010 = (50, -50)
        lulc_2010_array = numpy.array([
            [2, 2],
            [2, 2],
            [2, 2]], dtype=numpy.int32)

        lulc_2000_path = os.path.join(self.workspace_dir, '2000.tif')
        lulc_2010_path = os.path.join(self.workspace_dir, '2010.tif')
        for (array, origin, target_path) in [
                (lulc_2000_array, origin_2000, lulc_2000_path),
                (lulc_2010_array, origin_2010, lulc_2010_path)]:
            pygeoprocessing.geoprocessing.numpy_array_to_raster(
                array, nodata, pixelsize, origin, wkt, target_path)

        snapshot_csv_path = os.path.join(self.workspace_dir, 'snapshots.csv')
        with open(snapshot_csv_path, 'w') as snapshots_table:
            snapshots_table.write(textwrap.dedent(
                """\
                snapshot_year,raster_path
                2010,2010.tif
                2000,2000.tif
                """))

        lulc_attributes_path = os.path.join(self.workspace_dir,
                                            'lulc-attributes.csv')
        with open(lulc_attributes_path, 'w') as attributes_table:
            attributes_table.write(textwrap.dedent(
                """\
                lulc-class,lucode,is_coastal_blue_carbon_habitat
                mangrove,1,TRUE
                parkinglot,2,FALSE
                """))

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'lulc_lookup_table_path': lulc_attributes_path,
            'landcover_snapshot_csv': snapshot_csv_path,
        }
        preprocessor.execute(args)

        # Verify output raster dimensions.
        dimensions = set()
        for year in (2000, 2010):
            lulc_path = os.path.join(
                args['workspace_dir'], 'outputs_preprocessor',
                f'aligned_lulc_{year}.tif')
            dimensions.add(
                pygeoprocessing.get_raster_info(lulc_path)['raster_size'])
        self.assertEqual(dimensions, {(2, 3)})

        transition_table_path = os.path.join(
            args['workspace_dir'], 'outputs_preprocessor',
            'carbon_pool_transition_template.csv')
        with open(transition_table_path) as transition_table:
            self.assertEqual(
                transition_table.readline(),
                'lulc-class,mangrove,parkinglot\n')
            self.assertEqual(
                transition_table.readline(),
                'mangrove,,disturb\n')
            self.assertEqual(
                transition_table.readline(),
                'parkinglot,,NCC\n')

            # After the above lines is a blank line, then the legend.
            # Deliberately not testing the legend.
            self.assertEqual(transition_table.readline(), '\n')

    def test_sample_data(self):
        """CBC Preprocessor: Test on sample data."""
        from natcap.invest.coastal_blue_carbon import preprocessor

        snapshot_csv_path = os.path.join(
            REGRESSION_DATA, 'inputs', 'snapshots.csv')

        args = {
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'results_suffix': '150225',
            'lulc_lookup_table_path': os.path.join(
                REGRESSION_DATA, 'inputs', 'lulc_lookup.csv'),
            'landcover_snapshot_csv': snapshot_csv_path,
        }
        preprocessor.execute(args)

        # walk through all files in the workspace and assert that outputs have
        # the file suffix.
        non_suffixed_files = []
        outputs_dir = os.path.join(
            args['workspace_dir'], 'outputs_preprocessor')
        for root_dir, dirnames, filenames in os.walk(outputs_dir):
            for filename in filenames:
                if not filename.lower().endswith('.txt'):  # ignore logfile
                    basename, extension = os.path.splitext(filename)
                    if not basename.endswith('_150225'):
                        path_rel_to_workspace = os.path.relpath(
                            os.path.join(root_dir, filename),
                            args['workspace_dir'])
                        non_suffixed_files.append(path_rel_to_workspace)

        if non_suffixed_files:
            self.fail('%s files are missing suffixes: %s' %
                      (len(non_suffixed_files),
                       pprint.pformat(non_suffixed_files)))

        expected_landcover_codes = set(range(0, 24))
        found_landcover_codes = set(pandas.read_csv(
            os.path.join(outputs_dir, 'carbon_biophysical_table_template_150225.csv')
        )['lucode'].values)
        self.assertEqual(expected_landcover_codes, found_landcover_codes)

    def test_transition_table(self):
        """CBC Preprocessor: Test creation of transition table."""
        from natcap.invest.coastal_blue_carbon import preprocessor

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157)
        projection_wkt = srs.ExportToWkt()
        origin = (443723.127327877911739, 4956546.905980412848294)
        matrix_a = numpy.array([
            [0, 1],
            [0, 1],
            [0, 1]], dtype=numpy.int16)
        filename_a = os.path.join(self.workspace_dir, 'raster_a.tif')
        pygeoprocessing.numpy_array_to_raster(
            matrix_a, -1, (100, -100), origin, projection_wkt, filename_a)

        matrix_b = numpy.array([
            [0, 1],
            [1, 0],
            [-1, -1]], dtype=numpy.int16)
        filename_b = os.path.join(self.workspace_dir, 'raster_b.tif')
        pygeoprocessing.numpy_array_to_raster(
            matrix_b, -1, (100, -100), origin, projection_wkt, filename_b)

        landcover_table_path = os.path.join(self.workspace_dir,
                                            'lulc_table.csv')
        with open(landcover_table_path, 'w') as lulc_csv:
            lulc_csv.write('lucode,lulc-class,is_coastal_blue_carbon_habitat\n')
            lulc_csv.write('0,mangrove,True\n')
            lulc_csv.write('1,parking lot,False\n')

        landcover_df = preprocessor.MODEL_SPEC.get_input(
            'lulc_lookup_table_path').get_validated_dataframe(landcover_table_path)

        target_table_path = os.path.join(self.workspace_dir,
                                         'transition_table.csv')

        # Remove landcover code 1 from the table; expect error.
        landcover_df = landcover_df.drop(1)
        with self.assertRaises(ValueError) as context:
            preprocessor._create_transition_table(
                landcover_df, [filename_a, filename_b], target_table_path)

        self.assertIn('missing a row with the landuse code 1',
                      str(context.exception))

        # Re-load the landcover table
        landcover_df = preprocessor.MODEL_SPEC.get_input(
            'lulc_lookup_table_path').get_validated_dataframe(landcover_table_path)
        preprocessor._create_transition_table(
            landcover_df, [filename_a, filename_b], target_table_path)

        with open(target_table_path) as transition_table:
            self.assertEqual(
                transition_table.readline(),
                'lulc-class,mangrove,parking lot\n')
            self.assertEqual(
                transition_table.readline(),
                'mangrove,accum,disturb\n')
            self.assertEqual(
                transition_table.readline(),
                'parking lot,accum,NCC\n')

            # After the above lines is a blank line, then the legend.
            # Deliberately not testing the legend.
            self.assertEqual(transition_table.readline(), '\n')


class TestCBC2(unittest.TestCase):
    """Test Coastal Blue Carbon main model functions."""

    def setUp(self):
        """Create a temp directory for the workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove workspace after each test function."""
        shutil.rmtree(self.workspace_dir)

    def test_read_invalid_transition_matrix(self):
        """CBC: Test exceptions in invalid transition structure."""
        # The full biophysical table will have much, much more information.  To
        # keep the test simple, I'm only tracking the columns I know I'll need
        # in this function.
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon
        biophysical_table = pandas.DataFrame({
            1: {'lulc-class': 'a',
                'soil-yearly-accumulation': 2,
                'biomass-yearly-accumulation': 3,
                'soil-high-impact-disturb': 4,
                'biomass-high-impact-disturb': 5},
            2: {'lulc-class': 'b',
                'soil-yearly-accumulation': 6,
                'biomass-yearly-accumulation': 7,
                'soil-high-impact-disturb': 8,
                'biomass-high-impact-disturb': 9},
            3: {'lulc-class': 'c',
                'soil-yearly-accumulation': 10,
                'biomass-yearly-accumulation': 11,
                'soil-high-impact-disturb': 12,
                'biomass-high-impact-disturb': 13}
        }).T

        transition_csv_path = os.path.join(self.workspace_dir,
                                           'transitions.csv')
        with open(transition_csv_path, 'w') as transition_csv:
            transition_csv.write('lulc-class,a,b,c\n')
            transition_csv.write('missing code,NCC,accum,high-impact-disturb\n')
            transition_csv.write('b,,NCC,accum\n')
            transition_csv.write('c,accum,,NCC\n')
            transition_csv.write(',,,\n')
            transition_csv.write(',legend,,')  # simulate legend

        with self.assertRaises(ValueError) as cm:
            disturbance_matrices, accumulation_matrices = (
                 coastal_blue_carbon._read_transition_matrix(
                     transition_csv_path, biophysical_table))

        self.assertIn("'lulc-class' column has a value, 'missing code'",
                      str(cm.exception))

        with open(transition_csv_path, 'w') as transition_csv:
            transition_csv.write('lulc-class,missing code,b,c\n')
            transition_csv.write('a,NCC,accum,high-impact-disturb\n')
            transition_csv.write('b,,NCC,accum\n')
            transition_csv.write('c,accum,,NCC\n')
            transition_csv.write(',,,\n')
            transition_csv.write(',legend,,')  # simulate legend

        with self.assertRaises(ValueError) as cm:
            disturbance_matrices, accumulation_matrices = (
                 coastal_blue_carbon._read_transition_matrix(
                     transition_csv_path, biophysical_table))

        self.assertIn("The transition table's header row has a column name, "
                      "'missing code',", str(cm.exception))

    def test_read_transition_matrix(self):
        """CBC: Test transition matrix reading."""
        # The full biophysical table will have much, much more information.  To
        # keep the test simple, I'm only tracking the columns I know I'll need
        # in this function.
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon
        biophysical_table = pandas.DataFrame({
            1: {'lulc-class': 'a',
                'soil-yearly-accumulation': 2,
                'biomass-yearly-accumulation': 3,
                'soil-high-impact-disturb': 4,
                'biomass-high-impact-disturb': 5},
            2: {'lulc-class': 'b',
                'soil-yearly-accumulation': 6,
                'biomass-yearly-accumulation': 7,
                'soil-high-impact-disturb': 8,
                'biomass-high-impact-disturb': 9},
            3: {'lulc-class': 'c',
                'soil-yearly-accumulation': 10,
                'biomass-yearly-accumulation': 11,
                'soil-high-impact-disturb': 12,
                'biomass-high-impact-disturb': 13}
        }).T

        transition_csv_path = os.path.join(self.workspace_dir,
                                           'transitions.csv')
        with open(transition_csv_path, 'w') as transition_csv:
            transition_csv.write('lulc-class,a,b,c\n')
            transition_csv.write('a,NCC,accum,high-impact-disturb\n')
            transition_csv.write('b,,NCC,accum\n')
            transition_csv.write('c,accum,,NCC\n')
            transition_csv.write(',,,\n')
            transition_csv.write(',legend,,')  # simulate legend

        disturbance_matrices, accumulation_matrices = (
             coastal_blue_carbon._read_transition_matrix(
                 transition_csv_path, biophysical_table))

        expected_biomass_disturbance = numpy.zeros((4, 4), dtype=numpy.float32)
        expected_biomass_disturbance[1, 3] = (
            biophysical_table['biomass-high-impact-disturb'][1])
        numpy.testing.assert_allclose(
            expected_biomass_disturbance,
            disturbance_matrices['biomass'].toarray())

        expected_soil_disturbance = numpy.zeros((4, 4), dtype=numpy.float32)
        expected_soil_disturbance[1, 3] = (
            biophysical_table['soil-high-impact-disturb'][1])
        numpy.testing.assert_allclose(
            expected_soil_disturbance,
            disturbance_matrices['soil'].toarray())

        expected_biomass_accumulation = numpy.zeros(
            (4, 4), dtype=numpy.float32)
        expected_biomass_accumulation[3, 1] = (
            biophysical_table['biomass-yearly-accumulation'][1])
        expected_biomass_accumulation[1, 2] = (
            biophysical_table['biomass-yearly-accumulation'][2])
        expected_biomass_accumulation[2, 3] = (
            biophysical_table['biomass-yearly-accumulation'][3])
        numpy.testing.assert_allclose(
            expected_biomass_accumulation,
            accumulation_matrices['biomass'].toarray())

        expected_soil_accumulation = numpy.zeros((4, 4), dtype=numpy.float32)
        expected_soil_accumulation[3, 1] = (
            biophysical_table['soil-yearly-accumulation'][1])
        expected_soil_accumulation[1, 2] = (
            biophysical_table['soil-yearly-accumulation'][2])
        expected_soil_accumulation[2, 3] = (
            biophysical_table['soil-yearly-accumulation'][3])
        numpy.testing.assert_allclose(
            expected_soil_accumulation,
            accumulation_matrices['soil'].toarray())

    def test_emissions(self):
        """CBC: Check emissions calculations."""
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon
        volume_disturbed_carbon = numpy.array(
            [[5.5, coastal_blue_carbon.NODATA_FLOAT32_MIN]], dtype=numpy.float32)
        year_last_disturbed = numpy.array(
            [[10, coastal_blue_carbon.NODATA_UINT16_MAX]], dtype=numpy.uint16)
        half_life = numpy.array([[7.5, 7.5]], dtype=numpy.float32)
        current_year = 15

        result_matrix = coastal_blue_carbon._calculate_emissions(
            volume_disturbed_carbon, year_last_disturbed, half_life,
            current_year)

        # Calculated by hand.
        expected_array = numpy.array([
            [0.3058625, coastal_blue_carbon.NODATA_FLOAT32_MIN]],
            dtype=numpy.float32)
        numpy.testing.assert_allclose(
            result_matrix, expected_array, rtol=1E-6)

    def test_add_rasters(self):
        """CBC: Check that we can sum rasters."""
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31 S
        wkt = srs.ExportToWkt()

        raster_a_path = os.path.join(self.workspace_dir, 'a.tif')
        pygeoprocessing.numpy_array_to_raster(
            numpy.array([[5, 15, 12, 15]], dtype=numpy.uint8),
            15, (2, -2), (2, -2), wkt, raster_a_path)

        raster_b_path = os.path.join(self.workspace_dir, 'b.tif')
        pygeoprocessing.numpy_array_to_raster(
            numpy.array([[3, 4, 5, 5]], dtype=numpy.uint8),
            5, (2, -2), (2, -2), wkt, raster_b_path)

        target_path = os.path.join(self.workspace_dir, 'output.tif')
        coastal_blue_carbon._sum_n_rasters(
            [raster_a_path, raster_b_path], target_path)

        nodata = coastal_blue_carbon.NODATA_FLOAT32_MIN
        try:
            raster = gdal.OpenEx(target_path)
            numpy.testing.assert_allclose(
                raster.ReadAsArray(),
                numpy.array([[8, 4, 12, nodata]], dtype=numpy.float32))
        finally:
            raster = None

    @staticmethod
    def _create_model_args(target_dir):
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31 S
        wkt = srs.ExportToWkt()

        biophysical_table = [
            ['lucode', 'lulc-class', 'biomass-initial', 'soil-initial',
                'litter-initial', 'biomass-half-life',
                'biomass-low-impact-disturb', 'biomass-med-impact-disturb',
                'biomass-high-impact-disturb', 'biomass-yearly-accumulation',
                'soil-half-life', 'soil-low-impact-disturb',
                'soil-med-impact-disturb', 'soil-high-impact-disturb',
                'soil-yearly-accumulation', 'litter-yearly-accumulation'],
            [1, 'mangrove',
                64, 313, 3,  # initial
                15, 0.5, 0.5, 1, 2,  # biomass
                7.5, 0.3, 0.5, 0.66, 5.35,  # soil
                1],  # litter accum.
            [2, 'parking lot',
                0, 0, 0,  # initial
                0, 0, 0, 0, 0,  # biomass
                0, 0, 0, 0, 0,  # soil
                0],  # litter accum.
        ]
        biophysical_table_path = os.path.join(
            target_dir, 'biophysical.csv')
        with open(biophysical_table_path, 'w') as bio_table:
            for line_list in biophysical_table:
                line = ','.join(str(field) for field in line_list)
                bio_table.write(f'{line}\n')

        transition_matrix = [
            ['lulc-class', 'mangrove', 'parking lot'],
            ['mangrove', 'NCC', 'high-impact-disturb'],
            ['parking lot', 'accum', 'NCC']
        ]
        transition_matrix_path = os.path.join(
            target_dir, 'transitions.csv')
        with open(transition_matrix_path, 'w') as transition_table:
            for line_list in transition_matrix:
                line = ','.join(line_list)
                transition_table.write(f'{line}\n')

        baseline_landcover_raster_path = os.path.join(
            target_dir, 'baseline_lulc.tif')
        baseline_matrix = numpy.array([[1, 2]], dtype=numpy.uint8)
        pygeoprocessing.numpy_array_to_raster(
            baseline_matrix, 255, (2, -2), (2, -2), wkt,
            baseline_landcover_raster_path)

        snapshot_2010_raster_path = os.path.join(
            target_dir, 'snapshot_2010.tif')
        snapshot_2010_matrix = numpy.array([[2, 1]], dtype=numpy.uint8)
        pygeoprocessing.numpy_array_to_raster(
            snapshot_2010_matrix, 255, (2, -2), (2, -2), wkt,
            snapshot_2010_raster_path)

        snapshot_2020_raster_path = os.path.join(
            target_dir, 'snapshot_2020.tif')
        snapshot_2020_matrix = numpy.array([[1, 2]], dtype=numpy.uint8)
        pygeoprocessing.numpy_array_to_raster(
            snapshot_2020_matrix, 255, (2, -2), (2, -2), wkt,
            snapshot_2020_raster_path)

        snapshot_rasters_csv_path = os.path.join(
            target_dir, 'snapshot_rasters.csv')
        baseline_year = 2000
        with open(snapshot_rasters_csv_path, 'w') as snapshot_rasters_csv:
            snapshot_rasters_csv.write('snapshot_year,raster_path\n')
            snapshot_rasters_csv.write(
                f'{baseline_year},{baseline_landcover_raster_path}\n')
            snapshot_rasters_csv.write(
                f'2010,{snapshot_2010_raster_path}\n')
            snapshot_rasters_csv.write(
                f'2020,{snapshot_2020_raster_path}\n')

        args = {
            'landcover_transitions_table': transition_matrix_path,
            'landcover_snapshot_csv': snapshot_rasters_csv_path,
            'biophysical_table_path': biophysical_table_path,
            'analysis_year': 2030,
            'do_economic_analysis': True,
            'use_price_table': True,
            'price_table_path': os.path.join(target_dir,
                                             'price_table.csv'),
            'discount_rate': 4,
        }

        with open(args['price_table_path'], 'w') as price_table:
            price_table.write('year,price\n')
            prior_year_price = 1.0
            for year in range(baseline_year,
                              args['analysis_year']+1):
                price = prior_year_price * 1.04
                price_table.write(f'{year},{price}\n')
        return args

    def test_duplicate_lulc_classes(self):
        """CBC: Raise an execption if duplicate lulc-classes."""
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon
        args = TestCBC2._create_model_args(self.workspace_dir)
        args['workspace_dir'] = os.path.join(self.workspace_dir, 'workspace')
        with open(args['biophysical_table_path'], 'r') as table:
            lines = table.readlines()

        with open(args['biophysical_table_path'], 'a') as table:
            last_line_contents = lines[-1].strip().split(',')
            last_line_contents[0] = '3'  # assign a new code
            table.write(','.join(last_line_contents))

        with self.assertRaises(ValueError) as context:
            coastal_blue_carbon.execute(args)

        self.assertIn("`lulc-class` column must be unique",
                      str(context.exception))

    def test_model_no_analysis_year_no_price_table(self):
        """CBC: Test the model's execution."""
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon
        args = TestCBC2._create_model_args(self.workspace_dir)
        args['workspace_dir'] = os.path.join(self.workspace_dir, 'workspace')
        del args['analysis_year']  # final year is 2020.
        args['use_price_table'] = False
        args['inflation_rate'] = 5
        args['price'] = 10.0

        coastal_blue_carbon.execute(args)

        # Sample values calculated by hand.  Pixel 0 only accumulates.  Pixel 1
        # has no accumulation (per the biophysical table) and also has no
        # emissions.
        expected_sequestration_2000_to_2010 = numpy.array(
            [[83.5, 0]], dtype=numpy.float32)
        raster_path = os.path.join(
            args['workspace_dir'], 'output',
            ('total-net-carbon-sequestration-between-'
                '2000-and-2010.tif'))
        try:
            raster = gdal.OpenEx(raster_path)
            numpy.testing.assert_allclose(
                raster.ReadAsArray(),
                expected_sequestration_2000_to_2010)
        finally:
            raster = None

        expected_sequestration_2010_to_2020 = numpy.array(
            [[-176.9792, 73.5]], dtype=numpy.float32)
        raster_path = os.path.join(
            args['workspace_dir'], 'output',
            ('total-net-carbon-sequestration-between-'
                '2010-and-2020.tif'))
        try:
            raster = gdal.OpenEx(raster_path)
            numpy.testing.assert_allclose(
                raster.ReadAsArray(),
                expected_sequestration_2010_to_2020, rtol=1e-6)
        finally:
            raster = None

        # Total sequestration is the sum of all the previous sequestration.
        expected_total_sequestration = (
            expected_sequestration_2000_to_2010 +
            expected_sequestration_2010_to_2020)
        raster_path = os.path.join(
            args['workspace_dir'], 'output',
            'total-net-carbon-sequestration.tif')
        try:
            raster = gdal.OpenEx(raster_path)
            numpy.testing.assert_allclose(
                raster.ReadAsArray(),
                expected_total_sequestration, rtol=1e-6)
        finally:
            raster = None

        expected_net_present_value_at_2020 = numpy.array(
            [[-20506.314,  16123.521]], dtype=numpy.float32)
        raster_path = os.path.join(
            args['workspace_dir'], 'output', 'net-present-value-at-2020.tif')
        try:
            raster = gdal.OpenEx(raster_path)
            numpy.testing.assert_allclose(
                raster.ReadAsArray(),
                expected_net_present_value_at_2020, rtol=1e-6)
        finally:
            raster = None

    def test_model_one_transition_no_analysis_year(self):
        """CBC: Test the model on one transition with no analysis year.

        This test came up while looking into an issue reported on the forums:
        https://community.naturalcapitalproject.org/t/coastal-blue-carbon-negative-carbon-stocks/780/12
        """
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon
        args = TestCBC2._create_model_args(self.workspace_dir)
        args['workspace_dir'] = os.path.join(self.workspace_dir, 'workspace')

        prior_snapshots = coastal_blue_carbon.MODEL_SPEC.get_input(
            'landcover_snapshot_csv').get_validated_dataframe(
                args['landcover_snapshot_csv'])['raster_path'].to_dict()
        baseline_year = min(prior_snapshots.keys())
        baseline_raster = prior_snapshots[baseline_year]
        with open(args['landcover_snapshot_csv'], 'w') as snapshot_csv:
            snapshot_csv.write('snapshot_year,raster_path\n')
            snapshot_csv.write(f'{baseline_year},{baseline_raster}\n')
            snapshot_csv.write(f'2010,{prior_snapshots[2010]}\n')
        del args['analysis_year']

        # Use valuation parameters rather than price table.
        args['use_price_table'] = False
        args['inflation_rate'] = 4
        args['price'] = 1.0

        coastal_blue_carbon.execute(args)

        # Check sequestration raster
        expected_sequestration_2000_to_2010 = numpy.array(
            [[83.5, 0.]], dtype=numpy.float32)
        raster_path = os.path.join(
            args['workspace_dir'], 'output',
            ('total-net-carbon-sequestration-between-'
                '2000-and-2010.tif'))
        try:
            raster= gdal.OpenEx(raster_path)
            numpy.testing.assert_allclose(
                raster.ReadAsArray(),
                expected_sequestration_2000_to_2010)
        finally:
            raster = None

        # Check valuation raster
        # Discount rate here matches the inflation rate, so the value of the 10
        # years' accumulation is just 1*(10 years of accumulation).
        expected_net_present_value_at_2010 = numpy.array(
            [[835.0, 0.]], dtype=numpy.float32)
        raster_path = os.path.join(
            args['workspace_dir'], 'output', 'net-present-value-at-2010.tif')
        try:
            raster = gdal.OpenEx(raster_path)
            numpy.testing.assert_allclose(
                raster.ReadAsArray(),
                expected_net_present_value_at_2010, rtol=1e-6)
        finally:
            raster = None

    def test_model(self):
        """CBC: Test the model's execution."""
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon
        args = TestCBC2._create_model_args(self.workspace_dir)
        args['workspace_dir'] = os.path.join(self.workspace_dir, 'workspace')

        coastal_blue_carbon.execute(args)

        # Sample values calculated by hand.  Pixel 0 only accumulates.  Pixel 1
        # has no accumulation (per the biophysical table) and also has no
        # emissions.
        expected_sequestration_2000_to_2010 = numpy.array(
            [[83.5, 0]], dtype=numpy.float32)
        raster_path = os.path.join(
            args['workspace_dir'], 'output',
            ('total-net-carbon-sequestration-between-'
                '2000-and-2010.tif'))
        try:
            raster = gdal.OpenEx(raster_path)
            numpy.testing.assert_allclose(
                raster.ReadAsArray(),
                expected_sequestration_2000_to_2010)
        finally:
            raster = None

        expected_sequestration_2010_to_2020 = numpy.array(
            [[-176.9792, 73.5]], dtype=numpy.float32)
        raster_path = os.path.join(
            args['workspace_dir'], 'output',
            ('total-net-carbon-sequestration-between-'
                '2010-and-2020.tif'))
        try:
            raster = gdal.OpenEx(raster_path)
            numpy.testing.assert_allclose(
                raster.ReadAsArray(),
                expected_sequestration_2010_to_2020, rtol=1e-6)
        finally:
            raster = None

        expected_sequestration_2020_to_2030 = numpy.array(
            [[73.5, -28.698004]], dtype=numpy.float32)
        raster_path = os.path.join(
            args['workspace_dir'], 'output',
            ('total-net-carbon-sequestration-between-'
                '2020-and-2030.tif'))
        try:
            raster = gdal.OpenEx(raster_path)
            numpy.testing.assert_allclose(
                raster.ReadAsArray(),
                expected_sequestration_2020_to_2030, rtol=1e-6)
        finally:
            raster = None

        # Total sequestration is the sum of all the previous sequestration.
        expected_total_sequestration = (
            expected_sequestration_2000_to_2010 +
            expected_sequestration_2010_to_2020 +
            expected_sequestration_2020_to_2030)
        raster_path = os.path.join(
            args['workspace_dir'], 'output',
            'total-net-carbon-sequestration.tif')
        try:
            raster = gdal.OpenEx(raster_path)
            numpy.testing.assert_allclose(
                raster.ReadAsArray(),
                expected_total_sequestration, rtol=1e-6)
        finally:
            raster = None

        expected_net_present_value_at_2030 = numpy.array(
            [[-373.67245,  837.93445]], dtype=numpy.float32)
        raster_path = os.path.join(
            args['workspace_dir'], 'output', 'net-present-value-at-2030.tif')
        try:
            raster = gdal.OpenEx(raster_path)
            numpy.testing.assert_allclose(
                raster.ReadAsArray(),
                expected_net_present_value_at_2030, rtol=1e-6)
        finally:
            raster = None

        # For emissions, make sure that each of the emissions sum rasters in
        # the output folder match the sum of all annual emissions from the time
        # period.
        for emissions_raster_path in glob.glob(
                os.path.join(args['workspace_dir'], 'output',
                             'carbon-emissions-between-*.tif')):
            try:
                raster = gdal.OpenEx(emissions_raster_path)
                band = raster.GetRasterBand(1)
                emissions_in_period = band.ReadAsArray()
            finally:
                band = None
                raster = None

            basename = os.path.splitext(
                os.path.basename(emissions_raster_path))[0]
            parts = basename.split('-')
            start_year = int(parts[3])
            end_year = int(parts[5])

            summed_emissions_over_time_period = numpy.array(
                [[0.0, 0.0]], dtype=numpy.float32)
            for year in range(start_year, end_year):
                for pool in ('soil', 'biomass'):
                    yearly_emissions_raster = os.path.join(
                        args['workspace_dir'], 'intermediate',
                        f'emissions-{pool}-{year}.tif')
                    try:
                        raster = gdal.OpenEx(yearly_emissions_raster)
                        band = raster.GetRasterBand(1)
                        summed_emissions_over_time_period += band.ReadAsArray()
                    finally:
                        band = None
                        raster = None
            numpy.testing.assert_allclose(
                emissions_in_period,
                summed_emissions_over_time_period)

    def test_model_no_transitions(self):
        """CBC: Test model without transitions.

        When the model executes without transitions, we still evaluate carbon
        sequestration (accumulation only) for the whole baseline period.
        """
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon
        args = TestCBC2._create_model_args(self.workspace_dir)
        args['workspace_dir'] = os.path.join(self.workspace_dir, 'workspace')

        prior_snapshots = coastal_blue_carbon.MODEL_SPEC.get_input(
            'landcover_snapshot_csv').get_validated_dataframe(
                args['landcover_snapshot_csv'])['raster_path'].to_dict()
        baseline_year = min(prior_snapshots.keys())
        baseline_raster = prior_snapshots[baseline_year]
        with open(args['landcover_snapshot_csv'], 'w') as snapshot_csv:
            snapshot_csv.write('snapshot_year,raster_path\n')
            snapshot_csv.write(f'{baseline_year},{baseline_raster}\n')
        args['analysis_year'] = baseline_year + 10

        # Use valuation parameters rather than price table.
        args['use_price_table'] = False
        args['inflation_rate'] = 4
        args['price'] = 1.0

        coastal_blue_carbon.execute(args)

        # Check sequestration raster
        expected_sequestration_2000_to_2010 = numpy.array(
            [[83.5, 0.]], dtype=numpy.float32)
        raster_path = os.path.join(
            args['workspace_dir'], 'output',
            ('total-net-carbon-sequestration-between-'
                '2000-and-2010.tif'))
        try:
            raster= gdal.OpenEx(raster_path)
            numpy.testing.assert_allclose(
                raster.ReadAsArray(),
                expected_sequestration_2000_to_2010)
        finally:
            raster = None

        # Check valuation raster
        # Discount rate here matches the inflation rate, so the value of the 10
        # years' accumulation is just 1*(10 years of accumulation).
        expected_net_present_value_at_2010 = numpy.array(
            [[835.0, 0.]], dtype=numpy.float32)
        raster_path = os.path.join(
            args['workspace_dir'], 'output', 'net-present-value-at-2010.tif')
        try:
            raster = gdal.OpenEx(raster_path)
            numpy.testing.assert_allclose(
                raster.ReadAsArray(),
                expected_net_present_value_at_2010, rtol=1e-6)
        finally:
            raster = None

    def test_validation(self):
        """CBC: Test custom validation."""
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon
        args = TestCBC2._create_model_args(self.workspace_dir)
        args['workspace_dir'] = self.workspace_dir

        # verify validation passes on basic set of arguments.
        validation_warnings = coastal_blue_carbon.validate(args)
        self.assertEqual([], validation_warnings)

        # Now work through the extra validation warnings.
        # test validation: invalid analysis year
        prior_snapshots = coastal_blue_carbon.MODEL_SPEC.get_input(
            'landcover_snapshot_csv').get_validated_dataframe(
                args['landcover_snapshot_csv'])['raster_path'].to_dict()
        baseline_year = min(prior_snapshots)
        # analysis year must be >= the last transition year.
        args['analysis_year'] = baseline_year
        validation_warnings = coastal_blue_carbon.validate(args)
        self.assertIn(
            coastal_blue_carbon.INVALID_ANALYSIS_YEAR_MSG.format(
                analysis_year=2000, latest_year=2020),
            validation_warnings[0][1])

        # Create an invalid transitions table.
        invalid_raster_path = os.path.join(self.workspace_dir,
                                           'invalid_raster.tif')
        with open(invalid_raster_path, 'w') as raster:
            raster.write('not a raster')

        # Write over the landcover snapshot CSV
        with open(args['landcover_snapshot_csv'], 'w') as snapshot_table:
            snapshot_table.write('snapshot_year,raster_path\n')
            snapshot_table.write(
                f'{baseline_year},{prior_snapshots[baseline_year]}\n')
            snapshot_table.write(
                f"{baseline_year + 10},{invalid_raster_path}")

        # Write invalid entries to landcover transition table
        with open(args['landcover_transitions_table'], 'w') as transition_table:
            transition_table.write('lulc-class,Developed,Forest,Water\n')
            transition_table.write('Developed,NCC,,invalid\n')
            transition_table.write('Forest,accum,disturb,low-impact-disturb\n')
            transition_table.write('Water,disturb,med-impact-disturb,high-impact-disturb\n')
        transition_options = [
                'accum', 'high-impact-disturb', 'low-impact-disturb',
                'med-impact-disturb', 'ncc']
        validation_warnings = coastal_blue_carbon.validate(args)
        self.assertEqual(len(validation_warnings), 2)
        self.assertIn(
            'File could not be opened as a GDAL raster',
            validation_warnings[0][1])
        self.assertIn(
            coastal_blue_carbon.INVALID_TRANSITION_VALUES_MSG.format(
                model_transitions=transition_options,
                transition_values=['disturb', 'invalid']),
            validation_warnings[1][1])

    def test_track_first_disturbance(self):
        """CBC: Track disturbances over time."""
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon
        float32_nodata = coastal_blue_carbon.NODATA_FLOAT32_MIN

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31 S
        wkt = srs.ExportToWkt()

        disturbance_magnitude_path = os.path.join(
            self.workspace_dir, 'disturbance_magnitude.tif')
        disturbance_magnitude_matrix = numpy.array(
            [[0.5, float32_nodata, 0.0]], dtype=numpy.float32)
        pygeoprocessing.numpy_array_to_raster(
            disturbance_magnitude_matrix, float32_nodata, (2, -2), (2, -2), wkt,
            disturbance_magnitude_path)

        stocks_path = os.path.join(
            self.workspace_dir, 'stocks.tif')
        stocks_matrix = numpy.array(
            [[10, -1, 10]], dtype=numpy.float32)
        pygeoprocessing.numpy_array_to_raster(
            stocks_matrix, -1, (2, -2), (2, -2), wkt, stocks_path)

        current_year = 2010

        target_disturbance_path = os.path.join(
            self.workspace_dir, 'disturbance_volume.tif')
        target_year_of_disturbance = os.path.join(
            self.workspace_dir, 'year_of_disturbance.tif')

        coastal_blue_carbon._track_disturbance(
            disturbance_magnitude_path, stocks_path,
            None,  # No prior disturbance volume
            None,  # No prior disturbance years
            current_year, target_disturbance_path, target_year_of_disturbance)

        try:
            expected_disturbance = numpy.array(
                [[5.0, float32_nodata, 0.0]], dtype=numpy.float32)
            raster = gdal.OpenEx(target_disturbance_path)
            numpy.testing.assert_allclose(
                raster.ReadAsArray(),
                expected_disturbance)
        finally:
            raster = None

        try:
            uint16_nodata = coastal_blue_carbon.NODATA_UINT16_MAX
            expected_year_of_disturbance = numpy.array(
                [[2010, uint16_nodata, uint16_nodata]], dtype=numpy.uint16)
            raster = gdal.OpenEx(target_year_of_disturbance)
            numpy.testing.assert_allclose(
                raster.ReadAsArray(),
                expected_year_of_disturbance)
        finally:
            raster = None

    def test_track_later_disturbance(self):
        """CBC: Track disturbances over time."""
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon
        float32_nodata = coastal_blue_carbon.NODATA_FLOAT32_MIN
        uint16_nodata = coastal_blue_carbon.NODATA_UINT16_MAX

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31 S
        wkt = srs.ExportToWkt()

        disturbance_magnitude_path = os.path.join(
            self.workspace_dir, 'disturbance_magnitude.tif')
        disturbance_magnitude_matrix = numpy.array(
            [[0.5, float32_nodata, 0.0]], dtype=numpy.float32)
        pygeoprocessing.numpy_array_to_raster(
            disturbance_magnitude_matrix, float32_nodata, (2, -2), (2, -2), wkt,
            disturbance_magnitude_path)

        stocks_path = os.path.join(
            self.workspace_dir, 'stocks.tif')
        stocks_matrix = numpy.array(
            [[10, -1, 10]], dtype=numpy.float32)
        pygeoprocessing.numpy_array_to_raster(
            stocks_matrix, -1, (2, -2), (2, -2), wkt, stocks_path)

        prior_disturbance_volume_path = os.path.join(
            self.workspace_dir, 'prior_disturbance_vol.tif')
        prior_disturbance_vol_matrix = numpy.array(
            [[700, float32_nodata, 300]], numpy.float32)
        pygeoprocessing.numpy_array_to_raster(
            prior_disturbance_vol_matrix, float32_nodata, (2, -2), (2, -2),
            wkt, prior_disturbance_volume_path)

        prior_year_disturbance_path = os.path.join(
            self.workspace_dir, 'prior_disturbance_years.tif')
        prior_year_disturbance_matrix = numpy.array(
            [[2000, uint16_nodata, 1990]], numpy.uint16)
        pygeoprocessing.numpy_array_to_raster(
            prior_year_disturbance_matrix, uint16_nodata, (2, -2), (2, -2), wkt,
            prior_year_disturbance_path)

        target_disturbance_path = os.path.join(
            self.workspace_dir, 'disturbance_volume.tif')
        target_year_of_disturbance = os.path.join(
            self.workspace_dir, 'year_of_disturbance.tif')

        current_year = 2010
        coastal_blue_carbon._track_disturbance(
            disturbance_magnitude_path, stocks_path,
            prior_disturbance_volume_path,
            prior_year_disturbance_path,
            current_year, target_disturbance_path, target_year_of_disturbance)

        try:
            expected_disturbance = numpy.array(
                [[5.0, float32_nodata, 300]], dtype=numpy.float32)
            raster = gdal.OpenEx(target_disturbance_path)
            numpy.testing.assert_allclose(
                raster.ReadAsArray(),
                expected_disturbance)
        finally:
            raster = None

        try:
            expected_year_of_disturbance = numpy.array(
                [[2010, uint16_nodata, 1990]], dtype=numpy.uint16)
            raster = gdal.OpenEx(target_year_of_disturbance)
            numpy.testing.assert_allclose(
                raster.ReadAsArray(),
                expected_year_of_disturbance)
        finally:
            raster = None

    def test_validate_required_analysis_year(self):
        """CBC: analysis year validation (regression test for #1534)."""
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon

        args = TestCBC2._create_model_args(self.workspace_dir)
        args['workspace_dir'] = self.workspace_dir
        args['analysis_year'] = None
        # truncate the CSV so that it has only one snapshot year
        with open(args['landcover_snapshot_csv'], 'r') as file:
            lines = file.readlines()
        with open(args['landcover_snapshot_csv'], 'w') as file:
            file.writelines(lines[:2])
        validation_warnings = coastal_blue_carbon.validate(args)
        self.assertEqual(
            validation_warnings,
            [(['analysis_year'], coastal_blue_carbon.MISSING_ANALYSIS_YEAR_MSG)])

        args['analysis_year'] = 2000  # set analysis year equal to snapshot year
        validation_warnings = coastal_blue_carbon.validate(args)
        self.assertEqual(
            validation_warnings,
            [(['analysis_year'],
                coastal_blue_carbon.INVALID_ANALYSIS_YEAR_MSG.format(
                    analysis_year=2000, latest_year=2000))])

    def test_calculate_npv(self):
        """Test `_calculate_npv`"""
        from natcap.invest.coastal_blue_carbon import coastal_blue_carbon

        # make fake data
        net_sequestration_rasters = {
            2010: os.path.join(self.workspace_dir, "carbon_seq_2010.tif"),
            2011: os.path.join(self.workspace_dir, "carbon_seq_2011.tif"),
            2012: os.path.join(self.workspace_dir, "carbon_seq_2012.tif")
        }

        for year, path in net_sequestration_rasters.items():
            array = numpy.array([[year*.5, year*.25], [year-1, 50]])  # random array
            make_raster_from_array(path, array)

        prices_by_year = {
            2010: 50,
            2011: 80,
            2012: 95
        }
        discount_rate = 0.1
        baseline_year = 2010
        target_raster_years_and_paths = {
            2010: os.path.join(self.workspace_dir, "tgt_carbon_seq_2010.tif"),
            2011: os.path.join(self.workspace_dir, "tgt_carbon_seq_2011.tif"),
            2012: os.path.join(self.workspace_dir, "tgt_carbon_seq_2012.tif")
        }

        coastal_blue_carbon._calculate_npv(net_sequestration_rasters,
                                           prices_by_year, discount_rate,
                                           baseline_year,
                                           target_raster_years_and_paths)

        # read in the created target rasters
        actual_2011 = gdal.Open(target_raster_years_and_paths[2011])
        band = actual_2011.GetRasterBand(1)
        actual_2011 = band.ReadAsArray()

        actual_2012 = gdal.Open(target_raster_years_and_paths[2012])
        band = actual_2012.GetRasterBand(1)
        actual_2012 = band.ReadAsArray()

        # compare actual rasters to expected (based on running `_calculate_npv`)
        expected_2011 = numpy.array([[100525, 50262.5], [200950, 5000]])
        expected_2012 = numpy.array([[370206.818182, 185103.409091],
                                     [740045.454545, 18409.090909]])
        numpy.testing.assert_allclose(actual_2011, expected_2011)
        numpy.testing.assert_allclose(actual_2012, expected_2012)

    def test_calculate_accumulation_over_time(self):
        """Test `_calculate_accumulation_over_time`"""
        from natcap.invest.coastal_blue_carbon.coastal_blue_carbon import \
            _calculate_accumulation_over_time

        # generate fake data with nodata values
        nodata = float(numpy.finfo(numpy.float32).min)
        annual_biomass_matrix = numpy.array([[1, 2], [3, nodata]])
        annual_soil_matrix = numpy.array([[11, 12], [13, 14]])
        annual_litter_matrix = numpy.array([[.5, .9], [4, .9]])
        n_years = 3

        actual_accumulation = _calculate_accumulation_over_time(
            annual_biomass_matrix, annual_soil_matrix, annual_litter_matrix,
            n_years)

        expected_accumulation = numpy.array([[37.5, 44.7], [60, nodata]])
        numpy.testing.assert_allclose(actual_accumulation, expected_accumulation)

    def test_calculate_net_sequestration(self):
        """test `_calculate_net_sequestration`"""
        from natcap.invest.coastal_blue_carbon.coastal_blue_carbon import \
            _calculate_net_sequestration

        # make fake rasters that contain nodata pixels (-1)
        accumulation_raster_path = os.path.join(self.workspace_dir,
                                                "accumulation_raster.tif")
        accumulation_array = numpy.array([[40, -1], [70, -1]])
        make_raster_from_array(accumulation_raster_path, accumulation_array)

        emissions_raster_path = os.path.join(self.workspace_dir,
                                             "emissions_raster.tif")
        emissions_array = numpy.array([[-1, 8], [7, -1]])
        make_raster_from_array(emissions_raster_path, emissions_array)

        target_raster_path = os.path.join(self.workspace_dir,
                                          "target_raster.tif")

        # run `_calculate_net_sequestration`
        _calculate_net_sequestration(accumulation_raster_path,
                                     emissions_raster_path, target_raster_path)

        # compare actual to expected output net sequestration raster
        actual_sequestration = gdal.Open(target_raster_path)
        band = actual_sequestration.GetRasterBand(1)
        actual_sequestration = band.ReadAsArray()

        # calculated by running `_calculate_net_sequestration`
        nodata = float(numpy.finfo(numpy.float32).min)
        expected_sequestration = numpy.array([[40, -8], [-7, nodata]])

        numpy.testing.assert_allclose(actual_sequestration,
                                      expected_sequestration)
        
    def test_reclassify_accumulation_transition(self):
        """Test `_reclassify_accumulation_transition`"""
        from natcap.invest.coastal_blue_carbon.coastal_blue_carbon import \
                _reclassify_accumulation_transition, _reclassify_disturbance_magnitude

        # make fake raster data
        landuse_transition_from_raster = os.path.join(self.workspace_dir,
                                                      "landuse_transition_from.tif")
        landuse_transition_from_array = numpy.array([[1, 2], [3, 2]])
        make_raster_from_array(landuse_transition_from_raster,
                               landuse_transition_from_array)

        landuse_transition_to_raster = os.path.join(self.workspace_dir,
                                                    "landuse_transition_to.tif")
        landuse_transition_to_array = numpy.array([[1, 1], [2, 3]])
        make_raster_from_array(landuse_transition_to_raster,
                               landuse_transition_to_array)

        #make fake accumulation_rate_matrix
        accumulation_rate_matrix = dok_matrix((4, 4), dtype=numpy.float32)
        accumulation_rate_matrix[1, 2] = 0.5  # Forest -> Grassland
        accumulation_rate_matrix[1, 3] = 0.3  # Forest -> Agriculture

        accumulation_rate_matrix[2, 1] = 0.2  # Grassland -> Forest
        accumulation_rate_matrix[2, 3] = 0.4  # Grassland -> Agriculture

        accumulation_rate_matrix[3, 1] = 0.1  # Agriculture -> Forest
        accumulation_rate_matrix[3, 2] = 0.3  # Agriculture -> Grassland

        target_raster_path = os.path.join(self.workspace_dir, "output.tif")
        _reclassify_accumulation_transition(
            landuse_transition_from_raster, landuse_transition_to_raster,
            accumulation_rate_matrix, target_raster_path)
        
        # compare actual and expected target_raster
        actual_accumulation = gdal.Open(target_raster_path)
        band = actual_accumulation.GetRasterBand(1)
        actual_accumulation = band.ReadAsArray()

        expected_accumulation = numpy.array([[0, .2], [.3, .4]])

        numpy.testing.assert_allclose(actual_accumulation, expected_accumulation)
