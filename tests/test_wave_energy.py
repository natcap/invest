"""Module for Testing the InVEST Wave Energy module."""
import unittest
import tempfile
import shutil
import os
import re

import numpy
import numpy.testing
from osgeo import gdal
from osgeo import osr, ogr
from shapely.geometry import Polygon
from shapely.geometry import Point

import pygeoprocessing.testing
from pygeoprocessing.testing import sampledata

REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'wave_energy')
SAMPLE_DATA = os.path.join(REGRESSION_DATA, 'input')


def _make_empty_files(workspace_dir):
    """Within workspace, make intermediate and output folders with dummy files.

    Parameters:
        workspace_dir: path to workspace for creating intermediate/output folder.

    Returns:
        None.

    """
    intermediate_files = [
        'WEM_InputOutput_Pts.shp', 'aoi_clipped_to_extract_path.shp'
    ]

    raster_files = [
        'wp_rc.tif', 'wp_kw.tif', 'capwe_rc.tif', 'capwe_mwh.tif',
        'npv_rc.tif', 'npv_usd.tif'
    ]
    vector_files = ['GridPts_prj.shp', 'LandPts_prj.shp']
    table_files = ['capwe_rc.csv', 'wp_rc.csv', 'npv_rc.csv']
    output_files = raster_files + vector_files + table_files

    for folder, folder_files in zip(['intermediate', 'output'],
                                    [intermediate_files, output_files]):
        folder_path = os.path.join(workspace_dir, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for file_name in folder_files:
            with open(os.path.join(folder_path, file_name), 'w') as open_file:
                open_file.write('')


class WaveEnergyUnitTests(unittest.TestCase):
    """Unit tests for the Wave Energy module."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_pixel_size_based_on_coordinate_transform(self):
        """WaveEnergy: testing '_pixel_size_based_on_coordinate_transform' fn"""
        from natcap.invest import wave_energy

        srs = sampledata.SRS_WILLAMETTE
        srs_wkt = srs.projection
        spat_ref = osr.SpatialReference()
        spat_ref.ImportFromWkt(srs_wkt)

        # Define a Lat/Long WGS84 projection
        epsg_id = 4326
        reference = osr.SpatialReference()
        reference.ImportFromEPSG(epsg_id)
        # Get projection as WKT
        latlong_proj = reference.ExportToWkt()
        # Set origin to use for setting up geometries / geotransforms
        latlong_origin = (-70.5, 42.5)

        # Pixel size helper for defining lat/long pixel size
        def pixel_size(x):
            return (x, -1. * x)

        # Get a point from the clipped data object to use later in helping
        # determine proper pixel size
        matrix = numpy.array([[1, 1, 1, 1], [1, 1, 1, 1]])
        input_path = os.path.join(self.workspace_dir, 'input_raster.tif')
        # Create raster to use as testing input
        raster_path = pygeoprocessing.testing.create_raster_on_disk(
            [matrix],
            latlong_origin,
            latlong_proj,
            -1.0,
            pixel_size(0.033333),
            filename=input_path)

        raster_gt = pygeoprocessing.geoprocessing.get_raster_info(raster_path)[
            'geotransform']
        point = (raster_gt[0], raster_gt[3])
        raster_wkt = latlong_proj

        # Create a Spatial Reference from the rasters WKT
        raster_sr = osr.SpatialReference()
        raster_sr.ImportFromWkt(raster_wkt)

        # A coordinate transformation to help get the proper pixel size of
        # the reprojected raster
        coord_trans = osr.CoordinateTransformation(raster_sr, spat_ref)
        # Call the function to test
        result = wave_energy._pixel_size_based_on_coordinate_transform(
            raster_path, coord_trans, point)

        expected_res = (5553.933063, -1187.370813)

        # Compare
        for res, exp in zip(result, expected_res):
            pygeoprocessing.testing.assert_close(res, exp)

    def test_count_pixels_groups(self):
        """WaveEnergy: testing '_count_pixels_groups' function."""
        from natcap.invest import wave_energy

        raster_path = os.path.join(self.workspace_dir, 'pixel_groups.tif')
        srs = sampledata.SRS_WILLAMETTE

        group_values = [1, 3, 5, 7]
        matrix = numpy.array([[1, 3, 5, 9], [3, 7, 1, 5], [2, 4, 5, 7]])

        # Create raster to use for testing input
        raster_path = pygeoprocessing.testing.create_raster_on_disk(
            [matrix],
            srs.origin,
            srs.projection,
            -1,
            srs.pixel_size(100),
            datatype=gdal.GDT_Int32,
            filename=raster_path)

        results = wave_energy._count_pixels_groups(raster_path, group_values)

        expected_results = [2, 2, 3, 2]

        for res, exp_res in zip(results, expected_results):
            pygeoprocessing.testing.assert_close(res, exp_res, 1e-6)

    def test_calculate_min_distances(self):
        """WaveEnergy: testing '_calculate_min_distances' function."""
        from natcap.invest import wave_energy

        srs = sampledata.SRS_WILLAMETTE
        pos_x = srs.origin[0]
        pos_y = srs.origin[1]

        set_one = numpy.array([[pos_x, pos_y], [pos_x, pos_y - 100],
                               [pos_x, pos_y - 200]])
        set_two = numpy.array([[pos_x + 100,
                                pos_y], [pos_x + 100, pos_y - 100],
                               [pos_x + 100, pos_y - 200]])

        result_dist, result_id = wave_energy._calculate_min_distances(
            set_one, set_two)

        expected_result_dist = [100, 100, 100]
        expected_result_id = [0, 1, 2]

        for res, exp_res in zip(result_dist, expected_result_dist):
            self.assertEqual(res, exp_res)
        for res, exp_res in zip(result_id, expected_result_id):
            self.assertEqual(res, exp_res)

    def test_clip_vector_by_vector_polygons(self):
        """WaveEnergy: testing clipping polygons from polygons."""
        from natcap.invest import wave_energy

        aoi_path = os.path.join(REGRESSION_DATA, 'aoi_proj_to_extract.shp')
        extract_path = os.path.join(
            SAMPLE_DATA, 'WaveData', 'Global_extract.shp')

        result_path = os.path.join(self.workspace_dir, 'aoi_proj_clipped.shp')
        target_projection = pygeoprocessing.get_vector_info(
            extract_path)['projection']
        wave_energy._clip_vector_by_vector(
            aoi_path, extract_path, result_path, target_projection,
            self.workspace_dir)

        expected_path = os.path.join(REGRESSION_DATA, 'aoi_proj_clipped.shp')
        WaveEnergyRegressionTests._assert_point_vectors_equal(
            result_path, expected_path)

    def test_clip_vector_by_vector_points(self):
        """WaveEnergy: testing clipping points from polygons."""
        from natcap.invest import wave_energy

        srs = sampledata.SRS_WILLAMETTE

        pos_x = srs.origin[0]
        pos_y = srs.origin[1]
        fields_pt = {'id': 'int', 'myattr': 'string'}
        attrs_one = [{
            'id': 1,
            'myattr': 'hello'
        }, {
            'id': 2,
            'myattr': 'bye'
        }, {
            'id': 3,
            'myattr': 'highbye'
        }]

        fields_poly = {'id': 'int'}
        attrs_poly = [{'id': 1}]
        # Create geometry for the points, which will get clipped
        geom_one = [
            Point(pos_x + 20, pos_y - 20),
            Point(pos_x + 40, pos_y - 20),
            Point(pos_x + 100, pos_y - 20)
        ]
        # Create geometry for the polygons, which will be used to clip
        geom_two = [
            Polygon([(pos_x, pos_y), (pos_x + 60, pos_y),
                     (pos_x + 60, pos_y - 60), (pos_x, pos_y - 60),
                     (pos_x, pos_y)])
        ]

        shape_to_clip_path = os.path.join(self.workspace_dir,
                                          'shape_to_clip.shp')
        # Create the point shapefile
        shape_to_clip_path = pygeoprocessing.testing.create_vector_on_disk(
            geom_one,
            srs.projection,
            fields_pt,
            attrs_one,
            vector_format='ESRI Shapefile',
            filename=shape_to_clip_path)

        binding_shape_path = os.path.join(self.workspace_dir,
                                          'binding_shape.shp')
        # Create the polygon shapefile
        binding_shape_path = pygeoprocessing.testing.create_vector_on_disk(
            geom_two,
            srs.projection,
            fields_poly,
            attrs_poly,
            vector_format='ESRI Shapefile',
            filename=binding_shape_path)

        output_path = os.path.join(self.workspace_dir, 'vector.shp')
        # Call the function to test
        wave_energy._clip_vector_by_vector(
            shape_to_clip_path, binding_shape_path, output_path,
            srs.projection, self.workspace_dir)

        # Create the expected point shapefile
        fields_pt = {'id': 'int', 'myattr': 'string'}
        attrs_one = [{'id': 1, 'myattr': 'hello'}, {'id': 2, 'myattr': 'bye'}]
        geom_three = [
            Point(pos_x + 20, pos_y - 20),
            Point(pos_x + 40, pos_y - 20)
        ]
        # Need to save the expected shapefile in a sub folder since it must
        # have the same layer name / filename as what it will be compared
        # against.
        if not os.path.isdir(os.path.join(self.workspace_dir, 'exp_vector')):
            os.mkdir(os.path.join(self.workspace_dir, 'exp_vector'))

        expected_path = os.path.join(self.workspace_dir, 'exp_vector',
                                     'vector.shp')
        expected_shape = pygeoprocessing.testing.create_vector_on_disk(
            geom_three,
            srs.projection,
            fields_pt,
            attrs_one,
            vector_format='ESRI Shapefile',
            filename=expected_path)

        WaveEnergyRegressionTests._assert_point_vectors_equal(
            output_path, expected_shape)

    def test_clip_vector_by_vector_no_intersection(self):
        """WaveEnergy: testing '_clip_vector_by_vector' w/ no intersection."""
        from natcap.invest import wave_energy

        srs = sampledata.SRS_WILLAMETTE

        pos_x = srs.origin[0]
        pos_y = srs.origin[1]
        fields_pt = {'id': 'int', 'myattr': 'string'}
        attrs_one = [{'id': 1, 'myattr': 'hello'}]

        fields_poly = {'id': 'int'}
        attrs_poly = [{'id': 1}]
        # Create geometry for the points, which will get clipped
        geom_one = [Point(pos_x + 220, pos_y - 220)]
        # Create geometry for the polygons, which will be used to clip
        geom_two = [
            Polygon([(pos_x, pos_y), (pos_x + 60, pos_y),
                     (pos_x + 60, pos_y - 60), (pos_x, pos_y - 60),
                     (pos_x, pos_y)])
        ]

        shape_to_clip_path = os.path.join(self.workspace_dir,
                                          'shape_to_clip.shp')
        # Create the point shapefile
        shape_to_clip_path = pygeoprocessing.testing.create_vector_on_disk(
            geom_one,
            srs.projection,
            fields_pt,
            attrs_one,
            vector_format='ESRI Shapefile',
            filename=shape_to_clip_path)

        binding_shape_path = os.path.join(self.workspace_dir,
                                          'binding_shape.shp')
        # Create the polygon shapefile
        binding_shape_path = pygeoprocessing.testing.create_vector_on_disk(
            geom_two,
            srs.projection,
            fields_poly,
            attrs_poly,
            vector_format='ESRI Shapefile',
            filename=binding_shape_path)

        output_path = os.path.join(self.workspace_dir, 'vector.shp')
        # Call the function to test
        self.assertRaises(wave_energy.IntersectionError,
                          wave_energy._clip_vector_by_vector,
                          shape_to_clip_path,
                          binding_shape_path,
                          output_path,
                          srs.projection,
                          self.workspace_dir)

    def test_binary_wave_data_to_dict(self):
        """WaveEnergy: testing '_binary_wave_data_to_dict' function."""
        from natcap.invest import wave_energy

        wave_file_path = os.path.join(REGRESSION_DATA, 'example_ww3_binary.bin')

        result = wave_energy._binary_wave_data_to_dict(wave_file_path)

        exp_res = {
            'periods': numpy.array([.375, 1, 1.5, 2.0], dtype=numpy.float32),
            'heights': numpy.array([.375, 1], dtype=numpy.float32),
            'bin_matrix': {
                (102, 370):
                numpy.array(
                    [[0, 0, 0, 0], [0, 9, 3, 30]], dtype=numpy.float32),
                (102, 371):
                numpy.array(
                    [[0, 0, 0, 0], [0, 0, 3, 27]], dtype=numpy.float32)
            }
        }

        for key in ['periods', 'heights']:
            numpy.testing.assert_array_equal(result[key], exp_res[key])

        for key in [(102, 370), (102, 371)]:
            numpy.testing.assert_array_equal(result['bin_matrix'][key],
                                             exp_res['bin_matrix'][key])


class WaveEnergyRegressionTests(unittest.TestCase):
    """Regression tests for the Wave Energy module."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
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
            'wave_base_data_path': os.path.join(SAMPLE_DATA, 'WaveData'),
            'analysis_area_path': 'West Coast of North America and Hawaii',
            'machine_perf_path': os.path.join(
                SAMPLE_DATA, 'Machine_Pelamis_Performance.csv'),
            'machine_param_path': os.path.join(
                SAMPLE_DATA, 'Machine_Pelamis_Parameter.csv'),
            'dem_path': os.path.join(SAMPLE_DATA, 'resampled_global_dem.tif'),
            'n_workers': -1
        }
        return args

    def test_valuation(self):
        """WaveEnergy: testing valuation component."""
        from natcap.invest import wave_energy

        args = WaveEnergyRegressionTests.generate_base_args(self.workspace_dir)
        args['aoi_path'] = os.path.join(SAMPLE_DATA, 'AOI_WCVI.shp')
        args['valuation_container'] = True
        args['land_gridPts_path'] = os.path.join(
            SAMPLE_DATA, 'LandGridPts_WCVI.csv')
        args['machine_econ_path'] = os.path.join(
            SAMPLE_DATA, 'Machine_Pelamis_Economic.csv')
        args['number_of_machines'] = 28

        # Testing if intermediate/output were overwritten
        _make_empty_files(args['workspace_dir'])

        wave_energy.execute(args)

        raster_results = [
            'wp_rc.tif', 'wp_kw.tif', 'capwe_rc.tif', 'capwe_mwh.tif',
            'npv_rc.tif', 'npv_usd.tif'
        ]

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'valuation', raster_path), 1e-6)

        vector_results = ['GridPts_prj.shp', 'LandPts_prj.shp']

        for vector_path in vector_results:
            WaveEnergyRegressionTests._assert_point_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'valuation', vector_path))

        table_results = ['capwe_rc.csv', 'wp_rc.csv', 'npv_rc.csv']

        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, 'valuation', table_path))

    def test_aoi_no_val(self):
        """WaveEnergy: testing Biophysical component w AOI but w/o valuation."""
        from natcap.invest import wave_energy

        args = WaveEnergyRegressionTests.generate_base_args(self.workspace_dir)
        args['aoi_path'] = os.path.join(SAMPLE_DATA, 'AOI_WCVI.shp')

        wave_energy.execute(args)

        raster_results = [
            'wp_rc.tif', 'wp_kw.tif', 'capwe_rc.tif', 'capwe_mwh.tif'
        ]

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'aoi', raster_path), 1e-6)

        table_results = ['capwe_rc.csv', 'wp_rc.csv']

        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, 'aoi', table_path), 1e-6)

    def test_no_aoi_or_val(self):
        """WaveEnergy: testing Biophysical component w/o AOI or valuation."""
        from natcap.invest import wave_energy

        args = WaveEnergyRegressionTests.generate_base_args(self.workspace_dir)
        wave_energy.execute(args)

        raster_results = [
            'wp_rc.tif', 'wp_kw.tif', 'capwe_rc.tif', 'capwe_mwh.tif'
        ]

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'noaoi', raster_path), 1e-6)

        table_results = ['capwe_rc.csv', 'wp_rc.csv']

        for table_path in table_results:
            pygeoprocessing.testing.assert_csv_equal(
                os.path.join(args['workspace_dir'], 'output', table_path),
                os.path.join(REGRESSION_DATA, 'noaoi', table_path), 1e-6)

    def test_valuation_suffix(self):
        """WaveEnergy: testing suffix through Valuation."""
        from natcap.invest import wave_energy

        args = WaveEnergyRegressionTests.generate_base_args(self.workspace_dir)
        args['aoi_path'] = os.path.join(SAMPLE_DATA, 'AOI_WCVI.shp')
        args['valuation_container'] = True
        args['land_gridPts_path'] = os.path.join(
            SAMPLE_DATA, 'LandGridPts_WCVI.csv')
        args['machine_econ_path'] = os.path.join(
            SAMPLE_DATA, 'Machine_Pelamis_Economic.csv')
        args['number_of_machines'] = 28
        args['results_suffix'] = 'val'

        wave_energy.execute(args)

        raster_results = [
            'wp_rc_val.tif', 'wp_kw_val.tif', 'capwe_rc_val.tif',
            'capwe_mwh_val.tif', 'npv_rc_val.tif', 'npv_usd_val.tif'
        ]

        for raster_path in raster_results:
            self.assertTrue(
                os.path.exists(
                    os.path.join(args['workspace_dir'], 'output',
                                 raster_path)))

        vector_results = ['GridPts_prj_val.shp', 'LandPts_prj_val.shp']

        for vector_path in vector_results:
            self.assertTrue(
                os.path.exists(
                    os.path.join(args['workspace_dir'], 'output',
                                 vector_path)))

        table_results = ['capwe_rc_val.csv', 'wp_rc_val.csv', 'npv_rc_val.csv']

        for table_path in table_results:
            self.assertTrue(
                os.path.exists(
                    os.path.join(args['workspace_dir'], 'output', table_path)))

    @staticmethod
    def _assert_point_vectors_equal(a_path, b_path):
        """Assert that two point geometries in the vectors are equal.

        Parameters:
            a_path (str): a path to an OGR vector.
            b_path (str): a path to an OGR vector.

        Returns:
            None.

        Raises:
            AssertionError when the two point geometries are not equal up to
            desired precision (default is 6).
        """
        a_shape = ogr.Open(a_path)
        a_layer = a_shape.GetLayer(0)
        a_feat = a_layer.GetNextFeature()

        b_shape = ogr.Open(b_path)
        b_layer = b_shape.GetLayer(0)
        b_feat = b_layer.GetNextFeature()

        while a_feat is not None:
            # Get coordinates from point geometry and store them in a list
            a_geom = a_feat.GetGeometryRef()
            a_geom_list = re.findall(r'\d+\.\d+', a_geom.ExportToWkt())
            a_geom_list = [float(x) for x in a_geom_list]

            b_geom = b_feat.GetGeometryRef()
            b_geom_list = re.findall(r'\d+\.\d+', b_geom.ExportToWkt())
            b_geom_list = [float(x) for x in b_geom_list]

            try:
                numpy.testing.assert_array_almost_equal(
                    a_geom_list, b_geom_list)
            except AssertionError:
                a_feature_fid = a_feat.GetFID()
                b_feature_fid = b_feat.GetFID()
                raise AssertionError('Geometries are not equal in feature %s, '
                                     'regression feature %s in layer 0' %
                                     (a_feature_fid, b_feature_fid))
            a_feat = None
            b_feat = None
            a_feat = a_layer.GetNextFeature()
            b_feat = b_layer.GetNextFeature()

        a_shape = None
        b_shape = None


class WaveEnergyValidateTests(unittest.TestCase):
    """Wave Energy Validate: tests for ARGS_SPEC and validate."""

    def setUp(self):
        self.base_required_keys = [
            'workspace_dir',
            'machine_param_path',
            'wave_base_data_path',
            'analysis_area_path',
            'machine_perf_path',
            'dem_path',
        ]

    def test_missing_required_keys(self):
        """WaveEnergy: testing missing required keys from args."""
        from natcap.invest import wave_energy
        from natcap.invest import validation

        args = {}
        validation_error_list = wave_energy.validate(args)
        invalid_keys = validation.get_invalid_keys(validation_error_list)
        expected_missing_keys = set(self.base_required_keys)
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_required_keys_if_valuation(self):
        """WaveEnergy: testing missing required keys given valuation."""
        from natcap.invest import wave_energy
        from natcap.invest import validation

        args = {'valuation_container': True}
        validation_error_list = wave_energy.validate(args)
        invalid_keys = validation.get_invalid_keys(validation_error_list)
        expected_missing_keys = set(
            self.base_required_keys +
            ['number_of_machines', 'machine_econ_path', 'land_gridPts_path'])
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_incorrect_analysis_area_path_value(self):
        """WaveEnergy: testing incorrect analysis_area_path value."""
        from natcap.invest import wave_energy

        args = {}
        args['analysis_area_path'] = 'Incorrect Analysis Area'
        validation_error_list = wave_energy.validate(args)
        expected_message = 'Value must be one of'  # Start of option error msg
        actual_messages = ''
        for keys, error_strings in validation_error_list:
            actual_messages += error_strings
        self.assertTrue(expected_message in actual_messages)

    def test_validate_keys_missing_values(self):
        """WaveEnergy: testing validate when keys are missing values."""
        from natcap.invest import wave_energy

        args = {}
        args['wave_base_data_path'] = None
        args['dem_path'] = None

        validation_error_list = wave_energy.validate(args)
        expected_errors = [
            (['dem_path', 'wave_base_data_path'],
             'Input is required but has no value'),
        ]
        for expected_error in expected_errors:
            self.assertTrue(expected_error in validation_error_list)

    def test_validate_bad_aoi_format(self):
        """WaveEnergy: testing bad AOI vector format with validate."""
        from natcap.invest import wave_energy

        args = {}
        args['aoi_path'] = os.path.join(SAMPLE_DATA, 'bad_AOI_WCVI.shp')

        validation_error_list = wave_energy.validate(args)
        expected_errors = [
            (['aoi_path'], 'Layer must be projected in meters'),
        ]
        for expected_error in expected_errors:
            self.assertTrue(expected_error in validation_error_list)
