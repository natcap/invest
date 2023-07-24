# coding=UTF-8
"""Tests for the Urban Nature Access Model."""
import itertools
import math
import os
import random
import shutil
import tempfile
import textwrap
import unittest

import numpy
import pandas
import pygeoprocessing
import shapely.geometry
from natcap.invest import utils
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

_DEFAULT_ORIGIN = (444720, 3751320)
_DEFAULT_PIXEL_SIZE = (30, -30)
_DEFAULT_EPSG = 3116
_DEFAULT_SRS = osr.SpatialReference()
_DEFAULT_SRS.ImportFromEPSG(_DEFAULT_EPSG)


def _build_model_args(workspace):
    args = {
        'workspace_dir': os.path.join(workspace, 'workspace'),
        'results_suffix': 'suffix',
        'population_raster_path': os.path.join(
            workspace, 'population.tif'),
        'lulc_raster_path': os.path.join(workspace, 'lulc.tif'),
        'lulc_attribute_table': os.path.join(
            workspace, 'lulc_attributes.csv'),
        'decay_function': 'gaussian',
        'urban_nature_demand': 100,  # square meters
        'admin_boundaries_vector_path': os.path.join(
            workspace, 'aois.geojson'),
    }
    if not os.path.exists(workspace):
        os.makedirs(workspace)

    random.seed(-1)  # for our random number generation
    population_pixel_size = (90, -90)
    population_array_shape = (10, 10)
    population_array = numpy.array(
        random.choices(range(0, 100), k=100),
        dtype=numpy.int32).reshape(population_array_shape)
    population_srs = osr.SpatialReference()
    population_srs.ImportFromEPSG(_DEFAULT_EPSG)
    population_wkt = population_srs.ExportToWkt()
    pygeoprocessing.numpy_array_to_raster(
        base_array=population_array,
        target_nodata=-1,
        pixel_size=population_pixel_size,
        origin=_DEFAULT_ORIGIN,
        projection_wkt=population_wkt,
        target_path=args['population_raster_path'])

    lulc_pixel_size = _DEFAULT_PIXEL_SIZE
    lulc_array_shape = (30, 30)
    lulc_array = numpy.array(
        random.choices(range(0, 10), k=900),
        dtype=numpy.int32).reshape(lulc_array_shape)
    pygeoprocessing.numpy_array_to_raster(
        base_array=lulc_array,
        target_nodata=-1,
        pixel_size=lulc_pixel_size,
        origin=_DEFAULT_ORIGIN,
        projection_wkt=population_wkt,
        target_path=args['lulc_raster_path'])

    with open(args['lulc_attribute_table'], 'w') as attr_table:
        attr_table.write(textwrap.dedent(
            """\
            lucode,urban_nature,search_radius_m
            0,0,100
            1,1,100
            2,0,100
            3,1,100
            4,0,100
            5,1,100
            6,0,100
            7,1,100
            8,0,100
            9,1,100"""))

    admin_geom = [
        shapely.geometry.box(
            *pygeoprocessing.get_raster_info(
                args['lulc_raster_path'])['bounding_box'])]
    pygeoprocessing.shapely_geometry_to_vector(
        admin_geom, args['admin_boundaries_vector_path'],
        population_wkt, 'GeoJSON')

    return args


class UNATests(unittest.TestCase):
    """Tests for the Urban Nature Access Model."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the test result
        self.workspace_dir = tempfile.mkdtemp(suffix='\U0001f60e')  # smiley

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_resample_population_raster(self):
        """UNA: Test population raster resampling."""
        from natcap.invest import urban_nature_access

        random.seed(-1)  # for our random number generation

        source_population_raster_path = os.path.join(
            self.workspace_dir, 'population.tif')
        population_pixel_size = (90, -90)
        population_array_shape = (10, 10)

        array_of_100s = numpy.full(
            population_array_shape, 100, dtype=numpy.uint32)
        array_of_random_ints = numpy.array(
            random.choices(range(0, 100), k=100),
            dtype=numpy.uint32).reshape(population_array_shape)

        for population_array in (
                array_of_100s, array_of_random_ints):
            population_srs = osr.SpatialReference()
            population_srs.ImportFromEPSG(_DEFAULT_EPSG)
            population_wkt = population_srs.ExportToWkt()
            pygeoprocessing.numpy_array_to_raster(
                base_array=population_array,
                target_nodata=-1,
                pixel_size=population_pixel_size,
                origin=_DEFAULT_ORIGIN,
                projection_wkt=population_wkt,
                target_path=source_population_raster_path)

            for target_pixel_size in (
                    (30, -30),  # 1/3 the pixel size
                    (4, -4),  # way smaller
                    (100, -100)):  # bigger
                target_population_raster_path = os.path.join(
                    self.workspace_dir, 'resampled_population.tif')
                urban_nature_access._resample_population_raster(
                    source_population_raster_path,
                    target_population_raster_path,
                    lulc_pixel_size=target_pixel_size,
                    lulc_bb=pygeoprocessing.get_raster_info(
                        source_population_raster_path)['bounding_box'],
                    lulc_projection_wkt=population_wkt,
                    working_dir=os.path.join(self.workspace_dir, 'working'))

                resampled_population_array = (
                    pygeoprocessing.raster_to_numpy_array(
                        target_population_raster_path))

                # There should be no significant loss or gain of population due
                # to warping, but the fact that this is aggregating across the
                # whole raster (lots of pixels) means we need to lower the
                # relative tolerance.
                numpy.testing.assert_allclose(
                    population_array.sum(), resampled_population_array.sum(),
                    rtol=1e-3)

    def test_dichotomous_decay_simple(self):
        """UNA: Test dichotomous decay kernel on a simple case."""
        from natcap.invest import urban_nature_access

        expected_distance = 5
        kernel_filepath = os.path.join(self.workspace_dir, 'kernel.tif')

        urban_nature_access._create_kernel_raster(
            urban_nature_access._kernel_dichotomy, expected_distance,
            kernel_filepath)

        expected_array = numpy.array([
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=numpy.uint8)

        extracted_kernel_array = pygeoprocessing.raster_to_numpy_array(
            kernel_filepath)
        numpy.testing.assert_array_equal(
            expected_array, extracted_kernel_array)

    def test_dichotomous_decay_normalized(self):
        """UNA: Test normalized dichotomous kernel."""
        from natcap.invest import urban_nature_access

        expected_distance = 5
        kernel_filepath = os.path.join(self.workspace_dir, 'kernel.tif')

        urban_nature_access._create_kernel_raster(
            urban_nature_access._kernel_dichotomy,
            expected_distance, kernel_filepath, normalize=True)

        expected_array = numpy.array([
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=numpy.float32)
        expected_array /= numpy.sum(expected_array)

        extracted_kernel_array = pygeoprocessing.raster_to_numpy_array(
            kernel_filepath)
        numpy.testing.assert_allclose(
            expected_array, extracted_kernel_array)

    def test_dichotomous_decay_large(self):
        """UNA: Test dichotomous decay on a very large pixel radius."""
        from natcap.invest import urban_nature_access

        # kernel with > 268 million pixels.  This is big enough to force my
        # laptop to noticeably hang while swapping memory on an all in-memory
        # implementation.
        expected_distance = 2**13
        kernel_filepath = os.path.join(self.workspace_dir, 'kernel.tif')

        urban_nature_access._create_kernel_raster(
            urban_nature_access._kernel_dichotomy,
            expected_distance, kernel_filepath)

        expected_shape = (expected_distance*2+1, expected_distance*2+1)
        expected_n_1_pixels = math.pi*expected_distance**2

        kernel_info = pygeoprocessing.get_raster_info(kernel_filepath)
        n_1_pixels = 0
        for _, block in pygeoprocessing.iterblocks((kernel_filepath, 1)):
            n_1_pixels += numpy.count_nonzero(block)

        # 210828417 is only a slight overestimate from the area of the circle
        # at this radius: math.pi*expected_distance**2 = 210828714.13315654
        numpy.testing.assert_allclose(
            n_1_pixels, expected_n_1_pixels, rtol=1e-5)
        self.assertEqual(kernel_info['raster_size'], expected_shape)

    def test_density_decay_simple(self):
        """UNA: Test density decay."""
        from natcap.invest import urban_nature_access

        expected_distance = 200
        kernel_filepath = os.path.join(self.workspace_dir, 'kernel.tif')

        urban_nature_access._create_kernel_raster(
            urban_nature_access._kernel_density,
            expected_distance, kernel_filepath)

        expected_shape = (expected_distance*2+1,) * 2
        kernel_info = pygeoprocessing.get_raster_info(kernel_filepath)
        kernel_array = pygeoprocessing.raster_to_numpy_array(kernel_filepath)
        self.assertEqual(kernel_info['raster_size'], expected_shape)
        numpy.testing.assert_allclose(
            47123.867,  # obtained from manual inspection
            kernel_array.sum())
        self.assertEqual(0.75, kernel_array.max())
        self.assertEqual(0, kernel_array.min())

    def test_density_decay_normalized(self):
        """UNA: Test normalized density decay."""
        from natcap.invest import urban_nature_access

        expected_distance = 200
        kernel_filepath = os.path.join(self.workspace_dir, 'kernel.tif')

        urban_nature_access._create_kernel_raster(
            urban_nature_access._kernel_density,
            expected_distance, kernel_filepath, normalize=True)

        expected_shape = (expected_distance*2+1,) * 2
        kernel_info = pygeoprocessing.get_raster_info(kernel_filepath)
        kernel_array = pygeoprocessing.raster_to_numpy_array(kernel_filepath)
        self.assertEqual(kernel_info['raster_size'], expected_shape)
        numpy.testing.assert_allclose(1, kernel_array.sum())
        self.assertAlmostEqual(1.5915502e-05, kernel_array.max())
        self.assertEqual(0, kernel_array.min())

    def test_power_kernel(self):
        """UNA: Test the power kernel."""
        from natcap.invest import urban_nature_access

        beta = -5
        max_distance = 3
        distance = numpy.array([0, 1, 2, 3, 4])
        kernel = urban_nature_access._kernel_power(
            distance, max_distance, beta)
        # These regression values are calculated by hand
        expected_array = numpy.array([1, 1, (1/32), (1/243), 0])
        numpy.testing.assert_allclose(
            expected_array, kernel)

    def test_exponential_kernel(self):
        """UNA: Test the exponential decay kernel."""
        from natcap.invest import urban_nature_access

        max_distance = 3
        distance = numpy.array([0, 1, 2, 3, 4])
        kernel = urban_nature_access._kernel_exponential(
            distance, max_distance)
        # Regression values are calculated by hand
        expected_array = numpy.array(
            [1, 0.71653134, 0.5134171, 0.36787945, 0])
        numpy.testing.assert_allclose(
            expected_array, kernel)

    def test_gaussian_kernel(self):
        """UNA: Test the gaussian decay kernel."""
        from natcap.invest import urban_nature_access

        max_distance = 3
        distance = numpy.array([0, 1, 2, 3, 4])
        kernel = urban_nature_access._kernel_gaussian(
            distance, max_distance)
        # Regression values are calculated by hand
        expected_array = numpy.array(
            [1, 0.8626563, 0.4935753, 0, 0])
        numpy.testing.assert_allclose(
            expected_array, kernel)

    def test_urban_nature_balance(self):
        """UNA: Test the per-capita urban_nature balance functions."""
        from natcap.invest import urban_nature_access

        nodata = urban_nature_access.FLOAT32_NODATA
        urban_nature_supply = numpy.array([
            [nodata, 100.5],
            [75, 100]], dtype=numpy.float32)
        urban_nature_demand = 50

        population = numpy.array([
            [50, 100],
            [40.75, nodata]], dtype=numpy.float32)

        urban_nature_budget = (
            urban_nature_access._urban_nature_balance_percapita_op(
                urban_nature_supply, urban_nature_demand))
        expected_urban_nature_budget = numpy.array([
            [nodata, 50.5],
            [25, 50]], dtype=numpy.float32)
        numpy.testing.assert_allclose(
            urban_nature_budget, expected_urban_nature_budget)

        supply_demand = urban_nature_access._urban_nature_balance_totalpop_op(
            urban_nature_budget, population)
        expected_supply_demand = numpy.array([
            [nodata, 100 * 50.5],
            [25 * 40.75, nodata]], dtype=numpy.float32)
        numpy.testing.assert_allclose(
            supply_demand, expected_supply_demand)

    def test_reclassify_and_multpliy(self):
        """UNA: test reclassification/multiplication function."""
        from natcap.invest import urban_nature_access

        nodata = 255
        aois_array = numpy.array([
            [nodata, 1, 2, 3],
            [nodata, 1, 2, 3],
            [nodata, 1, 2, 3],
            [nodata, nodata, 2, 3]], dtype=numpy.uint8)
        reclassification_map = {
            1: 0.1,
            2: 0.3,
            3: 0.5,
        }
        supply_array = numpy.full(aois_array.shape, 3, dtype=numpy.float32)
        supply_array[1, 3] = 255

        aois_path = os.path.join(self.workspace_dir, 'aois.tif')
        supply_path = os.path.join(self.workspace_dir, 'supply.tif')

        for array, target_path in [(aois_array, aois_path),
                                   (supply_array, supply_path)]:
            pygeoprocessing.geoprocessing.numpy_array_to_raster(
                array, nodata, _DEFAULT_PIXEL_SIZE, _DEFAULT_ORIGIN,
                _DEFAULT_SRS.ExportToWkt(), target_path)

        target_raster_path = os.path.join(self.workspace_dir, 'target.tif')
        urban_nature_access._reclassify_and_multiply(
            aois_path, reclassification_map, supply_path, target_raster_path)

        float_nodata = urban_nature_access.FLOAT32_NODATA
        expected_array = numpy.array([
            [float_nodata, 0.3, 0.9, 1.5],
            [float_nodata, 0.3, 0.9, float_nodata],
            [float_nodata, 0.3, 0.9, 1.5],
            [float_nodata, float_nodata, 0.9, 1.5]], dtype=numpy.float32)
        numpy.testing.assert_allclose(
            expected_array,
            pygeoprocessing.raster_to_numpy_array(target_raster_path))

    def test_core_model(self):
        """UNA: Run through the model with base data."""
        from natcap.invest import urban_nature_access

        args = _build_model_args(self.workspace_dir)
        args['search_radius_mode'] = urban_nature_access.RADIUS_OPT_UNIFORM
        args['search_radius'] = 100

        urban_nature_access.execute(args)

        # Since we're doing a semi-manual alignment step, assert that the
        # aligned LULC and population rasters have the same pixel sizes,
        # origin and raster dimensions.
        # TODO: Remove these assertions once we're using align_and_resize
        # and it works as expected.
        aligned_lulc_raster_info = pygeoprocessing.get_raster_info(
            os.path.join(args['workspace_dir'], 'intermediate',
                         f"aligned_lulc_{args['results_suffix']}.tif"))
        aligned_population_raster_info = pygeoprocessing.get_raster_info(
            os.path.join(
                args['workspace_dir'], 'intermediate',
                f"aligned_population_{args['results_suffix']}.tif"))
        numpy.testing.assert_allclose(
            aligned_lulc_raster_info['pixel_size'],
            aligned_population_raster_info['pixel_size'])
        numpy.testing.assert_allclose(
            aligned_lulc_raster_info['raster_size'],
            aligned_population_raster_info['raster_size'])
        numpy.testing.assert_allclose(
            aligned_lulc_raster_info['geotransform'],
            aligned_population_raster_info['geotransform'])
        numpy.testing.assert_allclose(
            aligned_lulc_raster_info['bounding_box'],
            aligned_population_raster_info['bounding_box'])

        # Check that we're getting the appropriate summary values in the
        # admin units vector.
        layer_name = f"admin_boundaries_{args['results_suffix']}"
        admin_vector_path = os.path.join(
            args['workspace_dir'], 'output', f"{layer_name}.gpkg")
        admin_vector = gdal.OpenEx(admin_vector_path)
        admin_layer = admin_vector.GetLayer(layer_name)
        self.assertEqual(admin_layer.GetFeatureCount(), 1)

        # expected field values from eyeballing the results; random seed = 1
        expected_values = {
            'SUP_DEMadm_cap': -17.9078,
            'Pund_adm': 3991.827148,
            'Povr_adm': 1084.172852,
            urban_nature_access.ID_FIELDNAME: 0,
        }
        admin_feature = admin_layer.GetFeature(1)
        self.assertEqual(
            expected_values.keys(),
            admin_feature.items().keys()
        )
        for fieldname, expected_value in expected_values.items():
            numpy.testing.assert_allclose(
                admin_feature.GetField(fieldname), expected_value)

        # The sum of the under-and-oversupplied populations should be equal
        # to the total population count.
        population_array = pygeoprocessing.raster_to_numpy_array(
            args['population_raster_path'])
        numpy.testing.assert_allclose(
            (expected_values['Pund_adm'] + expected_values['Povr_adm']),
            population_array.sum())

        admin_vector = None
        admin_layer = None

    def test_split_urban_nature(self):
        from natcap.invest import urban_nature_access

        args = _build_model_args(self.workspace_dir)
        args['search_radius_mode'] = urban_nature_access.RADIUS_OPT_URBAN_NATURE

        # The split urban_nature feature requires an extra column in the
        # attribute table.
        attribute_table = pandas.read_csv(args['lulc_attribute_table'])
        new_search_radius_values = {
            value: 30*value for value in range(1, 10, 2)}
        new_search_radius_values[7] = 30 * 9  # make one a duplicate distance.
        attribute_table['search_radius_m'] = attribute_table['lucode'].map(
            new_search_radius_values)
        attribute_table.to_csv(args['lulc_attribute_table'], index=False)

        urban_nature_access.execute(args)

        admin_vector_path = os.path.join(
            args['workspace_dir'], 'output',
            f"admin_boundaries_{args['results_suffix']}.gpkg")
        admin_vector = gdal.OpenEx(admin_vector_path)
        admin_layer = admin_vector.GetLayer()
        self.assertEqual(admin_layer.GetFeatureCount(), 1)

        # expected field values from eyeballing the results; random seed = 1
        expected_values = {
            'SUP_DEMadm_cap': -18.045702,
            'Pund_adm': 4475.123047,
            'Povr_adm': 600.876587,
            urban_nature_access.ID_FIELDNAME: 0,
        }
        admin_feature = admin_layer.GetFeature(1)
        self.assertEqual(
            expected_values.keys(),
            admin_feature.items().keys()
        )
        for fieldname, expected_value in expected_values.items():
            numpy.testing.assert_allclose(
                admin_feature.GetField(fieldname), expected_value)

        # The sum of the under-and-oversupplied populations should be equal
        # to the total population count.
        population_array = pygeoprocessing.raster_to_numpy_array(
            args['population_raster_path'])
        numpy.testing.assert_allclose(
            (expected_values['Pund_adm'] + expected_values['Povr_adm']),
            population_array.sum())

        admin_vector = None
        admin_layer = None

    def test_split_population(self):
        """UNA: test split population optional module.

        Split population is not a radius mode, it's a summary statistics mode.
        Therefore, we test with another mode, such as uniform search radius.
        """
        from natcap.invest import urban_nature_access

        args = _build_model_args(self.workspace_dir)
        args['search_radius_mode'] = urban_nature_access.RADIUS_OPT_UNIFORM
        args['search_radius'] = 100
        args['aggregate_by_pop_group'] = True
        del args['results_suffix']

        admin_geom = [
            shapely.geometry.box(
                *pygeoprocessing.get_raster_info(
                    args['lulc_raster_path'])['bounding_box'])]
        fields = {
            'pop_female': ogr.OFTReal,
            'pop_male': ogr.OFTReal,
        }
        attributes = [
            {'pop_female': 0.56, 'pop_male': 0.44}
        ]
        pygeoprocessing.shapely_geometry_to_vector(
            admin_geom, args['admin_boundaries_vector_path'],
            pygeoprocessing.get_raster_info(
                args['population_raster_path'])['projection_wkt'],
            'GeoJSON', fields, attributes)

        urban_nature_access.execute(args)

        summary_vector = gdal.OpenEx(
            os.path.join(args['workspace_dir'], 'output',
                         'admin_boundaries.gpkg'))
        summary_layer = summary_vector.GetLayer()
        self.assertEqual(summary_layer.GetFeatureCount(), 1)
        summary_feature = summary_layer.GetFeature(1)

        def _read_and_sum_raster(path):
            array = pygeoprocessing.raster_to_numpy_array(path)
            nodata = pygeoprocessing.get_raster_info(path)['nodata'][0]
            return numpy.sum(array[~utils.array_equals_nodata(array, nodata)])

        intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')
        for (supply_type, supply_field), fieldname in itertools.product(
                [('over', 'Povr_adm'), ('under', 'Pund_adm')], fields.keys()):
            groupname = fieldname.replace('pop_', '')
            supply_raster_path = os.path.join(
                 intermediate_dir,
                 f'{supply_type}supplied_population.tif')
            group_supply_raster_path = os.path.join(
                 intermediate_dir,
                 f'{supply_type}supplied_population_{groupname}.tif')
            pop_proportion = summary_feature.GetField(fieldname)
            computed_value = summary_feature.GetField(
                f'{supply_field}_{groupname}')

            numpy.testing.assert_allclose(
                computed_value,
                _read_and_sum_raster(supply_raster_path) * pop_proportion,
                rtol=1e-6
            )
            numpy.testing.assert_allclose(
                computed_value,
                _read_and_sum_raster(group_supply_raster_path),
                rtol=1e-6
            )

    def test_radii_by_pop_group(self):
        """UNA: Test defining radii by population group."""
        from natcap.invest import urban_nature_access

        args = _build_model_args(self.workspace_dir)
        args['search_radius_mode'] = urban_nature_access.RADIUS_OPT_POP_GROUP
        args['population_group_radii_table'] = os.path.join(
            self.workspace_dir, 'pop_group_radii.csv')
        del args['results_suffix']

        with open(args['population_group_radii_table'], 'w') as pop_grp_table:
            pop_grp_table.write(
                textwrap.dedent("""\
                    pop_group,search_radius_m
                    pop_female,100
                    pop_male,100"""))

        admin_geom = [
            shapely.geometry.box(
                *pygeoprocessing.get_raster_info(
                    args['lulc_raster_path'])['bounding_box'])]
        fields = {
            'pop_female': ogr.OFTReal,
            'pop_male': ogr.OFTReal,
        }
        attributes = [
            {'pop_female': 0.56, 'pop_male': 0.44}
        ]
        pygeoprocessing.shapely_geometry_to_vector(
            admin_geom, args['admin_boundaries_vector_path'],
            pygeoprocessing.get_raster_info(
                args['population_raster_path'])['projection_wkt'],
            'GeoJSON', fields, attributes)

        urban_nature_access.execute(args)

        summary_vector = gdal.OpenEx(
            os.path.join(args['workspace_dir'], 'output',
                         'admin_boundaries.gpkg'))
        summary_layer = summary_vector.GetLayer()
        self.assertEqual(summary_layer.GetFeatureCount(), 1)
        summary_feature = summary_layer.GetFeature(1)

        expected_field_values = {
            'pop_female': attributes[0]['pop_female'],
            'pop_male': attributes[0]['pop_male'],
            'adm_unit_id': 0,
            'Pund_adm': 0,
            'Pund_adm_female': 2235.423095703125,
            'Pund_adm_male': 1756.404052734375,
            'Povr_adm': 0,
            'Povr_adm_female': 607.13671875,
            'Povr_adm_male': 477.0360107421875,
            'SUP_DEMadm_cap': -17.90779987933412,
            'SUP_DEMadm_cap_female': -17.907799675104435,
            'SUP_DEMadm_cap_male': -17.907800139262825,
        }
        self.assertEqual(
            set(defn.GetName() for defn in summary_layer.schema),
            set(expected_field_values.keys()))
        for fieldname, expected_value in expected_field_values.items():
            self.assertAlmostEqual(
                expected_value, summary_feature.GetField(fieldname))

    def test_modes_same_radii_same_results(self):
        """UNA: all modes have same results when consistent radii.

        Although the different modes have different ways of defining their
        search radii, the urban_nature_supply raster should be numerically
        equivalent if they all use the same search radii.

        This is a good gut-check of basic model behavior across modes.
        """
        from natcap.invest import urban_nature_access

        # This radius will be the same across all model runs.
        search_radius = 1000
        uniform_args = _build_model_args(
            os.path.join(self.workspace_dir, 'radius_uniform'))
        uniform_args['results_suffix'] = 'uniform'
        uniform_args['workspace_dir'] = os.path.join(
            self.workspace_dir, 'radius_uniform')
        uniform_args['search_radius_mode'] = (
            urban_nature_access.RADIUS_OPT_UNIFORM)
        uniform_args['search_radius'] = search_radius

        # build args for split urban_nature mode
        split_urban_nature_args = _build_model_args(
            os.path.join(self.workspace_dir, 'radius_urban_nature'))
        split_urban_nature_args['results_suffix'] = 'urban_nature'
        split_urban_nature_args['search_radius_mode'] = (
            urban_nature_access.RADIUS_OPT_URBAN_NATURE)
        attribute_table = pandas.read_csv(
            split_urban_nature_args['lulc_attribute_table'])
        new_search_radius_values = dict(
            (lucode, search_radius) for lucode in attribute_table['lucode'])
        attribute_table['search_radius_m'] = attribute_table['lucode'].map(
            new_search_radius_values)
        attribute_table.to_csv(
            split_urban_nature_args['lulc_attribute_table'], index=False)

        # build args for split population group mode
        pop_group_args = _build_model_args(
            os.path.join(self.workspace_dir, 'radius_popgroup'))
        pop_group_args['results_suffix'] = 'popgroup'
        pop_group_args['search_radius_mode'] = (
            urban_nature_access.RADIUS_OPT_POP_GROUP)
        pop_group_args['population_group_radii_table'] = os.path.join(
            self.workspace_dir, 'pop_group_radii.csv')

        table_path = pop_group_args['population_group_radii_table']
        with open(table_path, 'w') as pop_grp_table:
            pop_grp_table.write(
                textwrap.dedent(f"""\
                    pop_group,search_radius_m
                    pop_female,{search_radius}
                    pop_male,{search_radius}"""))
        admin_geom = [
            shapely.geometry.box(
                *pygeoprocessing.get_raster_info(
                    pop_group_args['lulc_raster_path'])['bounding_box'])]
        fields = {f'pop_{group}': ogr.OFTReal for group in ('female', 'male')}
        attributes = [{'pop_female': 0.56, 'pop_male': 0.44}]
        pygeoprocessing.shapely_geometry_to_vector(
            admin_geom, pop_group_args['admin_boundaries_vector_path'],
            pygeoprocessing.get_raster_info(
                pop_group_args['population_raster_path'])['projection_wkt'],
            'GeoJSON', fields, attributes)

        for args in (uniform_args, split_urban_nature_args, pop_group_args):
            urban_nature_access.execute(args)

            # make sure the output dir contains the correct files.
            for output_filename in (
                    urban_nature_access._OUTPUT_BASE_FILES.values()):
                basename, ext = os.path.splitext(
                    os.path.basename(output_filename))
                suffix = args['results_suffix']
                filepath = os.path.join(args['workspace_dir'], 'output',
                                        f'{basename}_{suffix}{ext}')
                self.assertTrue(os.path.exists(filepath))

            # check the urban_nature demand raster
            population = pygeoprocessing.raster_to_numpy_array(
                os.path.join(args['workspace_dir'], 'intermediate',
                             f'masked_population_{suffix}.tif'))
            demand = pygeoprocessing.raster_to_numpy_array(
                os.path.join(args['workspace_dir'], 'output',
                             f'urban_nature_demand_{suffix}.tif'))
            nodata = urban_nature_access.FLOAT32_NODATA
            valid_pixels = ~utils.array_equals_nodata(population, nodata)
            numpy.testing.assert_allclose(
                (population[valid_pixels].sum() *
                    float(args['urban_nature_demand'])),
                demand[valid_pixels].sum())

            # check the total-population urban_nature balance
            per_capita_balance = pygeoprocessing.raster_to_numpy_array(
                os.path.join(args['workspace_dir'], 'output',
                             f'urban_nature_balance_percapita_{suffix}.tif'))
            totalpop_balance = pygeoprocessing.raster_to_numpy_array(
                os.path.join(args['workspace_dir'], 'output',
                             f'urban_nature_balance_totalpop_{suffix}.tif'))
            numpy.testing.assert_allclose(
                per_capita_balance[valid_pixels] * population[valid_pixels],
                totalpop_balance[valid_pixels],
                rtol=1e-5)  # accommodate accumulation of numerical error

        uniform_radius_supply = pygeoprocessing.raster_to_numpy_array(
            os.path.join(uniform_args['workspace_dir'], 'output',
                         'urban_nature_supply_uniform.tif'))
        split_urban_nature_supply = pygeoprocessing.raster_to_numpy_array(
            os.path.join(split_urban_nature_args['workspace_dir'], 'output',
                         'urban_nature_supply_urban_nature.tif'))
        split_pop_groups_supply = pygeoprocessing.raster_to_numpy_array(
            os.path.join(pop_group_args['workspace_dir'], 'output',
                         'urban_nature_supply_popgroup.tif'))

        numpy.testing.assert_allclose(
            uniform_radius_supply, split_urban_nature_supply, rtol=1e-6)
        numpy.testing.assert_allclose(
            uniform_radius_supply, split_pop_groups_supply, rtol=1e-6)

    def test_polygon_overlap(self):
        """UNA: Test that we can check if polygons overlap."""
        from natcap.invest import urban_nature_access
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(_DEFAULT_EPSG)
        wkt = srs.ExportToWkt()

        origin_x, origin_y = _DEFAULT_ORIGIN
        polygon_1 = shapely.geometry.Point(origin_x, origin_y).buffer(10)
        polygon_2 = shapely.geometry.Point(origin_x+20, origin_y+20).buffer(50)
        polygon_3 = shapely.geometry.Point(origin_x+50, origin_y+50).buffer(10)

        vector_path = os.path.join(self.workspace_dir, 'vector_nonoverlapping.geojson')
        pygeoprocessing.shapely_geometry_to_vector(
            [polygon_1, polygon_3], vector_path, wkt, 'GeoJSON')
        self.assertFalse(urban_nature_access._geometries_overlap(vector_path))

        vector_path = os.path.join(self.workspace_dir, 'vector_overlapping.geojson')
        pygeoprocessing.shapely_geometry_to_vector(
            [polygon_1, polygon_2, polygon_3], vector_path, wkt, 'GeoJSON')
        self.assertTrue(urban_nature_access._geometries_overlap(vector_path))

    def test_invalid_search_radius_mode(self):
        """UNA: Assert an exception when invalid radius mode provided."""
        from natcap.invest import urban_nature_access

        args = _build_model_args(self.workspace_dir)
        args['search_radius_mode'] = 'some invalid mode'

        with self.assertRaises(ValueError) as cm:
            urban_nature_access.execute(args)

        self.assertIn('Invalid search radius mode provided', str(cm.exception))
        for mode_suffix in ('UNIFORM', 'URBAN_NATURE', 'POP_GROUP'):
            valid_mode_string = getattr(urban_nature_access,
                                        f'RADIUS_OPT_{mode_suffix}')
            self.assertIn(valid_mode_string, str(cm.exception))

    def test_square_pixels(self):
        """UNA: Assert we can make square pixels as expected."""
        from natcap.invest import urban_nature_access

        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        nodata = 255
        for (pixel_size, expected_pixel_size) in (
                ((10, -10), (10, -10)),
                ((-10, 10), (-10, 10)),
                ((5, -10), (7.5, -7.5)),
                ((-5, -10), (-7.5, -7.5))):
            pygeoprocessing.numpy_array_to_raster(
                numpy.ones((10, 10), dtype=numpy.uint8), nodata, pixel_size,
                _DEFAULT_ORIGIN, _DEFAULT_SRS.ExportToWkt(), raster_path)
            computed_pixel_size = (
                urban_nature_access._square_off_pixels(raster_path))
            self.assertEqual(computed_pixel_size, expected_pixel_size)

    def test_weighted_sum(self):
        """UNA: Assert weighted sum is correct."""
        from natcap.invest import urban_nature_access

        weights_paths = []
        source_paths = []

        for index in (1, 2):
            nodata = -index
            source_array = numpy.full((30, 30), index, dtype=numpy.float32)
            source_array[5][5] = nodata
            weight_array = numpy.full((30, 30), index/4, dtype=numpy.float32)
            weight_array[10][10] = nodata
            source_path = os.path.join(self.workspace_dir,
                                       f'source_{index}.tif')
            source_paths.append(source_path)
            weight_path = os.path.join(self.workspace_dir,
                                       f'weights_{index}.tif')
            weights_paths.append(weight_path)
            for array, path in ((source_array, source_path),
                                (weight_array, weight_path)):
                pygeoprocessing.numpy_array_to_raster(
                    base_array=array,
                    target_nodata=nodata,
                    pixel_size=_DEFAULT_PIXEL_SIZE,
                    origin=_DEFAULT_ORIGIN,
                    projection_wkt=_DEFAULT_SRS.ExportToWkt(),
                    target_path=path)

        target_path = os.path.join(self.workspace_dir, 'weighted_sum.tif')
        urban_nature_access._weighted_sum(source_paths, weights_paths,
                                          target_path)

        weighted_sum_array = pygeoprocessing.raster_to_numpy_array(target_path)
        weighted_sum_nodata = pygeoprocessing.get_raster_info(
            target_path)['nodata'][0]

        # check that we have the expected number of nodata pixels
        nodata_pixels = numpy.isclose(weighted_sum_array, weighted_sum_nodata)
        self.assertEqual(
            numpy.count_nonzero(nodata_pixels), 2)

        # Check that the sum is what we expect, given the expected nodata
        # pixels
        numpy.testing.assert_allclose(
            numpy.sum(weighted_sum_array[~nodata_pixels]), 1122.5)

    def test_write_vector(self):
        """UNA: test writing of various float types to the output vector."""
        # TODO
        pass

    def test_validate(self):
        """UNA: Basic test for validation."""
        from natcap.invest import urban_nature_access
        args = _build_model_args(self.workspace_dir)
        args['search_radius_mode'] = urban_nature_access.RADIUS_OPT_URBAN_NATURE
        self.assertEqual(urban_nature_access.validate(args), [])
