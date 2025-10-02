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

gdal.UseExceptions()
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
            9,1,100
            """))

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
                    working_dir=self.workspace_dir)

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

    def test_density_kernel(self):
        """UNA: Test the density kernel."""
        from natcap.invest import urban_nature_access

        max_distance = 3
        distance = numpy.array([0, 1, 2, 3, 4])
        kernel = urban_nature_access._kernel_density(distance, max_distance)
        # These regression values are calculated by hand
        expected_array = numpy.array([.75, 2/3, 5/12, 0, 0])
        numpy.testing.assert_allclose(expected_array, kernel)

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
        urban_nature_supply_percapita = numpy.array([
            [nodata, 100.5],
            [75, 100]], dtype=numpy.float32)
        urban_nature_demand = 50
        supply_path = os.path.join(self.workspace_dir, 'supply.path')
        target_path = os.path.join(self.workspace_dir, 'target.path')

        pygeoprocessing.numpy_array_to_raster(
            urban_nature_supply_percapita, nodata, _DEFAULT_PIXEL_SIZE,
            _DEFAULT_ORIGIN, _DEFAULT_SRS.ExportToWkt(), supply_path)

        urban_nature_access._calculate_urban_nature_balance_percapita(
            supply_path, urban_nature_demand, target_path)

        urban_nature_budget = pygeoprocessing.raster_to_numpy_array(
            target_path)

        expected_urban_nature_budget = numpy.array([
            [nodata, 50.5],
            [25, 50]], dtype=numpy.float32)
        numpy.testing.assert_allclose(
            urban_nature_budget, expected_urban_nature_budget)

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

        accessible_urban_nature_array = pygeoprocessing.raster_to_numpy_array(
            os.path.join(args['workspace_dir'], 'output',
                         'accessible_urban_nature_suffix.tif'))
        valid_mask = ~pygeoprocessing.array_equals_nodata(
            accessible_urban_nature_array, urban_nature_access.FLOAT32_NODATA)
        valid_pixels = accessible_urban_nature_array[valid_mask]
        self.assertAlmostEqual(numpy.sum(valid_pixels), 6221004.41259766)
        self.assertAlmostEqual(numpy.min(valid_pixels), 1171.7352294921875)
        self.assertAlmostEqual(numpy.max(valid_pixels), 11898.0712890625)

    def test_no_lulc_nodata(self):
        """UNA: verify behavior when the LULC has no nodata value."""
        from natcap.invest import urban_nature_access

        args = _build_model_args(self.workspace_dir)
        args['search_radius_mode'] = urban_nature_access.RADIUS_OPT_UNIFORM
        args['search_radius'] = 100

        raster = gdal.OpenEx(args['lulc_raster_path'], gdal.OF_RASTER)
        band = raster.GetRasterBand(1)
        band.DeleteNoDataValue()
        band = None
        raster = None
        urban_nature_access.execute(args)

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
                admin_feature.GetField(fieldname), expected_value, rtol=1e-6)

        # The sum of the under-and-oversupplied populations should be equal
        # to the total population count.
        population_array = pygeoprocessing.raster_to_numpy_array(
            args['population_raster_path'])
        numpy.testing.assert_allclose(
            (expected_values['Pund_adm'] + expected_values['Povr_adm']),
            population_array.sum())

        admin_vector = None
        admin_layer = None

        output_dir = os.path.join(args['workspace_dir'], 'output')
        self._assert_urban_nature(os.path.join(
            output_dir, 'accessible_urban_nature_lucode_1_suffix.tif'),
            72000.0, 0.0, 900.0)
        self._assert_urban_nature(os.path.join(
            output_dir, 'accessible_urban_nature_lucode_3_suffix.tif'),
            1034934.9864730835, 0.0, 4431.1650390625)
        self._assert_urban_nature(os.path.join(
            output_dir, 'accessible_urban_nature_lucode_5_suffix.tif'),
            2837622.9519348145, 0.0, 8136.6884765625)
        self._assert_urban_nature(os.path.join(
            output_dir, 'accessible_urban_nature_lucode_7_suffix.tif'),
            8112734.805541992, 2019.2935791015625, 17729.431640625)
        self._assert_urban_nature(os.path.join(
            output_dir, 'accessible_urban_nature_lucode_9_suffix.tif'),
            7744116.974121094, 1567.57958984375, 12863.4619140625)

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
            return numpy.sum(array[~pygeoprocessing.array_equals_nodata(array, nodata)])

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

    def _assert_urban_nature(self, path, sum_value, min_value, max_value):
        """Compare a raster's sum, min and max to given values.

        The raster is assumed to be an accessible urban nature raster.

        Args:
            path (str): The path to an urban nature raster.
            sum_value (float): The expected sum of the raster.
            min_value (float): The expected min of the raster.
            max_value (float): The expected max of the raster.

        Returns:
            ``None``

        Raises:
            AssertionError: When the raster's sum, min or max values are not
            numerically close to the expected values.
        """
        from natcap.invest import urban_nature_access

        accessible_urban_nature_array = (
            pygeoprocessing.raster_to_numpy_array(path))
        valid_mask = ~pygeoprocessing.array_equals_nodata(
            accessible_urban_nature_array,
            urban_nature_access.FLOAT32_NODATA)
        valid_pixels = accessible_urban_nature_array[valid_mask]
        self.assertAlmostEqual(numpy.sum(valid_pixels), sum_value)
        self.assertAlmostEqual(numpy.min(valid_pixels), min_value)
        self.assertAlmostEqual(numpy.max(valid_pixels), max_value)

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
            'Pund_adm': 3991.8271484375,
            'Pund_adm_female': 2235.423095703125,
            'Pund_adm_male': 1756.404052734375,
            'Povr_adm': 1084.1727294921875,
            'Povr_adm_female': 607.13671875,
            'Povr_adm_male': 477.0360107421875,
            'SUP_DEMadm_cap': -17.907799109781322,
            'SUP_DEMadm_cap_female': -17.90779830090304,
            'SUP_DEMadm_cap_male': -17.907800139262825,
        }
        self.assertEqual(
            set(defn.GetName() for defn in summary_layer.schema),
            set(expected_field_values.keys()))
        for fieldname, expected_value in expected_field_values.items():
            numpy.testing.assert_allclose(
                expected_value, summary_feature.GetField(fieldname), rtol=1e-6)

        output_dir = os.path.join(args['workspace_dir'], 'output')
        self._assert_urban_nature(os.path.join(
            output_dir, 'accessible_urban_nature_to_pop_male.tif'),
            6221004.412597656, 1171.7352294921875, 11898.0712890625)
        self._assert_urban_nature(os.path.join(
            output_dir, 'accessible_urban_nature_to_pop_female.tif'),
            6221004.412597656, 1171.7352294921875, 11898.0712890625)

    def test_radii_by_pop_group_exponential_kernal(self):
        """UNA: Regression test defining radii by population group.

        Issue for this bug: https://github.com/natcap/invest/issues/1502
        """
        from natcap.invest import urban_nature_access

        args = _build_model_args(self.workspace_dir)
        args['decay_function'] = urban_nature_access.KERNEL_LABEL_EXPONENTIAL
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
            'Pund_adm': 4801.7900390625,
            'Pund_adm_female': 2689.00244140625,
            'Pund_adm_male': 2112.78759765625,
            'Povr_adm': 274.2098693847656,
            'Povr_adm_female': 153.55752563476562,
            'Povr_adm_male': 120.65234375,
            'SUP_DEMadm_cap': -17.907799109781322,
            'SUP_DEMadm_cap_female': -17.90779830090304,
            'SUP_DEMadm_cap_male': -17.907800139262825,
        }
        self.assertEqual(
            set(defn.GetName() for defn in summary_layer.schema),
            set(expected_field_values.keys()))
        for fieldname, expected_value in expected_field_values.items():
            numpy.testing.assert_allclose(
                expected_value, summary_feature.GetField(fieldname), rtol=1e-6)

        output_dir = os.path.join(args['workspace_dir'], 'output')
        self._assert_urban_nature(os.path.join(
            output_dir, 'accessible_urban_nature_to_pop_male.tif'),
            17812884.000976562, 7740.4287109375, 25977.67578125)
        self._assert_urban_nature(os.path.join(
            output_dir, 'accessible_urban_nature_to_pop_female.tif'),
            17812884.000976562, 7740.4287109375, 25977.67578125)

    def test_modes_same_radii_same_results(self):
        """UNA: all modes have same results when consistent radii.

        Although the different modes have different ways of defining their
        search radii, the urban_nature_supply_percapita raster should be numerically
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
            for output_filename in ['urban_nature_supply_percapita.tif',
                                    'admin_boundaries.gpkg',
                                    'urban_nature_balance_percapita.tif',
                                    'urban_nature_balance_totalpop.tif',
                                    'urban_nature_demand.tif']:
                basename, ext = os.path.splitext(output_filename)
                suffix = args['results_suffix']
                filepath = os.path.join(
                    args['workspace_dir'], 'output', f'{basename}_{suffix}{ext}')
                self.assertTrue(os.path.exists(filepath))

            # check the urban_nature demand raster
            population = pygeoprocessing.raster_to_numpy_array(
                os.path.join(args['workspace_dir'], 'intermediate',
                             f'masked_population_{suffix}.tif'))
            demand = pygeoprocessing.raster_to_numpy_array(
                os.path.join(args['workspace_dir'], 'output',
                             f'urban_nature_demand_{suffix}.tif'))
            nodata = urban_nature_access.FLOAT32_NODATA
            valid_pixels = ~pygeoprocessing.array_equals_nodata(population, nodata)
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
                         'urban_nature_supply_percapita_uniform.tif'))
        split_urban_nature_supply_percapita = (
            pygeoprocessing.raster_to_numpy_array(
                os.path.join(
                    split_urban_nature_args['workspace_dir'], 'output',
                    'urban_nature_supply_percapita_urban_nature.tif')))
        split_pop_groups_supply = pygeoprocessing.raster_to_numpy_array(
            os.path.join(pop_group_args['workspace_dir'], 'output',
                         'urban_nature_supply_percapita_popgroup.tif'))

        numpy.testing.assert_allclose(
            uniform_radius_supply, split_urban_nature_supply_percapita,
            rtol=1e-6)
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
        """UNA: test writing of various numeric types to the output vector."""
        from natcap.invest import urban_nature_access
        args = _build_model_args(self.workspace_dir)

        admin_vector = gdal.OpenEx(args['admin_boundaries_vector_path'])
        admin_layer = admin_vector.GetLayer()
        fid = admin_layer.GetNextFeature().GetFID()
        admin_layer = None
        admin_vector = None

        feature_attrs = {
            fid: {
                'my-field-1': float(1.2345),
                'my-field-2': numpy.float32(2.34567),
                'my-field-3': numpy.float64(3.45678),
                'my-field-4': int(4),
                'my-field-5': numpy.int16(5),
                'my-field-6': numpy.int32(6),
            },
        }
        target_vector_path = os.path.join(self.workspace_dir, 'target.gpkg')
        urban_nature_access._write_supply_demand_vector(
            args['admin_boundaries_vector_path'], feature_attrs,
            target_vector_path)

        self.assertTrue(os.path.exists(target_vector_path))
        try:
            vector = gdal.OpenEx(target_vector_path)
            self.assertEqual(vector.GetLayerCount(), 1)
            layer = vector.GetLayer()
            self.assertEqual(len(layer.schema), len(feature_attrs[fid]))
            self.assertEqual(layer.GetFeatureCount(), 1)
            feature = layer.GetFeature(fid)
            for field_name, expected_field_value in feature_attrs[fid].items():
                self.assertEqual(
                    feature.GetField(field_name), expected_field_value)
        finally:
            feature = None
            layer = None
            vector = None

    def test_write_vector_for_single_raster_modes(self):
        """UNA: create a summary vector for single-raster summary stats."""

        from natcap.invest import urban_nature_access

        args = _build_model_args(self.workspace_dir)

        # Overwrite all population pixels with 0
        try:
            raster = gdal.Open(args['population_raster_path'], gdal.GA_Update)
            band = raster.GetRasterBand(1)
            array = band.ReadAsArray()
            array.fill(0.0)
            band.WriteArray(array)
        finally:
            raster = band = None

        args['search_radius_mode'] = urban_nature_access.RADIUS_OPT_UNIFORM
        args['search_radius'] = 1000

        urban_nature_access.execute(args)

        summary_vector = gdal.OpenEx(
            os.path.join(args['workspace_dir'], 'output',
                         'admin_boundaries_suffix.gpkg'))
        summary_layer = summary_vector.GetLayer()
        self.assertEqual(summary_layer.GetFeatureCount(), 1)
        summary_feature = summary_layer.GetFeature(1)

        expected_field_values = {
            'adm_unit_id': 0,
            'Pund_adm': 0,
            'Povr_adm': 0,
            'SUP_DEMadm_cap': None,  # OGR converts NaN to None.
        }
        self.assertEqual(
            set(defn.GetName() for defn in summary_layer.schema),
            set(expected_field_values.keys()))
        for fieldname, expected_value in expected_field_values.items():
            self.assertAlmostEqual(
                expected_value, summary_feature.GetField(fieldname))

    def test_urban_nature_proportion(self):
        """UNA: Run the model with urban nature proportion."""
        from natcap.invest import urban_nature_access

        args = _build_model_args(self.workspace_dir)
        args['search_radius_mode'] = urban_nature_access.RADIUS_OPT_UNIFORM
        args['search_radius'] = 1000
        with open(args['lulc_attribute_table'], 'a') as attr_table:
            attr_table.write("10,0.5,100\n")

        # make sure our inputs validate
        validation_results = urban_nature_access.validate(args)
        self.assertEqual(validation_results, [])

        urban_nature_access.execute(args)

    def test_reclassify_urban_nature(self):
        """UNA: Test for urban nature area reclassification."""
        from natcap.invest import urban_nature_access
        args = _build_model_args(self.workspace_dir)

        # Rewrite the lulc attribute table to use proportions of urban nature.
        with open(args['lulc_attribute_table'], 'w') as attr_table:
            attr_table.write(textwrap.dedent(
                """\
                lucode,urban_nature,search_radius_m
                0,0,100
                1,0.1,100
                2,0,100
                3,0.3,100
                4,0,100
                5,0.5,100
                6,0,100
                7,0.7,100
                8,0,100
                9,0.9,100
                """))

        urban_nature_area_path = os.path.join(
            self.workspace_dir, 'urban_nature_area.tif')

        for limit_to_lucodes in (None, set([1, 3])):
            urban_nature_access._reclassify_urban_nature_area(
                args['lulc_raster_path'], args['lulc_attribute_table'],
                urban_nature_area_path,
                only_these_urban_nature_codes=limit_to_lucodes)

            # The source lulc is randomized, so need to programmatically build
            # up the expected array.
            source_lulc_array = pygeoprocessing.raster_to_numpy_array(
                args['lulc_raster_path'])
            pixel_area = abs(_DEFAULT_PIXEL_SIZE[0] * _DEFAULT_PIXEL_SIZE[1])
            expected_array = numpy.zeros(source_lulc_array.shape,
                                         dtype=numpy.float32)
            for i in range(1, 10, 2):
                if limit_to_lucodes is not None:
                    if i not in limit_to_lucodes:
                        continue
                factor = float(f"0.{i}")
                expected_array[source_lulc_array == i] = factor * pixel_area

            reclassified_array = pygeoprocessing.raster_to_numpy_array(
                urban_nature_area_path)
            numpy.testing.assert_array_almost_equal(
                reclassified_array, expected_array)

    def test_validate(self):
        """UNA: Basic test for validation."""
        from natcap.invest import urban_nature_access
        args = _build_model_args(self.workspace_dir)
        args['search_radius_mode'] = (
            urban_nature_access.RADIUS_OPT_URBAN_NATURE)
        self.assertEqual(urban_nature_access.validate(args), [])

    def test_validate_uniform_search_radius(self):
        """UNA: Search radius is required when using uniform search radii."""
        from natcap.invest import urban_nature_access
        from natcap.invest import validation

        args = _build_model_args(self.workspace_dir)
        args['search_radius_mode'] = urban_nature_access.RADIUS_OPT_UNIFORM
        args['search_radius'] = ''

        warnings = urban_nature_access.validate(args)
        self.assertEqual(warnings, [(['search_radius'],
                                     validation.MESSAGES['MISSING_VALUE'])])
