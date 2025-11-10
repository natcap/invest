# coding=UTF-8
"""Tests for the Urban Mental Health Model."""
import os
import shutil
import tempfile
import unittest

import numpy
import pandas
import pygeoprocessing
from shapely import Polygon
from osgeo import gdal, ogr, osr

# FLOAT32_NODATA is used as a "custom" nodata vs PGP_FLOAT32_NODATA is
# pygeoprocessing's default nodata for a float32 raster and is used in the
# model beginning at the step where delta ndvi is calculated
FLOAT32_NODATA = float(numpy.finfo(numpy.float32).min)
PGP_FLOAT32_NODATA = pygeoprocessing.choose_nodata(numpy.float32)


def make_simple_vector(path_to_shp, fields={"id": ogr.OFTReal},
                       attribute_list=[{"id": 0}], epsg=26910,
                       shapely_geometry_list = [ # this polygon fits within middle pixel of raster
                           Polygon([(461351, 4923191), (461451, 4923191),
                                    (461451, 4923245), (461351, 4923245),
                                    (461351, 4923191)])]):
    
    #Polygon([(461251, 4923195), (461501, 4923195),
            # (461501, 4923445), (461251, 4923445),
            # (461251, 4923195)])]): --> this is too large! good for last test rn
    """
    Generate shapefile with one rectangular polygon

    Args:
        path_to_shp (str): path to target shapefile
        fields (dict): dictionary with attribute: datatype key/value pairs
        attribute_list (list): list of attribute: value dictionaries 
        epsg (int): EPSG code
        shapely_geometry_list (list): list of shapely geometry elements. For
            polygons, order of elements in list is:
            (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)
            Note: list length must equal length of values in attribute_list dicts.

    Returns:
        None.
    """
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    projection_wkt = srs.ExportToWkt()

    pygeoprocessing.shapely_geometry_to_vector(shapely_geometry_list,
                                               path_to_shp, projection_wkt,
                                               "ESRI Shapefile", fields,
                                               attribute_list)


def make_raster_from_array(base_raster_path, array):
    """Create a raster on designated path with array values.
    Args:
        base_raster_path (str): the raster path for making the new raster.
    Returns:
        None.
    """
    # UTM Zone 10N
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)
    projection_wkt = srs.ExportToWkt()

    origin = (461251, 4923445)
    pixel_size = (100, -100)
    no_data = FLOAT32_NODATA

    pygeoprocessing.numpy_array_to_raster(
        array.astype(numpy.float32), no_data, pixel_size, origin,
        projection_wkt, base_raster_path)


def make_synthetic_data_and_params(workspace_dir):

    # make synthetic input data
    baseline_prevalence_path = os.path.join(
        workspace_dir, "baseline_prevalence.shp")
    fields = {"id": ogr.OFTReal, "risk_rate": ogr.OFTReal}
    attribute_list = [{"id": 0, "risk_rate": 10}]
    make_simple_vector(baseline_prevalence_path, fields, attribute_list,
                       shapely_geometry_list=[
                           Polygon([(461251, 4923195), (461501, 4923195),
                                    (461501, 4923445), (461251, 4923445),
                                    (461251, 4923195)])])

    ndvi_base_array = numpy.array(
        [[.1, .2, .35], [.5, .6, .7],
         [.8, .9, .10], [.11, .12, FLOAT32_NODATA]])
    ndvi_base_path = os.path.join(workspace_dir, "ndvi_base.tif")
    make_raster_from_array(ndvi_base_path, ndvi_base_array)

    ndvi_alt_array = numpy.array(
        [[.12, .22, .1], [.2, .3, .8], [.9, .14, .14], [.16, .17, .3]])
    ndvi_alt_path = os.path.join(workspace_dir, "ndvi_alt.tif")
    make_raster_from_array(ndvi_alt_path, ndvi_alt_array)

    pop_array = ndvi_alt_array*100
    pop_path = os.path.join(workspace_dir, "population.tif")
    make_raster_from_array(pop_path, pop_array)

    aoi_path = os.path.join(workspace_dir, "aoi.shp")
    make_simple_vector(aoi_path)

    args = {
        'aoi_path': aoi_path,
        'baseline_prevalence_vector': baseline_prevalence_path,
        'effect_size': 0.94,
        'health_cost_rate': None,
        'lulc_alt': '',
        'lulc_attr_csv': '',
        'lulc_base': '',
        'ndvi_alt': ndvi_alt_path,
        'ndvi_base': ndvi_base_path,
        'population_raster': pop_path,
        'results_suffix': 'test1',
        'scenario': 'ndvi',
        'search_radius': 100, # 1 pixel
        'tc_raster': '',
        'tc_target': '',
        'workspace_dir': workspace_dir,
    }

    return args


gdal.UseExceptions()


class UMHTests(unittest.TestCase):
    """Tests for the Urban Mental Health Model."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the test result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_ndvi_preprocessing_and_delta_calc(self):
        """Test NDVI preprocessing and calculation with correct nodata

        Test that preprocessing of NDVI rasters works as expected and delta
        NDVI calculation is correct. Also test nodata propagates correctly,
        (i.e., output has nodata pixels anywhere either input is nodata).
        """
        from natcap.invest import urban_mental_health
        args = make_synthetic_data_and_params(self.workspace_dir)
        urban_mental_health.execute(args)

        intermediate = os.path.join(self.workspace_dir, "intermediate")
        suffix = args['results_suffix']
        # Assert that top row was removed as its outside of search_radius distance of AOI
        expected_base_aligned = numpy.array( #this is copied from base_ndvi without top row
            [[.5, .6, .7], 
            [.8, .9, .10], 
            [.11, .12, FLOAT32_NODATA]])
        actual_base_aligned = pygeoprocessing.raster_to_numpy_array(
            os.path.join(intermediate, f'ndvi_base_aligned_{suffix}.tif'))
        numpy.testing.assert_allclose(actual_base_aligned,
                                      expected_base_aligned, atol=1e-6)

        # Assert that convolution was correct (this expected result was
        # calculated by hand)
        expected_base_convolve = numpy.array(
            [[0.633333, 0.675, 0.4666667],
             [0.5775, 0.504, 0.5666667],
             [0.3433333, 0.3766667, FLOAT32_NODATA]])
        actual_base_convolve = pygeoprocessing.raster_to_numpy_array(
            os.path.join(intermediate, f'ndvi_base_buffer_mean_{suffix}.tif'))
        numpy.testing.assert_allclose(actual_base_convolve,
                                      expected_base_convolve, atol=1e-6)

        expected_alt_convolve = numpy.array(
            [[0.466667, 0.36, 0.413333],
             [0.35, 0.33, 0.345],
             [0.41, 0.1925, 0.203333]])

        output_delta_ndvi = os.path.join(
            intermediate, f'delta_ndvi_{suffix}.tif')

        actual_delta_ndvi = pygeoprocessing.raster_to_numpy_array(
            output_delta_ndvi)

        expected_delta_ndvi = expected_alt_convolve - expected_base_convolve
        mask = (expected_base_convolve == PGP_FLOAT32_NODATA) | (
            expected_alt_convolve == PGP_FLOAT32_NODATA)
        expected_delta_ndvi[mask] = PGP_FLOAT32_NODATA

        numpy.testing.assert_allclose(actual_delta_ndvi, expected_delta_ndvi,
                                      atol=1e-6)

    def test_option3(self):
        "Test umh option 3 (ndvi)"
        from natcap.invest import urban_mental_health

        args = make_synthetic_data_and_params(self.workspace_dir)

        urban_mental_health.execute(args)

        expected_baseline_cases = numpy.array(
            [[200, 300, 800], [900, 140, 140],
             [PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA]])
        actual_baseline_cases_path = os.path.join(
            self.workspace_dir, "intermediate",
            f"baseline_cases_{args['results_suffix']}.tif")
        actual_baseline_cases = pygeoprocessing.raster_to_numpy_array(
            actual_baseline_cases_path)
        numpy.testing.assert_allclose(actual_baseline_cases,
                                      expected_baseline_cases, atol=1e-6)

        expected_preventable_cases = numpy.array([
            [-21.72614, -64.55958, -26.8406096],
            [-136.040285, -15.914164, -20.58118],
            [PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA]])
        # ^ calculated using:
        # (1 - numpy.exp(
        #       numpy.log(0.94)*10*actual_delta_ndvi))*actual_baseline_cases
        # i.e., (1 - (exp(ln(RR0.1NE)10*NE))) * bc

        # results contains only center pixel left bc AOI is small
        expected_preventable_cases = numpy.array([
            [PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA],
            [PGP_FLOAT32_NODATA, -15.914164, PGP_FLOAT32_NODATA],
            [PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA]])
        actual_preventable_cases_path = os.path.join(
            self.workspace_dir,
            f"output/preventable_cases_{args['results_suffix']}.tif")
        actual_preventable_cases = pygeoprocessing.raster_to_numpy_array(
            actual_preventable_cases_path)

        numpy.testing.assert_allclose(actual_preventable_cases,
                                      expected_preventable_cases, atol=1e-5)

    def test_diff_prj_inputs(self):
        """Test model option 3 given inputs of different projections.

        Check that output preventable cases geotiff is clipped
        correctly given inputs of different projections
        """
        from natcap.invest import urban_mental_health

        args = make_synthetic_data_and_params(self.workspace_dir)

        # create AOI w different projection
        epsg = 5070
        xmin = -2151375
        ymax = 2699058.8
        xmax = xmin + 70
        ymin = ymax - 70
        geom = [Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax),
                         (xmin, ymax), (xmin, ymin)])]
        new_aoi_path = os.path.join(self.workspace_dir, "AOI_5070.shp")
        make_simple_vector(new_aoi_path, epsg=epsg, shapely_geometry_list=geom)
        args["aoi_path"] = new_aoi_path

        urban_mental_health.execute(args)

        actual_prev_cases = pygeoprocessing.raster_to_numpy_array(
            os.path.join(self.workspace_dir, "output",
                         "preventable_cases_test1.tif"))
        # most are nodata because AOI is < 1 pixel
        expected_prev_cases = numpy.array(
            [[PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA],
             [PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA],
             [PGP_FLOAT32_NODATA, -15.914164, PGP_FLOAT32_NODATA],
             [PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA]])
        numpy.testing.assert_allclose(actual_prev_cases, expected_prev_cases)

    def test_AOI_too_large(self):
        """Test that AOI larger than NDVI raises warning on option 3

        Test that if AOI is larger than the input NDVI, the model raises
        a warning but still runs correctly
        """
        from natcap.invest import urban_mental_health

        #assert UserWarning
        # make synthetic input data
        ndvi_base_array = numpy.array([[.1, .2, .35], [.5, .6, .7], [.8, .9, .10], [.11, .12, FLOAT32_NODATA]])
        ndvi_base_path = os.path.join(self.workspace_dir, "ndvi_base.tif")
        make_raster_from_array(ndvi_base_path, ndvi_base_array)

        aoi_path = os.path.join(self.workspace_dir, "aoi.shp")
        xmin = 461351 - 100
        make_simple_vector(aoi_path, shapely_geometry_list=[
                           Polygon([(xmin, 4923191), (461451, 4923191),
                                    (461451, 4923245), (xmin, 4923245),
                                    (xmin, 4923191)])])

        args = {
            'aoi_path': aoi_path,
            'baseline_prevalence_vector': '',
            'effect_size': '',
            'health_cost_rate': None,
            'lulc_alt': '',
            'lulc_attr_csv': '',
            'lulc_base': '',
            'ndvi_alt': ndvi_base_path,
            'ndvi_base': ndvi_base_path,
            'population_raster': '',
            'results_suffix': 'test1',
            'scenario': 'ndvi',
            'search_radius': 100,
            'tc_raster': '',
            'tc_target': '',
            'workspace_dir': self.workspace_dir,
        }

        with self.assertRaises(UserWarning) as context:
            urban_mental_health.execute(args)
        self.assertTrue(
            "The extent of bounding box of the AOI buffered by the search" in
            str(context.exception))

    def test_search_radius_smaller_than_resolution(self):
        """Test that search_radius < pixel size/2 of NDVI raises error on option 3"""
        from natcap.invest import urban_mental_health

        args = make_synthetic_data_and_params(self.workspace_dir)
        args["search_radius"] = 2
        with self.assertRaises(ValueError) as context:
            urban_mental_health.execute(args)
        self.assertTrue(
            "Search radius 2.0 yielded pixel_radius of zero. " in
            str(context.exception))

    def test_AOI_larger_than_population_raster(self):
        """Test analysis area = pop raster bbox if latter has smallest extent

        Test that a warning is raised as well"""
        from natcap.invest import urban_mental_health

        args = make_synthetic_data_and_params(self.workspace_dir)

        array = numpy.array(([40, 500], [90, 90]))
        pop_path = os.path.join(self.workspace_dir, "population.tif")
        make_raster_from_array(pop_path, array)

        with self.assertRaises(ValueError) as context:
            urban_mental_health.execute(args)
        self.assertTrue(
            "The bounding boxes of the preprocessed population raster" in
            str(context.exception))

    def test_AOI_larger_than_lulc_base_option3(self):
        """Test analysis area = lulc bbox if latter has smallest extent"""
        from natcap.invest import urban_mental_health

        args = make_synthetic_data_and_params(self.workspace_dir)

        array = numpy.array(([1, 2], [4, 2]))
        lulc_base = os.path.join(self.workspace_dir, "lulc_base.tif")
        make_raster_from_array(lulc_base, array)
        args["lulc_base"] = lulc_base

        with self.assertRaises(UserWarning) as context:
            urban_mental_health.execute(args)
        self.assertTrue(
            "The extent of bounding box of the AOI buffered by the search "
            "radius exceeds that of the lulc_base.tif" in
            str(context.exception))

    def test_masking_without_lulc(self):
        """Test NDVI threshold masking (given no lulc input for mask)"""
        from natcap.invest import urban_mental_health

        array = numpy.array(([.1, -.3, -6], [.1, .2, .9],
                             [.1, -.2, FLOAT32_NODATA], [.2, .5, .9]))
        ndvi_base = os.path.join(self.workspace_dir, "ndvi_base.tif")
        make_raster_from_array(ndvi_base, array)

        target_masked_ndvi = os.path.join(self.workspace_dir, "tgt_ndvi.tif")

        urban_mental_health.mask_ndvi(ndvi_base, target_masked_ndvi)

        # FLOAT32_NODATA is explicitly set in `make_raster_from_array`
        # and UMH uses native nodata until delta ndvi calculation
        expected_ndvi_masked = numpy.array(
            ([.1, FLOAT32_NODATA, FLOAT32_NODATA], [.1, .2, .9],
             [.1, FLOAT32_NODATA, FLOAT32_NODATA], [.2, .5, .9]))
        actual_ndvi_masked = pygeoprocessing.raster_to_numpy_array(
            target_masked_ndvi)

        numpy.testing.assert_allclose(actual_ndvi_masked, expected_ndvi_masked)

    def test_lulc_masking(self):
        """Test that lulc masks correctly"""
        from natcap.invest import urban_mental_health

        array = numpy.array(([.1, .3, .1], [.1, .2, .9],
                             [.1, .2, .4], [.2, .5, .9]))
        ndvi_base = os.path.join(self.workspace_dir, "ndvi_base.tif")
        make_raster_from_array(ndvi_base, array)

        array = numpy.array(([1, 2, 3], [1, 2, 3], [1, 2, 4], [2, 2, 4]))
        lulc_base = os.path.join(self.workspace_dir, "lulc_base.tif")
        make_raster_from_array(lulc_base, array)

        lulc_attr_table = pandas.DataFrame({"lucode": [1, 2, 3, 4],
                                            "exclude": [0, 0, 1, 1]})
        lulc_attr_path = os.path.join(self.workspace_dir, "lulc_attr_table.csv")
        lulc_attr_table.to_csv(lulc_attr_path)

        target_masked_ndvi = os.path.join(self.workspace_dir, "tgt_ndvi.tif")
        target_lulc_mask = os.path.join(self.workspace_dir, "tgt_mask.tif")

        urban_mental_health.mask_ndvi(ndvi_base, target_masked_ndvi, lulc_base,
                                      lulc_attr_path, target_lulc_mask)

        # FLOAT32_NODATA is explicitly set in `make_raster_from_array`
        # and UMH uses native nodata until delta ndvi calculation
        expected_ndvi_masked = numpy.array(
            ([.1, .3, FLOAT32_NODATA], [.1, .2, FLOAT32_NODATA],
             [.1, .2, FLOAT32_NODATA], [.2, .5, FLOAT32_NODATA]))
        actual_ndvi_masked = pygeoprocessing.raster_to_numpy_array(
            target_masked_ndvi)

        numpy.testing.assert_allclose(actual_ndvi_masked, expected_ndvi_masked)

    def test_calc_baseline_cases(self):
        """Test `calc_baseline_cases` equals prevalence * population

        baseline_cases = rasterized prevalence * population per pixel
        for valid pixels. Ensure that nodata propagates correctly
        (output should be nodata where pop or prevalance are nodata)
        """
        from natcap.invest import urban_mental_health

        pop_array = numpy.array(([FLOAT32_NODATA, 30, 10], [10, 20, 90],
                                 [100, 20, 40], [20, 5, .9]))
        population_raster = os.path.join(self.workspace_dir, "pop.tif")
        make_raster_from_array(population_raster, pop_array)

        baseline_prevalence_path = os.path.join(self.workspace_dir,
                                                "baseline_prevalence.shp")
        fields = {"id": ogr.OFTReal, "risk_rate": ogr.OFTReal}
        attribute_list = [{"id": 0, "risk_rate": 11}, {"id": 1, "risk_rate": 5}]
        # 2 non-overlapping polygons, first covers upper 3x3 pixels, second
        # polygon covers bottom left 1x2 pixels
        make_simple_vector(baseline_prevalence_path, fields, attribute_list,
                           shapely_geometry_list=[
                               Polygon([(461251, 4923155), (461521, 4923155),
                                        (461521, 4923445), (461251, 4923445),
                                        (461251, 4923155)]),
                               Polygon([(461251, 4922995), (461451, 4922995),
                                        (461451, 4923125), (461251, 4923125),
                                        (461251, 4922995)])
                                ])

        target_base_prevalence_raster = os.path.join(
            self.workspace_dir, "target_prevalence.tif")

        target_base_cases = os.path.join(
            self.workspace_dir, "target_baselinecases.tif")

        urban_mental_health.calc_baseline_cases(
            population_raster, baseline_prevalence_path,
            target_base_prevalence_raster, target_base_cases)

        # prevalence * pop, where prevalence is taken from polygons above
        expected_cases = numpy.array((
            [PGP_FLOAT32_NODATA, 30*11, 10*11],
            [10*11, 20*11, 90*11],
            [100*11, 20*11, 40*11],
            [20*5, 5*5, PGP_FLOAT32_NODATA]))
        actual_cases = pygeoprocessing.raster_to_numpy_array(
            target_base_cases)

        numpy.testing.assert_allclose(actual_cases, expected_cases)

    def test_calc_preventable_cases(self):
        """Test `calc_preventable_cases` correct and handles zeros and nodata

        Test that pixels outside of AOI are nodata and nodata pixels in either
        baseline_cases or delta_ndvi raster become nodata in the output.
        Test that anywhere delta_ndvi or baseline_cases are 0, output
        preventable cases is 0.
        Test that preventable_cases decreases with negative delta_ndvi and
        increases with positive delta_ndvi (given effect size < 1).
        """
        from natcap.invest import urban_mental_health

        ndvi_array = numpy.array(([FLOAT32_NODATA, .3, .1], [.2, 0, 0],
                                  [-.1, .3, .1], [-0.2, .5, 1]))
        delta_ndvi = os.path.join(self.workspace_dir, "ndvi_base.tif")
        make_raster_from_array(delta_ndvi, ndvi_array)

        bc_array = numpy.array(([5, FLOAT32_NODATA, 0], [1, 100, 10],
                                [50, 0, 10], [1, 1, 1]))
        baseline_cases = os.path.join(self.workspace_dir, "baseline_cases.tif")
        make_raster_from_array(baseline_cases, bc_array)

        aoi_path = os.path.join(self.workspace_dir, "aoi.shp")
        xmin = 461251  # origin of raster
        ymax = 4923445  # origin of raster
        xmax = xmin + 300
        ymin = ymax - 300  # cut off lowest row
        make_simple_vector(aoi_path,
                           shapely_geometry_list=[
                                Polygon([(xmin, ymin), (xmax, ymin),
                                         (xmax, ymax), (xmin, ymax),
                                         (xmin, ymin)])])

        effect_size = .9
        target_preventable_cases = os.path.join(self.workspace_dir,
                                                "prev_cases.tif")
        urban_mental_health.calc_preventable_cases(
            delta_ndvi, baseline_cases, effect_size,
            target_preventable_cases, aoi_path, self.workspace_dir)

        actual_prev_cases = pygeoprocessing.raster_to_numpy_array(
            target_preventable_cases)

        # expected_prev_cases calculated by hand using:
        # (1 - numpy.exp(numpy.log(effect_size) * 10 * delta_ndvi)) * baseline_cases
        expected_prev_cases = numpy.array(
            [[FLOAT32_NODATA, FLOAT32_NODATA, 0],
             [0.19, 0, 0],
             [-5.5555555555, 0, 1],
             [FLOAT32_NODATA, FLOAT32_NODATA, FLOAT32_NODATA]])

        numpy.testing.assert_allclose(actual_prev_cases, expected_prev_cases)

    def test_calc_preventable_cost(self):
        """ Test `calc_preventable_cost`

        prev cost = equals preventable_cases * health_cost_rate for valid
        pixels; nodata preserved."""
        from natcap.invest import urban_mental_health

        preventable_cases = os.path.join(self.workspace_dir, "prev_cases.tif")
        pc_array = numpy.array(([FLOAT32_NODATA, 40, 0], [1, 100, 10],
                                [50, 0, 10], [1, 1, 1]))
        make_raster_from_array(preventable_cases, pc_array)
        health_cost_rate = 90
        tgt_prev_cost = os.path.join(self.workspace_dir, "prev_cost.tif")

        urban_mental_health.calc_preventable_cost(
            preventable_cases, health_cost_rate, tgt_prev_cost)

        actual_cost = pygeoprocessing.raster_to_numpy_array(tgt_prev_cost)
        expected_cost = pc_array*health_cost_rate
        expected_cost[0, 0] = FLOAT32_NODATA

        numpy.testing.assert_allclose(actual_cost, expected_cost)

    def test_zonal_stats_preventable_cases_cost(self):
        """Test `zonal_stats_preventable_cases_cost`

          Test that function writes CSV and vector with `sum_cases` and
          `sum_cost` fields calculated correctly per polygon and calculates
          the sum total cases and costs aggregated across all polygons in AOI

          """
        from natcap.invest import urban_mental_health

        aoi_vector = os.path.join(self.workspace_dir, "aoi.shp")
        fields = {"id": ogr.OFTReal}
        attribute_list = [{"id": 0}, {"id": 1}]
        # 2 non-overlapping polygons, first covers upper 3x3 pixels, second
        # polygon covers bottom left 1x2 pixels
        make_simple_vector(aoi_vector, fields, attribute_list,
                           shapely_geometry_list=[
                               Polygon([(461251, 4923155), (461521, 4923155),
                                        (461521, 4923445), (461251, 4923445),
                                        (461251, 4923155)]),
                               Polygon([(461251, 4922995), (461451, 4922995),
                                        (461451, 4923125), (461251, 4923125),
                                        (461251, 4922995)])
                                ])

        preventable_cases = os.path.join(self.workspace_dir, "prev_cases.tif")
        pc_array = numpy.array(([FLOAT32_NODATA, 40, 0], [1, 100, 10],
                                [50, 0, 10], [1, 1, 1]))
        make_raster_from_array(preventable_cases, pc_array)

        preventable_cost = os.path.join(self.workspace_dir, "prev_cost.tif")
        cost_array = numpy.array(([FLOAT32_NODATA, 40.5, 0], [121, 100, 10],
                                  [540, 600, 150], [15, 2, 155]))
        make_raster_from_array(preventable_cost, cost_array)

        target_stats_csv = os.path.join(self.workspace_dir, "stats.csv")
        target_agg_vector_path = os.path.join(self.workspace_dir, "stats.gpkg")

        urban_mental_health.zonal_stats_preventable_cases_cost(
            aoi_vector, target_stats_csv, target_agg_vector_path,
            preventable_cases, preventable_cost)

        # Check CSV
        df = pandas.read_csv(target_stats_csv)
        actual_pc = df["sum_cases"].to_list()[0:2]
        actual_cost = df["sum_cost"][0:2]
        # target sum pc calculated via [sum(pc_array[:3, :3], sum(pc_array[3, :2]))]
        target_sum_pc = [211, 2]
        numpy.testing.assert_allclose(actual_pc, target_sum_pc)
        # target cost sum calculated using sum(cost_array[3, :2]) bc last
        # pixel cropped out by AOI
        target_sum_cost = [1561.5, 17]
        numpy.testing.assert_allclose(actual_cost, target_sum_cost)

        # check calculation of total cases in entire AOI
        actual_total_cases = df["total_cases"][2]
        actual_total_cost = df["total_cost"][2]
        numpy.testing.assert_allclose(actual_total_cases, sum(target_sum_pc))
        numpy.testing.assert_allclose(actual_total_cost, sum(target_sum_cost))

        # Check vector
        expected_attributes = [
            {"id": 0, "sum_cases": target_sum_pc[0],
             "sum_cost": target_sum_cost[0]},
            {"id": 1, "sum_cases": target_sum_pc[1],
             "sum_cost": target_sum_cost[1]}]

        # Get actual attributes
        data_source = ogr.Open(target_agg_vector_path, 0)

        layer = data_source.GetLayer(0)

        # Create a list to hold the data read from the file
        actual_attributes = []
        for feature in layer:
            actual_attributes.append({
                'id': feature.GetField('id'),
                'sum_cases': feature.GetField('sum_cases'),
                'sum_cost': feature.GetField('sum_cost'),
            })

        self.assertEqual(actual_attributes, expected_attributes)

    def test_execute_without_health_cost_skips_cost_outputs(self):
        """Test model option 3 without health input

        Test that `execute` runs without health cost input and produces CSV
        without cost column.
        """
        from natcap.invest import urban_mental_health

        args = make_synthetic_data_and_params(self.workspace_dir)
        urban_mental_health.execute(args)
        # Check CSV
        stats_csv = os.path.join(self.workspace_dir, "output",
                                 "preventable_cases_cost_sum_test1.csv")
        df = pandas.read_csv(stats_csv)
        self.assertIn("sum_cases", df.columns)
        self.assertNotIn("sum_cost", df.columns)
