"""Tests for the Urban Mental Health Model."""
import os
import shutil
import tempfile
import unittest

import numpy
import pandas
from pygam import LinearGAM, s
import pygeoprocessing
from shapely import Polygon
from osgeo import gdal, ogr, osr

gdal.UseExceptions()

# FLOAT32_NODATA is used as a "custom" nodata vs PGP_FLOAT32_NODATA is
# pygeoprocessing's default nodata for a float32 raster and is used in the
# model beginning at the step where delta ndvi is calculated
FLOAT32_NODATA = float(numpy.finfo(numpy.float32).min)
PGP_FLOAT32_NODATA = pygeoprocessing.choose_nodata(numpy.float32)
ORIGIN_X = 461251
ORIGIN_Y = 4923445


def make_simple_vector(path_to_shp, fields={"id": ogr.OFTReal},
                       attribute_list=[{"id": 0}], epsg=26910,
                       # this polygon fits within middle pixel of raster
                       shapely_geometry_list=[
                           Polygon([(ORIGIN_X+100, ORIGIN_Y-254),
                                    (ORIGIN_X+200, ORIGIN_Y-254),
                                    (ORIGIN_X+200, ORIGIN_Y-200),
                                    (ORIGIN_X+100, ORIGIN_Y-200),
                                    (ORIGIN_X+100, ORIGIN_Y-254)])]):
    """
    Generate shapefile with one rectangular polygon

    This shapefile covers just over 1/2 of a pixel in the default raster
    created via ``make_raster_from_array``. This is to allow for the smallest
    possible raster to be used in testing, so that when the AOI is buffered
    by ``search_radius``, the area of analysis becomes just 3x3 pixels.

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

    pixel_size = (100, -100)
    no_data = FLOAT32_NODATA

    pygeoprocessing.numpy_array_to_raster(
        array.astype(numpy.float32), no_data, pixel_size,
        (ORIGIN_X, ORIGIN_Y), projection_wkt, base_raster_path)


def make_synthetic_data_and_params(workspace_dir, model_option):
    """Make all data needed to run UMH model
    
    Args:
        workspace_dir (str): path to workspace directory
        model_option (int): One of: 1 (Tree cover and NDVI inputs), 2 (LULC inputs),
            or 3 (NDVI inputs) """

    # make synthetic input data
    baseline_prevalence_path = os.path.join(
        workspace_dir, "baseline_prevalence.shp")
    fields = {"id": ogr.OFTReal, "risk_rate": ogr.OFTReal}
    attribute_list = [{"id": 0, "risk_rate": 10}]
    make_simple_vector(baseline_prevalence_path, fields, attribute_list,
                       shapely_geometry_list=[
                           Polygon([(ORIGIN_X, ORIGIN_Y-250),
                                    (ORIGIN_X+250, ORIGIN_Y-250),
                                    (ORIGIN_X+250, ORIGIN_Y),
                                    (ORIGIN_X, ORIGIN_Y),
                                    (ORIGIN_X, ORIGIN_Y-250)])])

    pop_array = numpy.array(
            [[12, 22, 10], [20, 30, 80], [90, 14, 14], [16, 17, 30]])
    pop_path = os.path.join(workspace_dir, "population.tif")
    make_raster_from_array(pop_path, pop_array)

    aoi_path = os.path.join(workspace_dir, "aoi.shp")
    make_simple_vector(aoi_path)

    ndvi_base_array = numpy.array(
        [[.1, .2, .35], [.5, .6, .7],
         [.8, .9, .10], [.11, .12, FLOAT32_NODATA]])
    ndvi_base_path = os.path.join(workspace_dir, "ndvi_base.tif")
    make_raster_from_array(ndvi_base_path, ndvi_base_array)

    args = {
        'aoi_path': aoi_path,
        'baseline_prevalence_vector': baseline_prevalence_path,
        'effect_size': 0.94,
        'health_cost_rate': None,
        'ndvi_base': ndvi_base_path,
        'population_raster': pop_path,
        'results_suffix': 'test1',
        'search_radius': 100, # 1 pixel
        'workspace_dir': workspace_dir,
    }

    if model_option == 1:
        args['scenario'] = 'tcc_ndvi'
        args['tc_raster'] = '' #TODO
        args['tc_target'] = '' #TODO

    if model_option == 1:
        args['scenario'] = 'tcc_ndvi'
        tcc_array = numpy.array(
            [[.12, .22, .1], [.2, .3, .8], [.9, .14, .14], [.16, .17, .3]])
        tcc_path = os.path.join(workspace_dir, "tcc.tif")
        make_raster_from_array(tcc_path, tcc_array)
        args['tree_cover_raster'] = tcc_path
        args['tree_cover_target'] = 50

    elif model_option == 2:
        args['scenario'] = 'lulc'
        # make lulc arrays
        lulc_base_array = numpy.array(
            [[FLOAT32_NODATA, 2, 3], [1, 2, 2], [2, 1, 2], [1, 3, 1]])
        lulc_base_path = os.path.join(workspace_dir, "lulc_base.tif")
        make_raster_from_array(lulc_base_path, lulc_base_array)
        args['lulc_base'] = lulc_base_path

        lulc_alt_array = numpy.array(
            [[2, 2, 2], [1, 2, 1], [1, 2, 4], [1, 2, 3]])
        lulc_alt_path = os.path.join(workspace_dir, "lulc_alt.tif")
        make_raster_from_array(lulc_alt_path, lulc_alt_array)
        args['lulc_alt'] = lulc_alt_path

        # make attribute table
        lulc_attr_table = pandas.DataFrame({"lucode": [1, 2, 3, 4],
                                            "ndvi": [.1, .2, .35, .8],
                                            "exclude": [0, 0, 1, 1]})
        lulc_attr_path = os.path.join(workspace_dir, "lulc_attr_table.csv")
        lulc_attr_table.to_csv(lulc_attr_path)
        args['lulc_attr_csv'] = lulc_attr_path

    elif model_option == 3:
        args['scenario'] = 'ndvi'
        ndvi_alt_array = numpy.array(
            [[.12, .22, .1], [.2, .3, .8], [.9, .14, .14], [.16, .17, .3]])
        ndvi_alt_path = os.path.join(workspace_dir, "ndvi_alt.tif")
        make_raster_from_array(ndvi_alt_path, ndvi_alt_array)
        args['ndvi_alt'] = ndvi_alt_path

    else:
        raise ValueError("model_option must be one of: 1, 2, or 3")

    return args


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
        from natcap.invest.urban_mental_health import urban_mental_health
        args = make_synthetic_data_and_params(self.workspace_dir, 3)
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
        from natcap.invest.urban_mental_health import urban_mental_health

        args = make_synthetic_data_and_params(self.workspace_dir, 3)

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
        expected_preventable_cases = numpy.full((3, 3), PGP_FLOAT32_NODATA)
        expected_preventable_cases[1, 1] = -15.914164

        actual_preventable_cases_path = os.path.join(
            self.workspace_dir, "output",
            f"preventable_cases_{args['results_suffix']}.tif")
        actual_preventable_cases = pygeoprocessing.raster_to_numpy_array(
            actual_preventable_cases_path)

        numpy.testing.assert_allclose(actual_preventable_cases,
                                      expected_preventable_cases, atol=1e-5)

    def test_diff_prj_inputs(self):
        """Test model option 3 given inputs of different projections.

        Check that output preventable cases geotiff is clipped
        correctly given inputs of different projections
        """
        from natcap.invest.urban_mental_health import urban_mental_health

        args = make_synthetic_data_and_params(self.workspace_dir, 3)

        # create NDVI base with different projection
        ndvi_base_array = numpy.array(
            [[.1, .2, .35], [.5, .6, .7],
             [.8, .9, .10], [.11, .12, FLOAT32_NODATA]])

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(5070)
        pygeoprocessing.numpy_array_to_raster(
            ndvi_base_array.astype(numpy.float32), FLOAT32_NODATA,
            pixel_size=(100, -100), origin=(-2151490, 2699260),
            projection_wkt=srs.ExportToWkt(),
            target_path=args['ndvi_base'])

        urban_mental_health.execute(args)

        actual_prev_cases = pygeoprocessing.raster_to_numpy_array(
            os.path.join(self.workspace_dir, "output",
                         "preventable_cases_test1.tif"))
        # most are nodata because AOI is just under 1 pixel
        # shape is 3x4 (rather than 3x3) because when ndvi_base is resampled,
        # it causes extra nodata column to right of AOI
        expected_prev_cases = numpy.full((3, 4), PGP_FLOAT32_NODATA)
        expected_prev_cases[1, 1] = -47.76584
        numpy.testing.assert_allclose(actual_prev_cases, expected_prev_cases)

    def test_NDVI_extent_too_small(self):
        """Test that NDVI smaller than buffered AOI extent raises warning

        Test that if AOI is larger than the input NDVI extent by search_radius
        distance, the model raises a warning.
        """
        from natcap.invest.urban_mental_health import urban_mental_health

        # make synthetic input data
        args = make_synthetic_data_and_params(self.workspace_dir, 3)

        # overwrite AOI referenced in args with larger AOI
        xmin = ORIGIN_X
        xmax = ORIGIN_X + 200
        ymin = ORIGIN_Y - 254
        ymax = ORIGIN_Y - 200
        make_simple_vector(args['aoi_path'], shapely_geometry_list=[
                           Polygon([(xmin, ymin), (xmax, ymin),
                                    (xmax, ymax), (xmin, ymax),
                                    (xmin, ymin)])])

        with self.assertLogs(urban_mental_health.LOGGER,
                             level="WARNING") as context:
            urban_mental_health.execute(args)

        # Check that a WARNING-level log contains warning_text
        warning_text = "The extent of bounding box of the AOI buffered by " \
            "the search radius exceeds that of the ndvi_base.tif"
        self.assertTrue(
            any(warning_text in message for message in context.output),
            f"Expected warning text not found in logs: {context.output}"
        )

    def test_search_radius_smaller_than_resolution(self):
        """Test that search_radius < pixel size/2 of NDVI raises error on option 3"""
        from natcap.invest.urban_mental_health import urban_mental_health

        args = make_synthetic_data_and_params(self.workspace_dir, 3)
        args["search_radius"] = 2
        with self.assertRaises(ValueError) as context:
            urban_mental_health.execute(args)
        self.assertTrue(
            "Search radius 2.0 yielded pixel_radius of zero. " in
            str(context.exception))

    def test_population_raster_too_small(self):
        """Test if pop raster is smaller than AOI, model runs

        Model will run, but output extent will have nodata where
        population raster is nodata. That is, valid extent of outputs match
        extent of population raster input.
        """
        from natcap.invest.urban_mental_health import urban_mental_health

        args = make_synthetic_data_and_params(self.workspace_dir, 3)

        # make AOI larger than default so pop raster can only cover part of it
        xmin = ORIGIN_X + 100  # same as default
        ymax = ORIGIN_Y - 100
        xmax = ORIGIN_X + 200  # same as default
        ymin = ORIGIN_Y - 254  # same as default
        make_simple_vector(os.path.join(self.workspace_dir, "aoi.shp"),
                           shapely_geometry_list=[
                                Polygon([(xmin, ymin), (xmax, ymin),
                                         (xmax, ymax), (xmin, ymax),
                                         (xmin, ymin)])])
        
        urban_mental_health.execute(args)

        # check output prev cases
        preventable_cases_path = os.path.join(
            self.workspace_dir, "output",
            f"preventable_cases_{args['results_suffix']}.tif")
        actual_prev_cases = pygeoprocessing.raster_to_numpy_array(
            preventable_cases_path)

        expected_prev_cases = numpy.full((4, 3), PGP_FLOAT32_NODATA)
        expected_prev_cases[1, 1] = -49.75519
        expected_prev_cases[2, 1] = -15.91417

        numpy.testing.assert_allclose(actual_prev_cases, expected_prev_cases,
                                      atol=1e-5)

        # Now compare results to if population raster _doesn't_ cover AOI fully
        # make small population raster that covers top pixel of AOI but not
        # lower pixel
        array = numpy.array(([40], [500]))
        # UTM Zone 10N
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(26910)
        projection_wkt = srs.ExportToWkt()
        origin = (ORIGIN_X+100, ORIGIN_Y)
        pygeoprocessing.numpy_array_to_raster(
            array.astype(numpy.float32), FLOAT32_NODATA, (100, -100), origin,
            projection_wkt, args['population_raster'])

        urban_mental_health.execute(args)

        # check output prev cases
        preventable_cases_path = os.path.join(
            self.workspace_dir, "output",
            f"preventable_cases_{args['results_suffix']}.tif")
        actual_prev_cases = pygeoprocessing.raster_to_numpy_array(
            preventable_cases_path)

        expected_prev_cases = numpy.full((4, 3), PGP_FLOAT32_NODATA)
        # data only exists where population raster covers AOI
        expected_prev_cases[1, 1] = -829.2532

        numpy.testing.assert_allclose(actual_prev_cases, expected_prev_cases)

    def test_AOI_larger_than_lulc_base_option3(self):
        """Test warning raised but model runs if LULC raster too small"""
        from natcap.invest.urban_mental_health import urban_mental_health

        args = make_synthetic_data_and_params(self.workspace_dir, 3)

        array = numpy.array(([1, 2], [4, 2]))
        lulc_base = os.path.join(self.workspace_dir, "lulc_base.tif")
        make_raster_from_array(lulc_base, array)
        args["lulc_base"] = lulc_base

        with self.assertLogs(urban_mental_health.LOGGER,
                             level="WARNING") as context:
            urban_mental_health.execute(args)

        # Check that a WARNING-level log contains warning_text
        warning_text = "The extent of bounding box of the AOI buffered by " \
            "the search radius exceeds that of the lulc_base.tif"
        self.assertTrue(
            any(warning_text in message for message in context.output),
            f"Expected warning text not found in logs: {context.output}"
        )

        # Check that model still ran and expected output was created
        output_path = os.path.join(
            self.workspace_dir, "output",
            f"preventable_cases_{args['results_suffix']}.tif"
        )
        self.assertTrue(os.path.isfile(output_path))

    def test_masking_without_lulc(self):
        """Test NDVI threshold masking (given no lulc input for mask)"""
        from natcap.invest.urban_mental_health import urban_mental_health

        array = numpy.array(([.1, -.3, -6], [.1, .2, .9],
                             [.1, -.2, FLOAT32_NODATA], [.2, .5, .9]))
        ndvi_base = os.path.join(self.workspace_dir, "ndvi_base.tif")
        make_raster_from_array(ndvi_base, array)

        target_masked_ndvi = os.path.join(self.workspace_dir, "tgt_ndvi.tif")

        urban_mental_health.mask_ndvi(ndvi_base, target_masked_ndvi,
                                      None, None, None)

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
        from natcap.invest.urban_mental_health import urban_mental_health

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

        lulc_attr_df = pandas.read_csv(lulc_attr_path)

        urban_mental_health.mask_ndvi(ndvi_base, target_masked_ndvi, lulc_base,
                                      lulc_attr_df, target_lulc_mask)

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
        from natcap.invest.urban_mental_health import urban_mental_health

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
                               Polygon([(ORIGIN_X, ORIGIN_Y - 290),
                                        (ORIGIN_X+270, ORIGIN_Y-290),
                                        (ORIGIN_X+270, ORIGIN_Y),
                                        (ORIGIN_X, ORIGIN_Y),
                                        (ORIGIN_X, ORIGIN_Y-290)]),
                               Polygon([(ORIGIN_X, ORIGIN_Y-450),
                                        (ORIGIN_X+200, ORIGIN_Y-450),
                                        (ORIGIN_X+200, ORIGIN_Y-320),
                                        (ORIGIN_X, ORIGIN_Y-320),
                                        (ORIGIN_X, ORIGIN_Y-450)])
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
        from natcap.invest.urban_mental_health import urban_mental_health

        ndvi_array = numpy.array(([FLOAT32_NODATA, .3, .1], [.2, 0, 0],
                                  [-.1, .3, .1], [-0.2, .5, 1]))
        delta_ndvi = os.path.join(self.workspace_dir, "delta_ndvi.tif")
        make_raster_from_array(delta_ndvi, ndvi_array)

        bc_array = numpy.array(([5, FLOAT32_NODATA, 0], [1, 100, 10],
                                [50, 0, 10], [1, 1, 1]))
        baseline_cases = os.path.join(self.workspace_dir, "baseline_cases.tif")
        make_raster_from_array(baseline_cases, bc_array)

        aoi_path = os.path.join(self.workspace_dir, "aoi.shp")
        xmin = ORIGIN_X  # origin of raster
        ymax = ORIGIN_Y  # origin of raster
        xmax = ORIGIN_X + 300
        ymin = ORIGIN_Y - 300  # cut off lowest row
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
        from natcap.invest.urban_mental_health import urban_mental_health

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
        from natcap.invest.urban_mental_health import urban_mental_health

        aoi_vector = os.path.join(self.workspace_dir, "aoi.shp")
        fields = {"id": ogr.OFTReal}
        attribute_list = [{"id": 0}, {"id": 1}]
        # 2 non-overlapping polygons, first covers upper 3x3 pixels, second
        # polygon covers bottom left 1x2 pixels
        make_simple_vector(aoi_vector, fields, attribute_list,
                           shapely_geometry_list=[
                               Polygon([(ORIGIN_X, ORIGIN_Y-290),
                                        (ORIGIN_X+270, ORIGIN_Y-290),
                                        (ORIGIN_X+270, ORIGIN_Y),
                                        (ORIGIN_X, ORIGIN_Y),
                                        (ORIGIN_X, ORIGIN_Y-290)]),
                               Polygon([(ORIGIN_X, ORIGIN_Y-450),
                                        (ORIGIN_X+200, ORIGIN_Y-450),
                                        (ORIGIN_X+200, ORIGIN_Y-320),
                                        (ORIGIN_X, ORIGIN_Y-320),
                                        (ORIGIN_X, ORIGIN_Y-450)])
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
        from natcap.invest.urban_mental_health import urban_mental_health

        args = make_synthetic_data_and_params(self.workspace_dir, 3)
        urban_mental_health.execute(args)
        # Check CSV
        stats_csv = os.path.join(self.workspace_dir, "output",
                                 "preventable_cases_cost_sum_test1.csv")
        df = pandas.read_csv(stats_csv)
        self.assertIn("sum_cases", df.columns)
        self.assertNotIn("sum_cost", df.columns)

    def test_option2_basic_inputs(self):
        """Test UMH option 2 (LULC inputs) with basic LULC and attr table inputs"

        Test that LULC rasters are reclassified to NDVI based on attribute
        table values. Then test that delta NDVI is calculated correctly.
        """
        from natcap.invest.urban_mental_health import urban_mental_health
        args = make_synthetic_data_and_params(self.workspace_dir, 2)

        urban_mental_health.execute(args)

        # LULC reclassified to NDVI based on attr table, then convolved
        # Expected result calculated by hand
        expected_ndvi_alt_buffer_mean = numpy.array(
            [[.133333, .15, .15],
             [0.125, .175, PGP_FLOAT32_NODATA],
             [.13333333, .16666667, PGP_FLOAT32_NODATA]])
        actual_ndvi_alt_buffer_mean_path = os.path.join(
            self.workspace_dir, "intermediate",
            f"ndvi_alt_buffer_mean_{args['results_suffix']}.tif")
        actual_ndvi_alt_buffer_mean = pygeoprocessing.raster_to_numpy_array(
            actual_ndvi_alt_buffer_mean_path)
        numpy.testing.assert_allclose(
            actual_ndvi_alt_buffer_mean,
            expected_ndvi_alt_buffer_mean, atol=1e-6)

        # Expected delta NDVI calculated by hand
        expected_delta_ndvi = numpy.array(
            [[-0.0333333, 0, -0.05],
             [0, 0, PGP_FLOAT32_NODATA],
             [-0.01666669, PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA]])

        actual_delta_ndvi_path = os.path.join(
            self.workspace_dir, "intermediate",
            f"delta_ndvi_{args['results_suffix']}.tif")
        actual_delta_ndvi = pygeoprocessing.raster_to_numpy_array(
            actual_delta_ndvi_path)
        numpy.testing.assert_allclose(actual_delta_ndvi,
                                      expected_delta_ndvi, atol=1e-6)

    def test_option2_lulc_relcassified_by_ndvi_raster(self):
        """Test UMH option 2 (LULC) if reclassifying with NDVI raster

        Test that UMH falls back to reclassifying LULC based on NDVI raster
        if ``ndvi`` column not provided in attribute table.
        """
        from natcap.invest.urban_mental_health import urban_mental_health
        args = make_synthetic_data_and_params(self.workspace_dir, 2)
        # open LULC attribute table and drop ndvi column
        lulc_attr_table = pandas.read_csv(args['lulc_attr_csv'])
        lulc_attr_table = lulc_attr_table.drop(columns=['ndvi'])
        lulc_attr_table.to_csv(args['lulc_attr_csv'])

        urban_mental_health.execute(args)

        # assert that model ran using NDVI raster for reclassification
        # (calculated by hand using base_ndvi (_not_ alt_ndvi) averages)
        ndvi_lucode_1 = 0.503333  # (.5 + .9 + .1133 + nodata) / 3
        ndvi_lucode_2 = 0.55  # (.6 + .7 + .8 + .12) / 4
        expected_ndvi_alt = numpy.array(
            [[ndvi_lucode_1, ndvi_lucode_2, ndvi_lucode_1],
             [ndvi_lucode_1, ndvi_lucode_2, PGP_FLOAT32_NODATA],
             [ndvi_lucode_1, ndvi_lucode_2, PGP_FLOAT32_NODATA]]
        )
        actual_ndvi_alt_path = os.path.join(
            self.workspace_dir, "intermediate",
            f"ndvi_alt_aligned_masked_{args['results_suffix']}.tif")
        actual_ndvi_alt = pygeoprocessing.raster_to_numpy_array(
            actual_ndvi_alt_path)
        numpy.testing.assert_allclose(actual_ndvi_alt, expected_ndvi_alt,
                                      atol=1e-6)

        expected_ndvi_base = numpy.array(
            [[ndvi_lucode_1, ndvi_lucode_2, ndvi_lucode_2],
             [ndvi_lucode_2, ndvi_lucode_1, ndvi_lucode_2],
             [ndvi_lucode_1, PGP_FLOAT32_NODATA, ndvi_lucode_1]])
        actual_ndvi_base_path = os.path.join(
            self.workspace_dir, "intermediate",
            f"ndvi_base_aligned_masked_{args['results_suffix']}.tif")
        actual_ndvi_base = pygeoprocessing.raster_to_numpy_array(
            actual_ndvi_base_path)
        numpy.testing.assert_allclose(actual_ndvi_base, expected_ndvi_base,
                                      atol=1e-6)

    def test_calculate_mean_ndvi_by_lulc_class(self):
        """Test `_calculate_mean_ndvi_by_lulc_class`

        Test that mean NDVI is calculated per LULC class correctly.
        """
        from natcap.invest.urban_mental_health import urban_mental_health

        lulc_array = numpy.array(
            [[1, 2, 3], [1, 2, 3], [1, 2, 4], [2, 2, 4]])
        lulc_path = os.path.join(self.workspace_dir, "lulc.tif")
        make_raster_from_array(lulc_path, lulc_array)

        ndvi_array = numpy.array(
            [[.75, .3, .1], [.1, .2, .9],
             [.1, .2, .4], [.2, .5, .9]])
        ndvi_path = os.path.join(self.workspace_dir, "ndvi.tif")
        make_raster_from_array(ndvi_path, ndvi_array)

        # dictionary with exclude codes
        lulc_dict = {
            1: 0,
            2: 0,
            3: 1,
            4: 1
        }

        actual_mean_ndvi = urban_mental_health._calculate_mean_ndvi_by_lulc_class(
            lulc_path, ndvi_path, lulc_dict)

        expected_mean_ndvi = {1: 0.3166667, 2: 0.28, 3: PGP_FLOAT32_NODATA,
                              4: PGP_FLOAT32_NODATA}

        # assert all key:value pairs are equal
        for key in expected_mean_ndvi:
            numpy.testing.assert_allclose(
                actual_mean_ndvi[key], expected_mean_ndvi[key], atol=1e-6)

    # def test_option1_tcc_input(self):
    #     "Test umh option 1 (tcc + ndvi inputs)"
    #     from natcap.invest.urban_mental_health import urban_mental_health

    #     args = make_synthetic_data_and_params(self.workspace_dir, 1)

    #     urban_mental_health.execute(args)

    #     expected_delta_ndvi = numpy.array(
    #         [[-0.1, 0.0, -0.2],
    #          [0.0, 0.1, PGP_FLOAT32_NODATA],
    #          [-0.05, PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA]]) # TODO: verify
    #     actual_delta_ndvi_path = os.path.join(
    #         self.workspace_dir, "intermediate",
    #         f"delta_ndvi_negatives_masked_{args['results_suffix']}.tif")
    #     actual_delta_ndvi = pygeoprocessing.raster_to_numpy_array(
    #         actual_delta_ndvi_path)

    #     numpy.testing.assert_allclose(actual_delta_ndvi,
    #                                   expected_delta_ndvi, atol=1e-6)

    #     expected_baseline_cases = numpy.array(
    #         [[200, 300, 800], [900, 140, 140],
    #          [PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA]])
    #     actual_baseline_cases_path = os.path.join(
    #         self.workspace_dir, "intermediate",
    #         f"baseline_cases_{args['results_suffix']}.tif")
    #     actual_baseline_cases = pygeoprocessing.raster_to_numpy_array(
    #         actual_baseline_cases_path)
    #     numpy.testing.assert_allclose(actual_baseline_cases,
    #                                   expected_baseline_cases, atol=1e-6)

    #     expected_preventable_cases = numpy.array([
    #         [-21.72614, -64.55958, -26.8406096],
    #         [-136.040285, -15.914164, -20.58118],
    #         [PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA, PGP_FLOAT32_NODATA]])

    #     # results contains only center pixel left bc AOI is small
    #     expected_preventable_cases = numpy.full((3, 3), PGP_FLOAT32_NODATA)
    #     expected_preventable_cases[1, 1] = -15.914164

    #     actual_preventable_cases_path = os.path.join(
    #         self.workspace_dir, "output",
    #         f"preventable_cases_{args['results_suffix']}.tif")
    #     actual_preventable_cases = pygeoprocessing.raster_to_numpy_array(
    #         actual_preventable_cases_path)

    #     numpy.testing.assert_allclose(actual_preventable_cases,
    #                                   expected_preventable_cases, atol=1e-5)

    def test_apply_tc_target_to_alt_ndvi(self):
        """Test `_fit_tc_to_ndvi_curve` and `_apply_tc_target_to_alt_ndvi`"""
        from natcap.invest.urban_mental_health import urban_mental_health

        ndvi_ar = numpy.array(((1, 1, .5), (.5, 0, .2)), dtype=float)
        ndvi_path = os.path.join(self.workspace_dir, "ndvi.tif")
        tc_ar = numpy.array(((90, 91, 50), (45, 1, 20)), dtype=float)
        tc_path = os.path.join(self.workspace_dir, "tc.tif")
        pop_ar = numpy.array(((10, 0, 10), (100, 600, 500)), dtype=int)
        pop_path = os.path.join(self.workspace_dir, "pop.tif")
        result_fig_path = os.path.join(self.workspace_dir, "tc_to_ndvi_curve.png")
        actual_alt_ndvi_path = os.path.join(self.workspace_dir, "intermediate",
                                            "ndvi_alt_buffer_mean.tif")

        tc_target = 30

        make_raster_from_array(ndvi_path, ndvi_ar)
        make_raster_from_array(tc_path, tc_ar)
        make_raster_from_array(pop_path, pop_ar)
        centers, curve = urban_mental_health._fit_tc_to_ndvi_curve(
            ndvi_path, tc_path, pop_path, result_fig_path,
            nbins=5, nsplines=10)

        # edges: [0, 20, 40, 60, 80, 100] # because there are 5 bins in range of perentages [0, 100]
        # idx (i.e. index of which bin TCC values fall into): [4, 4, 2, 2, 0, 1]
        # expected ndvi_pop_sum: [0*600, 0.2*500, 0.5*10 + 0.5*100, 0 , 1*10 + 1*0]
        # pop_sum calculated by summing the population values that fall into each TCC "bin"
        expected_pop_sum = numpy.array([600, 500, 100+10, 0, 10+0])
        # expected curve pre-interp: numpy.array((0*600, 0.2*500, 0.5*10 + 0.5*100, 0 , 1*10 + 1*0))/numpy.array((600, 500, 110, 0, 10))
        expected_interp_curve = numpy.array([0, 0.2, 0.5, 0.75, 1])
        # ^ calculated via:
        # ((0*600, 0.2*500, 0.5*10 + 0.5*100, 0 , 1*10 + 1*0))/((600, 500, 110, 0, 10))
        # expected_interp_curve[3] = (expected_interp_curve[2] + expected_interp_curve[4])/2

        expected_centers = numpy.array([10, 30, 50, 70, 90])
        numpy.testing.assert_allclose(centers, expected_centers)

        # GAM smoothing on binned means
        x = (expected_centers[expected_pop_sum > 0]).reshape(-1, 1)
        y = expected_interp_curve[expected_pop_sum > 0]
        w = expected_pop_sum[expected_pop_sum > 0]

        gam = LinearGAM(s(0, n_splines=10))
        gam.fit(x, y, weights=w)

        expected_curve_smooth = gam.predict(expected_centers.reshape(-1, 1))
        numpy.testing.assert_allclose(curve, expected_curve_smooth)

        urban_mental_health._apply_tc_target_to_alt_ndvi(
            ndvi_path, pop_path, tc_path, tc_target, actual_alt_ndvi_path,
            result_fig_path, nbins=5, nsplines=10)

        ndvi_tgt_val = float(numpy.interp(tc_target, centers,
                                          expected_curve_smooth))
        f_tc = numpy.interp(tc_ar, centers, expected_curve_smooth)

        ndvi_diff = float(ndvi_tgt_val) - f_tc

        expected_alt_ndvi = (ndvi_ar + ndvi_diff).astype(numpy.float32)
        print(expected_alt_ndvi, 'is exp alt ndvi')

        actual_alt_ndvi = pygeoprocessing.raster_to_numpy_array(
            actual_alt_ndvi_path)
        numpy.testing.assert_allclose(actual_alt_ndvi, expected_alt_ndvi)

    def test_diff_prj_inputs_opt1(self):
        """Test model option 1 given inputs of different projections.

        Check that output preventable cases geotiff is clipped
        correctly given inputs of different projections
        """
        from natcap.invest.urban_mental_health import urban_mental_health

        args = make_synthetic_data_and_params(self.workspace_dir, 1)

        # create NDVI base with different projection
        ndvi_base_array = numpy.array(
            [[.1, .2, .35], [.5, .6, .7],
             [.8, .9, .10], [.11, .12, FLOAT32_NODATA]])

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(5070)
        pygeoprocessing.numpy_array_to_raster(
            ndvi_base_array.astype(numpy.float32), FLOAT32_NODATA,
            pixel_size=(100, -100), origin=(-2151490, 2699260),
            projection_wkt=srs.ExportToWkt(),
            target_path=args['ndvi_base'])

        urban_mental_health.execute(args)

        actual_prev_cases = pygeoprocessing.raster_to_numpy_array(
            os.path.join(self.workspace_dir, "output",
                         "preventable_cases_test1.tif"))
        # most are nodata because AOI is just under 1 pixel
        # shape is 3x4 (rather than 3x3) because when ndvi_base is resampled,
        # it causes extra nodata column to right of AOI
        expected_prev_cases = numpy.full((3, 4), PGP_FLOAT32_NODATA)
        expected_prev_cases[1, 1] = 242.5831 #from output
        numpy.testing.assert_allclose(actual_prev_cases, expected_prev_cases,
                                      atol=1e-4)
