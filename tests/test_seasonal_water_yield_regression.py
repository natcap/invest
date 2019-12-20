"""InVEST Seasonal water yield model tests that use the InVEST sample data."""
import unittest
import tempfile
import shutil
import os

import numpy
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import pygeoprocessing.testing

REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'seasonal_water_yield')


def make_simple_shp(base_shp_path, origin):
    """Make a 100x100 ogr rectangular geometry shapefile.

    Parameters:
        base_shp_path (str): path to the shapefile.

    Returns:
        None.

    """
    # Create a new shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = driver.CreateDataSource(base_shp_path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)  # Spatial reference UTM Zone 10N
    layer = data_source.CreateLayer('layer', srs, ogr.wkbPolygon)

    # Add an FID field to the layer
    field_name = 'FID'
    field = ogr.FieldDefn(field_name)
    layer.CreateField(field)

    # Create a rectangular geometry
    lon, lat = origin[0], origin[1]
    width = 100
    rect = ogr.Geometry(ogr.wkbLinearRing)
    rect.AddPoint(lon, lat)
    rect.AddPoint(lon + width, lat)
    rect.AddPoint(lon + width, lat - width)
    rect.AddPoint(lon, lat - width)
    rect.AddPoint(lon, lat)

    # Create the feature from the geometry
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(rect)
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetField(field_name, '1')
    feature.SetGeometry(poly)
    layer.CreateFeature(feature)

    feature = None
    data_source = None


def make_raster_from_array(base_array, base_raster_path):
    """Make a raster from an array on a designated path.

    Parameters:
        array (numpy.ndarray): the 2D array for making the raster.
        raster_path (str): path to the raster to be created.

    Returns:
        None.

    """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)  # UTM Zone 10N
    project_wkt = srs.ExportToWkt()

    pygeoprocessing.testing.create_raster_on_disk(
        [base_array],
        (1180000, 690000),
        project_wkt,
        -1,
        (1, -1),  # Each pixel is 1x1 m
        filename=base_raster_path)


def make_lulc_raster(lulc_ras_path):
    """Make a 100x100 LULC raster with two LULC codes on the raster path.

    Parameters:
        lulc_raster_path (str): path to the LULC raster.

    Returns:
        None.
    """
    size = 100
    lulc_array = numpy.zeros((size, size), dtype=numpy.int8)
    lulc_array[size // 2:, :] = 1
    make_raster_from_array(lulc_array, lulc_ras_path)


def make_soil_raster(soil_ras_path):
    """Make a 100x100 soil group raster with four soil groups on th raster path.

    Parameters:
        soil_ras_path (str): path to the soil group raster.

    Returns:
        None.
    """
    size = 100
    soil_groups = 4
    soil_array = numpy.zeros((size, size))
    for i, row in enumerate(soil_array):
        row[:] = i % soil_groups + 1
    make_raster_from_array(soil_array, soil_ras_path)


def make_gradient_raster(grad_ras_path):
    """Make a raster with different values on each row on the raster path.

    The raster values on each column are in an ascending order from 0 to the
    nth column, based on the size of the array. This function can be used for
    making DEM or climate zone rasters.

    Parameters:
        grad_ras_path (str): path to the gradient raster.

    Returns:
        None.
    """
    size = 100
    grad_array = numpy.resize(numpy.arange(size), (size, size))
    make_raster_from_array(grad_array, grad_ras_path)


def make_eto_rasters(eto_dir_path):
    """Make twelve 100x100 rasters of monthly evapotranspiration.

    Parameters:
        eto_dir_path (str): path to the directory for saving the rasters.

    Returns:
        None.
    """
    size = 100
    for month in range(1, 13):
        eto_raster_path = os.path.join(eto_dir_path,
                                       'eto' + str(month) + '.tif')
        eto_array = numpy.full((size, size), month)
        make_raster_from_array(eto_array, eto_raster_path)


def make_precip_rasters(precip_dir_path):
    """Make twelve 100x100 rasters of monthly precipitation.

    Parameters:
        precip_dir_path (str): path to the directory for saving the rasters.

    Returns:
        None.
    """
    size = 100
    for month in range(1, 13):
        precip_raster_path = os.path.join(precip_dir_path,
                                          'precip_mm_' + str(month) + '.tif')
        precip_array = numpy.full((size, size), month + 10)
        make_raster_from_array(precip_array, precip_raster_path)


def make_recharge_raster(recharge_ras_path):
    """Make a 100x100 raster of user defined recharge.

    Parameters:
        recharge_ras_path (str): path to the directory for saving the rasters.

    Returns:
        None.
    """
    size = 100
    recharge_array = numpy.full((size, size), 200)
    make_raster_from_array(recharge_array, recharge_ras_path)


def make_rain_csv(rain_csv_path):
    """Make a synthesized rain events csv on the designated csv path.

    Parameters:
        rain_csv_path (str): path to the rain events csv.

    Returns:
        None.
    """
    with open(rain_csv_path, 'w') as open_table:
        open_table.write('month,events\n')
        for month in range(1, 13):
            open_table.write(str(month) + ',' + '1\n')


def make_biophysical_csv(biophysical_csv_path):
    """Make a synthesized biophysical csv on the designated path.

    Parameters:
        biophysical_csv (str): path to the biophysical csv.

    Returns:
        None.
    """
    with open(biophysical_csv_path, 'w') as open_table:
        open_table.write(
            'lucode,Description,CN_A,CN_B,CN_C,CN_D,Kc_1,Kc_2,Kc_3,Kc_4,')
        open_table.write('Kc_5,Kc_6,Kc_7,Kc_8,Kc_9,Kc_10,Kc_11,Kc_12\n')

        open_table.write('0,"lulc 1",50,50,0,0,0.7,0.7,0.7,0.7,0.7,0.7,0.7,')
        open_table.write('0.7,0.7,0.7,0.7,0.7\n')

        open_table.write('1,"lulc 2",72,82,0,0,0.4,0.4,0.4,0.4,0.4,0.4,0.4,')
        open_table.write('0.4,0.4,0.4,0.4,0.4\n')


def make_bad_biophysical_csv(biophysical_csv_path):
    """Make a bad biophysical csv with bad values to test error handling.

    Parameters:
        biophysical_csv (str): path to the corrupted biophysical csv.

    Returns:
        None.
    """
    with open(biophysical_csv_path, 'w') as open_table:
        open_table.write(
            'lucode,Description,CN_A,CN_B,CN_C,CN_D,Kc_1,Kc_2,Kc_3,Kc_4,')
        open_table.write('Kc_5,Kc_6,Kc_7,Kc_8,Kc_9,Kc_10,Kc_11,Kc_12\n')
        # look at that 'fifty'
        open_table.write(
            '0,"lulc 1",fifty,50,0,0,0.7,0.7,0.7,0.7,0.7,0.7,0.7,')
        open_table.write('0.7,0.7,0.7,0.7,0.7\n')
        open_table.write('1,"lulc 2",72,82,0,0,0.4,0.4,0.4,0.4,0.4,0.4,0.4,')
        open_table.write('0.4,0.4,0.4,0.4,0.4\n')


def make_alpha_csv(alpha_csv_path):
    """Make a monthly alpha csv on the designated path.

    Parameters:
        alpha_csv_path (str): path to the alpha csv.

    Returns:
        None.
    """
    with open(alpha_csv_path, 'w') as open_table:
        open_table.write('month,alpha\n')
        for month in range(1, 13):
            open_table.write(str(month) + ',0.083333333\n')


def make_climate_zone_csv(cz_csv_path):
    """Make a climate zone csv with number of rain events per months and CZs.

    Parameters:
        cz_csv_path (str): path to the climate zone csv.

    Returns:
        None.
    """
    climate_zones = 100
    # Random rain events for each month
    rain_events = [14, 17, 14, 15, 20, 18, 4, 6, 5, 16, 16, 20]
    with open(cz_csv_path, 'w') as open_table:
        open_table.write('cz_id,jan,feb,mar,apr,may,jun,jul,aug,sep,oct,nov,dec\n')

        for cz in range(climate_zones):
            rain_events = [x + 1 for x in rain_events]
            rain_events_str = [str(val) for val in [cz] + rain_events]
            rain_events_str = ','.join(rain_events_str) + '\n'
            open_table.write(rain_events_str)


def make_agg_results_csv(result_csv_path,
                         climate_zones=False,
                         recharge=False,
                         vector_exists=False):
    """Make csv file that has the expected aggregated_results.shp table.

    The csv table is in the form of fid,vri_sum,qb_val per line.

    Parameters:
        csv_path (str): path to the aggregated results csv file.
        climate_zones (bool): True if model is executed in climate zone mode.
        recharge (bool): True if user inputs recharge zone shapefile.
        vector_preexists (bool): True if aggregate results exists.

    Returns:
        None.
    """
    with open(result_csv_path, 'w') as open_table:
        if climate_zones:
            open_table.write('0,1.0,54.4764\n')
        elif recharge:
            open_table.write('0,0.00000,200.00000')
        elif vector_exists:
            open_table.write('0,2000000.00000,200.00000')
        else:
            open_table.write('0,1.0,51.359875\n')


class SeasonalWaterYieldUnusualDataTests(unittest.TestCase):
    """Tests for InVEST Seasonal Water Yield model that cover cases where
    input data are in an unusual corner case"""

    def setUp(self):
        """Make tmp workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Delete workspace after test is done."""
        shutil.rmtree(self.workspace_dir, ignore_errors=True)

    def test_ambiguous_precip_data(self):
        """SWY test case where there are more than 12 precipitation files"""
        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        precip_dir_path = os.path.join(self.workspace_dir, 'precip_dir')
        test_precip_dir_path = os.path.join(self.workspace_dir,
                                            'test_precip_dir')
        os.makedirs(precip_dir_path)
        make_precip_rasters(precip_dir_path)
        shutil.copytree(precip_dir_path, test_precip_dir_path)
        shutil.copy(
            os.path.join(test_precip_dir_path, 'precip_mm_3.tif'),
            os.path.join(test_precip_dir_path, 'bonus_precip_mm_3.tif'))

        # A placeholder args that has the property that the aoi_path will be
        # the same name as the output aggregate vector
        args = {
            'workspace_dir': self.workspace_dir,
            'alpha_m': '1/12',
            'beta_i': '1.0',
            'gamma': '1.0',
            'precip_dir': test_precip_dir_path,  # test constructed one
            'threshold_flow_accumulation': '1000',
            'user_defined_climate_zones': False,
            'user_defined_local_recharge': False,
            'monthly_alpha': False,
        }

        watershed_shp_path = os.path.join(args['workspace_dir'],
                                          'watershed.shp')
        make_simple_shp(watershed_shp_path, (1180000.0, 690000.0))
        args['aoi_path'] = watershed_shp_path

        biophysical_csv_path = os.path.join(args['workspace_dir'],
                                            'biophysical_table.csv')
        make_biophysical_csv(biophysical_csv_path)
        args['biophysical_table_path'] = biophysical_csv_path

        dem_ras_path = os.path.join(args['workspace_dir'], 'dem.tif')
        make_gradient_raster(dem_ras_path)
        args['dem_raster_path'] = dem_ras_path

        eto_dir_path = os.path.join(args['workspace_dir'], 'eto_dir')
        os.makedirs(eto_dir_path)
        make_eto_rasters(eto_dir_path)
        args['et0_dir'] = eto_dir_path

        lulc_ras_path = os.path.join(args['workspace_dir'], 'lulc.tif')
        make_lulc_raster(lulc_ras_path)
        args['lulc_raster_path'] = lulc_ras_path

        rain_csv_path = os.path.join(args['workspace_dir'],
                                     'rain_events_table.csv')
        make_rain_csv(rain_csv_path)
        args['rain_events_table_path'] = rain_csv_path

        soil_ras_path = os.path.join(args['workspace_dir'], 'soil_group.tif')
        make_soil_raster(soil_ras_path)
        args['soil_group_path'] = soil_ras_path

        with self.assertRaises(ValueError):
            seasonal_water_yield.execute(args)

    def test_precip_data_missing(self):
        """SWY test case where there is a missing precipitation file"""

        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        precip_dir_path = os.path.join(self.workspace_dir, 'precip_dir')
        test_precip_dir_path = os.path.join(self.workspace_dir,
                                            'test_precip_dir')
        os.makedirs(precip_dir_path)
        make_precip_rasters(precip_dir_path)
        shutil.copytree(precip_dir_path, test_precip_dir_path)
        os.remove(os.path.join(test_precip_dir_path, 'precip_mm_3.tif'))

        # A placeholder args that has the property that the aoi_path will be
        # the same name as the output aggregate vector
        args = {
            'workspace_dir': self.workspace_dir,
            'alpha_m': '1/12',
            'beta_i': '1.0',
            'gamma': '1.0',
            'precip_dir': test_precip_dir_path,  # test constructed one
            'threshold_flow_accumulation': '1000',
            'user_defined_climate_zones': False,
            'user_defined_local_recharge': False,
            'monthly_alpha': False,
        }

        watershed_shp_path = os.path.join(args['workspace_dir'],
                                          'watershed.shp')
        make_simple_shp(watershed_shp_path, (1180000.0, 690000.0))
        args['aoi_path'] = watershed_shp_path

        biophysical_csv_path = os.path.join(args['workspace_dir'],
                                            'biophysical_table.csv')
        make_biophysical_csv(biophysical_csv_path)
        args['biophysical_table_path'] = biophysical_csv_path

        dem_ras_path = os.path.join(args['workspace_dir'], 'dem.tif')
        make_gradient_raster(dem_ras_path)
        args['dem_raster_path'] = dem_ras_path

        eto_dir_path = os.path.join(args['workspace_dir'], 'eto_dir')
        os.makedirs(eto_dir_path)
        make_eto_rasters(eto_dir_path)
        args['et0_dir'] = eto_dir_path

        lulc_ras_path = os.path.join(args['workspace_dir'], 'lulc.tif')
        make_lulc_raster(lulc_ras_path)
        args['lulc_raster_path'] = lulc_ras_path

        rain_csv_path = os.path.join(args['workspace_dir'],
                                     'rain_events_table.csv')
        make_rain_csv(rain_csv_path)
        args['rain_events_table_path'] = rain_csv_path

        soil_ras_path = os.path.join(args['workspace_dir'], 'soil_group.tif')
        make_soil_raster(soil_ras_path)
        args['soil_group_path'] = soil_ras_path

        with self.assertRaises(ValueError):
            seasonal_water_yield.execute(args)

    def test_aggregate_vector_preexists(self):
        """SWY test that model deletes a preexisting aggregate output result"""
        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        # Set up data so there is enough code to do an aggregate over the
        # rasters but the output vector already exists
        aoi_path = os.path.join(self.workspace_dir, 'watershed.shp')
        make_simple_shp(aoi_path, (1180000.0, 690000.0))
        l_path = os.path.join(self.workspace_dir, 'L.tif')
        make_recharge_raster(l_path)
        aggregate_vector_path = os.path.join(self.workspace_dir,
                                             'aggregated_results.shp')
        make_simple_shp(aggregate_vector_path, (1180000.0, 690000.0))
        seasonal_water_yield._aggregate_recharge(aoi_path, l_path, l_path,
                                                 aggregate_vector_path)

        # test if aggregate is expected
        agg_results_csv_path = os.path.join(self.workspace_dir,
                                            'agg_results_base.csv')
        make_agg_results_csv(agg_results_csv_path, vector_exists=True)
        result_vector = ogr.Open(aggregate_vector_path)
        result_layer = result_vector.GetLayer()
        incorrect_value_list = []

        with open(agg_results_csv_path, 'r') as agg_result_file:
            for line in agg_result_file:
                fid, vri_sum, qb_val = [float(x) for x in line.split(',')]
                feature = result_layer.GetFeature(int(fid))
                for field, value in [('vri_sum', vri_sum), ('qb', qb_val)]:
                    if not numpy.isclose(
                            feature.GetField(field), value, rtol=1e-6):
                        incorrect_value_list.append(
                            'Unexpected value on feature %d, '
                            'expected %f got %f' % (fid, value,
                                                    feature.GetField(field)))
                ogr.Feature.__swig_destroy__(feature)
                feature = None

        result_layer = None
        ogr.DataSource.__swig_destroy__(result_vector)
        result_vector = None

        if incorrect_value_list:
            raise AssertionError('\n' + '\n'.join(incorrect_value_list))

    def test_duplicate_aoi_assertion(self):
        """SWY ensure model halts when AOI path identical to output vector"""
        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        # A placeholder args that has the property that the aoi_path will be
        # the same name as the output aggregate vector
        args = {
            'workspace_dir': self.workspace_dir,
            'aoi_path': os.path.join(
                        self.workspace_dir, 'aggregated_results_foo.shp'),
            'results_suffix': 'foo',
            'alpha_m': '1/12',
            'beta_i': '1.0',
            'gamma': '1.0',
            'threshold_flow_accumulation': '1000',
            'user_defined_climate_zones': False,
            'user_defined_local_recharge': False,
            'monthly_alpha': False,
        }

        biophysical_csv_path = os.path.join(args['workspace_dir'],
                                            'biophysical_table.csv')
        make_biophysical_csv(biophysical_csv_path)
        args['biophysical_table_path'] = biophysical_csv_path

        dem_ras_path = os.path.join(args['workspace_dir'], 'dem.tif')
        make_gradient_raster(dem_ras_path)
        args['dem_raster_path'] = dem_ras_path

        eto_dir_path = os.path.join(args['workspace_dir'], 'eto_dir')
        os.makedirs(eto_dir_path)
        make_eto_rasters(eto_dir_path)
        args['et0_dir'] = eto_dir_path

        lulc_ras_path = os.path.join(args['workspace_dir'], 'lulc.tif')
        make_lulc_raster(lulc_ras_path)
        args['lulc_raster_path'] = lulc_ras_path

        precip_dir_path = os.path.join(args['workspace_dir'], 'precip_dir')
        os.makedirs(precip_dir_path)
        make_precip_rasters(precip_dir_path)
        args['precip_dir'] = precip_dir_path

        rain_csv_path = os.path.join(args['workspace_dir'],
                                     'rain_events_table.csv')
        make_rain_csv(rain_csv_path)
        args['rain_events_table_path'] = rain_csv_path

        soil_ras_path = os.path.join(args['workspace_dir'], 'soil_group.tif')
        make_soil_raster(soil_ras_path)
        args['soil_group_path'] = soil_ras_path

        with self.assertRaises(ValueError):
            seasonal_water_yield.execute(args)


class SeasonalWaterYieldRegressionTests(unittest.TestCase):
    """Regression tests for InVEST Seasonal Water Yield model"""

    def setUp(self):
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def generate_base_args(workspace_dir):
        """Generate an args list that is consistent across all three regression
        tests"""
        args = {
            'alpha_m': '1/12',
            'beta_i': '1.0',
            'gamma': '1.0',
            'results_suffix': '',
            'threshold_flow_accumulation': '50',
            'workspace_dir': workspace_dir,
        }

        watershed_shp_path = os.path.join(workspace_dir, 'watershed.shp')
        make_simple_shp(watershed_shp_path, (1180000.0, 690000.0))
        args['aoi_path'] = watershed_shp_path

        biophysical_csv_path = os.path.join(workspace_dir,
                                            'biophysical_table.csv')
        make_biophysical_csv(biophysical_csv_path)
        args['biophysical_table_path'] = biophysical_csv_path

        dem_ras_path = os.path.join(workspace_dir, 'dem.tif')
        make_gradient_raster(dem_ras_path)
        args['dem_raster_path'] = dem_ras_path

        eto_dir_path = os.path.join(workspace_dir, 'eto_dir')
        os.makedirs(eto_dir_path)
        make_eto_rasters(eto_dir_path)
        args['et0_dir'] = eto_dir_path

        lulc_ras_path = os.path.join(workspace_dir, 'lulc.tif')
        make_lulc_raster(lulc_ras_path)
        args['lulc_raster_path'] = lulc_ras_path

        precip_dir_path = os.path.join(workspace_dir, 'precip_dir')
        os.makedirs(precip_dir_path)
        make_precip_rasters(precip_dir_path)
        args['precip_dir'] = precip_dir_path

        rain_csv_path = os.path.join(workspace_dir, 'rain_events_table.csv')
        make_rain_csv(rain_csv_path)
        args['rain_events_table_path'] = rain_csv_path

        soil_ras_path = os.path.join(workspace_dir, 'soil_group.tif')
        make_soil_raster(soil_ras_path)
        args['soil_group_path'] = soil_ras_path

        return args

    def test_base_regression(self):
        """SWY base regression test on sample data

        Executes SWY in default mode and checks that the output files are
        generated and that the aggregate shapefile fields are the same as the
        regression case."""
        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        # use predefined directory so test can clean up files during teardown
        args = SeasonalWaterYieldRegressionTests.generate_base_args(
            self.workspace_dir)

        # Ensure the model can pass when a nodata value is not defined.
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(26910)  # UTM Zone 10N
        project_wkt = srs.ExportToWkt()

        size = 100
        lulc_array = numpy.zeros((size, size), dtype=numpy.int8)
        lulc_array[size // 2:, :] = 1

        driver = gdal.GetDriverByName('GTiff')
        new_raster = driver.Create(
            args['lulc_raster_path'], lulc_array.shape[0],
            lulc_array.shape[1], 1, gdal.GDT_Byte)
        band = new_raster.GetRasterBand(1)
        band.WriteArray(lulc_array)
        geotransform = [1180000, 1, 0, 690000, 0, -1]
        new_raster.SetGeoTransform(geotransform)
        band = None
        new_raster = None
        driver = None

        # make args explicit that this is a base run of SWY
        args['user_defined_climate_zones'] = False
        args['user_defined_local_recharge'] = False
        args['monthly_alpha'] = False
        args['results_suffix'] = ''

        seasonal_water_yield.execute(args)

        # generate aggregated results csv table for assertion
        agg_results_csv_path = os.path.join(
            args['workspace_dir'], 'agg_results_base.csv')
        make_agg_results_csv(agg_results_csv_path)

        SeasonalWaterYieldRegressionTests._assert_regression_results_equal(
            os.path.join(args['workspace_dir'], 'aggregated_results.shp'),
            agg_results_csv_path)

    def test_bad_biophysical_table(self):
        """SWY bad biophysical table with non-numerical values."""
        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        # use predefined directory so test can clean up files during teardown
        args = SeasonalWaterYieldRegressionTests.generate_base_args(
            self.workspace_dir)
        # make args explicit that this is a base run of SWY
        args['user_defined_climate_zones'] = False
        args['user_defined_local_recharge'] = False
        args['monthly_alpha'] = False
        args['results_suffix'] = ''
        make_bad_biophysical_csv(args['biophysical_table_path'])

        with self.assertRaises(ValueError) as context:
            seasonal_water_yield.execute(args)
        self.assertTrue(
            'expecting all floating point numbers' in str(context.exception))

    def test_monthly_alpha_regression(self):
        """SWY monthly alpha values regression test on sample data

        Executes SWY using the monthly alpha table and checks that the output
        files are generated and that the aggregate shapefile fields are the
        same as the regression case."""
        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        # use predefined directory so test can clean up files during teardown
        args = SeasonalWaterYieldRegressionTests.generate_base_args(
            self.workspace_dir)
        # make args explicit that this is a base run of SWY
        args['user_defined_climate_zones'] = False
        args['user_defined_local_recharge'] = False
        args['monthly_alpha'] = True
        args['results_suffix'] = ''

        alpha_csv_path = os.path.join(args['workspace_dir'],
                                      'monthly_alpha.csv')
        make_alpha_csv(alpha_csv_path)
        args['monthly_alpha_path'] = alpha_csv_path

        seasonal_water_yield.execute(args)

        # generate aggregated results csv table for assertion
        agg_results_csv_path = os.path.join(args['workspace_dir'],
                                            'agg_results_base.csv')
        make_agg_results_csv(agg_results_csv_path)

        SeasonalWaterYieldRegressionTests._assert_regression_results_equal(
            os.path.join(args['workspace_dir'], 'aggregated_results.shp'),
            agg_results_csv_path)

    def test_climate_zones_regression(self):
        """SWY climate zone regression test on sample data

        Executes SWY in climate zones mode and checks that the output files are
        generated and that the aggregate shapefile fields are the same as the
        regression case."""
        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        # use predefined directory so test can clean up files during teardown
        args = SeasonalWaterYieldRegressionTests.generate_base_args(
            self.workspace_dir)
        # modify args to account for climate zones defined
        cz_csv_path = os.path.join(args['workspace_dir'],
                                   'climate_zone_events.csv')
        make_climate_zone_csv(cz_csv_path)
        args['climate_zone_table_path'] = cz_csv_path

        cz_ras_path = os.path.join(args['workspace_dir'], 'dem.tif')
        make_gradient_raster(cz_ras_path)
        args['climate_zone_raster_path'] = cz_ras_path

        args['user_defined_climate_zones'] = True
        args['user_defined_local_recharge'] = False
        args['monthly_alpha'] = False
        args['results_suffix'] = 'cz'

        seasonal_water_yield.execute(args)

        # generate aggregated results csv table for assertion
        agg_results_csv_path = os.path.join(args['workspace_dir'],
                                            'agg_results_cz.csv')
        make_agg_results_csv(agg_results_csv_path, climate_zones=True)

        SeasonalWaterYieldRegressionTests._assert_regression_results_equal(
            os.path.join(args['workspace_dir'], 'aggregated_results_cz.shp'),
            agg_results_csv_path)

    def test_user_recharge(self):
        """SWY user recharge regression test on sample data

        Executes SWY in user defined local recharge mode and checks that the
        output files are generated and that the aggregate shapefile fields
        are the same as the regression case."""
        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        # use predefined directory so test can clean up files during teardown
        args = SeasonalWaterYieldRegressionTests.generate_base_args(
            self.workspace_dir)
        # modify args to account for user recharge
        args['user_defined_climate_zones'] = False
        args['monthly_alpha'] = False
        args['results_suffix'] = ''
        args['user_defined_local_recharge'] = True
        recharge_ras_path = os.path.join(args['workspace_dir'], 'L.tif')
        make_recharge_raster(recharge_ras_path)
        args['l_path'] = recharge_ras_path

        seasonal_water_yield.execute(args)

        # generate aggregated results csv table for assertion
        agg_results_csv_path = os.path.join(args['workspace_dir'],
                                            'agg_results_l.csv')
        make_agg_results_csv(agg_results_csv_path, recharge=True)

        SeasonalWaterYieldRegressionTests._assert_regression_results_equal(
            os.path.join(args['workspace_dir'], 'aggregated_results.shp'),
            agg_results_csv_path)

    @staticmethod
    def _assert_regression_results_equal(
            result_vector_path, agg_results_path):
        """Test the state of the workspace against the expected list of files
        and aggregated results.

        Parameters:
            result_vector_path (string): path to the summary shapefile
                produced by the SWY model.
            agg_results_path (string): path to a csv file that has the
                expected aggregated_results.shp table in the form of
                fid,vri_sum,qb_val per line

        Returns:
            None

        Raises:
            AssertionError if any files are missing or results are out of
            range by `tolerance_places`
        """
        # we expect a file called 'aggregated_results.shp'
        result_vector = gdal.OpenEx(result_vector_path, gdal.OF_VECTOR)
        result_layer = result_vector.GetLayer()

        # The tolerance of 3 digits after the decimal was determined by
        # experimentation on the application with the given range of numbers.
        # This is an apparently reasonable approach as described by ChrisF:
        # http://stackoverflow.com/a/3281371/42897
        # and even more reading about picking numerical tolerance (it's hard):
        # https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
        tolerance_places = 3

        with open(agg_results_path, 'r') as agg_result_file:
            for line in agg_result_file:
                fid, vri_sum, qb_val = [float(x) for x in line.split(',')]
                feature = result_layer.GetFeature(int(fid))
                for field, value in [('vri_sum', vri_sum), ('qb', qb_val)]:
                    numpy.testing.assert_almost_equal(
                        feature.GetField(field),
                        value,
                        decimal=tolerance_places)
                ogr.Feature.__swig_destroy__(feature)
                feature = None

        result_layer = None
        result_vector = None


class SWYValidationTests(unittest.TestCase):
    """Tests for the SWY Model ARGS_SPEC and validation."""

    def setUp(self):
        """Create a temporary workspace."""
        self.workspace_dir = tempfile.mkdtemp()
        self.base_required_keys = [
            'workspace_dir',
            'gamma',
            'alpha_m',
            'soil_group_path',
            'user_defined_climate_zones',
            'rain_events_table_path',
            'biophysical_table_path',
            'monthly_alpha',
            'lulc_raster_path',
            'dem_raster_path',
            'beta_i',
            'et0_dir',
            'aoi_path',
            'precip_dir',
            'threshold_flow_accumulation',
            'user_defined_local_recharge',
        ]

    def tearDown(self):
        """Remove the temporary workspace after a test."""
        shutil.rmtree(self.workspace_dir)

    def test_missing_keys(self):
        """SWY Validate: assert missing required keys."""
        from natcap.invest.seasonal_water_yield import seasonal_water_yield
        from natcap.invest import validation

        validation_errors = seasonal_water_yield.validate({})  # empty args dict.
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(self.base_required_keys)
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_climate_zones(self):
        """SWY Validate: assert missing required keys given climate zones."""
        from natcap.invest.seasonal_water_yield import seasonal_water_yield
        from natcap.invest import validation

        validation_errors = seasonal_water_yield.validate(
            {'user_defined_climate_zones': True})
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys +
            ['climate_zone_table_path', 'climate_zone_raster_path'])
        expected_missing_keys.difference_update(
            {'user_defined_climate_zones', 'rain_events_table_path'})
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_local_recharge(self):
        """SWY Validate: assert missing required keys given local recharge."""
        from natcap.invest.seasonal_water_yield import seasonal_water_yield
        from natcap.invest import validation

        validation_errors = seasonal_water_yield.validate(
            {'user_defined_local_recharge': True})
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys + ['l_path'])
        expected_missing_keys.difference_update(
            {'user_defined_local_recharge',
             'et0_dir',
             'precip_dir',
             'rain_events_table_path',
             'soil_group_path'})
        self.assertEqual(invalid_keys, expected_missing_keys)

    def test_missing_keys_monthly_alpha_table(self):
        """SWY Validate: assert missing required keys given monthly alpha."""
        from natcap.invest.seasonal_water_yield import seasonal_water_yield
        from natcap.invest import validation

        validation_errors = seasonal_water_yield.validate(
            {'monthly_alpha': True})
        invalid_keys = validation.get_invalid_keys(validation_errors)
        expected_missing_keys = set(
            self.base_required_keys + ['monthly_alpha_path'])
        expected_missing_keys.difference_update(
            {'monthly_alpha', 'alpha_m'})
        self.assertEqual(invalid_keys, expected_missing_keys)
