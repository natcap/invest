"""Module for Regression Testing the InVEST Habitat Quality model."""
import unittest
import tempfile
import shutil
import os

from osgeo import gdal
from osgeo import osr
from osgeo import ogr
from shapely.geometry import Polygon
import numpy
import pygeoprocessing


def make_raster_from_array(
        base_array, base_raster_path, nodata_val=-1, gdal_type=gdal.GDT_Int32):
    """Make a raster from an array on a designated path.

    Args:
        base_array (numpy.ndarray): the 2D array for making the raster.
        nodata_val (int; float): nodata value for the raster.
        gdal_type (gdal datatype; int): gdal datatype for the raster.
        base_raster_path (str): the path for the raster to be created.

    Returns:
        None.
    """
    # Projection to user for generated sample data UTM Zone 10N
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)
    project_wkt = srs.ExportToWkt()
    origin = (1180000, 690000)

    pygeoprocessing.numpy_array_to_raster(
        base_array, nodata_val, (1, -1), origin, project_wkt, base_raster_path)


def make_access_shp(access_shp_path):
    """Create a 100x100 accessibility polygon shapefile with two access values.

    Args:
        access_shp_path (str): the path for the shapefile.

    Returns:
        None.
    """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)
    projection_wkt = srs.ExportToWkt()
    origin = (1180000, 690000)
    pos_x = origin[0]
    pos_y = origin[1]

    # Setup parameters for creating point shapefile
    fields = {'FID': ogr.OFTInteger64, 'ACCESS': ogr.OFTReal}
    attrs = [{'FID': 0, 'ACCESS': 0.2}, {'FID': 1, 'ACCESS': 1.0}]

    poly_geoms = {
        'poly_1': [(pos_x, pos_y), (pos_x + 100, pos_y),
                   (pos_x + 100, pos_y - 100 / 2.0),
                   (pos_x, pos_y - 100 / 2.0),
                   (pos_x, pos_y)],
        'poly_2': [(pos_x, pos_y - 50.0), (pos_x + 100, pos_y - 50.0),
                   (pos_x + 100, (pos_y - 50.0) - (100 / 2.0)),
                   (pos_x, (pos_y - 50.0) - (100 / 2.0)),
                   (pos_x, pos_y - 50.0)]}

    poly_geometries = [
        Polygon(poly_geoms['poly_1']), Polygon(poly_geoms['poly_2'])]

    # Create point shapefile to use for testing input
    pygeoprocessing.shapely_geometry_to_vector(
        poly_geometries, access_shp_path, projection_wkt, 'ESRI Shapefile',
        fields=fields, attribute_list=attrs, ogr_geom_type=ogr.wkbPolygon)


def make_threats_raster(
        folder_path, make_empty_raster=False, side_length=100,
        threat_values=None, nodata_val=-1, dtype=numpy.int8,
        gdal_type=gdal.GDT_Int32):
    """Create a side_lengthXside_length raster on designated path.

    Args:
        folder_path (str): the folder path for saving the threat rasters.
        make_empty_raster=False (bool): Whether to write a raster file
            that has no values at all.
        side_length=100 (int): The length of the sides of the threat raster.
        threat_values=None (None or list): If None, threat values of 1 will be
            used for the two threat rasters created.  Otherwise, a 2-element
            list should include numeric threat values for the two threat
            rasters.
        nodata_val (number): a number for the output nodata value
        dtype (numpy datatype): numpy datatype for the array.
        gdal_type (gdal datatype; int): gdal datatype for the raster.

    Returns:
        None.
    """
    threat_names = ['threat_1', 'threat_2']
    if threat_values is None:
        threat_values = [1, 1]

    for time_index, suffix in enumerate(['_c', '_f']):
        for (i, threat), value in zip(enumerate(threat_names), threat_values):
            threat_array = numpy.zeros((side_length, side_length), dtype=dtype)
            raster_path = os.path.join(folder_path, threat + suffix + '.tif')
            # making variations among threats and current vs future
            col_start = 5 * (i + 1)
            col_end = col_start + (3 * (time_index + 1))
            threat_array[20:side_length-20, col_start:col_end] = value
            if make_empty_raster:
                open(raster_path, 'a').close()  # writes an empty raster.
            else:
                make_raster_from_array(
                    threat_array, raster_path, nodata_val=nodata_val,
                    gdal_type=gdal_type)


def make_sensitivity_samp_csv(
        csv_path, include_threat=True, missing_lines=False):
    """Create a simplified sensitivity csv file with five land cover types.

    Args:
        csv_path (str): the path of sensitivity csv.
        include_threat (bool): whether the "threat" column is included in csv.
        missing_lines (bool): whether to intentionally leave out lulc rows.

    Returns:
        None.

    """
    if include_threat:
        with open(csv_path, 'w') as open_table:
            open_table.write('LULC,NAME,HABITAT,threat_1,threat_2\n')
            open_table.write('1,"lulc 1",1,1,1\n')
            if not missing_lines:
                open_table.write('2,"lulc 2",0.5,0.5,1\n')
                open_table.write('3,"lulc 3",0,0.3,1\n')
    else:
        with open(csv_path, 'w') as open_table:
            open_table.write('LULC,NAME,HABITAT\n')
            open_table.write('1,"lulc 1",1\n')
            if not missing_lines:
                open_table.write('2,"lulc 2",0.5\n')
                open_table.write('3,"lulc 3",0\n')


def assert_array_sum(base_raster_path, desired_sum, include_nodata=True):
    """Assert that the sum of a raster is equal to the specified value.

    Args:
        base_raster_path (str): the filepath of the raster to be asserted.
        desired_sum (float): the value to be compared with the raster sum.
        include_nodata (bool): whether to inlude nodata in the sum.

    Returns:
        None.

    """
    base_raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)
    base_band = base_raster.GetRasterBand(1)
    base_array = base_band.ReadAsArray()
    nodata = base_band.GetNoDataValue()
    if not include_nodata:
        base_array = base_array[~numpy.isclose(base_array, nodata)]

    raster_sum = numpy.sum(base_array)
    numpy.testing.assert_allclose(raster_sum, desired_sum, rtol=0, atol=1e-3)


class HabitatQualityTests(unittest.TestCase):
    """Tests for the Habitat Quality model."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_habitat_quality_regression(self):
        """Habitat Quality: base regression test with simplified data."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], threat_values=[0.5, 1.0],
            dtype=numpy.float32, gdal_type=gdal.GDT_Float32)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        habitat_quality.execute(args)

        # Assert values were obtained by summing each output raster.
        for output_filename, assert_value in {
                'deg_sum_c_regression.tif': 18.91135,
                'deg_sum_f_regression.tif': 33.931896,
                'quality_c_regression.tif': 7499.983,
                'quality_f_regression.tif': 4999.9893,
                'rarity_c_regression.tif': 3333.3335,
                'rarity_f_regression.tif': 3333.3335}.items():
            raster_path = os.path.join(args['workspace_dir'], output_filename)
            # Check that the raster's computed values are what we expect.
            # In this case, the LULC and threat rasters should have been
            # expanded to be beyond the bounds of the original threat values,
            # so we should exclude those new nodata pixel values.
            assert_array_sum(raster_path, assert_value, include_nodata=False)

    def test_habitat_quality_presence_absence_regression(self):
        """Habitat Quality: base regression test with simplified data.

        Threat rasters are set to 0 or 1.
        """
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], threat_values=[1, 1], dtype=numpy.int8,
            gdal_type=gdal.GDT_Int32)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        habitat_quality.execute(args)

        # Assert values were obtained by summing each output raster.
        for output_filename, assert_value in {
                'deg_sum_c_regression.tif': 27.153614,
                'deg_sum_f_regression.tif': 46.279358,
                'quality_c_regression.tif': 7499.9414,
                'quality_f_regression.tif': 4999.955,
                'rarity_c_regression.tif': 3333.3335,
                'rarity_f_regression.tif': 3333.3335}.items():
            raster_path = os.path.join(args['workspace_dir'], output_filename)
            # Check that the raster's computed values are what we expect.
            # In this case, the LULC and threat rasters should have been
            # expanded to be beyond the bounds of the original threat values,
            # so we should exclude those new nodata pixel values.
            assert_array_sum(raster_path, assert_value, include_nodata=False)

    def test_habitat_quality_nworkers_regression(self):
        """Habitat Quality: n_workers=2 regression test w/ simplified data."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': 2,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], threat_values=[1, 1],
            dtype=numpy.int8, gdal_type=gdal.GDT_Int32)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        habitat_quality.execute(args)

        # Assert values were obtained by summing each output raster.
        for output_filename, assert_value in {
                'deg_sum_c_regression.tif': 27.153614,
                'deg_sum_f_regression.tif': 46.279358,
                'quality_c_regression.tif': 7499.9414,
                'quality_f_regression.tif': 4999.955,
                'rarity_c_regression.tif': 3333.3335,
                'rarity_f_regression.tif': 3333.3335}.items():
            raster_path = os.path.join(args['workspace_dir'], output_filename)
            # Check that the raster's computed values are what we expect.
            # In this case, the LULC and threat rasters should have been
            # expanded to be beyond the bounds of the original threat values,
            # so we should exclude those new nodata pixel values.
            assert_array_sum(raster_path, assert_value, include_nodata=False)

    def test_habitat_quality_undefined_threat_nodata(self):
        """Habitat Quality: test for undefined threat nodata."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], threat_values=[1, 1],
            dtype=numpy.int8, gdal_type=gdal.GDT_Int32, nodata_val=None)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        # create the threat CSV table
        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        habitat_quality.execute(args)

        # Assert values were obtained by summing each output raster.
        for output_filename, assert_value in {
                'deg_sum_c_regression.tif': 27.153614,
                'deg_sum_f_regression.tif': 46.279358,
                'quality_c_regression.tif': 7499.9414,
                'quality_f_regression.tif': 4999.955,
                'rarity_c_regression.tif': 3333.3335,
                'rarity_f_regression.tif': 3333.3335}.items():
            raster_path = os.path.join(args['workspace_dir'], output_filename)
            # Check that the raster's computed values are what we expect.
            # In this case, the LULC and threat rasters should have been
            # expanded to be beyond the bounds of the original threat values,
            # so we should exclude those new nodata pixel values.
            assert_array_sum(raster_path, assert_value, include_nodata=False)

    def test_habitat_quality_lulc_bbox(self):
        """Habitat Quality: regression test for bbox sizes."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], side_length=50, threat_values=[1, 1])

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        habitat_quality.execute(args)

        base_lulc_bbox = pygeoprocessing.get_raster_info(
            args['lulc_bas_path'])['bounding_box']

        # Assert values were obtained by summing each output raster.
        for output_filename in ['deg_sum_c_regression.tif',
                                'deg_sum_f_regression.tif',
                                'quality_c_regression.tif',
                                'quality_f_regression.tif',
                                'rarity_c_regression.tif',
                                'rarity_f_regression.tif']:
            raster_path = os.path.join(args['workspace_dir'], output_filename)

            # Check that the output raster has the same bounding box as the
            # LULC rasters.
            raster_info = pygeoprocessing.get_raster_info(raster_path)
            raster_bbox = raster_info['bounding_box']
            numpy.testing.assert_allclose(
                raster_bbox, base_lulc_bbox, rtol=0, atol=1e-6)

    def test_habitat_quality_numeric_threats(self):
        """Habitat Quality: regression test on numeric threat names."""
        from natcap.invest import habitat_quality
        threat_array = numpy.zeros((100, 100), dtype=numpy.int8)

        threatnames = ['1111', '2222']
        threat_values = [1, 1]
        side_length = 100
        for time_index, suffix in enumerate(['_c', '_f']):
            for (i, threat), value in zip(
                    enumerate(threatnames), threat_values):
                threat_array = numpy.zeros(
                    (side_length, side_length), dtype=numpy.int8)
                raster_path = os.path.join(
                    self.workspace_dir, threat + suffix + '.tif')
                # making variations among threats and current vs future
                col_start = 5 * (i + 1)
                col_end = col_start + (3 * (time_index + 1))
                threat_array[20:side_length-20, col_start:col_end] = value
                make_raster_from_array(
                    threat_array, raster_path, nodata_val=-1,
                    gdal_type=gdal.GDT_Int32)

        threat_csv_path = os.path.join(self.workspace_dir, 'threats.csv')
        with open(threat_csv_path, 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,%s,linear,,1111_c.tif,1111_f.tif\n' % threatnames[0])
            open_table.write(
                '0.07,1.0,%s,exponential,,2222_c.tif,2222_f.tif\n'
                % threatnames[1])

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'threats_table_path': threat_csv_path,
            'workspace_dir': self.workspace_dir,
            'sensitivity_table_path': os.path.join(
                self.workspace_dir, 'sensitivity_samp.csv'),
            'access_vector_path': os.path.join(
                self.workspace_dir, 'access_samp.shp'),
            'n_workers': -1,
        }
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        with open(args['sensitivity_table_path'], 'w') as open_table:
            open_table.write(
                'LULC,NAME,HABITAT,%s,%s\n' % tuple(threatnames))
            open_table.write('1,"lulc 1",1,1,1\n')
            open_table.write('2,"lulc 2",0.5,0.5,1\n')
            open_table.write('3,"lulc 3",0,0.3,1\n')

        habitat_quality.execute(args)

        # Assert values were obtained by summing each output raster.
        for output_filename, assert_value in {
                'deg_sum_c_regression.tif': 27.153614,
                'deg_sum_f_regression.tif': 46.279358,
                'quality_c_regression.tif': 7499.9414,
                'quality_f_regression.tif': 4999.955,
                'rarity_c_regression.tif': 3333.3335,
                'rarity_f_regression.tif': 3333.3335
        }.items():
            assert_array_sum(os.path.join(
                args['workspace_dir'], output_filename), assert_value)

    def test_habitat_quality_missing_sensitivity_threat(self):
        """Habitat Quality: ValueError w/ missing threat in sensitivity."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        # Include a missing threat to the sensitivity csv table
        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(
            args['sensitivity_table_path'], include_threat=False)

        args['lulc_cur_path'] = os.path.join(
            args['workspace_dir'], 'lc_samp_cur_b.tif')

        lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
        lulc_array[50:, :] = 2
        make_raster_from_array(lulc_array, args['lulc_cur_path'])

        make_threats_raster(args['workspace_dir'])

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')
        # create the threat CSV table
        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        with self.assertRaises(KeyError):
            habitat_quality.execute(args)

    def test_habitat_quality_missing_threat(self):
        """Habitat Quality: expected ValueError on missing threat raster."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        args['lulc_cur_path'] = os.path.join(
            args['workspace_dir'], 'lc_samp_cur_b.tif')

        lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
        lulc_array[50:, :] = 2
        make_raster_from_array(lulc_array, args['lulc_cur_path'])

        make_threats_raster(args['workspace_dir'])

        # Include a missing threat to the threats csv table
        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        # create the threat CSV table
        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.07,0.8,missing_threat,linear,,missing_threat_c.tif,'
                'missing_threat_f.tif\n')

        with self.assertRaises(ValueError):
            habitat_quality.execute(args)

    def test_habitat_quality_threat_values_outside_range(self):
        """Habitat Quality: expected ValueError on threat values 0<=x<=1."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], threat_values=[0.1, 1.2],
            dtype=numpy.float32, gdal_type=gdal.GDT_Float32)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        # create the threat CSV table
        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        with self.assertRaises(ValueError):
            habitat_quality.execute(args)

    def test_habitat_quality_threat_max_dist(self):
        """Habitat Quality: expected ValueError on max_dist <=0."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(args['workspace_dir'])

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        # create the threat CSV table
        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.0,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        with self.assertRaises(ValueError) as cm:
            habitat_quality.execute(args)

        self.assertIn("max distance for threat: 'threat_1' is less",
                      str(cm.exception))

    def test_habitat_quality_invalid_decay_type(self):
        """Habitat Quality: expected ValueError on invalid decay type."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        args['lulc_cur_path'] = os.path.join(
            args['workspace_dir'], 'lc_samp_cur_b.tif')

        lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
        lulc_array[50:, :] = 2
        make_raster_from_array(lulc_array, args['lulc_cur_path'])

        make_threats_raster(args['workspace_dir'])

        # Include an invalid decay function name to the threats csv table.
        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        # create the threat CSV table
        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,invalid,,threat_1_c.tif,threat_1_f.tif\n')

        with self.assertRaises(ValueError):
            habitat_quality.execute(args)

    def test_habitat_quality_no_base_column(self):
        """Habitat Quality: no baseline LULC and no column should pass."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], threat_values=[1, 1],
            dtype=numpy.int8, gdal_type=gdal.GDT_Int32)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        # leave out BASE_PATH column
        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,threat_2_c.tif,'
                'threat_2_f.tif\n')

        try:
            habitat_quality.execute(args)
            self.assertTrue(True)
        except Exception as e:
            self.fail("HQ failed when using threat data CSV missing BASE_PATH"
                      f" column. \n {str(e)}")

    def test_habitat_quality_no_fut_column(self):
        """Habitat Quality: no future LULC and no column should pass."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], threat_values=[1, 1],
            dtype=numpy.int8, gdal_type=gdal.GDT_Int32)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        # leave out FUT_PATH column
        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif\n')

        try:
            habitat_quality.execute(args)
            self.assertTrue(True)
        except Exception as e:
            self.fail("HQ failed when using threat data CSV missing FUT_PATH"
                      f" column. \n {str(e)}")

    def test_habitat_quality_bad_rasters(self):
        """Habitat Quality: raise error on threats that aren't real rasters."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        args['lulc_cur_path'] = os.path.join(
            args['workspace_dir'], 'lc_samp_cur_b.tif')

        lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
        lulc_array[50:, :] = 2
        make_raster_from_array(lulc_array, args['lulc_cur_path'])

        # Make an empty threat raster in the workspace folder.
        make_threats_raster(
            args['workspace_dir'], make_empty_raster=True)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        # create the threat CSV table
        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        with self.assertRaises(ValueError) as cm:
            habitat_quality.execute(args)

        actual_message = str(cm.exception)
        self.assertIn(
            'There was an Error locating a threat raster from '
            'the path in CSV for column: cur_path and threat: threat_1',
            actual_message)

    def test_habitat_quality_lulc_current_only(self):
        """Habitat Quality: on missing base and future LULC rasters."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        args['lulc_cur_path'] = os.path.join(
            args['workspace_dir'], 'lc_samp_cur_b.tif')

        lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
        lulc_array[50:, :] = 2
        make_raster_from_array(lulc_array, args['lulc_cur_path'])

        make_threats_raster(args['workspace_dir'])

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        # create the threat CSV table
        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        habitat_quality.execute(args)

        # Reasonable to just check quality out in this case
        assert_array_sum(
            os.path.join(args['workspace_dir'], 'quality_c.tif'),
            7499.524)

    def test_habitat_quality_case_insensitivty(self):
        """Habitat Quality: with table columns that have camel case."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')

        with open(args['sensitivity_table_path'], 'w') as open_table:
            open_table.write('Lulc,Name,habitat,Threat_1,THREAT_2\n')
            open_table.write('1,"lulc 1",1,1,1\n')
            open_table.write('2,"lulc 2",0.5,0.5,1\n')
            open_table.write('3,"lulc 3",0,0.3,1\n')

        args['lulc_cur_path'] = os.path.join(
            args['workspace_dir'], 'lc_samp_cur_b.tif')

        lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
        lulc_array[50:, :] = 2
        make_raster_from_array(lulc_array, args['lulc_cur_path'])

        make_threats_raster(args['workspace_dir'])

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        # create the threat CSV table
        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'Max_Dist,Weight,threat,Decay,BASE_PATH,cur_PATH,fut_path\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        habitat_quality.execute(args)

        # Reasonable to just check quality out in this case
        assert_array_sum(
            os.path.join(args['workspace_dir'], 'quality_c.tif'),
            7499.524)

    def test_habitat_quality_lulc_baseline_current(self):
        """Habitat Quality: on missing future LULC raster."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        scenarios = ['_bas_', '_cur_']  # Missing '_fut_'
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        make_threats_raster(args['workspace_dir'])

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        # create the threat CSV table
        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        habitat_quality.execute(args)

        # Reasonable to just check quality out in this case
        assert_array_sum(
            os.path.join(args['workspace_dir'], 'quality_c.tif'),
            7499.524)
        assert_array_sum(
            os.path.join(args['workspace_dir'], 'rarity_c.tif'),
            3333.3335)

    def test_habitat_quality_missing_lucodes_in_table(self):
        """Habitat Quality: on missing lucodes in the sensitivity table."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            path = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            args['lulc' + scenario + 'path'] = path
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

            # Add a nodata value to this raster to make sure we don't include
            # the nodata value in the error message.
            raster = gdal.OpenEx(path, gdal.OF_RASTER | gdal.GA_Update)
            band = raster.GetRasterBand(1)
            band_nodata = 255
            band.SetNoDataValue(band_nodata)  # band nodata before this is -1
            current_array = band.ReadAsArray()
            current_array[49][49] = band_nodata
            band.WriteArray(current_array)
            band = None
            raster = None

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(
            args['sensitivity_table_path'], missing_lines=True)

        make_threats_raster(args['workspace_dir'])

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        # create the threat CSV table
        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        with self.assertRaises(ValueError) as cm:
            habitat_quality.execute(args)

        # 2 is the missing landcover codes.
        # Raster nodata is 255 and should NOT appear in this list.
        self.assertIn(
            "The missing values found in the LULC_c raster but not the table"
            " are: [2]", str(cm.exception))

    def test_habitat_quality_validate(self):
        """Habitat Quality: validate raise exception as expected."""
        from natcap.invest import habitat_quality

        args = {
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        validation_results = habitat_quality.validate(args)
        self.assertEqual(len(validation_results), 1)

        keys_without_value = set([
            'lulc_cur_path', 'threats_table_path', 'sensitivity_table_path',
            'half_saturation_constant'])
        self.assertEqual(set(validation_results[0][0]), keys_without_value)

    def test_habitat_quality_validate_complete(self):
        """Habitat Quality: test regular validation."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], threat_values=[1, 1],
            dtype=numpy.int8, gdal_type=gdal.GDT_Int32)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        # create the threat CSV table
        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        validate_result = habitat_quality.validate(args, limit_to=None)
        self.assertFalse(
            validate_result,  # List should be empty if validation passes
            f"expected no failed validations instead got {validate_result}.")

    def test_habitat_quality_validation_wrong_spatial_types(self):
        """Habitat Quality: test validation for wrong GIS types."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], threat_values=[1, 1],
            dtype=numpy.int8, gdal_type=gdal.GDT_Int32)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        # create the threat CSV table
        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        args['lulc_cur_path'], args['access_vector_path'] = (
            args['access_vector_path'], args['lulc_cur_path'])

        validate_result = habitat_quality.validate(args, limit_to=None)
        self.assertTrue(
            validate_result,
            "expected failed validations instead didn't get any")
        for (validation_keys, error_msg), phrase in zip(
                validate_result, ['GDAL vector', 'GDAL raster']):
            self.assertIn(phrase, error_msg)

    def test_habitat_quality_validation_missing_sens_header(self):
        """Habitat Quality: test validation for sens threat header."""
        from natcap.invest import habitat_quality, utils

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(
            args['sensitivity_table_path'], include_threat=False)

        make_threats_raster(
            args['workspace_dir'], threat_values=[1, 1],
            dtype=numpy.int8, gdal_type=gdal.GDT_Int32)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        # create the threat CSV table
        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        # At least one threat header is expected, so there should be a message
        validate_result = habitat_quality.validate(args, limit_to=None)
        self.assertEqual(len(validate_result), 1)
        self.assertEqual(validate_result[0][0], ['sensitivity_table_path'])
        self.assertTrue(utils.matches_format_string(
            validate_result[0][1],
            habitat_quality.MISSING_SENSITIVITY_TABLE_THREATS_MSG))

    def test_habitat_quality_validation_bad_threat_path(self):
        """Habitat Quality: test validation for bad threat paths."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        # intentialy do not make the threat rasters

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        # create the threat CSV table
        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        validate_result = habitat_quality.validate(args, limit_to=None)
        self.assertTrue(
            validate_result,
            "expected failed validations instead didn't get any.")
        self.assertEqual(
            habitat_quality.MISSING_THREAT_RASTER_MSG.format(
                threat_list=[
                    ('threat_1', 'cur_path'),
                    ('threat_2', 'cur_path'),
                    ('threat_1', 'fut_path'),
                    ('threat_2', 'fut_path')]),
            validate_result[0][1])

    def test_habitat_quality_missing_cur_threat_path(self):
        """Habitat Quality: test for missing threat paths in current."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], threat_values=[1, 1],
            dtype=numpy.int8, gdal_type=gdal.GDT_Int32)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        with self.assertRaises(ValueError) as cm:
            habitat_quality.execute(args)

        actual_message = str(cm.exception)
        self.assertIn(
            'There was an Error locating a threat raster from '
            'the path in CSV for column: cur_path and threat: threat_1',
            actual_message)

    def test_habitat_quality_missing_fut_threat_path(self):
        """Habitat Quality: test for missing threat paths in future."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], threat_values=[1, 1],
            dtype=numpy.int8, gdal_type=gdal.GDT_Int32)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        with self.assertRaises(ValueError) as cm:
            habitat_quality.execute(args)

        actual_message = str(cm.exception)
        self.assertIn(
            'There was an Error locating a threat raster from '
            'the path in CSV for column: fut_path and threat: threat_1',
            actual_message)

    def test_habitat_quality_misspelled_cur_threat_path(self):
        """Habitat Quality: test for a misspelled current threat path."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], threat_values=[1, 1],
            dtype=numpy.int8, gdal_type=gdal.GDT_Int32)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_cur.tif,threat_1_c.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        with self.assertRaises(ValueError) as cm:
            habitat_quality.execute(args)

        actual_message = str(cm.exception)
        self.assertIn(
            'There was an Error locating a threat raster from '
            'the path in CSV for column: cur_path and threat: threat_1',
            actual_message)

    def test_habitat_quality_validate_missing_cur_threat_path(self):
        """Habitat Quality: test validate for missing threat paths in cur."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], threat_values=[1, 1],
            dtype=numpy.int8, gdal_type=gdal.GDT_Int32)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        validate_result = habitat_quality.validate(args, limit_to=None)
        self.assertTrue(
            validate_result,
            "expected failed validations instead didn't get any.")
        self.assertEqual(
            habitat_quality.MISSING_THREAT_RASTER_MSG.format(
                threat_list=[('threat_1', 'cur_path')]),
            validate_result[0][1])

    def test_habitat_quality_validate_missing_fut_threat_path(self):
        """Habitat Quality: test validate for missing threat paths in fut."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], threat_values=[1, 1],
            dtype=numpy.int8, gdal_type=gdal.GDT_Int32)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        validate_result = habitat_quality.validate(args, limit_to=None)
        self.assertTrue(
            validate_result,
            "expected failed validations instead didn't get any.")
        self.assertEqual(
            habitat_quality.MISSING_THREAT_RASTER_MSG.format(
                threat_list=[('threat_1', 'fut_path')]),
            validate_result[0][1])

    def test_habitat_quality_validate_misspelled_cur_threat_path(self):
        """Habitat Quality: test validate for a misspelled cur threat path."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], threat_values=[1, 1],
            dtype=numpy.int8, gdal_type=gdal.GDT_Int32)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_cur.tif,threat_1_c.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        validate_result = habitat_quality.validate(args, limit_to=None)
        self.assertTrue(
            validate_result,
            "expected failed validations instead didn't get any.")
        self.assertEqual(
            habitat_quality.MISSING_THREAT_RASTER_MSG.format(
                threat_list=[('threat_1', 'cur_path')]),
            validate_result[0][1], validate_result[0][1])

    def test_habitat_quality_validate_duplicate_threat_path(self):
        """Habitat Quality: test validate for duplicate threat paths."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        # create threat rasters for the test
        threat_names = ['threat_1', 'threat_2']
        threat_values = [1, 1]

        threat_array = numpy.zeros((10, 10), dtype=numpy.int8)

        for suffix in ['_c', '_f']:
            for (i, threat), value in zip(
                    enumerate(threat_names), threat_values):
                raster_path = os.path.join(
                    args['workspace_dir'], threat + suffix + '.tif')
                # making variations among threats
                threat_array[100//(i+1):, :] = value
                make_raster_from_array(
                    threat_array, raster_path, nodata_val=-1)

        # create threats table for the test
        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_c.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        validate_result = habitat_quality.validate(args, limit_to=None)
        self.assertTrue(
            validate_result,
            "expected failed validations instead didn't get any.")
        self.assertEqual(
            habitat_quality.DUPLICATE_PATHS_MSG + str(['threat_1_c.tif']),
            validate_result[0][1])

    def test_habitat_quality_duplicate_threat_path(self):
        """Habitat Quality: test for duplicate threat paths."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        # create threat rasters for the test
        threat_names = ['threat_1', 'threat_2']
        threat_values = [1, 1]

        threat_array = numpy.zeros((10, 10), dtype=numpy.int8)

        for suffix in ['_c', '_f']:
            for (i, threat), value in zip(
                    enumerate(threat_names), threat_values):
                raster_path = os.path.join(
                    args['workspace_dir'], threat + suffix + '.tif')
                # making variations among threats
                threat_array[100//(i+1):, :] = value
                make_raster_from_array(
                    threat_array, raster_path, nodata_val=-1)

        # create threats table for the test
        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_c.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        with self.assertRaises(ValueError) as cm:
            habitat_quality.execute(args)

        actual_message = str(cm.exception)
        # assert that a duplicate error message was raised
        self.assertIn(habitat_quality.DUPLICATE_PATHS_MSG, actual_message)
        # assert that the path for the duplicate was in the error message
        self.assertIn('threat_1_c.tif', actual_message)

    def test_habitat_quality_argspec_spatial_overlap(self):
        """Habitat Quality: raise error on incorrect spatial overlap."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        # make future LULC with different origin
        args['lulc_fut_path'] = os.path.join(
            args['workspace_dir'], 'lc_samp_fut_b.tif')
        lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
        lulc_array[50:, :] = 3
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(26910)  # UTM Zone 10N
        project_wkt = srs.ExportToWkt()

        gtiff_driver = gdal.GetDriverByName('GTiff')
        ny, nx = lulc_array.shape
        new_raster = gtiff_driver.Create(
            args['lulc_fut_path'], nx, ny, 1, gdal.GDT_Int32)

        new_raster.SetProjection(project_wkt)
        origin = (1180200, 690200)
        new_raster.SetGeoTransform([origin[0], 1.0, 0.0, origin[1], 0.0, -1.0])
        new_band = new_raster.GetRasterBand(1)
        new_band.SetNoDataValue(-1)
        new_band.WriteArray(lulc_array)
        new_raster.FlushCache()
        new_band = None
        new_raster = None

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        # create threat rasters for the test
        threat_names = ['threat_1', 'threat_2']
        threat_values = [1, 1]

        threat_array = numpy.zeros((100, 100), dtype=numpy.int8)

        for suffix in ['_c', '_f']:
            for (i, threat), value in zip(
                    enumerate(threat_names), threat_values):
                raster_path = os.path.join(
                    args['workspace_dir'], threat + suffix + '.tif')
                # making variations among threats
                threat_array[100//(i+1):, :] = value
                make_raster_from_array(
                    threat_array, raster_path, nodata_val=-1)

        # create threats table for the test
        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        validate_result = habitat_quality.validate(args)
        self.assertTrue(
            validate_result,
            "expected failed validations instead didn't get any.")
        self.assertIn("Bounding boxes do not intersect", validate_result[0][1])

    def test_habitat_quality_argspec_missing_projection(self):
        """Habitat Quality: raise error on missing projection."""
        from natcap.invest import habitat_quality, validation

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        # make lulc cur without projection
        lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
        lulc_array[50:, :] = 1
        args['lulc_cur_path'] = os.path.join(
            args['workspace_dir'], 'lc_samp_cur_b.tif')

        lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
        lulc_array[50:, :] = 1
        # inentionally do not set spatial reference

        gtiff_driver = gdal.GetDriverByName('GTiff')
        ny, nx = lulc_array.shape
        new_raster = gtiff_driver.Create(
            args['lulc_cur_path'], nx, ny, 1, gdal.GDT_Int32)

        origin = (1180200, 690200)
        new_raster.SetGeoTransform([origin[0], 1.0, 0.0, origin[1], 0.0, -1.0])
        new_band = new_raster.GetRasterBand(1)
        new_band.SetNoDataValue(-1)
        new_band.WriteArray(lulc_array)
        new_raster.FlushCache()
        new_band = None
        new_raster = None

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        # create threat rasters for the test
        threat_names = ['threat_1', 'threat_2']
        threat_values = [1, 1]

        threat_array = numpy.zeros((100, 100), dtype=numpy.int8)

        for suffix in ['_c', '_f']:
            for (i, threat), value in zip(
                    enumerate(threat_names), threat_values):
                raster_path = os.path.join(
                    args['workspace_dir'], threat + suffix + '.tif')
                # making variations among threats
                threat_array[100//(i+1):, :] = value
                make_raster_from_array(
                    threat_array, raster_path, nodata_val=-1)

        # create threats table for the test
        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif,'
                'threat_2_f.tif\n')

        validate_result = habitat_quality.validate(args)
        expected = [(['lulc_cur_path'], validation.MESSAGES['INVALID_PROJECTION'])]
        self.assertEqual(validate_result, expected)

    def test_habitat_quality_argspec_missing_threat_header(self):
        """Habitat Quality: test validate for a threat header."""
        from natcap.invest import habitat_quality, validation

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], threat_values=[1, 1],
            dtype=numpy.int8, gdal_type=gdal.GDT_Int32)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,threat_1_cur.tif,threat_1_c.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,threat_2_c.tif,threat_2_f.tif\n')

        validate_result = habitat_quality.validate(args, limit_to=None)
        expected = [(
            ['threats_table_path'],
            validation.MESSAGES['MATCHED_NO_HEADERS'].format(
                header='column', header_name='decay'))]
        self.assertEqual(validate_result, expected)

    def test_habitat_quality_validate_missing_base_column(self):
        """Habitat Quality: test validate for a missing base column."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], threat_values=[1, 1],
            dtype=numpy.int8, gdal_type=gdal.GDT_Int32)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,CUR_PATH,FUT_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,threat_1_c.tif,threat_1_f.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,threat_2_c.tif,'
                'threat_2_f.tif\n')

        validate_result = habitat_quality.validate(args, limit_to=None)
        expected = [(
            ['threats_table_path'],
            habitat_quality.MISSING_COLUMN_MSG.format(column_name='base_path')
        )]
        self.assertEqual(validate_result, expected)

    def test_habitat_quality_validate_missing_fut_column(self):
        """Habitat Quality: test validate for a missing fut column."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': '',
            'workspace_dir': self.workspace_dir,
            'n_workers': -1,
        }

        args['access_vector_path'] = os.path.join(
            args['workspace_dir'], 'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            lulc_array = numpy.ones((100, 100), dtype=numpy.int8)
            lulc_array[50:, :] = lulc_val
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_raster_from_array(
                lulc_array, args['lulc' + scenario + 'path'])

        args['sensitivity_table_path'] = os.path.join(
            args['workspace_dir'], 'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        make_threats_raster(
            args['workspace_dir'], threat_values=[1, 1],
            dtype=numpy.int8, gdal_type=gdal.GDT_Int32)

        args['threats_table_path'] = os.path.join(
            args['workspace_dir'], 'threats_samp.csv')

        with open(args['threats_table_path'], 'w') as open_table:
            open_table.write(
                'MAX_DIST,WEIGHT,THREAT,DECAY,BASE_PATH,CUR_PATH\n')
            open_table.write(
                '0.04,0.7,threat_1,linear,,threat_1_c.tif\n')
            open_table.write(
                '0.07,1.0,threat_2,exponential,,threat_2_c.tif')

        validate_result = habitat_quality.validate(args, limit_to=None)
        expected = [(
            ['threats_table_path'],
            habitat_quality.MISSING_COLUMN_MSG.format(column_name='fut_path'))]
        self.assertEqual(validate_result, expected)
