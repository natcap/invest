"""Module for Regression and Unit Testing for the InVEST HRA model."""
import os
import shutil
import tempfile
import re
import unittest

import numpy
from osgeo import gdal, ogr, osr
import pygeoprocessing.testing
import pygeoprocessing
from shapely.geometry import Point, LineString

TEST_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'hra')

# Location to start drawing shp and tiff files
ORIGIN = (1180000.0, 690000.0)
# Spatial reference UTM Zone 10N for synthetic rasters and vectors
EPSG_CODE = 26910


def _make_simple_vector(target_vector_path, projected=True):
    """Make a 10x10 ogr rectangular geometry shapefile.

    Parameters:
        target_vector_path (str): path to the output shapefile.

        projected (bool): if true, define projection information for the vector
            based on an ESPG code.

    Returns:
        None.

    """
    # Create a new shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    vector = driver.CreateDataSource(target_vector_path)
    srs = osr.SpatialReference()
    if projected:
        srs.ImportFromEPSG(EPSG_CODE)
    layer = vector.CreateLayer('layer', srs, ogr.wkbPolygon)

    # Add an FID field to the layer
    field_name = 'FID'
    field = ogr.FieldDefn(field_name)
    layer.CreateField(field)

    # Create a rectangular geometry
    lon, lat = ORIGIN[0], ORIGIN[1]
    width = 10
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
    vector = None


def _make_rating_vector(target_vector_path):
    """Make a 10x10 ogr rectangular geometry shapefile with `rating` field.

    Parameters:
        target_vector_path (str): path to the output shapefile.

    Returns:
        None.

    """
    width = 10

    # Create a new shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    vector = driver.CreateDataSource(target_vector_path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG_CODE)  # Spatial reference UTM Zone 10N
    layer = vector.CreateLayer('layer', srs, ogr.wkbPolygon)

    # Add an FID field to the layer
    field_name = 'rating'
    field = ogr.FieldDefn(field_name)
    layer.CreateField(field)

    # Create a left rectangle with rating of 1 and a right rectangle with a
    # rating of 3
    lon, lat = ORIGIN[0], ORIGIN[1]
    for rating in [1, 3]:
        rect = ogr.Geometry(ogr.wkbLinearRing)
        rect.AddPoint(lon, lat)
        rect.AddPoint(lon + width/2, lat)
        rect.AddPoint(lon + width/2, lat - width)
        rect.AddPoint(lon, lat - width)
        rect.AddPoint(lon, lat)

        # Create the feature from the geometry
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(rect)
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField(field_name, rating)
        feature.SetGeometry(poly)
        layer.CreateFeature(feature)

        # Shift the origin to the right by half width
        lon += width/2

    feature = None
    vector = None


def _make_aoi_vector(target_vector_path, projected=True, subregion_field=True):
    """Make a 20x20 ogr rectangular geometry shapefile with `rating` field.

    Parameters:
        target_vector_path (str): path to the output shapefile.

        projected (bool): if true, define projection information for the vector
            based on an ESPG code.

        subregion_field (bool): if true, create a field called `name` in the
            layer, which represents subregions.

    Returns:
        None.

    """
    # Create a new shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    vector = driver.CreateDataSource(target_vector_path)

    srs = osr.SpatialReference()
    if projected:
        srs.ImportFromEPSG(EPSG_CODE)  # Spatial reference UTM Zone 10N
    layer = vector.CreateLayer('layer', srs, ogr.wkbPolygon)

    # Use `name` as field name to represent subregion field
    field_name = 'name' if subregion_field else 'random'

    field = ogr.FieldDefn(field_name)
    layer.CreateField(field)

    # Create three rectangles, each shift to the right by half width
    width = 20
    # Shift the origin to the left and to the top by width/4
    lon, lat = ORIGIN[0]-width/4, ORIGIN[1]+width/4
    for region in ["region A", "region B", "region C"]:
        rect = ogr.Geometry(ogr.wkbLinearRing)
        rect.AddPoint(lon, lat)
        rect.AddPoint(lon + width/2, lat)
        rect.AddPoint(lon + width/2, lat - width)
        rect.AddPoint(lon, lat - width)
        rect.AddPoint(lon, lat)

        # Create the feature from the geometry
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(rect)
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField(field_name, region)
        feature.SetGeometry(poly)
        layer.CreateFeature(feature)

        # Shift the origin to the right by half width
        lon += width/2

    feature = None
    vector = None


def _make_raster_from_array(base_array, target_raster_path, projected=True):
    """Make a raster from an array on a designated path.

    Parameters:
        array (numpy.ndarray): the 2D array for making the raster.

        raster_path (str): path to the output raster.

        projected (bool): if true, define projection information for the raster
            based on an ESPG code.

    Returns:
        None.

    """
    srs = osr.SpatialReference()
    if projected:
        srs.ImportFromEPSG(EPSG_CODE)  # UTM Zone 10N, unit = meter
    project_wkt = srs.ExportToWkt()

    pygeoprocessing.testing.create_raster_on_disk(
        band_matrices=[base_array],
        origin=ORIGIN,
        projection_wkt=project_wkt,
        nodata=-1,
        pixel_size=(1, -1),
        filename=target_raster_path)


def _make_info_csv(info_table_path, workspace_dir, missing_columns=False,
                   wrong_layer_type=False, wrong_buffer_value=False,
                   projected=True, rel_path=False):
    """Make a synthesized information csv on the designated path.

    Parameters:
        info_table_path (str): path to the csv with information on habitats and
            stressors.

        workspace_dir (str): path to the workspace for creating file paths.

        missing_columns (bool): if true, write wrong column headers to the CSV.

        wrong_layer_type (bool): if true, write a type different from `habitat` or
            `stressor`.

        wrong_buffer_value (bool): if true, write a string to the buffer column

        projected (bool): if true, define projection information when creating
            vectors and rasters.

        rel_path (bool): if true, write relative raster and vector paths to
            the table. File locations are relative to the folder of the table.

    Returns:
        None.

    """
    # Make a Shapefile and a GeoTIFF file for each layer and write to the table
    with open(info_table_path, 'w') as table:
        if missing_columns:
            table.write('missing habitat,PATH,TYPE,"missing buffer"\n')
        else:
            table.write('NAME,PATH,TYPE,"STRESSOR BUFFER (meters)"\n')

        if wrong_layer_type:
            layer_types = ['habitat', 'wrong type']
        else:
            layer_types = ['habitat', 'stressor']

        for layer_type in layer_types:
            # Create a shapefile for habitat_0 and stressor_0
            abs_vector_path = os.path.join(
                workspace_dir, layer_type + '_0') + '.shp'

            if projected:
                _make_simple_vector(abs_vector_path, projected=True)
            else:
                _make_simple_vector(abs_vector_path, projected=False)

            # Write relative path to the table
            if rel_path:
                rel_vector_path = os.path.relpath(
                    abs_vector_path, workspace_dir)
                # Write the information about the shapefile layer to the table
                table.write(
                    layer_type + '_0,' + rel_vector_path + ',' + layer_type)
            else:
                table.write(
                    layer_type + '_0,' + abs_vector_path + ',' + layer_type)

            # Add buffer of 3 meters to stressor_0
            if layer_type == 'stressor':
                # Write a string buffer value to the cell for error test
                if wrong_buffer_value:
                    table.write(',"wrong buffer"')
                else:
                    table.write(',3')

            # Create a tiff for habitat_1 and stressor_1
            size = 5
            array = numpy.zeros((size * 2, size * 2), dtype=numpy.int8)
            array[size:, :] = 1
            abs_raster_path = os.path.join(
                workspace_dir, layer_type + '_1') + '.tif'
            if projected:
                _make_raster_from_array(array, abs_raster_path, projected=True)
            else:
                _make_raster_from_array(array, abs_raster_path, projected=False)

            if rel_path:
                rel_raster_path = os.path.relpath(
                    abs_raster_path, workspace_dir)
                # Write relevant info to the table
                table.write(
                    '\n' + layer_type + '_1,' + rel_raster_path + ',' +
                    layer_type)
            else:
                table.write(
                    '\n' + layer_type + '_1,' + abs_raster_path + ',' +
                    layer_type)

            # Add buffer of 5 meters to stressor_1
            if layer_type == 'stressor':
                table.write(',5')
            table.write('\n')


def _make_criteria_csv(
        criteria_table_path, workspace_dir, missing_criteria=False,
        missing_index=False, missing_layer_names=False,
        missing_criteria_header=False, unknown_criteria=False,
        wrong_criteria_type=False, wrong_weight=False, large_rating=False,
        rel_path=False):
    """Make a synthesized information CSV on the designated path.

    Parameters:

        info_table_path (str): path to the CSV or Excel file with information
            on habitats and stressors.

        workspace_dir (str): path to the folder for saving spatially explicit
            criteria files.

        missing_criteria (bool): if true, let stressor_1 only have C criteria
            so that E criteria is missing.

        missing_index (bool): if true, remove `HABITAT NAME` and `HABITAT
            STRESSOR OVERLAP PROPERTIES` from the CSV file.

        missing_layer_names (bool): if true, rename `habitat_0` to `habitat`
            and `stressor_1` to `stressor` to cause unmatched names between
            criteria and info CSVs.

        missing_criteria_header (bool): if true, remove the column header
            `CRITERIA TYPE` from the CSV file.

        unknown_criteria (bool): if true, add a criteria row that belongs to
            no stressors.

        wrong_criteria_type (bool): if true, provide a criteria type that's not
            either C or E.

        wrong_weight (bool): if true, provide a weight score that's not a
            number.

        rel_path (bool): if true, write relative raster and vector paths to
            the table. File locations are relative to the folder of the table.

    Returns:
        None

    """
    # Create spatially explicit criteria raster and vector files in workspace.
    # Make a rating raster file on criteria 1 of habitat_0
    abs_rating_raster_path = os.path.join(workspace_dir, 'hab_0_crit_1.tif')
    size = 5
    array = numpy.full((size * 2, size * 2), 2, dtype=numpy.int8)
    array[size:, :] = 3
    _make_raster_from_array(array, abs_rating_raster_path)

    # Make a rating shapefile on criteria 3 of habitat_1
    abs_rating_vector_path = os.path.join(workspace_dir, 'hab_1_crit_3.shp')
    _make_rating_vector(abs_rating_vector_path)

    with open(criteria_table_path, 'w') as table:
        if missing_index:
            table.write(
                '"missing index",habitat_0,,,habitat_1,,,"CRITERIA TYPE",\n')
        elif missing_criteria_header:
            table.write(
                '"HABITAT NAME",habitat_0,,,habitat_1,,,"missing type",\n')
        elif missing_layer_names:
            table.write(
                '"HABITAT NAME",habitat,,,habitat_1,,,"CRITERIA TYPE",\n')
        else:
            table.write(
                '"HABITAT NAME",habitat_0,,,habitat_1,,,"CRITERIA TYPE",\n')
        table.write('"HABITAT RESILIENCE ATTRIBUTES",Rating,DQ,Weight,Rating,'
                    'DQ,Weight,E/C\n')

        # Write relative path to the table
        if rel_path:
            rel_rating_raster_path = os.path.relpath(
                abs_rating_raster_path, workspace_dir)
            table.write('"criteria 1",'+rel_rating_raster_path+',2,2,3,2,2,C\n')
        else:
            table.write('"criteria 1",'+abs_rating_raster_path+',2,2,3,2,2,C\n')
        table.write('"criteria 2",0,2,2,1,2,2,C\n')

        if missing_index:
            table.write('missing index\n')
        else:
            table.write('HABITAT STRESSOR OVERLAP PROPERTIES\n')

        if unknown_criteria:
            table.write('"extra criteria",1,2,2,0,2,2,E\n')
        table.write('stressor_0,Rating,DQ,Weight,Rating,DQ,Weight,E/C\n')

        if rel_path:
            rel_rating_vector_path = os.path.relpath(
                abs_rating_vector_path, workspace_dir)
            table.write('"criteria 3",2,2,2,'+rel_rating_vector_path+',2,2,C\n')
        else:
            table.write('"criteria 3",2,2,2,'+abs_rating_vector_path+',2,2,C\n')
        table.write('"criteria 4",1,2,2,0,2,2,E\n')

        if missing_layer_names:
            table.write('stressor,Rating,DQ,Weight,Rating,DQ,Weight,E/C\n')
        else:
            table.write('stressor_1,Rating,DQ,Weight,Rating,DQ,Weight,E/C\n')
        table.write('"criteria 5",3,2,2,3,2,2,C\n')

        if missing_criteria:
            # Only write C criteria for stressor_1 to test exception
            table.write('"criteria 6",3,2,2,3,2,2,C\n')
        elif wrong_criteria_type:
            # Produce a wrong criteria type "A"
            table.write('"criteria 6",3,2,2,3,2,2,A\n')
        elif wrong_weight:
            # Produce a wrong weight score
            table.write('"criteria 6",3,2,nan,3,2,2,E\n')
        elif large_rating:
            # Make a large rating score
            table.write('"criteria 6",99999,2,2,3,2,2,E\n')
        else:
            table.write('"criteria 6",3,2,2,3,2,2,E\n')


class HraUnitTests(unittest.TestCase):
    """Unit tests for the Wind Energy module."""

    def setUp(self):
        """Overriding setUp function to create temporary workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_missing_criteria_header(self):
        """HRA: exception raised when missing criteria from criteria CSV."""
        from natcap.invest.hra import _get_criteria_dataframe, _get_overlap_dataframe

        # Create a criteria CSV that misses a criteria type
        bad_criteria_table_path = os.path.join(
            self.workspace_dir, 'bad_criteria.csv')
        _make_criteria_csv(
            bad_criteria_table_path, self.workspace_dir, missing_criteria=True)

        with self.assertRaises(ValueError) as cm:
            criteria_df = _get_criteria_dataframe(bad_criteria_table_path)
            _get_overlap_dataframe(
                criteria_df, ['habitat_0', 'habitat_1'],
                {'stressor_0': ['criteria 3', 'criteria 4'],
                 'stressor_1': ['criteria 5', 'criteria 6']},
                3, self.workspace_dir, self.workspace_dir, '')

        expected_message = 'The following stressor-habitat pair(s)'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_unknown_criteria_from_criteria_csv(self):
        """HRA: exception raised with unknown criteria from criteria CSV."""
        from natcap.invest.hra import _get_criteria_dataframe, _get_attributes_from_df

        # Create a criteria CSV that has a criteria row that shows up before
        # any stressors
        bad_criteria_table_path = os.path.join(
            self.workspace_dir, 'bad_criteria.csv')
        _make_criteria_csv(
            bad_criteria_table_path, self.workspace_dir, unknown_criteria=True)

        with self.assertRaises(ValueError) as cm:
            criteria_df = _get_criteria_dataframe(bad_criteria_table_path)
            _get_attributes_from_df(criteria_df, ['habitat_0', 'habitat_1'],
                                    ['stressor_0', 'stressor_1'])

        expected_message = 'The "extra criteria" criteria does not belong to '
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_missing_index_from_criteria_csv(self):
        """HRA: correct error message when missing indexes from criteria CSV."""
        from natcap.invest.hra import _get_criteria_dataframe

        # Use a criteria CSV that misses two indexes
        bad_criteria_table_path = os.path.join(
            self.workspace_dir, 'bad_criteria.csv')
        _make_criteria_csv(
            bad_criteria_table_path, self.workspace_dir, missing_index=True)

        with self.assertRaises(ValueError) as cm:
            _get_criteria_dataframe(bad_criteria_table_path)

        expected_message = (
            'The Criteria table is missing the following '
            'value(s) in the first column:')
        actual_message = str(cm.exception)
        self.assertTrue(
            expected_message in actual_message, actual_message)

    def test_missing_criteria_header_from_criteria_csv(self):
        """HRA: correct error message when missing indexes from criteria CSV."""
        from natcap.invest.hra import _get_criteria_dataframe

        # Use a criteria CSV that misses two indexes
        bad_criteria_table_path = os.path.join(
            self.workspace_dir, 'bad_criteria.csv')
        _make_criteria_csv(
            bad_criteria_table_path, self.workspace_dir,
            missing_criteria_header=True)

        with self.assertRaises(ValueError) as cm:
            _get_criteria_dataframe(bad_criteria_table_path)

        expected_message = 'missing the column header "CRITERIA TYPE"'
        actual_message = str(cm.exception)
        self.assertTrue(
            expected_message in actual_message, actual_message)

    def test_criteria_excel_file(self):
        """HRA: test excel files read correctly by _get_criteria_dataframe."""
        from natcap.invest.hra import _get_criteria_dataframe

        # Make an info CSV file and read it as a dataframe
        criteria_csv_path = os.path.join(self.workspace_dir, 'criteria.csv')
        _make_criteria_csv(
            criteria_csv_path, self.workspace_dir, rel_path=True)
        expected_df = _get_criteria_dataframe(criteria_csv_path).astype(str)

        # Since we don't have openpyxl library, use the existing excel file
        # from TEST_DATA folder, and copy it to self.workspace_dir so
        # the function won't raise exceptions about vector or raster files
        # in the table not existing
        criteria_excel_path = os.path.join(TEST_DATA, 'criteria_excel.xlsx')
        copied_criteria_excel_path = os.path.join(
            self.workspace_dir, 'criteria_excel.xlsx')
        shutil.copyfile(criteria_excel_path, copied_criteria_excel_path)
        out_df = _get_criteria_dataframe(copied_criteria_excel_path).astype(str)

        self.assertTrue(
            out_df.equals(expected_df),
            'The dataframes from criteria CSV and excel files are different.')

    def test_info_excel_file(self):
        """HRA: test excel files read correctly by _get_info_dataframe."""
        from natcap.invest.hra import _get_info_dataframe

        # Make an info CSV file and read it as a dataframe
        info_csv_path = os.path.join(self.workspace_dir, 'info.csv')
        _make_info_csv(
            info_csv_path, workspace_dir=self.workspace_dir, rel_path=True)
        expected_df = _get_info_dataframe(
            info_csv_path, self.workspace_dir, self.workspace_dir,
            self.workspace_dir, '')[0].astype(str)

        # Since we don't have openpyxl library, use the existing excel file
        # from TEST_DATA folder, and copy it to self.workspace_dir so
        # the function won't raise exceptions about vector or raster files
        # in the table not existing
        info_excel_path = os.path.join(TEST_DATA, 'info_excel.xlsx')
        copied_info_excel_path = os.path.join(
            self.workspace_dir, 'info_excel.xlsx')
        shutil.copyfile(info_excel_path, copied_info_excel_path)
        out_df = _get_info_dataframe(
            copied_info_excel_path, self.workspace_dir, self.workspace_dir,
            self.workspace_dir, '')[0].astype(str)

        self.assertTrue(
            out_df.equals(expected_df),
            'The dataframes read from info CSV and excel files are different.')

    def test_missing_columns_from_info_csv(self):
        """HRA: exception raised when columns are missing from info CSV."""
        from natcap.invest.hra import _get_info_dataframe

        # Test missing columns from info CSV
        bad_info_table_path = os.path.join(
            self.workspace_dir, 'bad_criteria.csv')
        _make_info_csv(
            bad_info_table_path, workspace_dir=self.workspace_dir,
            missing_columns=True)

        with self.assertRaises(ValueError) as cm:
            _get_info_dataframe(
                bad_info_table_path, self.workspace_dir, self.workspace_dir,
                self.workspace_dir, '')

        expected_message = 'Missing column header(s) from the Info CSV file:'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_wrong_layer_type_in_info_csv(self):
        """HRA: exception raised when layer type is wrong info CSV."""
        from natcap.invest.hra import _get_info_dataframe

        # Test missing columns from info CSV
        bad_info_table_path = os.path.join(
            self.workspace_dir, 'bad_criteria.csv')
        _make_info_csv(
            bad_info_table_path, workspace_dir=self.workspace_dir,
            wrong_layer_type=True)

        with self.assertRaises(ValueError) as cm:
            _get_info_dataframe(
                bad_info_table_path, self.workspace_dir, self.workspace_dir,
                self.workspace_dir, '')

        expected_message = "is having ['wrong type']"
        actual_message = str(cm.exception)
        self.assertTrue(
            expected_message in actual_message, actual_message)

    def test_wrong_buffer_in_info_csv(self):
        """HRA: exception raised when buffers are not number in info CSV."""
        from natcap.invest.hra import _get_info_dataframe

        # Test missing columns from info CSV
        bad_info_table_path = os.path.join(
            self.workspace_dir, 'bad_criteria.csv')
        _make_info_csv(
            bad_info_table_path, workspace_dir=self.workspace_dir,
            wrong_buffer_value=True)

        with self.assertRaises(ValueError) as cm:
            _get_info_dataframe(
                bad_info_table_path, self.workspace_dir, self.workspace_dir,
                self.workspace_dir, '')

        expected_message = "should be a number for stressors"
        actual_message = str(cm.exception)
        self.assertTrue(
            expected_message in actual_message, actual_message)

    def test_wrong_criteria_type_type(self):
        """HRA: exception raised when type is not C or E from criteria CSV."""
        from natcap.invest.hra import _get_criteria_dataframe, _get_overlap_dataframe

        # Use a criteria CSV that's missing a criteria type
        bad_criteria_table_path = os.path.join(
            self.workspace_dir, 'bad_criteria.csv')
        _make_criteria_csv(
            bad_criteria_table_path, self.workspace_dir,
            wrong_criteria_type=True)

        with self.assertRaises(ValueError) as cm:
            criteria_df = _get_criteria_dataframe(bad_criteria_table_path)
            _get_overlap_dataframe(
                criteria_df, ['habitat_0', 'habitat_1'],
                {'stressor_0': ['criteria 3', 'criteria 4'],
                 'stressor_1': ['criteria 5', 'criteria 6']},
                3, self.workspace_dir, self.workspace_dir, '')

        expected_message = 'Criteria Type in the criteria scores table'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_wrong_weight_from_criteria_csv(self):
        """HRA: exception raised when weight is not a number from CSV."""
        from natcap.invest.hra import _get_criteria_dataframe, _get_overlap_dataframe

        # Use a criteria CSV that's missing a criteria type
        bad_criteria_table_path = os.path.join(
            self.workspace_dir, 'bad_criteria.csv')
        _make_criteria_csv(
            bad_criteria_table_path, self.workspace_dir, wrong_weight=True)

        with self.assertRaises(ValueError) as cm:
            criteria_df = _get_criteria_dataframe(bad_criteria_table_path)
            _get_overlap_dataframe(
                criteria_df, ['habitat_0', 'habitat_1'],
                {'stressor_0': ['criteria 3', 'criteria 4'],
                 'stressor_1': ['criteria 5', 'criteria 6']},
                3, self.workspace_dir, self.workspace_dir, '')

        expected_message = (
            'Weight column for habitat "habitat_0" and stressor "stressor_1"')
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_large_rating_from_criteria_csv(self):
        """HRA: exception raised when rating is larger than maximum rating."""
        from natcap.invest.hra import _get_criteria_dataframe, _get_overlap_dataframe

        # Use a criteria CSV that's missing a criteria type
        bad_criteria_table_path = os.path.join(
            self.workspace_dir, 'bad_criteria.csv')
        _make_criteria_csv(
            bad_criteria_table_path, self.workspace_dir, large_rating=True)

        with self.assertRaises(ValueError) as cm:
            criteria_df = _get_criteria_dataframe(bad_criteria_table_path)
            _get_overlap_dataframe(
                criteria_df, ['habitat_0', 'habitat_1'],
                {'stressor_0': ['criteria 3', 'criteria 4'],
                 'stressor_1': ['criteria 5', 'criteria 6']},
                3, self.workspace_dir, self.workspace_dir, '')

        expected_message = 'rating 99999.0 larger than the maximum rating 3'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_to_abspath(self):
        """HRA: test exception raised in _to_abspath function."""
        from natcap.invest.hra import _to_abspath

        with self.assertRaises(ValueError) as cm:
            _to_abspath('non_exist_raster.tif', self.workspace_dir)

        expected_message = 'does not exist.'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_simplify_geometry(self):
        """HRA: test _simplify_geometry function."""
        from natcap.invest.hra import _simplify_geometry

        complicated_vector_path = os.path.join(
            TEST_DATA, 'complicated_vector.gpkg')
        expected_simplified_vector_path = os.path.join(
            TEST_DATA, 'simplified_vector.gpkg')
        target_simplified_vector_path = os.path.join(
            self.workspace_dir, 'simplified_vector.gpkg')
        # Create an existing target vector to test if it's properly removed
        open(target_simplified_vector_path, 'a').close()

        tolerance = 3000  # in meters
        _simplify_geometry(
            complicated_vector_path, tolerance, target_simplified_vector_path)

        pygeoprocessing.testing.assert_vectors_equal(
            target_simplified_vector_path, expected_simplified_vector_path,
            1E-6)

    def test_simplify_geometry_points(self):
        """HRA: test _simplify_geometry does not alter geometry given points."""
        from natcap.invest.hra import _simplify_geometry

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(EPSG_CODE)
        base_points_path = os.path.join(self.workspace_dir, 'base_points.gpkg')
        points = [Point(0.0, 0.0), Point(10.0, 10.0)]
        pygeoprocessing.testing.sampledata.create_vector_on_disk(
            points, srs.ExportToWkt(),
            filename=base_points_path, vector_format='GPKG')

        target_simplified_vector_path = os.path.join(
            self.workspace_dir, 'simplified_vector.gpkg')

        tolerance = 3000  # in meters
        _simplify_geometry(
            base_points_path, tolerance, target_simplified_vector_path)

        pygeoprocessing.testing.assert_vectors_equal(
            target_simplified_vector_path, base_points_path,
            1E-6)

    def test_simplify_geometry_lines(self):
        """HRA: test _simplify_geometry does not alter geometry given lines."""
        from natcap.invest.hra import _simplify_geometry

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(EPSG_CODE)
        base_lines_path = os.path.join(self.workspace_dir, 'base_lines.gpkg')
        lines = [LineString([(0.0, 0.0), (10.0, 10.0)])]
        pygeoprocessing.testing.sampledata.create_vector_on_disk(
            lines, srs.ExportToWkt(),
            filename=base_lines_path, vector_format='GPKG')

        target_simplified_vector_path = os.path.join(
            self.workspace_dir, 'simplified_vector.gpkg')

        tolerance = 3000  # in meters
        _simplify_geometry(
            base_lines_path, tolerance, target_simplified_vector_path)

        pygeoprocessing.testing.assert_vectors_equal(
            target_simplified_vector_path, base_lines_path,
            1E-6)


class HraRegressionTests(unittest.TestCase):
    """Tests for the Pollination model."""

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
        """Generate args dict that is consistent across all regression tests."""
        args = {
            'workspace_dir': workspace_dir,
            'results_suffix': '',
            'info_table_path': os.path.join(workspace_dir, 'info.csv'),
            'criteria_table_path': os.path.join(workspace_dir, 'criteria.csv'),
            'max_rating': 3,
            'risk_eq': 'Euclidean',
            'decay_eq': 'Linear',
            'aoi_vector_path': os.path.join(workspace_dir, 'aoi.shp'),
            'resolution': 1,
            'n_workers': -1,
            'visualize_outputs': True,
        }

        return args

    def test_hra_regression_euclidean_linear(self):
        """HRA: regression testing synthetic data with linear, euclidean eqn."""
        import natcap.invest.hra

        args = HraRegressionTests.generate_base_args(self.workspace_dir)
        # Also test on GeoJSON outputs for visualization
        args['visualize_outputs'] = True

        # Also test relative file paths in Info CSV file
        _make_info_csv(
            args['info_table_path'], args['workspace_dir'], rel_path=True)
        _make_criteria_csv(args['criteria_table_path'], args['workspace_dir'])
        _make_aoi_vector(args['aoi_vector_path'])
        args['n_workers'] = ''  # tests empty string for `n_workers`

        natcap.invest.hra.execute(args)

        output_rasters = [
            'TOTAL_RISK_habitat_0', 'TOTAL_RISK_habitat_1',
            'TOTAL_RISK_Ecosystem', 'RECLASS_RISK_habitat_0',
            'RECLASS_RISK_habitat_1', 'RECLASS_RISK_Ecosystem']

        output_vectors = [
            'RECLASS_RISK_habitat_0', 'RECLASS_RISK_habitat_1',
            'RECLASS_RISK_Ecosystem', 'STRESSOR_stressor_0',
            'STRESSOR_stressor_1']

        # Assert rasters are equal
        output_raster_paths = [
            os.path.join(args['workspace_dir'], 'outputs', raster_name + '.tif')
            for raster_name in output_rasters]
        expected_raster_paths = [os.path.join(
            TEST_DATA, raster_name + '_euc_lin.tif') for raster_name in
            output_rasters]

        # Append a intermediate raster to test the linear decay equation
        output_raster_paths.append(
            os.path.join(args['workspace_dir'], 'intermediate_outputs',
                         'C_habitat_0_stressor_1.tif'))
        expected_raster_paths.append(
            os.path.join(TEST_DATA, 'C_habitat_0_stressor_1_euc_lin.tif'))

        for output_raster, expected_raster in zip(
                output_raster_paths, expected_raster_paths):
            pygeoprocessing.testing.assert_rasters_equal(
                output_raster, expected_raster)

        # Assert GeoJSON vectors are equal
        output_vector_paths = [os.path.join(
            args['workspace_dir'], 'visualization_outputs',
            vector_name + '.geojson') for vector_name in output_vectors]
        expected_vector_paths = [
            os.path.join(TEST_DATA, vector_name + '_euc_lin.geojson') for
            vector_name in output_vectors]

        for output_vector, expected_vector in zip(
                output_vector_paths, expected_vector_paths):
            HraRegressionTests._assert_vectors_equal(
                output_vector, expected_vector, precision=6)

        # Assert summary statistics CSV equal
        output_csv_path = os.path.join(
            args['workspace_dir'], 'outputs', 'SUMMARY_STATISTICS.csv')
        expected_csv_path = os.path.join(
            TEST_DATA, 'SUMMARY_STATISTICS_euc_lin.csv')
        pygeoprocessing.testing.assert_csv_equal(
            output_csv_path, expected_csv_path, rel_tol=1E-6)

    def test_hra_no_subregion_multiplicative_exponential(self):
        """HRA: regression testing with exponential, multiplicative eqn."""
        import natcap.invest.hra

        args = HraRegressionTests.generate_base_args(self.workspace_dir)
        _make_info_csv(args['info_table_path'], self.workspace_dir)
        _make_criteria_csv(args['criteria_table_path'], self.workspace_dir)
        _make_aoi_vector(args['aoi_vector_path'])
        args['risk_eq'] = 'Multiplicative'
        args['decay_eq'] = 'Exponential'
        args['resolution'] = 1

        aoi_vector_path = os.path.join(
            self.workspace_dir, 'no_subregion_aoi.shp')
        # Test if `Total Region` gets written in output stats CSV
        _make_aoi_vector(aoi_vector_path, subregion_field=False)
        args['aoi_vector_path'] = aoi_vector_path

        natcap.invest.hra.execute(args)

        output_rasters = [
            'TOTAL_RISK_habitat_0', 'TOTAL_RISK_habitat_1',
            'TOTAL_RISK_Ecosystem', 'RECLASS_RISK_habitat_0',
            'RECLASS_RISK_habitat_1', 'RECLASS_RISK_Ecosystem']

        # Assert rasters are equal
        output_raster_paths = [
            os.path.join(self.workspace_dir, 'outputs', raster_name + '.tif')
            for raster_name in output_rasters]
        expected_raster_paths = [os.path.join(
            TEST_DATA, raster_name + '_mul_exp.tif') for raster_name in
            output_rasters]

        # Append a intermediate raster to test the linear decay equation
        output_raster_paths.append(
            os.path.join(self.workspace_dir, 'intermediate_outputs',
                         'C_habitat_0_stressor_1.tif'))
        expected_raster_paths.append(
            os.path.join(TEST_DATA, 'C_habitat_0_stressor_1_mul_exp.tif'))

        for output_raster, expected_raster in zip(
                output_raster_paths, expected_raster_paths):
            pygeoprocessing.testing.assert_rasters_equal(
                output_raster, expected_raster)

        # Assert summary statistics CSV equal
        output_csv_path = os.path.join(
            self.workspace_dir, 'outputs', 'SUMMARY_STATISTICS.csv')
        expected_csv_path = os.path.join(
            TEST_DATA, 'SUMMARY_STATISTICS_mul_exp.csv')
        pygeoprocessing.testing.assert_csv_equal(
            output_csv_path, expected_csv_path, rel_tol=1E-6)

    def test_aoi_no_projection(self):
        """HRA: testing AOI vector without projection."""
        import natcap.invest.hra

        args = HraRegressionTests.generate_base_args(self.workspace_dir)
        _make_info_csv(args['info_table_path'], self.workspace_dir)
        _make_criteria_csv(args['criteria_table_path'], self.workspace_dir)

        # Make unprojected AOI vector
        bad_aoi_vector_path = os.path.join(
            self.workspace_dir, 'missing_projection_aoi.shp')
        _make_aoi_vector(bad_aoi_vector_path, projected=False)
        args['aoi_vector_path'] = bad_aoi_vector_path

        with self.assertRaises(ValueError) as cm:
            natcap.invest.hra.execute(args)

        expected_message = 'Dataset must have a valid projection'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_layer_no_projection(self):
        """HRA: testing habitats and stressors without projection."""
        import natcap.invest.hra

        args = HraRegressionTests.generate_base_args(self.workspace_dir)
        _make_criteria_csv(args['criteria_table_path'], self.workspace_dir)
        _make_aoi_vector(args['aoi_vector_path'])

        # Make unprojected files and write their filepaths to info csv.
        bad_info_table_path = os.path.join(self.workspace_dir, 'bad_info.csv')
        _make_info_csv(bad_info_table_path, self.workspace_dir, projected=False)
        args['info_table_path'] = bad_info_table_path

        with self.assertRaises(ValueError) as cm:
            natcap.invest.hra.execute(args)

        expected_message = 'The following layer does not have a projection'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_unmatched_layer_names(self):
        """HRA: testing unmatched layer names between info and criteria CSV."""
        import natcap.invest.hra

        args = HraRegressionTests.generate_base_args(self.workspace_dir)
        _make_info_csv(args['info_table_path'], self.workspace_dir)
        _make_aoi_vector(args['aoi_vector_path'])

        # Make habitat and stressor layer names in criteria CSV different from
        # that in info CSV.
        bad_criteria_table_path = os.path.join(
            self.workspace_dir, 'bad_criteria.csv')
        _make_criteria_csv(bad_criteria_table_path, self.workspace_dir,
                           missing_layer_names=True)
        args['criteria_table_path'] = bad_criteria_table_path

        with self.assertRaises(ValueError) as cm:
            natcap.invest.hra.execute(args)

        # Two layers that are expected to be missing from criteria CSV
        for missing_layer in ['habitat_0', 'stressor_1']:
            expected_message = (
                "missing from the criteria table: ['" + missing_layer)
            actual_message = str(cm.exception)
            self.assertTrue(expected_message in actual_message, actual_message)

    def test_invalid_args(self):
        """HRA: testing invalid arguments."""
        import natcap.invest.hra

        args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': '',
            'info_table_path': os.path.join(
                TEST_DATA, 'file_not_exist.csv'),  # invalid file path
            'criteria_table_path': os.path.join(
                TEST_DATA, 'exposure_consequence_criteria.csv'),
            'max_rating': 'not a number',  # invalid value
            'risk_eq': 'Typo',  # invalid value
            'decay_eq': 'Linear',
            'aoi_vector_path': os.path.join(
                TEST_DATA, 'file_not_exist.shp'),  # invalid file path
            'resolution': 1,
            'n_workers': -1
        }

        with self.assertRaises(ValueError) as cm:
            natcap.invest.hra.execute(args)

        expected_invalid_parameters = [
            'info_table_path', 'risk_eq', 'max_rating', 'aoi_vector_path']
        actual_message = str(cm.exception)

        for invalid_parameter in expected_invalid_parameters:
            self.assertTrue(
                invalid_parameter in actual_message, actual_message)

    def test_missing_args(self):
        """HRA: testing invalid arguments."""
        import natcap.invest.hra

        args = {
            # missing workspace_dir
            'results_suffix': '',
            'info_table_path': os.path.join(
                TEST_DATA, 'habitat_stressor_info.csv'),
            'criteria_table_path': os.path.join(
                TEST_DATA, 'exposure_consequence_criteria.csv'),
            'max_rating': 3,
            'risk_eq': 'Euclidean',
            'decay_eq': 'Linear',
            'aoi_vector_path': os.path.join(
                TEST_DATA, 'aoi.shp'),
            'resolution': 1,
            'n_workers': -1
        }

        with self.assertRaises(ValueError) as cm:
            natcap.invest.hra.execute(args)
        self.assertEquals(len(cm.exception.args), 1)

    def test_validate(self):
        """HRA: testing validation."""
        import natcap.invest.hra

        args = HraRegressionTests.generate_base_args(self.workspace_dir)
        _make_info_csv(args['info_table_path'], self.workspace_dir)
        _make_criteria_csv(args['criteria_table_path'], self.workspace_dir)
        _make_aoi_vector(args['aoi_vector_path'])

        validation_warnings = natcap.invest.hra.validate(args)
        self.assertTrue([] == validation_warnings)

    def test_validate_max_rating_value(self):
        """HRA: testing validation with max_rating less than 1 in args."""
        import natcap.invest.hra

        args = HraRegressionTests.generate_base_args(self.workspace_dir)
        args['max_rating'] = '-1'

        validation_error_list = natcap.invest.hra.validate(args)
        expected_error = (['max_rating'],
                          'Value does not meet condition value > 0')
        self.assertTrue(expected_error in validation_error_list)

    def test_validate_negative_resolution(self):
        """HRA: testing validation with negative value in resolution in args."""
        import natcap.invest.hra

        args = HraRegressionTests.generate_base_args(self.workspace_dir)
        args['resolution'] = '-110'

        validation_error_list = natcap.invest.hra.validate(args)
        expected_error = (['resolution'],
                          'Value does not meet condition value > 0')
        self.assertTrue(expected_error in validation_error_list)

    @staticmethod
    def _assert_vectors_equal(a_vector_path, b_vector_path, precision=6):
        """Assert that geometries in two vectors are equal, order-insensitive.

        Parameters:
            a_vector_path (str): a path to a valid OGR vector.
            b_vector_path (str): a path to a valid OGR vector.

        Returns:
            None.

        Raises:
            AssertionError when the two point geometries are not equal up to
                desired precision (default is 1E-6).

        """
        a_vector = gdal.OpenEx(a_vector_path, gdal.OF_VECTOR)
        a_layer = a_vector.GetLayer(0)
        a_feat = a_layer.GetNextFeature()

        b_vector = gdal.OpenEx(b_vector_path, gdal.OF_VECTOR)
        b_layer = b_vector.GetLayer(0)
        b_feat = b_layer.GetNextFeature()

        while a_feat is not None:
            # Get coordinates from geometry and store them in a list
            a_geom = a_feat.GetGeometryRef()
            a_geom_list = re.findall(r'\d+\.\d+', a_geom.ExportToWkt())
            a_geom_list = [float(x) for x in a_geom_list]

            b_geom = b_feat.GetGeometryRef()
            b_geom_list = re.findall(r'\d+\.\d+', b_geom.ExportToWkt())
            b_geom_list = [float(x) for x in b_geom_list]

            try:
                numpy.testing.assert_array_almost_equal(
                    a_geom_list, b_geom_list, decimal=precision)
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

        a_layer = None
        b_layer = None
        a_vector = None
        b_vector = None
