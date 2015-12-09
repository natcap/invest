"""Module for Regression Testing the InVEST Wind Energy module."""
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

from nose.tools import nottest
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

SAMPLE_DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(os.path.dirname(__file__), 'data', 'wind_energy')

def _create_csv(fields, data, fname):
    """Create a new CSV table from a dictionary.

    Parameters:
        fname (string): a file path for the new table to be written to disk

        fields (list): a list of the column names. The order of the fields
            in the list will be the order in how they are written. ex:
            ['id', 'precip', 'total']

        data (dictionary): a dictionary representing the table.
            The dictionary should be constructed with unique numerical keys
            that point to a dictionary which represents a row in the table:
            data = {0 : {'id':1, 'precip':43, 'total': 65},
                    1 : {'id':2, 'precip':65, 'total': 94}}

    Returns:
        Nothing
    """
    csv_file = open(fname, 'wb')

    #  Sort the keys so that the rows are written in order
    row_keys = data.keys()
    row_keys.sort()

    csv_writer = csv.DictWriter(csv_file, fields)
    #  Write the columns as the first row in the table
    csv_writer.writerow(dict((fn, fn) for fn in fields))

    # Write the rows from the dictionary
    for index in row_keys:
        csv_writer.writerow(data[index])

    csv_file.close()

def _create_vertical_csv(data, fname):
    """Create a new CSV table where the fields are in the left column.

    This CSV table is created with fields / keys running vertically
        down the first column. The second column has the corresponding
        values.

    Parameters:
        data (Dictionary): a Dictionary where each key is the name
            of a field and set in the first column. The second
            column is set with the value of that key.

        fname (string): a file path for the new table to be written to disk

    Returns:
        Nothing
    """

    csv_file = open(fname, 'wb')

    writer = csv.writer(csv_file)
    for key, val in data.iteritems():
        writer.writerow([key, val])

    csv_file.close()


def _csv_wind_data_to_binary(wind_data_file_uri, binary_file_uri):
    """Convert and compress the wind data into binary format,
        packing in a specific manner such that the InVEST3.0 wind energy
        model can properly unpack it

        wind_data_file_uri - a URI to a CSV file with the formatted wind data
            Data should have the following order of columns:
            ["Latitude","Longitude","Ram-010m","Ram-020m","Ram-030m","Ram-040m",
            "Ram-050m","Ram-060m","Ram-070m","Ram-080m","Ram-090m","Ram-100m",
            "Ram-110m","Ram-120m","Ram-130m","Ram-140m","Ram-150m","K"]
            (required)

        binary_file_uri - a URI to write out the binary file (.bin) (required)

        returns - Nothing"""

    # Open the wave watch three files
    wind_file = open(wind_data_file_uri,'rU')
    # Open the binary output file as writeable
    bin_file = open(binary_file_uri, 'wb')

    # This is the expected column header list for the binary wind energy file.
    # It is expected that the data will be in this order so that we can properly
    # unpack the information into a dictionary
    # ["LONG","LATI","Ram-010m","Ram-020m","Ram-030m","Ram-040m",
    #  "Ram-050m","Ram-060m","Ram-070m","Ram-080m","Ram-090m","Ram-100m",
    #  "Ram-110m","Ram-120m","Ram-130m","Ram-140m","Ram-150m","K-010m"]

    #burn header line
    header_line = wind_file.readline()

    while True:
        # Read each line of the csv file for wind data
        line = wind_file.readline()

        # If end of file, break out
        if len(line) == 0:
            break

        # Get the data values as floats
        float_list = map(float,line.split(','))
        # Swap long / lat values
        float_list[0], float_list[1] = float_list[1], float_list[0]
        # Pack up the data values as float types
        s=struct.pack('f'*len(float_list), *float_list)
        bin_file.write(s)

class WindEnergyUnitTests(unittest.TestCase):
    """Unit tests for the Wind Energy module."""

    def setUp(self):
        """Overriding setUp function to create temporary workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    ReferenceData = collections.namedtuple('ReferenceData',
                                           'projection origin pixel_size')

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
        pixel_size=lambda x: (x, -1. * x)
    )

    SRS_UTM19 = ReferenceData(
        projection=projection_wkt(32619),
        origin=(376749.5, 4706383.2),
        pixel_size=lambda x: (x, -1. * x)
    )

    def _create_ptm(fields, attributes):
        """
        Create a point shapefile

        This point shapefile is created with the following characteristis:
            * SRS is in the SRS_WILLAMETTE.
            * Vector type is Point
            * Points are 100m apart

        Parameters:
            fields (dict or None): a python dictionary mapping string fieldname
                to a string datatype representation of the target ogr fieldtype.
                Example: {'ws_id': 'int'}.  See
                ``pygeoprocessing.testing.sampledata.VECTOR_FIELD_TYPES.keys()``
                for the complete list of all allowed types.  If None, the datatype
                will be determined automatically based on the types of the
                attribute values.
            attributes (list of dicts): a list of python dictionary mapping
                fieldname to field value.  The field value's type must match the
                type defined in the fields input.  It is an error if it doesn't.

        Returns:
            A string filepath to the vector on disk
        """
        srs = sampledata.SRS_WILLAMETTE

        pos_x = srs.origin[0]
        pos_y = srs.origin[1]

        geometries = [Point(pos_x + 50, pos_y - 50), Point(pos_x + 50, pos_y - 150)]

        return pygeoprocessing.testing.create_vector_on_disk(
                geometries, srs.projection, fields, attributes,
                vector_format='ESRI Shapefile')



    def _create_pt_vector(fields, attributes):
        """
        Create a point shapefile

        This point shapefile is created with the following characteristis:
            * SRS is in the SRS_WILLAMETTE.
            * Vector type is Point
            * Points are 100m apart

        Parameters:
            fields (dict or None): a python dictionary mapping string fieldname
                to a string datatype representation of the target ogr fieldtype.
                Example: {'ws_id': 'int'}.  See
                ``pygeoprocessing.testing.sampledata.VECTOR_FIELD_TYPES.keys()``
                for the complete list of all allowed types.  If None, the datatype
                will be determined automatically based on the types of the
                attribute values.
            attributes (list of dicts): a list of python dictionary mapping
                fieldname to field value.  The field value's type must match the
                type defined in the fields input.  It is an error if it doesn't.

        Returns:
            A string filepath to the vector on disk
        """
        srs = sampledata.SRS_WILLAMETTE

        pos_x = srs.origin[0]
        pos_y = srs.origin[1]

        geometries = [Point(pos_x, pos_y), Point(pos_x + 100, pos_y),
                      Point(pos_x, pos_y - 100), Point(pos_x + 100, pos_y - 100)]

        return pygeoprocessing.testing.create_vector_on_disk(
                geometries, srs.projection, fields, attributes,
                vector_format='ESRI Shapefile')

    def _create_polygon_utm_vector(fields, attributes):
        """
        Create a vector of 2 polygons

        This vector is created with the following characteristis:
            * SRS is in the SRS_WILLAMETTE.
            * Vector type is Polygon
            * Polygons are 100m x 50m

        Parameters:
            fields (dict or None): a python dictionary mapping string fieldname
                to a string datatype representation of the target ogr fieldtype.
                Example: {'ws_id': 'int'}.  See
                ``pygeoprocessing.testing.sampledata.VECTOR_FIELD_TYPES.keys()``
                for the complete list of all allowed types.  If None, the datatype
                will be determined automatically based on the types of the
                attribute values.
            attributes (list of dicts): a list of python dictionary mapping
                fieldname to field value.  The field value's type must match the
                type defined in the fields input.  It is an error if it doesn't.

        Returns:
            A string filepath to the vector on disk
        """
        srs = SRS_UTM19

        pos_x = srs.origin[0]
        pos_y = srs.origin[1]

        poly_geoms = {
            'poly_1': [(pos_x, pos_y), (pos_x, pos_y),
                       (pos_x + 250, pos_y - 100), (pos_x + 200, pos_y -100),
                       (pos_x + 200, pos_y)]}

        geometries = [Polygon(poly_geoms['poly_1']), Polygon(poly_geoms['poly_2'])]

        return pygeoprocessing.testing.create_vector_on_disk(
                geometries, srs.projection, fields, attributes,
                vector_format='ESRI Shapefile')



    def _create_polygon_vector(fields, attributes):
        """
        Create a vector of 2 polygons

        This vector is created with the following characteristis:
            * SRS is in the SRS_WILLAMETTE.
            * Vector type is Polygon
            * Polygons are 100m x 50m

        Parameters:
            fields (dict or None): a python dictionary mapping string fieldname
                to a string datatype representation of the target ogr fieldtype.
                Example: {'ws_id': 'int'}.  See
                ``pygeoprocessing.testing.sampledata.VECTOR_FIELD_TYPES.keys()``
                for the complete list of all allowed types.  If None, the datatype
                will be determined automatically based on the types of the
                attribute values.
            attributes (list of dicts): a list of python dictionary mapping
                fieldname to field value.  The field value's type must match the
                type defined in the fields input.  It is an error if it doesn't.

        Returns:
            A string filepath to the vector on disk
        """
        srs = sampledata.SRS_WILLAMETTE

        pos_x = srs.origin[0]
        pos_y = srs.origin[1]

        poly_geoms = {
            'poly_1': [(pos_x + 200, pos_y), (pos_x + 250, pos_y),
                       (pos_x + 250, pos_y - 100), (pos_x + 200, pos_y -100),
                       (pos_x + 200, pos_y)],
            'poly_2': [(pos_x, pos_y - 150), (pos_x + 100, pos_y - 150),
                       (pos_x + 100, pos_y - 200), (pos_x, pos_y - 200),
                       (pos_x, pos_y - 150)]}

        geometries = [Polygon(poly_geoms['poly_1']), Polygon(poly_geoms['poly_2'])]

        return pygeoprocessing.testing.create_vector_on_disk(
                geometries, srs.projection, fields, attributes,
                vector_format='ESRI Shapefile')


    def _create_latlong_raster(matrix, dtype=gdal.GDT_Int32, nodata=-1):
        """
        Create a raster for the hydropower_water_yield model.

        This raster is created with the following characteristics:
            * SRS is in the SRS_LATLONG.
            * Nodata is -1.
            * Raster type is `gdal.GDT_Int32`
            * Pixel size is 0.033333333

        Parameters:
            matrix (numpy.array): A numpy array to use as a landcover matrix.
                The output raster created will be saved with these pixel values.

        Returns:
            A string filepath to a new LULC raster on disk.
        """

        srs = SRS_LATLONG
        return pygeoprocessing.testing.create_raster_on_disk(
            [matrix], srs.origin, srs.projection, nodata,
            srs.pixel_size(0.033333), datatype=dtype)

    def _create_raster(matrix, dtype, nodata, fpath):
        """
        Create a raster for the hydropower_water_yield model.

        This raster is created with the following characteristics:
            * SRS is in the SRS_WILLAMETTE.
            * Nodata is -1.
            * Raster type is `gdal.GDT_Int32`
            * Pixel size is 30m

        Parameters:
            matrix=None (numpy.array): A numpy array to use as a landcover matrix.
                The output raster created will be saved with these pixel values.
                If None, a default matrix will be used.

        Returns:
            A string filepath to a new LULC raster on disk.
        """
        srs = sampledata.SRS_WILLAMETTE
        return pygeoprocessing.testing.create_raster_on_disk(
            [lulc_matrix], srs.origin, srs.projection, lulc_nodata,
            srs.pixel_size(100), datatype=dtype, filename=fpath)

    #@nottest
    def test_calculate_distances_land_grid(self):
        """WindEnergy: testing 'calculate_distances_land_grid' function."""
        from natcap.invest.wind_energy import wind_energy

        fields = {'id': 'real', 'L2G': 'real'}
        attrs = [{'id': 1, 'L2G': 10}, {'id': 2, 'L2G': 20}]
        srs = sampledata.SRS_WILLAMETTE
        pos_x = srs.origin[0]
        pos_y = srs.origin[1]
        geometries = [
            Point(pos_x + 50, pos_y - 50), Point(pos_x + 50, pos_y - 150)]

        temp_dir = self.workspace_dir
        shape_path = os.path.join(temp_dir, 'temp_shape.shp')
        land_shape_uri = pygeoprocessing.testing.create_vector_on_disk(
            geometries, srs.projection, fields, attrs,
            vector_format='ESRI Shapefile', filename=shape_path)

        matrix = numpy.array([[1,1,1,1],[1,1,1,1]])
        raster_path = os.path.join(temp_dir, 'temp_raster.tif')

        harvested_masked_uri = pygeoprocessing.testing.create_raster_on_disk(
            [matrix], srs.origin, srs.projection, -1, srs.pixel_size(100),
            datatype=gdal.GDT_Int32, filename=raster_path)

        tmp_dist_final_uri = os.path.join(temp_dir, 'dist_final.tif')

        wind_energy.calculate_distances_land_grid(
            land_shape_uri, harvested_masked_uri, tmp_dist_final_uri)

        #Compare
        result = gdal.Open(tmp_dist_final_uri)
        res_band = result.GetRasterBand(1)
        res_array = res_band.ReadAsArray()
        exp_array = numpy.array([[10, 110, 210, 310],[20, 120, 220, 320]])
        numpy.testing.assert_array_equal(res_array, exp_array)
        res_band = None
        result = None

    #@nottest
    def test_point_to_polygon_distance(self):
        """WindEnergy: testing 'point_to_polygon_distance' function."""
        from natcap.invest.wind_energy import wind_energy

        fields = {'vec_id': 'int'}
        attr_pt= [{'vec_id': 1}, {'vec_id': 2}, {'vec_id': 3}, {'vec_id': 4}]
        attr_poly= [{'vec_id': 1}, {'vec_id': 2}]

        srs = sampledata.SRS_WILLAMETTE

        pos_x = srs.origin[0]
        pos_y = srs.origin[1]

        poly_geoms = {
            'poly_1': [(pos_x + 200, pos_y), (pos_x + 250, pos_y),
                       (pos_x + 250, pos_y - 100), (pos_x + 200, pos_y -100),
                       (pos_x + 200, pos_y)],
            'poly_2': [(pos_x, pos_y - 150), (pos_x + 100, pos_y - 150),
                       (pos_x + 100, pos_y - 200), (pos_x, pos_y - 200),
                       (pos_x, pos_y - 150)]}

        poly_geometries = [
            Polygon(poly_geoms['poly_1']), Polygon(poly_geoms['poly_2'])]
        temp_dir = self.workspace_dir
        poly_file = os.path.join(temp_dir, 'poly_shape.shp')
        poly_ds_uri = pygeoprocessing.testing.create_vector_on_disk(
            poly_geometries, srs.projection, fields, attr_poly,
            vector_format='ESRI Shapefile', filename=poly_file)

        point_geometries = [
            Point(pos_x, pos_y), Point(pos_x + 100, pos_y),
            Point(pos_x, pos_y - 100), Point(pos_x + 100, pos_y - 100)]
        point_file = os.path.join(temp_dir, 'point_shape.shp')

        point_ds_uri = pygeoprocessing.testing.create_vector_on_disk(
            point_geometries, srs.projection, fields, attr_pt,
            vector_format='ESRI Shapefile', filename=point_file)

        results = wind_energy.point_to_polygon_distance(
            poly_ds_uri, point_ds_uri)

        exp_results = [.15, .1, .05, .05]

        for dist_a, dist_b in zip(results, exp_results):
            pygeoprocessing.testing.assert_close(dist_a, dist_b, 1e-9)

    #@nottest
    def test_add_field_to_shape_given_list(self):
        """WindEnergy: testing 'add_field_to_shape_given_list' function."""
        from natcap.invest.wind_energy import wind_energy

        fields = {'pt_id': 'int'}
        attributes= [{'pt_id': 1}, {'pt_id': 2}, {'pt_id': 3}, {'pt_id': 4}]

        srs = sampledata.SRS_WILLAMETTE

        pos_x = srs.origin[0]
        pos_y = srs.origin[1]

        geometries = [Point(pos_x, pos_y), Point(pos_x + 100, pos_y),
                      Point(pos_x, pos_y - 100), Point(pos_x + 100, pos_y - 100)]
        temp_dir = self.workspace_dir
        point_file = os.path.join(temp_dir, 'point_shape.shp')
        shape_ds_uri = pygeoprocessing.testing.create_vector_on_disk(
            geometries, srs.projection, fields, attributes,
            vector_format='ESRI Shapefile', filename=point_file)

        value_list = [10, 20, 30, 40]
        field_name = "num_turb"

        wind_energy.add_field_to_shape_given_list(
            shape_ds_uri, value_list, field_name)

        #compare
        results = {1: {'num_turb': 10}, 2: {'num_turb': 20},
                   3: {'num_turb': 30}, 4: {'num_turb': 40}}

        shape = ogr.Open(shape_ds_uri)
        layer_count = shape.GetLayerCount()

        for layer_num in range(layer_count):
            layer = shape.GetLayer(layer_num)

            feat = layer.GetNextFeature()
            while feat is not None:
                pt_id = feat.GetField('pt_id')

                try:
                    field_val = feat.GetField(field_name)
                    pygeoprocessing.testing.assert_close(
                        results[pt_id][field_name], field_val, 1e-9)
                except ValueError:
                    raise AssertionError(
                        'Could not find field %s' % field_name)

                feat = layer.GetNextFeature()

     #@nottest
    def test_combine_dictionaries(self):
        """WindEnergy: testing 'combine_dictionaries' function"""
        from natcap.invest.wind_energy import wind_energy

        dict_1 = {"name": "bob", "age": 3, "sex": "female"}
        dict_2 = {"hobby": "crawling", "food": "milk"}

        result = wind_energy.combine_dictionaries(dict_1, dict_2)

        expected_result = {"name": "bob", "age": 3, "sex": "female",
                           "hobby": "crawling", "food": "milk"}

        self.assertDictEqual(expected_result, result)

    #@nottest
    def test_combine_dictionaries_duplicates(self):
        """WindEnergy: testing 'combine_dictionaries' function w/ duplicates."""
        from natcap.invest.wind_energy import wind_energy

        dict_1 = {"name": "bob", "age": 3, "sex": "female"}
        dict_2 = {"hobby": "crawling", "food": "milk", "age": 4}

        result = wind_energy.combine_dictionaries(dict_1, dict_2)

        expected_result = {"name": "bob", "age": 3, "sex": "female",
                           "hobby": "crawling", "food": "milk"}

        self.assertDictEqual(expected_result, result)

    #@nottest
    def test_read_csv_wind_parameters(self):
        """WindEnergy: testing 'read_csv_wind_parameter' function"""
        from natcap.invest.wind_energy import wind_energy

        csv_uri = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input',
            'global_wind_energy_parameters.csv')

        parameter_list = [
            'air_density', 'exponent_power_curve', 'decommission_cost',
            'operation_maintenance_cost', 'miscellaneous_capex_cost']

        result = wind_energy.read_csv_wind_parameters(csv_uri, parameter_list)

        expected_result = {
            'air_density': '1.225', 'exponent_power_curve': '2',
            'decommission_cost': '.037', 'operation_maintenance_cost': '.035',
            'miscellaneous_capex_cost': '.05'
        }
        self.assertDictEqual(expected_result, result)

    #nottest
    def test_create_wind_farm_box(self):
        """WindEnergy: testing 'create_wind_farm_box' function."""
        from natcap.invest.wind_energy import wind_energy

        srs = sampledata.SRS_WILLAMETTE
        spat_ref = osr.SpatialReference()
        spat_ref.ImportFromWkt(srs.projection)

        pos_x = srs.origin[0]
        pos_y = srs.origin[1]

        fields = {'id': 'real'}
        attributes = [{'id': 1}]
        geometries = [LinearRing([(pos_x + 100, pos_y), (pos_x + 100, pos_y + 150),
                      (pos_x + 200, pos_y + 150), (pos_x + 200, pos_y),
                      (pos_x + 100, pos_y)])]

        temp_dir = self.workspace_dir
        farm_1 = os.path.join(temp_dir, 'farm_1')
        os.mkdir(farm_1)

        farm_file = os.path.join(farm_1, 'vector.shp')
        farm_ds_uri = pygeoprocessing.testing.create_vector_on_disk(
            geometries, srs.projection, fields, attributes,
            vector_format='ESRI Shapefile', filename=farm_file)

        start_point = (pos_x + 100, pos_y)
        x_len = 100
        y_len = 150

        farm_2 = os.path.join(temp_dir, 'farm_2')
        os.mkdir(farm_2)
        out_uri = os.path.join(farm_2, 'vector.shp')

        wind_energy.create_wind_farm_box(
            spat_ref, start_point, x_len, y_len, out_uri)
        #compare
        pygeoprocessing.testing.assert_vectors_equal(
            out_uri, farm_ds_uri, 1e-9)

    #@nottest
    def test_get_highest_harvested_geom(self):
        """WindEnergy: testing 'get_highest_harvested_geom' function."""
        from natcap.invest.wind_energy import wind_energy

        srs = sampledata.SRS_WILLAMETTE

        fields = {'pt_id': 'int', 'Harv_MWhr': 'real'}
        attributes = [{'pt_id': 1, 'Harv_MWhr': 20.5},
                      {'pt_id': 2, 'Harv_MWhr': 24.5},
                      {'pt_id': 3, 'Harv_MWhr': 13},
                      {'pt_id': 4, 'Harv_MWhr': 15}]

        pos_x = srs.origin[0]
        pos_y = srs.origin[1]

        geometries = [Point(pos_x, pos_y), Point(pos_x + 100, pos_y),
                      Point(pos_x, pos_y - 100), Point(pos_x + 100, pos_y - 100)]
        temp_dir = self.workspace_dir
        point_file = os.path.join(temp_dir, 'point_shape.shp')
        shape_ds_uri = pygeoprocessing.testing.create_vector_on_disk(
            geometries, srs.projection, fields, attributes,
            vector_format='ESRI Shapefile', filename=point_file)

        result = wind_energy.get_highest_harvested_geom(shape_ds_uri)

        ogr_point = ogr.Geometry(ogr.wkbPoint)
        ogr_point.AddPoint_2D(443823.12732787791, 4956546.9059804128)

        if not ogr_point.Equals(result):
            raise AssertionError(
                'Expected geometry %s is not equal to the result %s' % (
                ogr_point, result))

    #@nottest
    def test_read_binary_wind_data(self):
        """WindEnergy: testing 'read_binary_wind_data' function."""
        from natcap.invest.wind_energy import wind_energy

        fields = ["LATI","LONG","Ram-010m","Ram-020m","Ram-030m","Ram-040m",
            "Ram-050m","Ram-060m","Ram-070m","Ram-080m","Ram-090m","Ram-100m",
            "Ram-110m","Ram-120m","Ram-130m","Ram-140m","Ram-150m","K"]
        attributes = {0:
            {"LATI": 31.794439, "LONG": 123.761261,
             "Ram-010m": 6.355385, "Ram-020m": 6.858911, "Ram-030m": 7.171751,
             "Ram-040m": 7.40233, "Ram-050m": 7.586274, "Ram-060m": 7.739956,
             "Ram-070m": 7.872318, "Ram-080m": 7.988804, "Ram-090m": 8.092981,
             "Ram-100m": 8.187322, "Ram-110m": 8.27361, "Ram-120m": 8.353179,
             "Ram-130m": 8.427051, "Ram-140m": 8.496029, "Ram-150m": 8.560752,
             "K": 1.905783},
             1:
            {"LATI": 32.795679,"LONG": 123.814537,
             "Ram-010m": 6.430588, "Ram-020m": 6.940056, "Ram-030m": 7.256615,
             "Ram-040m": 7.489923, "Ram-050m": 7.676043, "Ram-060m": 7.831544,
             "Ram-070m": 7.965473, "Ram-080m": 8.083337, "Ram-090m": 8.188746,
             "Ram-100m": 8.284203, "Ram-110m": 8.371511, "Ram-120m": 8.452021,
             "Ram-130m": 8.526766, "Ram-140m": 8.596559, "Ram-150m": 8.662045,
             "K": 0.999436},
             2:
            {"LATI": 33.796978, "LONG": 123.867828,
             "Ram-010m": 6.53508, "Ram-020m": 7.05284, "Ram-030m": 7.374526,
             "Ram-040m": 7.611625, "Ram-050m": 7.80077, "Ram-060m": 7.958797,
             "Ram-070m": 8.094902, "Ram-080m": 8.214681, "Ram-090m": 8.321804,
             "Ram-100m": 8.418812, "Ram-110m": 8.50754, "Ram-120m": 8.589359,
             "Ram-130m": 8.665319, "Ram-140m": 8.736247, "Ram-150m": 8.8028,
             "K": 2.087774}}


        # Get the wave watch three uri from the first command line argument
        temp_dir = self.workspace_dir
        wind_data_file_uri = os.path.join(temp_dir, 'wind_data_csv.csv')
        _create_csv(fields, attributes, wind_data_file_uri)
        # Get the out binary uri from the second command line argument
        binary_file_uri = os.path.join(temp_dir, 'binary_file.bin')
        # Call the function to properly convert and compress data
        _csv_wind_data_to_binary(wind_data_file_uri, binary_file_uri)

        field_list = ['LATI', 'LONG', 'Ram-080m', 'K-010m']

        results = wind_energy.read_binary_wind_data(binary_file_uri, field_list)

        expected_results = {
            (31.794439, 123.761261): {
                'LONG': 123.761261, 'LATI': 31.794439, 'Ram-080m': 7.988804,
                'K-010m': 1.905783},
            (32.795679, 123.814537): {
                'LONG': 123.814537, 'LATI': 32.795679, 'Ram-080m': 8.083337,
                'K-010m': 0.999436},
            (33.796978, 123.867828): {
                'LONG': 123.867828, 'LATI': 33.7969, 'Ram-080m': 8.214681,
                'K-010m': 2.087774}
        }
        #compare

        result_keys = results.keys()
        expected_keys = expected_results.keys()
        result_keys.sort()
        expected_keys.sort()
        print result_keys
        print expected_keys

        for res_key, exp_key in zip(result_keys, expected_keys):
            for a,b in zip(res_key, exp_key):
                pygeoprocessing.testing.assert_close(a,b,1e-5)
            for field in field_list:
                pygeoprocessing.testing.assert_close(
                    results[res_key][field], expected_results[exp_key][field], 1e-5)


class WindEnergyRegressionTests(unittest.TestCase):
    """Regression tests for the Wind Energy module."""

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
            'wind_data_uri': os.path.join(
                SAMPLE_DATA, 'WindEnergy', 'input',
                'ECNA_EEZ_WEBPAR_Aug27_2012.bin'),
            'bathymetry_uri': os.path.join(
                SAMPLE_DATA, 'Base_Data', 'Marine', 'DEMs',
                'global_dem'),
            'global_wind_parameters_uri': os.path.join(
                SAMPLE_DATA, 'WindEnergy', 'input',
                'global_wind_energy_parameters.csv'),
            'turbine_parameters_uri': os.path.join(
                SAMPLE_DATA, 'WindEnergy', 'input',
                '3_6_turbine.csv'),
            'number_of_turbines': 80,
            'min_depth': 3,
            'max_depth': 60
            }
        return args

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_val_avggrid_dist_windsched(self):
        """WindEnergy: testing Valuation using avg grid distance and wind sched."""
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'New_England_US_Aoi.shp')
        args['land_polygon_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 'global_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2
        args['discount_rate'] = 0.07
        args['avg_grid_distance'] = 4
        args['price_table'] = True
        args['wind_schedule'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'price_table_example.csv')

        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons.tif',
            'levelized_cost_price_per_kWh.tif', 'npv_US_millions.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'pricetable', raster_path))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'pricetable', vector_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_no_aoi(self):
        """WindEnergy: testing base case w/ no AOI, distances, or valuation."""
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        wind_energy.execute(args)

        raster_results = [
            'density_W_per_m2.tif',	'harvested_energy_MWhr_per_yr.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'noaoi', raster_path))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'noaoi', vector_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_no_land_polygon(self):
        """WindEnergy: testing case w/ AOI but w/o land poly or distances."""
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'New_England_US_Aoi.shp')

        wind_energy.execute(args)

        raster_results = [
            'density_W_per_m2.tif',	'harvested_energy_MWhr_per_yr.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'nolandpoly', raster_path))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'nolandpoly', vector_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_no_distances(self):
        """WindEnergy: testing case w/ AOI, land poly, but w/o distances."""
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'New_England_US_Aoi.shp')
        args['land_polygon_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 'global_polygon.shp')

        wind_energy.execute(args)

        raster_results = [
            'density_W_per_m2.tif',	'harvested_energy_MWhr_per_yr.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'nodistances', raster_path))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'nodistances', vector_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_no_valuation(self):
        """WindEnergy: testing case w/ AOI, land poly, and distances."""
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'New_England_US_Aoi.shp')
        args['land_polygon_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 'global_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000

        wind_energy.execute(args)

        raster_results = [
            'density_W_per_m2.tif', 'harvested_energy_MWhr_per_yr.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'novaluation', raster_path))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'novaluation', vector_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_val_gridpts_windsched(self):
        """WindEnergy: testing Valuation w/ grid points and wind sched."""
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'New_England_US_Aoi.shp')
        args['land_polygon_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 'global_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2
        args['discount_rate'] = 0.07
        args['grid_points_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'NE_sub_pts.csv')
        args['price_table'] = True
        args['wind_schedule'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'price_table_example.csv')

        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons.tif',
            'levelized_cost_price_per_kWh.tif',	'npv_US_millions.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'pricetablegridpts', raster_path))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'pricetablegridpts', vector_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_val_avggriddist_windprice(self):
        """WindEnergy: testing Valuation w/ avg grid distances and wind price."""
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'New_England_US_Aoi.shp')
        args['land_polygon_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 'global_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2
        args['discount_rate'] = 0.07
        args['avg_grid_distance'] = 4
        args['price_table'] = False
        args['wind_price'] = 0.187
        args['rate_change'] = 0.2

        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons.tif',
            'levelized_cost_price_per_kWh.tif', 'npv_US_millions.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'priceval', raster_path))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'priceval', vector_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_val_gridpts_windprice(self):
        """WindEnergy: testing Valuation w/ grid pts and wind price."""
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'New_England_US_Aoi.shp')
        args['land_polygon_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 'global_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2
        args['discount_rate'] = 0.07
        args['grid_points_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'NE_sub_pts.csv')
        args['price_table'] = False
        args['wind_price'] = 0.187
        args['rate_change'] = 0.2

        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons.tif',
            'levelized_cost_price_per_kWh.tif',	'npv_US_millions.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'pricevalgridpts', raster_path))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'pricevalgridpts', vector_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_val_land_grid_points(self):
        """WindEnergy: testing Valuation w/ grid/land pts and wind price."""
        from natcap.invest.wind_energy import wind_energy

        args = WindEnergyRegressionTests.generate_base_args(self.workspace_dir)

        args['aoi_uri'] = os.path.join(
            SAMPLE_DATA, 'WindEnergy', 'input', 'New_England_US_Aoi.shp')
        args['land_polygon_uri'] = os.path.join(
            SAMPLE_DATA, 'Base_Data', 'Marine', 'Land', 'global_polygon.shp')
        args['min_distance'] = 0
        args['max_distance'] = 200000
        args['valuation_container'] = True
        args['foundation_cost'] = 2
        args['discount_rate'] = 0.07
        # there was no sample data that provided landing points, thus for
        # testing, grid points in 'NE_sub_pts.csv' were duplicated and marked
        # as land points. So the distances will be zero, keeping the result
        # the same but testing that section of code
        args['grid_points_uri'] = os.path.join(
            REGRESSION_DATA, 'grid_land_pts.csv')
        args['price_table'] = False
        args['wind_price'] = 0.187
        args['rate_change'] = 0.2

        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons.tif',
            'levelized_cost_price_per_kWh.tif',	'npv_US_millions.tif']

        for raster_path in raster_results:
            pygeoprocessing.testing.assert_rasters_equal(
                os.path.join(args['workspace_dir'], 'output', raster_path),
                os.path.join(REGRESSION_DATA, 'pricevalgridpts', raster_path))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm.shp',
            'wind_energy_points.shp']

        for vector_path in vector_results:
            pygeoprocessing.testing.assert_vectors_equal(
                os.path.join(args['workspace_dir'], 'output', vector_path),
                os.path.join(REGRESSION_DATA, 'pricevalgridpts', vector_path))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_suffix(self):
        """WindEnergy: testing suffix handling."""
        from natcap.invest.wind_energy import wind_energy

        args = {
            'workspace_dir': self.workspace_dir,
            'wind_data_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'wind_data_smoke.bin'),
            'bathymetry_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'dem_smoke.tif'),
            'global_wind_parameters_uri': os.path.join(
                SAMPLE_DATA, 'WindEnergy', 'input',
                'global_wind_energy_parameters.csv'),
            'turbine_parameters_uri': os.path.join(
                SAMPLE_DATA, 'WindEnergy', 'input',
                '3_6_turbine.csv'),
            'number_of_turbines': 80,
            'min_depth': 3,
            'max_depth': 200,
            'aoi_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'aoi_smoke.shp'),
            'land_polygon_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'landpoly_smoke.shp'),
            'min_distance': 0,
            'max_distance': 200000,
            'valuation_container': True,
            'foundation_cost': 2,
            'discount_rate': 0.07,
            'avg_grid_distance': 4,
            'price_table': True,
            'wind_schedule': os.path.join(
                SAMPLE_DATA, 'WindEnergy', 'input', 'price_table_example.csv'),
            'suffix': 'test'
        }
        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons_test.tif', 'density_W_per_m2_test.tif',
            'harvested_energy_MWhr_per_yr_test.tif',
            'levelized_cost_price_per_kWh_test.tif', 'npv_US_millions_test.tif']

        for raster_path in raster_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', raster_path)))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm_test.shp',
            'wind_energy_points_test.shp']

        for vector_path in vector_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', vector_path)))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_suffix_underscore(self):
        """WindEnergy: testing that suffix w/ underscore is handled correctly."""
        from natcap.invest.wind_energy import wind_energy

        args = {
            'workspace_dir': self.workspace_dir,
            'wind_data_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'wind_data_smoke.bin'),
            'bathymetry_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'dem_smoke.tif'),
            'global_wind_parameters_uri': os.path.join(
                SAMPLE_DATA, 'WindEnergy', 'input',
                'global_wind_energy_parameters.csv'),
            'turbine_parameters_uri': os.path.join(
                SAMPLE_DATA, 'WindEnergy', 'input',
                '3_6_turbine.csv'),
            'number_of_turbines': 80,
            'min_depth': 3,
            'max_depth': 200,
            'aoi_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'aoi_smoke.shp'),
            'land_polygon_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'landpoly_smoke.shp'),
            'min_distance': 0,
            'max_distance': 200000,
            'valuation_container': True,
            'foundation_cost': 2,
            'discount_rate': 0.07,
            'avg_grid_distance': 4,
            'price_table': True,
            'wind_schedule': os.path.join(
                SAMPLE_DATA, 'WindEnergy', 'input', 'price_table_example.csv'),
            'suffix': '_test'
        }
        wind_energy.execute(args)

        raster_results = [
            'carbon_emissions_tons_test.tif', 'density_W_per_m2_test.tif',
            'harvested_energy_MWhr_per_yr_test.tif',
            'levelized_cost_price_per_kWh_test.tif', 'npv_US_millions_test.tif']

        for raster_path in raster_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', raster_path)))

        vector_results = [
            'example_size_and_orientation_of_a_possible_wind_farm_test.shp',
            'wind_energy_points_test.shp']

        for vector_path in vector_results:
            self.assertTrue(os.path.exists(
                os.path.join(args['workspace_dir'], 'output', vector_path)))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_field_error_missing_bio_param(self):
        """WindEnergy: testing that FieldError raised when missing bio param."""
        from natcap.invest.wind_energy import wind_energy

        # for testing raised exceptions, running on a set of data that was
        # created by hand and has no numerical validity. Helps test the
        # raised exception quicker
        args = {
            'workspace_dir': self.workspace_dir,
            'wind_data_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'wind_data_smoke.bin'),
            'bathymetry_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'dem_smoke.tif'),
            'global_wind_parameters_uri': os.path.join(
                SAMPLE_DATA, 'WindEnergy', 'input',
                'global_wind_energy_parameters.csv'),
            'number_of_turbines': 80,
            'min_depth': 3,
            'max_depth': 200,
            'aoi_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'aoi_smoke.shp'),
            'land_polygon_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'landpoly_smoke.shp'),
            'min_distance': 0,
            'max_distance': 200000
        }
        # creating a stand in turbine parameter csv file that is missing
        # a biophysical field / value. This should raise the exception
        tmp, fname = tempfile.mkstemp(suffix='.csv', dir=args['workspace_dir'])
        os.close(tmp)
        data = {
            'hub_height': 80, 'cut_in_wspd': 4.0, 'rated_wspd': 12.5,
            'cut_out_wspd': 25.0, 'turbine_rated_pwr': 3.6, 'turbine_cost': 8.0,
            'turbines_per_circuit': 8
        }

        _create_vertical_csv(data, fname)

        args['turbine_parameters_uri'] = fname

        self.assertRaises(wind_energy.FieldError, wind_energy.execute, args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_non_divisible_by_ten_hub_height_error(self):
        """WindEnergy: raise HubHeightError when value not divisible by 10."""
        from natcap.invest.wind_energy import wind_energy

        # for testing raised exceptions, running on a set of data that was
        # created by hand and has no numerical validity. Helps test the
        # raised exception quicker
        args = {
            'workspace_dir': self.workspace_dir,
            'wind_data_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'wind_data_smoke.bin'),
            'bathymetry_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'dem_smoke.tif'),
            'global_wind_parameters_uri': os.path.join(
                SAMPLE_DATA, 'WindEnergy', 'input',
                'global_wind_energy_parameters.csv'),
            'number_of_turbines': 80,
            'min_depth': 3,
            'max_depth': 200,
            'aoi_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'aoi_smoke.shp'),
            'land_polygon_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'landpoly_smoke.shp'),
            'min_distance': 0,
            'max_distance': 200000
        }

        # creating a stand in turbine parameter csv file that is missing
        # a biophysical field / value. This should raise the exception
        tmp, fname = tempfile.mkstemp(suffix='.csv', dir=args['workspace_dir'])
        os.close(tmp)
        data = {
            'hub_height': 83, 'cut_in_wspd': 4.0, 'rated_wspd': 12.5,
            'cut_out_wspd': 25.0, 'turbine_rated_pwr': 3.6, 'turbine_cost': 8.0,
            'turbines_per_circuit': 8, 'rotor_diameter': 40
        }

        _create_vertical_csv(data, fname)

        args['turbine_parameters_uri'] = fname

        self.assertRaises(wind_energy.HubHeightError, wind_energy.execute, args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_missing_valuation_params(self):
        """WindEnergy: testing that FieldError is thrown when val params miss."""
        from natcap.invest.wind_energy import wind_energy

        # for testing raised exceptions, running on a set of data that was
        # created by hand and has no numerical validity. Helps test the
        # raised exception quicker
        args = {
            'workspace_dir': self.workspace_dir,
            'wind_data_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'wind_data_smoke.bin'),
            'bathymetry_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'dem_smoke.tif'),
            'global_wind_parameters_uri': os.path.join(
                SAMPLE_DATA, 'WindEnergy', 'input',
                'global_wind_energy_parameters.csv'),
            'turbine_parameters_uri': os.path.join(
                REGRESSION_DATA, 'turbine_params_val_missing.csv'),
            'number_of_turbines': 80,
            'min_depth': 3,
            'max_depth': 200,
            'aoi_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'aoi_smoke.shp'),
            'land_polygon_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'landpoly_smoke.shp'),
            'min_distance': 0,
            'max_distance': 200000,
            'valuation_container': True,
            'foundation_cost': 2,
            'discount_rate': 0.07,
            'avg_grid_distance': 4,
            'price_table': True,
            'wind_schedule': os.path.join(
                SAMPLE_DATA, 'WindEnergy', 'input', 'price_table_example.csv'),
            'suffix': '_test'
        }
        # creating a stand in turbine parameter csv file that is missing
        # a valuation field / value. This should raise the exception
        tmp, fname = tempfile.mkstemp(suffix='.csv', dir=args['workspace_dir'])
        os.close(tmp)
        data = {
            'hub_height': 80, 'cut_in_wspd': 4.0, 'rated_wspd': 12.5,
            'cut_out_wspd': 25.0, 'turbine_rated_pwr': 3.6,
            'turbines_per_circuit': 8, 'rotor_diamater': 40
        }

        _create_vertical_csv(data, fname)

        args['turbine_parameters_uri'] = fname

        self.assertRaises(wind_energy.FieldError, wind_energy.execute, args)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_time_period_exceptoin(self):
        """WindEnergy: raised TimePeriodError if 'time' and 'wind_sched' differ."""
        from natcap.invest.wind_energy import wind_energy

        # for testing raised exceptions, running on a set of data that was
        # created by hand and has no numerical validity. Helps test the
        # raised exception quicker
        args = {
            'workspace_dir': self.workspace_dir,
            'wind_data_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'wind_data_smoke.bin'),
            'bathymetry_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'dem_smoke.tif'),
            'turbine_parameters_uri': os.path.join(
                SAMPLE_DATA, 'WindEnergy', 'input',
                '3_6_turbine.csv'),
            'number_of_turbines': 80,
            'min_depth': 3,
            'max_depth': 200,
            'aoi_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'aoi_smoke.shp'),
            'land_polygon_uri': os.path.join(
                REGRESSION_DATA, 'smoke', 'landpoly_smoke.shp'),
            'min_distance': 0,
            'max_distance': 200000,
            'valuation_container': True,
            'foundation_cost': 2,
            'discount_rate': 0.07,
            'avg_grid_distance': 4,
            'price_table': True,
            'wind_schedule': os.path.join(
                SAMPLE_DATA, 'WindEnergy', 'input', 'price_table_example.csv'),
            'suffix': '_test'
        }
        # creating a stand in global wind params table that has a different
        # 'time' value than what is given in the wind schedule table.
        # This should raise the exception
        tmp, fname = tempfile.mkstemp(suffix='.csv', dir=args['workspace_dir'])
        os.close(tmp)
        data = {
            'air_density': 1.225, 'exponent_power_curve': 2,
            'decommission_cost': .037, 'operation_maintenance_cost': .035,
            'miscellaneous_capex_cost': .05, 'installation_cost': .20,
            'infield_cable_length': 0.91, 'infield_cable_cost': 0.26,
            'mw_coef_ac': .81, 'mw_coef_dc': 1.09, 'cable_coef_ac': 1.36,
            'cable_coef_dc': .89, 'ac_dc_distance_break': 60,
            'time_period': 10, 'rotor_diameter_factor': 7,
            'carbon_coefficient': 6.8956e-4,
            'air_density_coefficient': 1.194e-4, 'loss_parameter': .05
        }

        _create_vertical_csv(data, fname)

        args['global_wind_parameters_uri'] = fname

        self.assertRaises(wind_energy.TimePeriodError, wind_energy.execute, args)

