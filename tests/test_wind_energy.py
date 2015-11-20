"""Module for Regression Testing the InVEST Wind Energy module."""
import unittest
import tempfile
import shutil
import os
import collections

import pygeoprocessing.testing
from pygeoprocessing.testing import scm
import numpy
import numpy.testing
from shapely.geometry import Polygon
from shapely.geometry import Point
from nose.tools import nottest
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

SAMPLE_DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(os.path.dirname(__file__), 'data', 'wind_energy')

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

    @nottest
    def test_calculate_distances_land_grid(self):
        """Testing 'calculate_distances_land_grid' function"""
        from natcap.invest.wind_energy import wind_energy

        fields = {'id': 'real', 'L2G': 'real'}
        attrs = [{'id': 1, 'L2G': 10}, {'id': 2, 'L2G': 20}]
        srs = sampledata.SRS_WILLAMETTE
        pos_x = srs.origin[0]
        pos_y = srs.origin[1]
        geometries = [Point(pos_x + 50, pos_y - 50), Point(pos_x + 50, pos_y - 150)]

        temp_dir = self.workspace_dir
        tmp, temp_shape = tempfile.mkstemp(suffix='.shp', dir=temp_dir)
        os.close(tmp)
        land_shape_uri = pygeoprocessing.testing.create_vector_on_disk(
                geometries, srs.projection, fields, attrs,
                vector_format='ESRI Shapefile', filename=temp_shape)

        #land_shape_uri = _create_ptm(fields, attrs)

        matrix = numpy.array([[1,1,1,1],[1,1,1,1]])
        raster_path = tempfile.mkstemp(suffix='.tif', dir=temp_dir)
        #harvested_masked_uri = _create_raster(matrix, raster_path)
        srs = sampledata.SRS_WILLAMETTE
        harvested_masked_uri = pygeoprocessing.testing.create_raster_on_disk(
            [lulc_matrix], srs.origin, srs.projection, -1,
            srs.pixel_size(100), datatype=gdal.GDT_Int32, filename=raster_path)

        tmp, tmp_dist_final_uri = tempfile.mkstemp(suffix='.tif', dir=temp_dir)
        os.close(tmp)

        wind_energy.calculate_distances_land_grid(
            land_shape_uri, harvested_masked_uri, tmp_dist_final_uri)

        #Compare
        result = gdal.open(tmp_dist_final_uri)
        res_band = gdal.GetRasterBand(1)
        res_array = res_band.ReadAsArray()
        exp_array = numpy.array([[10, 110, 210, 310],[20, 120, 220, 320]])
        numpy.testing.assert_array_equal(res_array, exp_array)


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
            'turbines_per_circuit': 8}

        args['turbine_parameters_uri'] = _create_vertical_csv(data, fname)

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
            'turbine_parameters_uri': os.path.join(
                REGRESSION_DATA, 'turbine_params_hubheight.csv'),
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
            'turbines_per_circuit': 8, 'rotator_diameter': 40}

        self.assertRaises(wind_energy.HubHeightError, wind_energy.execute, args)
#???????????
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    @nottest
    def test_hubheight_100(self):
        """WindEnergy: testing that a HubHeightError is raised on bad bio params."""
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
                REGRESSION_DATA, 'turbine_params_hubheight.csv'),
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
#??????????????????
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    #@nottest
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
            'turbines_per_circuit': 8, 'rotor_diamater': 40}

        args['turbine_parameters_uri'] = _create_vertical_csv(data, fname)


        self.assertRaises(wind_energy.FieldError, wind_energy.execute, args)


    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    #@nottest
    def test_time_loss(self):
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
                REGRESSION_DATA, 'global_params_time_loss.csv'),
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

        self.assertRaises(wind_energy.TimePeriodError, wind_energy.execute, args)

