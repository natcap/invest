"""Tests for natcap.invest.hydropower.hydropower_water_yield.

    This module is intended for unit testing of the hydropower_water_yield
    model. The goal is to thoroughly test each of the modules functions
    and verify correctness.
"""
import unittest
import tempfile
import shutil
import os
import csv

from osgeo import gdal
from osgeo import ogr
import numpy
import numpy.testing
import pygeoprocessing.testing
from pygeoprocessing.testing import sampledata
from shapely.geometry import Polygon
from nose.tools import nottest

def _create_watershed(
        fields=None, attributes=None, subshed=True, execute=False):
    """
    Create a watershed shapefile

    This Watershed Shapefile is created with the following characteristics:
        * SRS is in the SRS_WILLAMETTE.
        * Vector type is Polygon
        * Polygons are 100m x 100m

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
        subshed (boolean): True or False depicting whether the created vector
            should be representative of a sub watershed or watershed.
            Watershed will return one polygon, sub watershed will return 2 or 3
            depending on the value of 'execute'.
        execute (boolean): Determines how to construct the sub shed polygons.
            If True, create 2 sub watershed polygons, else create 3

    Returns:
        A string filepath to the vector on disk
    """

    srs = sampledata.SRS_WILLAMETTE

    pos_x = srs.origin[0]
    pos_y = srs.origin[1]

    if subshed:
        if not execute:
            poly_geoms = {
                'poly_1': [(pos_x, pos_y), (pos_x + 100, pos_y),
                           (pos_x + 100, pos_y - 100), (pos_x, pos_y - 100),
                           (pos_x, pos_y)],
                'poly_2': [(pos_x + 100, pos_y), (pos_x + 200, pos_y),
                           (pos_x + 200, pos_y - 100),
                           (pos_x + 100, pos_y - 100), (pos_x + 100, pos_y)],
                'poly_3': [(pos_x, pos_y - 100), (pos_x + 100, pos_y -100),
                           (pos_x + 100, pos_y - 200), (pos_x, pos_y - 200),
                           (pos_x, pos_y - 100)]}

            geometries = [Polygon(poly_geoms['poly_1']),
                          Polygon(poly_geoms['poly_2']),
                          Polygon(poly_geoms['poly_3'])]
        else:
            poly_geoms = {
                'poly_1': [(pos_x, pos_y), (pos_x + 200, pos_y),
                           (pos_x + 200, pos_y - 100), (pos_x, pos_y - 100),
                           (pos_x, pos_y)],
                'poly_2': [(pos_x, pos_y - 100), (pos_x + 200, pos_y - 100),
                           (pos_x + 200, pos_y - 200), (pos_x, pos_y - 200),
                           (pos_x, pos_y - 100)]}

            geometries = [Polygon(poly_geoms['poly_1']),
                          Polygon(poly_geoms['poly_2'])]
    else:
        poly_geoms = {
            'poly_1': [(pos_x, pos_y), (pos_x + 200, pos_y),
                       (pos_x + 200, pos_y - 200), (pos_x, pos_y - 200),
                       (pos_x, pos_y)]}

        geometries = [Polygon(poly_geoms['poly_1'])]

    return pygeoprocessing.testing.create_vector_on_disk(
        geometries, srs.projection, fields, attributes,
        vector_format='ESRI Shapefile')


def _create_csv(fields, data):
    """
    Create a new csv table from a dictionary

    Parameters:
        fields (list): a python list of the column names. The order of the
            fields in the list will be the order in how they are written. ex:
            ['id', 'precip', 'total']
        data (dict): a python dictionary representing the table. The dictionary
            should be constructed with unique numerical keys that point to
            another dictionary which represents a row in the table:
            data = {0 : {'id':1, 'precip':43, 'total': 65},
                    1 : {'id':2, 'precip':65, 'total': 94}}
            The unique outer keys determine the ordering of the rows to be
            written.

    Returns:
        A filepath to the csv file on disk
    """
    temp, filename = tempfile.mkstemp(suffix='.csv')
    os.close(temp)

    csv_file = open(filename, 'wb')

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

    return filename


def _create_input_table(component):
    """
    Creates a csv file for the biophysical, scarcity, or valuation inputs

    Parameters:
        component (string): a String indicating which csv table to build
            and return. Either "biophysical", "scarcity", or "valuation"

    Returns:
        A filepath to the csv table on disk.
    """
    if component == "biophysical":
        bio_fields = ['lucode', 'Kc', 'root_depth', 'LULC_veg']

        bio_data = {0: {'lucode':0, 'Kc': 0.3, 'root_depth': 500,
                        'LULC_veg': 0},
                    1: {'lucode':1, 'Kc': 0.75, 'root_depth': 5000,
                        'LULC_veg': 1},
                    2: {'lucode':2, 'Kc': 0.85, 'root_depth': 5000,
                        'LULC_veg': 1}}

        return _create_csv(bio_fields, bio_data)

    elif component == "scarcity":
        demand_fields = ['lucode', 'demand']

        demand_data = {0: {'lucode':0, 'demand': 500},
                       1: {'lucode':1, 'demand': 0},
                       2: {'lucode':2, 'demand': 0}}

        return _create_csv(demand_fields, demand_data)

    else:
        val_fields = ['ws_id', 'time_span', 'discount', 'efficiency',
                      'fraction', 'cost', 'height', 'kw_price']

        val_data = {0: {'ws_id': 1, 'time_span': 20, 'discount': 5,
                        'efficiency': 0.8, 'fraction': 0.6, 'cost': 0,
                        'height': 25, 'kw_price': 0.07}}

        return _create_csv(val_fields, val_data)


def _create_result_watersheds(component, sub_shed=False):
    """
    Creates a watershed / subwatershed shapefile of results correlating
        to the 'component'

    Parameters:
        component (string): a String indicating which results to construct
            for what components of the model were run. Can be "water_yield",
            "scarcity", or "valuation".
        sub_shed=False (boolean): a Boolean indicating whether or not to
            create a shapefile using subwatershed characteristics.

    Returns:
        A filepath to a shapefile on disk
    """
    if sub_shed:
        sub_res_fields = {'subws_id': 'int', 'precip_mn': 'real',
                          'PET_mn': 'real', 'AET_mn': 'real',
                          'wyield_mn': 'real', 'wyield_vol': 'real',
                          'num_pixels': 'int'}
        sub_res_attr = [{'subws_id': 1, 'precip_mn': 1500, 'PET_mn': 510,
                         'AET_mn': 409.02562, 'wyield_mn': 1090.97438,
                         'wyield_vol': 21819.4876, 'num_pixels': 2},
                        {'subws_id': 2, 'precip_mn': 3000, 'PET_mn': 935,
                         'AET_mn': 580.30143, 'wyield_mn': 2419.69857,
                         'wyield_vol': 48393.9713999, 'num_pixels': 2}]

        return _create_watershed(
            fields=sub_res_fields, attributes=sub_res_attr, subshed=True,
            execute=True)

    res_fields = {'ws_id': 'int', 'precip_mn': 'real', 'PET_mn': 'real',
                  'AET_mn': 'real', 'wyield_mn': 'real', 'wyield_vol': 'real',
                  'num_pixels': 'int'}
    res_attr = [{'ws_id': 1, 'precip_mn': 2250, 'PET_mn': 722.5,
                 'AET_mn': 494.663525, 'wyield_mn': 1755.336475,
                 'wyield_vol': 70213.459, 'num_pixels': 4}]

    if component == "water_yield":
        return _create_watershed(
            fields=res_fields, attributes=res_attr, subshed=False,
            execute=True)

    else:
        scarcity_fields = ['consum_vol', 'consum_mn', 'rsupply_vl',
                           'rsupply_mn']
        scarcity_values = [500, 125, 65213.459, 1630.336475]
        for field, value in zip(scarcity_fields, scarcity_values):
            res_fields[field] = 'real'
            res_attr[0][field] = value

        if component == "valuation":
            valuation_fields = ['hp_energy', 'hp_val']
            valuation_values = [212.856730176, 67.73453119]
            for field, value in zip(valuation_fields, valuation_values):
                res_fields[field] = 'real'
                res_attr[0][field] = value
            return _create_watershed(
                fields=res_fields, attributes=res_attr, subshed=False,
                execute=True)

        else:
            return _create_watershed(
                fields=res_fields, attributes=res_attr, subshed=False,
                execute=True)


def _create_result_tables(component, sub_shed=False):
    """
    Creates a csv table of results correlating to the 'component'

    Parameters:
        component (string): a String indicating which results to construct
            for what components of the model were run. Can be "water_yield",
            "scarcity", or "valuation".
        sub_shed=False (boolean): a Boolean indicating whether or not to
            create a csv table using subwatershed characteristics.

    Returns:
        A filepath to a csv table on disk
    """
    if sub_shed:
        csv_fields_subws = ['subws_id', 'num_pixels', 'precip_mn', 'PET_mn',
                            'AET_mn', 'wyield_mn', 'wyield_vol']
        csv_data_subws = {0: {'subws_id': 1, 'precip_mn': 1500, 'PET_mn': 510,
                              'AET_mn': 409.02562, 'wyield_mn': 1090.97438,
                              'wyield_vol': 21819.4876, 'num_pixels': 2},
                          1: {'subws_id': 2, 'precip_mn': 3000, 'PET_mn': 935,
                              'AET_mn': 580.30143, 'wyield_mn': 2419.69857,
                              'wyield_vol': 48393.9713999, 'num_pixels': 2}}

        return _create_csv(csv_fields_subws, csv_data_subws)

    csv_fields_ws = ['ws_id', 'num_pixels', 'precip_mn', 'PET_mn', 'AET_mn',
                     'wyield_mn', 'wyield_vol']

    csv_data_ws = {0: {'ws_id': 1, 'precip_mn': 2250, 'PET_mn': 722.5,
                       'AET_mn': 494.663525, 'wyield_mn': 1755.336475,
                       'wyield_vol': 70213.459, 'num_pixels': 4}}

    if component == "water_yield":
        return _create_csv(csv_fields_ws, csv_data_ws)

    else:
        scarcity_fields = ['consum_vol', 'consum_mn', 'rsupply_vl',
                           'rsupply_mn']
        scarcity_values = [500, 125, 65213.459, 1630.336475]
        for field, value in zip(scarcity_fields, scarcity_values):
            csv_fields_ws[field] = 'real'
            csv_data_ws[field] = value

        if component == "valuation":
            valuation_fields = ['hp_energy', 'hp_val']
            valuation_values = [212.856730176, 67.73453119]
            for field, value in zip(valuation_fields, valuation_values):
                csv_fields_ws[field] = 'real'
                csv_data_ws[field] = value
            return _create_csv(csv_fields_ws, csv_data_ws)
        else:
            return _create_csv(csv_fields_ws, csv_data_ws)


def _create_raster(matrix, dtype=gdal.GDT_Int32, nodata=-1):
    """
    Create a raster for the hydropower_water_yield model.

    This raster is created with the following characteristics:
        * SRS is in the SRS_WILLAMETTE.
        * Nodata is -1.
        * Raster type is defaulted with `gdal.GDT_Int32`
        * Pixel size is 100m

    Parameters:
        matrix (numpy.array): A numpy array to use as the raster matrix.
            The output raster created will be saved with these pixel values.
        dtype=gdal.GDT_Int32 (GDAL datatype): a GDAL datatype to use
            for the new raster. Common types are gdal.GDT_Int32 and
            gdal.GDT_Float32.
        nodata=-1 (int or float or None): a number to set as the rasters
            nodata value. If None then no nodata value is set.

    Returns:
        A string filepath to a new raster on disk.
    """

    lulc_matrix = matrix
    lulc_nodata = nodata
    srs = sampledata.SRS_WILLAMETTE
    return pygeoprocessing.testing.create_raster_on_disk(
        [lulc_matrix], srs.origin, srs.projection, lulc_nodata,
        srs.pixel_size(100), datatype=dtype)

class HydropowerUnitTests(unittest.TestCase):
    """Tests for natcap.invest.hydropower.hydropower_water_yield"""


    def test_extract_datasource_table(self):
        """Testing the 'extract_datasource_table_by_key' function, which
            returns a dictionary based on a Shapefiles attributes"""
        from natcap.invest.hydropower import hydropower_water_yield

        fields = {'ws_id': 'int', 'wyield_mn': 'real', 'wyield_vol': 'real',
                  'consum_mn': 'real', 'consum_vol': 'real'}

        attributes = [
            {'ws_id': 1, 'wyield_mn': 1000, 'wyield_vol': 1000,
             'consum_vol': 100, 'consum_mn': 10000},
            {'ws_id': 2, 'wyield_mn': 1000, 'wyield_vol': 800,
             'consum_vol': 420, 'consum_mn': 10000},
            {'ws_id': 3, 'wyield_mn': 1000, 'wyield_vol': 600,
             'consum_vol': 350, 'consum_mn': 10000}]

        watershed_uri = _create_watershed(fields, attributes)

        key_field = 'ws_id'
        wanted_list = ['wyield_vol', 'consum_vol']
        results = hydropower_water_yield.extract_datasource_table_by_key(
            watershed_uri, key_field, wanted_list)

        expected_res = {1: {'wyield_vol': 1000, 'consum_vol': 100},
                        2: {'wyield_vol': 800, 'consum_vol': 420},
                        3: {'wyield_vol': 600, 'consum_vol': 350}}

        for res_key in results.keys():
            if res_key not in expected_res.keys():
                raise AssertionError(
                    'Extra Key %s found in returned results' % res_key)
            for sub_key in results[res_key].keys():
                if sub_key not in expected_res[res_key].keys():
                    raise AssertionError(
                        'Extra key %s found in returned results' % sub_key)
                pygeoprocessing.testing.assert_close(
                    expected_res[res_key][sub_key], results[res_key][sub_key])

        os.remove(watershed_uri)

    def test_write_new_table(self):
        """Testing 'write_new_table' function, which produces a CSV file."""
        from natcap.invest.hydropower import hydropower_water_yield

        tmp, filename = tempfile.mkstemp()
        os.close(tmp)

        fields = ['id', 'precip', 'volume']

        data = {0: {'id':1, 'precip': 100, 'volume': 150},
                1: {'id':2, 'precip': 150, 'volume': 350},
                2: {'id':3, 'precip': 170, 'volume': 250}}

        hydropower_water_yield.write_new_table(filename, fields, data)

        csv_file = open(filename, 'rb')
        reader = csv.DictReader(csv_file)

        for row, key in zip(reader, [0, 1, 2]):
            for sub_key in data[key].keys():
                if sub_key not in row:
                    raise AssertionError(
                        'Key %s not found in CSV table' % sub_key)
                pygeoprocessing.testing.assert_close(
                    float(data[key][sub_key]), float(row[sub_key]))

        csv_file.close()
        os.remove(filename)


    def test_add_dict_to_shape(self):
        """Testing the 'add_dict_to_shape' function."""
        from natcap.invest.hydropower import hydropower_water_yield

        fields = {'ws_id': 'int'}

        attributes = [{'ws_id': 1}, {'ws_id': 2}, {'ws_id': 3}]

        watershed_uri = _create_watershed(fields, attributes)

        field_dict = {1: 50.0, 2: 10.5, 3: 15.8}

        field_name = 'precip'
        key = 'ws_id'

        hydropower_water_yield.add_dict_to_shape(
            watershed_uri, field_dict, field_name, key)

        results = {
            1: {'precip': 50.0},
            2: {'precip': 10.5},
            3: {'precip': 15.8}}

        shape = ogr.Open(watershed_uri)
        layer_count = shape.GetLayerCount()

        for layer_num in range(layer_count):
            layer = shape.GetLayer(layer_num)

            feat = layer.GetNextFeature()
            while feat is not None:
                ws_id = feat.GetField('ws_id')

                try:
                    field_val = feat.GetField(field_name)
                    pygeoprocessing.testing.assert_close(
                        results[ws_id][field_name], field_val)
                except ValueError:
                    raise AssertionError(
                        'Could not find field %s' % field_name)

                feat = None
                feat = layer.GetNextFeature()

        shape = None
        os.remove(watershed_uri)
