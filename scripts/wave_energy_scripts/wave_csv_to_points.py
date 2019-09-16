"""Python module that creates a point shapefile from a CSV

    Example from command line:
    >> python wave_csv_to_points.py wave_formatted_csv_data.txt my_layer shape_out.shp

    Example two:
    >> python wave_csv_to_points.py wave_formatted_csv_data.csv WCVI WCVI_points.shp
"""

import csv
import os
import sys
import shutil

from osgeo import ogr
from osgeo import osr


def create_wave_point_ds(wave_data_csv_uri, layer_name, output_uri):
    """Creates a point shapefile from a wave energy data csv file that is
        properly formatted. The point shapefile created is not projected
        and uses latitude and longitude for its geometry.

        wave_data_csv_uri - a URI to a comma separated file of wave point data
            that has been properly formatted (required) Example format:
            ID,I,J,LONG,LATI,HSAVG,TPAVG
            1,102,370,24.3,54.3,10.2,11.1
            2,102,370,24.3,54.3,10.2,11.1

        layer_name - a string for the name of the point shapefile
            layer (required)

        output_uri - a URI for the output path of the point shapefile (required)

        return - Nothing"""

    # Initiate a dictionary to build up data from the csv file
    dict_data = {}

    # Open the csv file
    point_file = open(wave_data_csv_uri)

    # Get a handle on the csv file by using dictReader which handles each line
    # as a dictionary where the column headers are the keys
    reader = csv.DictReader(point_file)

    # A list of column headers that we want to remain integers all other column
    # headers will become floats
    int_list = ['ID', 'I', 'J']

    # Iterate over the file by line
    for row in reader:
        # For each line's dictionary, iterate over the key-value pairs
        for k,v in row.items():
            # If the key represents a value that should be an integer, cast to
            # an int, else cast to a float
            if k in int_list:
                row[k] = int(v)
            else:
                row[k] = float(v)
        # Build up the new dictionary
        dict_data[row['ID']] = row

    # If the output_uri exists delete it
    if os.path.isfile(output_uri):
        os.remove(output_uri)
    elif os.path.isdir(output_uri):
        shutil.rmtree(output_uri)

    # Create the ogr driver for the new point shapefile
    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    # Create the datasource for the new point shapefile
    output_datasource = output_driver.CreateDataSource(output_uri)

    # Set the spatial reference to WGS84 (lat/long)
    source_sr = osr.SpatialReference()
    source_sr.SetWellKnownGeogCS("WGS84")

    # Create the new point shapefile layer
    output_layer = output_datasource.CreateLayer(
            layer_name, source_sr, ogr.wkbPoint)

    # Get the keys of the dictionary which are the 'ID' values
    outer_keys = dict_data.keys()

    # Using the list of keys from above, get the first keys sub dictionary and
    # get it's keys. These 'inner' keys are the column headers from the file
    # and will be added to the point shapefile as fields
    field_list = dict_data[outer_keys[0]].keys()

    # Create a dictionary to store what variable types the fields are
    type_dict = {}

    for field in field_list:
        # Get a value from the field
        val = dict_data[outer_keys[0]][field]
        # Check to see if the value is a String of characters or a number. This
        # will determine the type of field created in the shapefile
        if isinstance(val, str):
            type_dict[field] = 'str'
        else:
            type_dict[field] = 'number'

    for field in field_list:
        field_type = None
        # Distinguish if the field type is of type String or other. If Other, we
        # are assuming it to be a float
        if type_dict[field] == 'str':
            field_type = ogr.OFTString
        else:
            field_type = ogr.OFTReal

        # Create a new field in the point shapefile
        output_field = ogr.FieldDefn(field, field_type)
        output_layer.CreateField(output_field)

    # For each inner dictionary (for each point) create a point and set its
    # fields
    for point_dict in dict_data.values():
        # Get latitude / longitude values
        latitude = float(point_dict['LATI'])
        longitude = float(point_dict['LONG'])

        # Make a new point geometry set the long / lat
        geom = ogr.Geometry(ogr.wkbPoint)
        geom.AddPoint_2D(longitude, latitude)

        # Create a new point feature
        output_feature = ogr.Feature(output_layer.GetLayerDefn())

        for field_name in point_dict:
            field_index = output_feature.GetFieldIndex(field_name)
            # Set the value for each field for this particular point
            output_feature.SetField(field_index, point_dict[field_name])

        # Set geometry and create / set the feature
        output_feature.SetGeometryDirectly(geom)
        output_layer.CreateFeature(output_feature)
        output_feature = None

    output_layer.SyncToDisk()


if __name__ == '__main__':
    # Argument 1 from the command line, the wave energy csv data
    wave_data_csv_uri = sys.argv[1]

    # Argument 2 from the command line, a string for the layer name
    layer_name = sys.argv[2]

    # Argument 3 from the command line, the output URI for the point shapefile
    out_uri = sys.argv[3]

    # Call the function to create our point shapefile from CSV
    create_wave_point_ds(wave_data_csv_uri, layer_name, out_uri)
