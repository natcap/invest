"""InVEST Nearshore Wave and Erosion model non-core."""

import sys
import os
import shutil
import logging
import numpy
import json
import csv
import h5py
from h5py import h5z

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import pygeoprocessing
import nearshore_wave_and_erosion_core

logging.getLogger("pygeoprocessing").setLevel(logging.WARNING)
logging.getLogger("raster_cython_utils").setLevel(logging.WARNING)
LOGGER = logging.getLogger('natcap.invest.nearshore_wave_and_erosion')
logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')



# TODO: Get rid of this function!!!! (in CV too)
def clip_datasource(aoi_ds, orig_ds, output_uri):
    """Clip an OGR Datasource of geometry type polygon by another OGR Datasource
        geometry type polygon. The aoi_ds should be a shapefile with a layer
        that has only one polygon feature

        aoi_ds - an OGR Datasource that is the clipping bounding box
        orig_ds - an OGR Datasource to clip
        out_uri - output uri path for the clipped datasource

        returns - a clipped OGR Datasource """
    orig_layer = orig_ds.GetLayer()
    aoi_layer = aoi_ds.GetLayer()

    # If the file already exists remove it
    if os.path.isfile(output_uri):
        os.remove(output_uri)
    elif os.path.isdir(output_uri):
        shutil.rmtree(output_uri)

    # Create a new shapefile from the orginal_datasource
    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    output_datasource = output_driver.CreateDataSource(output_uri)

    # Get the original_layer definition which holds needed attribute values
    original_layer_dfn = orig_layer.GetLayerDefn()

    # Create the new layer for output_datasource using same name and geometry
    # type from original_datasource as well as spatial reference
    output_layer = output_datasource.CreateLayer(
            original_layer_dfn.GetName(), orig_layer.GetSpatialRef(),
            original_layer_dfn.GetGeomType())

    # Get the number of fields in original_layer
    original_field_count = original_layer_dfn.GetFieldCount()

    # For every field, create a duplicate field and add it to the new
    # shapefiles layer
    for fld_index in range(original_field_count):
        original_field = original_layer_dfn.GetFieldDefn(fld_index)
        output_field = ogr.FieldDefn(
                original_field.GetName(), original_field.GetType())
        output_field.SetWidth(original_field.GetWidth())
        output_field.SetPrecision(original_field.GetPrecision())
        output_layer.CreateField(output_field)

    # Get the feature and geometry of the aoi
    aoi_feat = aoi_layer.GetFeature(0)
    aoi_geom = aoi_feat.GetGeometryRef()

    # Iterate over each feature in original layer
    missing_geometries = 0 # Count the number of features without a geometry
    total_geometries = 0
    for orig_feat in orig_layer:
        total_geometries += 1
        # Get the geometry for the feature
        orig_geom = orig_feat.GetGeometryRef()
        if orig_geom is None:
            missing_geometries += 1 # Will report the missing geometries later
            continue
        # Check to see if the feature and the aoi intersect. This will return a
        # new geometry if there is an intersection. If there is not an
        # intersection it will return an empty geometry or it will return None
        # and print an error to standard out
        intersect_geom = aoi_geom.Intersection(orig_geom)

        if not intersect_geom == None and not intersect_geom.IsEmpty():
            # Copy original_datasource's feature and set as new shapes feature
            output_feature = ogr.Feature(
                    feature_def=output_layer.GetLayerDefn())
            output_feature.SetGeometry(intersect_geom)
            # Since the original feature is of interest add it's fields and
            # Values to the new feature from the intersecting geometries
            for fld_index2 in range(output_feature.GetFieldCount()):
                orig_field_value = orig_feat.GetField(fld_index2)
                output_feature.SetField(fld_index2, orig_field_value)

            output_layer.CreateFeature(output_feature)
            output_feature = None
    # If missing geometries, report it
    if missing_geometries:
        message = 'Missing ' + str(missing_geometries) + \
        ' features out of' + str(total_geometries) + ' geometries in' + \
        output_uri
        LOGGER.warning(message)

    return output_datasource

# TODO: write a unit test for this function
def projections_match(projection_list, silent_mode = True):
    """Check that two gdal datasets are projected identically.
       Functionality adapted from Doug's
       biodiversity_biophysical.check_projections

        Inputs:
            - projection_list: list of wkt projections to compare
            - silent_mode: id True (default), don't output anything, otherwise
              output if and why some projections are not the same.

        Output:
            - False the datasets are not projected identically.
    """
    assert len(projection_list) > 1

    srs_1 = osr.SpatialReference()
    srs_2 = osr.SpatialReference()

    srs_1.ImportFromWkt(projection_list[0])

    for projection in projection_list:
        srs_2.ImportFromWkt(projection)

        if srs_1.IsProjected() != srs_2.IsProjected():
            if not silent_mode:
                message = \
                'Different projection: One of the Rasters is Not Projected'
                LOGGER.debug(message)
            return False
        if srs_1.GetLinearUnits() != srs_2.GetLinearUnits():
            #LOGGER.debug('Different proj.: Proj units do not match %s:%s', \
            #         srs_1.GetLinearUnits(), srs_2.GetLinearUnits())
            return False

        if srs_1.GetAttrValue("PROJECTION") != srs_2.GetAttrValue("PROJECTION"):
            #LOGGER.debug('Projections are not the same')
            return False
        # At this point, everything looks identical. look further into the
        # projection parameters:
        for parameter in ['false_easting', 'false_northing', \
            'standard_parallel_1', 'standard_parallel_2', \
            'latitude_of_center', 'longitude_of_center']:
            if srs_1.GetProjParm(parameter) != srs_2.GetProjParm(parameter):
                return False
    return True

def raster_wkt(raster):
    """ Return the projection of a raster in the OpenGIS WKT format.

        Input:
            - raster: raster file

        Output:
            - a projection encoded as a WKT-compliant string."""
    return raster.GetProjection()

def shapefile_wkt(shapefile):
    """ Return the projection of a shapefile in the OpenGIS WKT format.

        Input:
            - raster: raster file

        Output:
            - a projection encoded as a WKT-compliant string."""
    layer = shapefile.GetLayer()
    sr = layer.GetSpatialRef()
    return sr.ExportToWkt()

def adjust_raster_to_aoi(in_dataset_uri, aoi_datasource_uri, cell_size, \
    out_dataset_uri):
    """Adjust in_dataset_uri to match aoi_dataset_uri's extents, cell size and
    projection.

        Inputs:
            - in_dataset_uri: the uri of the dataset to adjust
            - aoi_dataset_uri: uri to the aoi we want to use to adjust
                in_dataset_uri
            - out_dataset_uri: uri to the adjusted dataset

        Returns:
            - out_dataset_uri"""

    # Split the path apart from the filename
    out_head, out_tail = os.path.split(out_dataset_uri)
    _, tail = os.path.split(in_dataset_uri)

    # Split the file basename from the file extension
    out_base, _ = os.path.splitext(out_tail)
    base, _ = os.path.splitext(tail)

    # Preliminary variable initialization
    aoi_wkt = shapefile_wkt(ogr.Open(aoi_datasource_uri))
    input_wkt = raster_wkt(gdal.Open(in_dataset_uri))
    message = "Cannot extract a Well Known Transformation (wkt) from " + \
        str(in_dataset_uri) + ". Please check the file has a valid WKT."
    assert input_wkt, message

    # Reproject AOI to input dataset projection
    reprojected_aoi_uri = os.path.join(out_head, base + '_reprojected_aoi.shp')
    pygeoprocessing.geoprocessing.reproject_datasource_uri(aoi_datasource_uri, input_wkt, \
        reprojected_aoi_uri)

    # Clip dataset with reprojected AOI
    clipped_dataset_uri = os.path.join(out_head, out_base + '_clipped_unprojected.tif')
    pygeoprocessing.geoprocessing.clip_dataset_uri(in_dataset_uri, reprojected_aoi_uri, \
        clipped_dataset_uri, False)

    # Reproject clipped dataset to AOI's projection
    pygeoprocessing.geoprocessing.reproject_dataset_uri(clipped_dataset_uri, \
        cell_size, aoi_wkt, 'bilinear', out_dataset_uri)
    # Done, return the dataset uri
    return out_dataset_uri

# TODO: write a unit test for this function
def adjust_shapefile_to_aoi(data_uri, aoi_uri, output_uri, \
    empty_raster_allowed = False):
    """Adjust the shapefile's data to the aoi, i.e.reproject & clip data points.

        Inputs:
            - data_uri: uri to the shapefile to adjust
            - aoi_uri: uir to a single polygon shapefile
            - base_path: directory where the intermediate files will be saved
            - output_uri: dataset that is clipped and/or reprojected to the
            aoi if necessary.
            - empty_raster_allowed: boolean flag that, if False (default),
              causes the function to break if output_uri is empty, or return
              an empty raster otherwise.

        Returns: output_uri
    """
    # Data and aoi are the same URIs, data is good as it is, return it.
    if data_uri == aoi_uri:
        return data_uri
    # Split the path apart from the filename
    head, tail = os.path.split(output_uri)
    # Split the file basename from the file extension
    base, _ = os.path.splitext(tail)
    # Open URIs and get the projections
    data = ogr.Open(data_uri)
    message = "OGR Can't open " + data_uri
    assert data is not None, message
    aoi = ogr.Open(aoi_uri)
    data_wkt = shapefile_wkt(data)
    aoi_wkt = shapefile_wkt(aoi)

    if projections_match([data_wkt, aoi_wkt]):
        # Same projections, just clip
        clip_datasource(aoi, data, output_uri)
    else:
        # Reproject the aoi to be in data's projection
        projected_aoi_uri = os.path.join(head, base + '_projected_aoi')
        # TODO: include this in raster utils
        # Removing output_uri if it already exists
        if os.path.isdir(projected_aoi_uri):
            shutil.rmtree(projected_aoi_uri)
        pygeoprocessing.geoprocessing.reproject_datasource(aoi, data_wkt, projected_aoi_uri)
        # Clip all the shapes outside the aoi
        out_uri = os.path.join(head, base + '_clipped')
        clip_datasource(ogr.Open(projected_aoi_uri), data, out_uri)
        # Convert the datasource back to the original projection (aoi's)
        # Removing output_uri if it already exists
        if os.path.isdir(output_uri):
            shutil.rmtree(output_uri)
        pygeoprocessing.geoprocessing.reproject_datasource(ogr.Open(out_uri), aoi_wkt, \
        output_uri)
    # Ensure the resulting file's 1st layer is not empty
    out_shapefile = ogr.Open(output_uri)
    out_layer = out_shapefile.GetLayer(0)
    out_feature_count = out_layer.GetFeatureCount()
    out_layer = None
    out_shapefile = None
    # Break if returning an empty raster is not allowed
    if not empty_raster_allowed:
        message = 'Error: first layer of ' + output_uri + ' is empty. Are ' + \
        data_uri + ' and ' + aoi_uri + ' mis-aligned?'
        assert out_feature_count > 0, message
    return output_uri

def get_layer_and_index_from_field_name(field_name, shapefile):
    """Given a field name, return its layer and field index.
        Inputs:
            - field_name: string to look for.
            - shapefile: where to look for the field.

        Output:
            - A tuple (layer, field_index) if the field exist in 'shapefile'.
            - (None, None) otherwise."""
    # Look into every layer
    layer_count = shapefile.GetLayerCount()
    for l in range(layer_count):
        layer = shapefile.GetLayer(l)
        # Make sure the layer is not empty
        feature_count = layer.GetFeatureCount()
        if feature_count > 0:
            feature = layer.GetFeature(0)
            # Enumerate every field
            field_count = feature.GetFieldCount()
            for f in range(field_count):
                field_defn = feature.GetFieldDefnRef(f)
                if field_defn.GetNameRef() == field_name:
                    return (l, f)
    # Didn't find fields, enumerate their names and the feature counts
    layer_count = shapefile.GetLayerCount()
    for l in range(layer_count):
        layer = shapefile.GetLayer(l)
        feature_count = layer.GetFeatureCount()
        print("Layer " + str(l) + " has " + str(feature_count) + " features.")
        if feature_count > 0:
            feature = layer.GetFeature(0)
            # Enumerate every field
            field_count = feature.GetFieldCount()
            print('fields:')
            for f in range(field_count):
                field_defn = feature.GetFieldDefnRef(f)
                print(field_defn.GetNameRef())
    # Nothing found
    return (None, None)

def combined_rank(R_k):
    """Compute the combined habitats ranks as described in equation (3)

        Inputs:
            - R_k: the list of ranks

        Output:
            - R_hab as decribed in the user guide's equation 3."""
    R = 4.8 -0.5 *math.sqrt( (1.5 *max(5-R_k))**2 + \
                    sum((5-R_k)**2) -(max(5-R_k))**2)
    if R <=0:
        print('rank', R, 'R_k', R_k)
    return R

# TODO: write a unit test for this function
def raster_from_shapefile_uri(shapefile_uri, aoi_uri, cell_size, output_uri, \
    field=None, all_touched=False, nodata = 0., datatype = gdal.GDT_Float32):
    """Burn default or user-defined data from a shapefile on a raster.

        Inputs:
            - shapefile: the dataset to be discretized
            - aoi_uri: URI to an AOI shapefile
            - cell_size: coarseness of the discretization (in meters)
            - output_uri: uri where the raster will be saved
            - field: optional field name (string) where to extract the data
                from.
            - all_touched: optional boolean that indicates if we use GDAL's
              ALL_TOUCHED parameter when rasterizing.

        Output: A shapefile where:
            If field is specified, the field data is used as burn value.
            If field is not specified, then:
                - shapes on the first layer are encoded as 1s
                - the rest is encoded as 0"""
    shapefile = ogr.Open(shapefile_uri)
    message = "Can't open shapefile " + shapefile_uri
    assert shapefile, message
    if aoi_uri == shapefile_uri:
        aoi = shapefile
    else:
        aoi = ogr.Open(aoi_uri)
    # Create the raster that will contain the new data
    raster = \
        pygeoprocessing.geoprocessing.create_raster_from_vector_extents(cell_size,
        cell_size, datatype, nodata, \
        output_uri, aoi)
    layer = shapefile.GetLayer(0)
    # Add the all_touched option
    options = ['ALL_TOUCHED='+str(all_touched).upper()]
    if field:
        # Burn the data in 'field' to a raster
        layer_id, _ = \
        get_layer_and_index_from_field_name(field, shapefile)
        layer = shapefile.GetLayer(layer_id)
        options = ['ATTRIBUTE='+field] + options
        gdal.RasterizeLayer(raster, [1], layer, options = options)
    else:
        gdal.RasterizeLayer(raster, [1], layer, burn_values = [1], \
            options = options)
    return output_uri

def preprocess_dataset(dataset_uri, aoi_uri, cell_size, output_uri):
    """Function that preprocesses an input dataset (clip,
    reproject, resample) so that it is ready to be used in the model

        Inputs:
            -dataset_uri: uri to the input dataset to be pre-processed
            -aoi_uri: uri to an aoi polygon datasource that is used for
                clipping and reprojection.
            -cell_size: output dataset cell size in meters (integer)
            -output_uri: uri to the pre-processed output dataset.

        Returns output_uri
    """
    # Adjust the dataset to the aoi and save the result
    adjust_raster_to_aoi(dataset_uri, aoi_uri, cell_size, output_uri)

    return output_uri

def preprocess_polygon_datasource(datasource_uri, aoi_uri, cell_size, \
    output_uri, field_name = None, all_touched = False, nodata = 0., \
    empty_raster_allowed = False):
    """Function that converts a polygon shapefile to a dataset by clipping,
    reprojecting, resampling, burning, and extrapolating burnt values.

        Inputs:
            -datasource_uri: uri to the datasource to be pre-processed
            -aoi_uri: uri to an aoi polygon datasource that is used for
                clipping and reprojection.
            -cell_size: output dataset cell size in meters (integer)
            -output_uri: uri to the pre-processed output dataset.
            -field_name: name of the field in the attribute table to get the
            values from. If a number, use it as a constant. If Null, use 1.
            -all_touched: boolean flag used in gdal's vectorize_rasters
            options flag
            -nodata: float used as nodata in the output raster
            -empty_raster_allowed: flag that allows the function to return an
            empty raster if set to True, or break if set to False. False is the
            default.

        Returns output_uri"""
    # Split the path apart from the filename
    head, tail = os.path.split(output_uri)
    # Split the file basename from the file extension
    base, _ = os.path.splitext(tail)
    # Insert a suffix to the file basename and glue the new path together
    aoi_adjusted_uri = os.path.join(head, base + '_aoi_adjusted')
    # Adjust the shapefile to the aoi and save the result
    aoi_adjusted_uri = \
    adjust_shapefile_to_aoi(datasource_uri, aoi_uri, aoi_adjusted_uri, \
    empty_raster_allowed)
    # Burn the adjusted file to a raster
    raster_from_shapefile_uri(aoi_adjusted_uri, aoi_uri, cell_size, \
        output_uri, field_name, all_touched, nodata)

    return output_uri

def extract_raster_information(raster_uri):
    raster_nodata = \
        pygeoprocessing.geoprocessing.get_nodata_from_uri(raster_uri)

    cell_size = \
        pygeoprocessing.geoprocessing.get_cell_size_from_uri(raster_uri)

    return (raster_nodata, cell_size)


def hdf5_is_gzip_capable():
    """Test if the current hdf5 distribution can compress data using gzip.
        Used code from Andrew Collette's python script posted in this forum:
        https://groups.google.com/forum/#!topic/h5py/m_Sqt2qZNQk

        Inputs: None

        Returns True if the gzip filter is available, False otherwise
    """
    return h5z.filter_avail(h5z.FILTER_DEFLATE)


def execute(args):
    """This function invokes the coastal protection model given uri inputs
        specified by the user's guide.

    args - a dictionary object of arguments

    args['workspace_dir']     - The file location where the outputs will
                                be written (Required)
    """

    if not hdf5_is_gzip_capable():
        anaconda_gzip_issue = \
            "https://groups.google.com/forum/#!topic/h5py/m_Sqt2qZNQk"
        anaconda_website = \
            "https://store.continuum.io/cshop/anaconda/"
        h5py_users_manual = \
            "docs.h5py.org/en/latest/high/dataset.html#filter-pipeline"

        LOGGER.critical("The gzip filter is not available on your current" + \
            " distribution of HDF5. Some anaconda versions of h5py" + \
            " don't have gzip capability (%s). In this case," + \
            " you can download the latest version of anaconda here: %s." + \
            " For more information on HDF5 filters, see here: %s", \
            anaconda_gzip_issue, anaconda_website, h5py_users_manual)
        assert hdf5_is_gzip_capable(), "See log message above"

    # Add the Output directory to the given workspace
    args['output_dir'] = \
        os.path.join(args['workspace_dir'], 'output')
    if not os.path.isdir(args['output_dir']):
        os.makedirs(args['output_dir'])

    # Add the intermediate directory as well
    args['intermediate_dir'] = \
        os.path.join(args['workspace_dir'], 'intermediate')
    if not os.path.isdir(args['intermediate_dir']):
        os.makedirs(args['intermediate_dir'])

    # Initializations
    # This is the finest useful scale at which the model can extract bathy data
    args['cell_size'] = max(args['model_resolution'], \
        pygeoprocessing.geoprocessing.get_cell_size_from_uri(args['bathymetry_uri']))

    # Look for mis-aligned shore up to 10 pixels inland
    args['max_land_profile_len'] = int(10 * args['cell_size'])
    args['max_land_profile_height'] = 20 # Maximum inland elevation


    # Habitats shouldn't overlap. If they do, pick the highest priority one.
    # Habitat information is an array of (habitat_name, habitat_info) tuples
    # habitat_info is a distionary with all the information necessary to
    # resolve habitat_name:
    #   -first entry: ('habitat':habitat_type)
    #   -subsequent entries: ('field_name':field_value)
    #
    # ------- Habitat type values conventions: --------
    #
    # Each habitat has a type. The habitat type is used in situations where
    # habitats overlap: there can only be one habitat type at a given location.
    # To resolve the ovrerlap, the model propritizes some habitats over others
    # by assigning them a higher 'type' value.
    #
    # ------- Habitats constraints conventions: --------
    # The field 'constraints' indicates where a specific layer can be found.
    # The limits are expressed in various units. For instance:
    #
    #   -for the constraints land/water: the unit is a distance from the shore
    #       in meters
    #   -for MLLW: the unit is depth, which is expressed as a number of MLLW.
    #       For instance, a value of 2 for 'MLLW' means twice the MLLW depth.
    #   -For MHHW: the unit is an elevation expressed as a number of MHHW.
    #       Similarly to 'MLLW', a value of 3 for 'MHHW' indicates an
    #       elevation 3 times that of MHHW.
    #
    # Note: All the distances and elevations/depth are relative to the shore.
    #
    # The values can be either positive or negative:
    #   -Positive values indicate that the habitat is allowed from the shore
    #       up to a certain distance away from the shore:
    #       'land':10 -> the habitat is allowed to grow from 0 to 10m from the
    #           shore landward.
    #       'water':20 -> the habitat is allowed to grow from 0 to 20m from the
    #           shore seaward.
    #       'MLLW':2 -> the habitat is allowed to grow within depths that range
    #           from 0 to 2 units of MLLW. If MLLW is 1.2,
    #           the depth is 1.2x2=2.4m
    #       'MHHW':0.5 -> the habitat is allowed to grow within an elevations
    #           that range from 0 to 0.5 units of MHHW. If MLLW is 0.2,
    #           the depth is 0.2x0.5=0.1m
    #
    # Negative values are the opposite: instead of specifying an area FROM
    #   the shore out, it specifies an area UP TO the shore. In other words,
    #   The area
    #       'land':-10 -> the habitat is allowed to grow from 10m inland from
    #           the shore up to infinity landward.
    #       'MLLW':-2 -> the habitat cannot grow at depths shallower
    #           than 2 units of MLLW. If MLLW is 1.2, the depth is 1.2x2=2.4m
    #
    args['habitat_information'] = [
        ('kelp',
            {'shapefile type':'seagrass', 'type':1},
            {'constraints':{'land':0.}}),
        ('seagrass',
            {'shapefile type':'seagrass', 'type':2},
            {'constraints':{'land':0.}}),
        # underwater structures: 'type' doesn't exist in the
        # 'underwater structures' dictionary, will be assigned a default value
        # based on the habitat positional index in the 'habitat_information'
        ('underwater structures',
            {'shapefile type':'underwater structures'},
            {'constraints':{'land':0.}}),
        ('coral reef',
            {'shapefile type':'coral reef'},
            {'constraints':{'land':0.}}),
        ('levee',
            {'shapefile type':'man-made structure', 'type':5},
            {'constraints':{'land':-50.0, 'water':0.}}),
        ('beach',
            {'shapefile type':'beach'},
            {'constraints':{'MHHW':1, 'MLLW':1}}),
        ('seawall',
            {'shapefile type':'man-made structure', 'type':7},
            {'constraints':{'land':25.0, 'water':25.0}}),
        ('marsh',
            {'shapefile type':'marsh'},
            {'constraints':{'MLLW':2}}),
        ('mangrove',
            {'shapefile type':'mangrove'},
            {'constraints':{'MLLW':2}})
        ]



    # Collect and compile habitat information
    args['habitat_priority'] = {}   # dictionary of (habitat_name: priority)
    args['habitat_shapefiles'] = set() # Shapefile types that are habitats
    args['shapefile_priorities'] = {} # Per-shapefile habitat priorities

    for i in range(len(args['habitat_information'])):

        habitat_name = args['habitat_information'][i][0]
        shapefile_type = args['habitat_information'][i][1]['shapefile type']

        # Build the 'shapefile type':habitat_priority dictionary
        # The lowest the habitat priority, the more susceptible it will be replaced
        # by a habitat of higher priority
        args['habitat_priority'][habitat_name] = i + 1

        # List of shapefile types that are natural habitats
        args['habitat_shapefiles'].add(shapefile_type)

        # Dictionary that stores the set of all valid habitat codes (priority)
        # for every habitat shapefile
        if shapefile_type in args['shapefile_priorities']:
            args['shapefile_priorities'][shapefile_type].add( \
                args['habitat_priority'][habitat_name])
        else:
            args['shapefile_priorities'][shapefile_type] = set([ \
                args['habitat_priority'][habitat_name]])

#    print("args['habitat_priority']", args['habitat_priority'])
#    print("args['habitat_shapefiles']", args['habitat_shapefiles'])
#    print("args['shapefile_priorities']", args['shapefile_priorities'])

    # List all shapefiles in the habitats directory
    files = []

    # Collect habitat files
    habitat_files = os.listdir(args['habitats_directory_uri'])
    for file_name in habitat_files:
        files.append(os.path.join(args['habitats_directory_uri'], file_name))

    # Add additional files
    files.append(args['landmass_uri'])
    files.append(args['climatic_forcing_uri'])

    # Process each habitat
    args['shapefile_required_fields'] = { \
        # MHHW >= MSL >= MLLW, all floating point numbers
        'tidal information': set([ \
            'MHHW', \
            'MSL', \
            'MLLW']),
        'climatic forcing': set([ \
            'Surge', \
            'WindSpeed', \
            'WavePeriod', \
            'WaveHeight']), \
        # Type for soil type has to be either 1 (mud), 2 (sand), or 3 (gravel)
        'soil type': set([ \
            'DryDensity', \
            'ErosionCst', \
            'SedSize', \
            'ForshrSlop', \
            'BermLength', \
            'BermHeight', \
            'DuneHeight', \
            'Type']),
        # Seagrass type has to be either 1 (kelp) or 2 (eelgrass)
        # see 'type' values defined in args['habitat_information'] above
        'seagrass': set([ \
            'StemHeight', \
            'StemDiam', \
            'StemDensty', \
            'StemDrag', \
            'Type']),
        # This habitat has no 'type' field, so its default type is set to 3
        'underwater structures': set([ \
            'Geometry', \
            'ShoreDist', \
            'Height', \
            'BaseWidth', \
            'CrestWidth']),
        'coral reef': set([ \
            'FricCoverd',
            'FricUncov',
            'SLRKeepUp',
            'DegrUncov']),
        'man-made structures': set([ \
            'Type',
            'Height',
            'SideSlope',
            'OvertopLim']),
        'beach': set([ \
            'SedSize', \
            'ForshrSlop', \
            'BermLength', \
            'BermHeight', \
            'DuneHeight']),
        'marsh': set([ \
            'SurfRough',
            'SurRedFact',
            'StemHeight', \
            'StemDiam', \
            'StemDensty', \
            'StemDrag']),
        'mangrove': set([ \
            'SurfRough',
            'SurRedFact',
            'RootHeight',
            'RootDiam',
            'RootDensty',
            'RootsDrag',
            'TrnkHeight',
            'TrunkDiam',
            'TrnkDensty',
            'TrunkDrag',
            'CnpyHeight',
            'CanopyDiam',
            'CnpyDensty',
            'CanopyDrag',
            'DnstyReduc']),
        'terrestrial structures': set([ \
            'Type',
            'TransParam',
            'Height'])}


    # Preprocessing the AOI
    args['aoi_raster_uri'] = os.path.join(args['intermediate_dir'], 'aoi.tif')
    if not os.path.isfile(args['aoi_raster_uri']):
        LOGGER.debug('Pre-processing the AOI...')
        args['aoi_raster_uri'] = \
            preprocess_polygon_datasource(args['aoi_uri'], args['aoi_uri'], \
            args['cell_size'], args['aoi_raster_uri'])

    # Preprocess the landmass
    args['landmass_raster_uri'] = \
        os.path.join(args['intermediate_dir'], 'landmass.tif')
    if not os.path.isfile(args['landmass_raster_uri']):
        LOGGER.debug('Pre-processing landmass...')
        preprocess_polygon_datasource(args['landmass_uri'], \
            args['aoi_uri'], args['cell_size'], \
            os.path.join(args['intermediate_dir'], 'landmass.tif'))

    # Preprocess bathymetry
    args['bathymetry_raster_uri'] = \
        os.path.join(args['intermediate_dir'], 'bathymetry.tif')
    if not os.path.isfile(args['bathymetry_raster_uri']):
        LOGGER.debug('Pre-processing bathymetry...')
        bathy_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(args['bathymetry_uri'])
        preprocess_dataset(args['bathymetry_uri'], \
            args['aoi_uri'], args['cell_size'], args['bathymetry_raster_uri'])



    shapefile_required_fields = args['shapefile_required_fields']

    # Assign a positional index to every shapefile field
    assign_shapefile_field_index(args)

#    print("args['field_index']")
#    for category in args['field_index']:
#        print('  category', category)
#        if category == 'natural habitats':
#            for priority in args['field_index'][category]:
#                if 'name' in args['field_index'][category][priority]:
#                    print('    habitat name:', \
#                        args['field_index'][category][priority]['name'])
#                print('    priority', priority)
#                for field in args['field_index'][category][priority]['fields']:
#                    print('      ', \
#                        field, args['field_index'][category][priority]['fields'][field])
#        else:
#            for field in args['field_index'][category]['fields']:
#                print('      ', field, args['field_index'][category]['fields'][field])

    # Save the dictionary
    field_index_dictionary_uri = \
        os.path.join(args['intermediate_dir'], 'field_indices')
    json.dump(args['field_index'], open(field_index_dictionary_uri, 'w'))


    # Compute habitat field count
    args['habitat_field_count'] = \
        max([len(shapefile_required_fields[shp]) \
            for shp in shapefile_required_fields \
                if shp in args['habitat_shapefiles']])

    args['climatic_forcing_field_count'] = \
        len(shapefile_required_fields['climatic forcing'])

    args['tidal_forcing_field_count'] = \
        len(shapefile_required_fields['tidal information'])

    # Exclude the field 'Type'
    args['soil_field_count'] = len(shapefile_required_fields['soil type']) - 1


    # Precompute aoi nodata
    aoi_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(args['aoi_raster_uri'])
    bathymetry_nodata, cell_size = \
        extract_raster_information(args['bathymetry_raster_uri'])

    args['cell_size'] = cell_size
    args['bathymetry_nodata'] = bathymetry_nodata
    args['files'] = files

    # Rasters that will be aligned and clipped and resampled
    args['in_raster_list'] = []


    # Only does this for the profile generator...
    if 'Profile generator' in args['modules_to_run']:
        detect_shapefile_types(args)
        build_habitat_constraints(args)
        combine_habitat_contraints(args)
        uniformize_input_rasters(args)

    # We're done with boiler-plate code, now we can delve into core processing
    nearshore_wave_and_erosion_core.execute(args)



def detect_shapefile_types(args):
    """Detect the shapefile types.

        Inputs:
            -args['intermediate_dir']:  path to the current run's intermediate
                directory
            -args['aoi_uri']: URI of the original aoi shapefile
            -args['aoi_raster_uri']: URI of the rasterized aoi layer
            -args['cell_size']: Size of the bathymetry raster cell, in meters
            -args['shapefile_required_fields']: dictionary of the fields
                required to make the model interpret a shapefile as a given
                habitat. An example is:

                'seagrass': set([ \
                    'StemHeight', \
                    'StemDiam', \
                    'StemDensty', \
                    'StemDrag', \
                    'Type']),

            -args['habitat_priority']: priority value for every habitat.
                args['habitat_priority'][habitat_name] = 1 (kelp), 2 (seagrass),...

            -args['shapefile_priorities']: set of priority values expected in a
                specific shapefile: 8 for marsh, 1 and 2 for seagrass, etc.

            -args['in_raster_list']: list of allthe rasters created so far


        Create the following entries:
            -args['shapefile types']: shapefile types, grouped by category.

                type = args['shapefile types'][category][basename]
                -category: shapefile type (seagrass, marsh, coral reef...)
                -basename: filename on disk, without the path.

            -args['shapefiles']: List of all the raster URIs created from each
                shapefile field:
                uri = args['shapefile types'][category][basename][field_name]

        Returns nothing
    """

    files = args['files']
    in_raster_list = args['in_raster_list']

    # Keep track of the original shapefile type, grouped by category:
    # type = args['shapefile types'][category][basename]
    args['shapefile types'] = {}

    # Keep track of all the raster URIs created from each shapefile field:
    # uri = args['shapefile types'][category][basename][field_name]
    args['shapefiles'] = {}

    # Collect all the different fields and assign a weight to each
    field_values = {} # weight for each field_value
    shapefile_type_checksum = {} # checksum for each shapefile type
    power = 1
    shapefile_fields = set()

    for shapefile_type in args['shapefile_required_fields']:
        required_fields = args['shapefile_required_fields'][shapefile_type]
        field_signature = 0
        for required_field in required_fields:
            if required_field not in shapefile_fields:
                shapefile_fields.add(required_field)
                field_values[required_field] = power
                power *= 2
            field_signature += field_values[required_field]

        shapefile_type_checksum[field_signature] = shapefile_type


    # Looking at fields in shapefiles and compute their checksum to see if
    # they're of a known type
    known_shapefiles = set() # Set of known shapefiles that will be rasterized
    in_raster_list.append(args['aoi_raster_uri'])
    in_habitat_type = [] # habitat type. See args['habitat_information']
    for file_uri in files:
        basename = os.path.basename(file_uri)
        basename, ext = os.path.splitext(basename)
        if ext == '.shp':
            # Find the number of fields in this shapefile
            shapefile = ogr.Open(file_uri)
            assert shapefile, "can't open " + file_uri
            layer = shapefile.GetLayer(0)
            layer_def = layer.GetLayerDefn()
            field_count = layer_def.GetFieldCount()

            # Extract field names and build checksum
            shapefile_checksum = 0
            recognized_fields = set()
            unknown_fields = []
            for field_id in range(field_count):
                field_defn = layer_def.GetFieldDefn(field_id)
                field_name = field_defn.GetName()

                if field_name in field_values:
                    shapefile_checksum += field_values[field_name]
                    recognized_fields.add(field_name)
                else:
                    unknown_fields.append(field_name)

            # Check if the recognized fields fit a type of shapefile
            is_known_shapefile = False # Used to notify that a shapefile is unknown

            missing_fields = {} # Stores missing fields for each habitat

            for shapefile_type in args['shapefile_required_fields']:

                # All the necessary fields for this shapefile have been identified
                if args['shapefile_required_fields'][shapefile_type] <= recognized_fields:

                    is_known_shapefile = True

                    LOGGER.debug('Detected that %s is %s', file_uri, shapefile_type)

                    if shapefile_type in args['habitat_shapefiles']:
                        category = 'natural habitats'
                    else:
                        category = shapefile_type

                    # If new category (new shapefile type), add it to the dictionary
                    if category not in args['shapefiles']:
                        args['shapefiles'][category] = {}
                        args['shapefile types'][category] = {}

                    # New file, add it to the category
                    if basename not in args['shapefiles'][category]:
                        args['shapefiles'][category][basename] = {}
                        args['shapefile types'][category][basename] = \
                            shapefile_type

                    # Rasterize the known shapefile for each field name
#                    LOGGER.info('Processing %s...', file_uri)
                    for field_name in args['shapefile_required_fields'][shapefile_type]:
                        # Rasterize the shapefile's field
                        # If this habitat has subtypes, then the field 'type'
                        # is used to determine priority.
                        output_uri = os.path.join(args['intermediate_dir'], \
                            basename + '_' + field_name.lower() + '.tif')

                        if os.path.isfile(output_uri):
                            LOGGER.debug('Skipping file %s.', \
                                basename + '_' + field_name.lower() + '.tif')
                        else:
                            # Rasterize the current shapefile field
                            LOGGER.debug('rasterizing field %s to %s', \
                                field_name, output_uri)
                            preprocess_polygon_datasource(file_uri, \
                                args['aoi_uri'], args['cell_size'], \
                                output_uri, field_name = field_name, \
                                all_touched = True, nodata = -99999.0)

                        # Keep this raster uri
                        args['shapefiles'][category][basename][field_name] = \
                            output_uri
                        in_raster_list.append(output_uri)

                    # If priority raster not already added, add it now
                    if shapefile_type in args['habitat_priority']:
                        # Create the uri
                        output_uri = os.path.join(args['intermediate_dir'], \
                                basename + '_' + 'type' + '.tif')
                        if not os.path.isfile(output_uri):
                            LOGGER.debug('Creating type raster to %s', output_uri)
                            # Copy data over from most recent raster
                            shutil.copy(in_raster_list[-1], output_uri)
                            # Extract array
                            nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(output_uri)
                            raster = gdal.Open(output_uri, gdal.GA_Update)
                            band = raster.GetRasterBand(1)
                            array = band.ReadAsArray()
                            # Overwrite data with priority value
                            array[array != nodata] = \
                                args['habitat_priority'][shapefile_type]
                            LOGGER.debug('assigning %i to %s_type.tif', \
                                args['habitat_priority'][shapefile_type], basename)
                            band.WriteArray(array)
                            # clean-up
                            array = None
                            band = None
                            raster = None
                        # Add new uri to uri list
                        args['shapefiles'][category][basename]['Type'] = output_uri
                        in_raster_list.append(output_uri)

                    # Only valid habitat types should be stored in the raster
                    if shapefile_type in args['habitat_shapefiles']:
                        detected_uniques = set( \
                            pygeoprocessing.geoprocessing.unique_raster_values_count( \
                                output_uri).keys())

                        expected_uniques = \
                            args['shapefile_priorities'][shapefile_type]

                        message = 'Unexpected types in ' + \
                            str(detected_uniques) + \
                            ' found for habitat ' + str(shapefile_type) + \
                            '. Expected ' + str(expected_uniques)

                        assert detected_uniques <= expected_uniques, message

                    # This shapefile was recognized and processed,
                    # now stop and move on to the next shapefile
                    break

                else:
                    missing_fields[shapefile_type] = \
                        args['shapefile_required_fields'][shapefile_type] - \
                        recognized_fields

            if not is_known_shapefile:
                LOGGER.debug("Can't recognize shapefile %s", file_uri)
                filename = os.path.split(file_uri)[1]
                # Enumerate what's missing to comply for each shapefile type:
                for shapefile_type in missing_fields:
                    if len(missing_fields[shapefile_type]) < 5:
                        LOGGER.debug("To be %s, %s needs the fields %s", \
                            shapefile_type, filename, \
                            str(sorted(list(missing_fields[shapefile_type]))))



def build_habitat_constraints(args):
    """Build constraint maps that will be used to create the constraints masks

        Inputs:
            -args['habitat_information']: List of Habitath information, like:
                ('kelp',
                {'shapefile type':'seagrass', 'type':1},
                {'constraints':{'land':0.}})

            -args['shapefile types']['natural habitats']: dictionary of all
                the habitat shapefile types:

                args['shapefile types'][category][basename] = shapefile_type

                where:
                    -category is 'natural habitats'
                    -basename is the habitat filename on disk (no path)
                    -shapefile_type: keys of args['shapefile_required_fields']
                        examples are 'seagrass', 'coral reef', 'beach',
                        'marsh', 'mangroves', 'underwater structures' etc.

            -args['intermediate_dir']: path to the current run's intermediate
                directory
            -args['bathymetry_raster_uri']: URI of the rasterized bathymetry
                layer
            -args['aoi_raster_uri']: URI of the rasterized aoi layer
            -args['in_raster_list']: list of allthe rasters created so far


        Create the args dictionary entry args['constraints_type']
            of constraint URIs.

            Example keys are 'land', 'water', MHHW', 'MLLW'.

            args['constraints_type']['land'] = constraint_raster_uri

            where constraint_raster_uri is a raster that has been scaled
            so that the constraint threshold ('value' in the code) can
            directly be applied to get the mask.

        Returns nothing.

    """
    cell_size = args['cell_size']
    bathymetry_nodata = args['bathymetry_nodata']
    in_raster_list = args['in_raster_list']

    args['constraints_type'] = {}

    # Keep land and nodata
    def keep_land_and_nodata(x, aoi):
        result = numpy.zeros(x.shape) # Add everything
        result[x != bathymetry_nodata] = 1 # Remove land and water
        result[x > 0] = 0 # Add land

        return result

    # Keep water and nodata
    def keep_water_and_nodata(x, aoi):
        result = numpy.zeros(x.shape) # Add everything
        result[x > 0] = 1 # Remove land

        return result

    # Keep water only, discards nodata
    def keep_water(x, aoi):
        result = numpy.ones(x.shape) # Remove everything
        result[x != bathymetry_nodata] = 0 # Add land and water
        result[x > 0] = 1 # Remove land

        return result

    # Keep land only, discards nodata
    def keep_land(x, aoi):
        result = numpy.ones(x.shape) # Remove everything
        result[x > 0] = 0 # Add land

        return result

    # Scales a raster inplace by 'scaling_factor'
    def scale_raster_inplace(raster_uri, scaling_factor):
        temp_uri = pygeoprocessing.geoprocessing.temporary_filename()

        pygeoprocessing.geoprocessing.vectorize_datasets([raster_uri], \
            lambda x: x * scaling_factor, temp_uri, gdal.GDT_Float32, -1, \
            cell_size, 'intersection', vectorize_op = False)

        os.remove(raster_uri)
        os.rename(temp_uri, raster_uri)

    # Computing constraints rasters so they can directly be used as masks
    # for the habitats that have spatial constraints.
#    print("args['shapefile types']", args['shapefile types'])

    # TODO: Should remove this limitation
    assert 'natural habitats' in args['shapefile types'], \
        "No natural habitats detected."

    for detected_habitat in args['shapefile types']['natural habitats']:

        detected_habitat_type = args['shapefile types']['natural habitats'][detected_habitat]


        for habitat_information in args['habitat_information']:
            habitat_type = habitat_information[0]
            if habitat_type != detected_habitat_type:
                continue

            habitat_constraints = habitat_information[2]['constraints']

            # Detected a land-related distance constraint
            if 'land' in habitat_constraints:
                constraint_uri = os.path.join(args['intermediate_dir'], \
                    'land_distance_map.tif')

                # Create the constraint raster if it doesn't exist already
    #            if not os.path.isfile(constraint_uri):
    #                print('Creating land constraint', constraint_uri)

                land_distance_mask_uri = \
                    os.path.join(args['intermediate_dir'], \
                        'land_distance_mask.tif')

                pygeoprocessing.geoprocessing.vectorize_datasets( \
                    [args['bathymetry_raster_uri'], args['aoi_raster_uri']], \
                    keep_land, land_distance_mask_uri, gdal.GDT_Float32, \
                    -1, cell_size, 'intersection', vectorize_op = False)

                # Use the mask to compute distance over land
                pygeoprocessing.geoprocessing.distance_transform_edt(land_distance_mask_uri, \
                    constraint_uri)

                scale_raster_inplace(constraint_uri, cell_size)

                # Keep this out of the if statement
                args['constraints_type']['land'] = constraint_uri

            # Detect a sea-related distance constraint
            if 'water' in habitat_constraints:
                constraint_uri = os.path.join(args['intermediate_dir'], \
                    'water_distance_map.tif')

                # Create the constraint raster if it doesn't exist already
    #            if not os.path.isfile(constraint_uri):
    #                print('Creating water constraint', constraint_uri)

                water_distance_mask_uri = \
                    os.path.join(args['intermediate_dir'], \
                        'water_distance_mask.tif')

                pygeoprocessing.geoprocessing.vectorize_datasets( \
                    [args['bathymetry_raster_uri'], args['aoi_raster_uri']], \
                    keep_water, water_distance_mask_uri, gdal.GDT_Float32, \
                    -1, cell_size, 'intersection', vectorize_op = False)

                # Use the mask to compute distance over land
                pygeoprocessing.geoprocessing.distance_transform_edt(water_distance_mask_uri, \
                    constraint_uri)

                scale_raster_inplace(constraint_uri, cell_size)

                # Keep this out of the if statement
                args['constraints_type']['water'] = constraint_uri


            # Detect a mean high water-related depth constraint
            if 'MHHW' in habitat_constraints:
                constraint_uri = os.path.join(args['intermediate_dir'], \
                    'MHHW_depth_map.tif')

                # Create the constraint raster if it doesn't exist already
                if not os.path.isfile(constraint_uri):

                    MHHW_depth_mask_uri = \
                        os.path.join(args['intermediate_dir'], \
                            'MHHW_depth_mask.tif')

                    pygeoprocessing.geoprocessing.vectorize_datasets( \
                        [args['bathymetry_raster_uri'], args['aoi_raster_uri']], \
                        keep_land, MHHW_depth_mask_uri, gdal.GDT_Float32, \
                        bathymetry_nodata, cell_size, 'intersection', vectorize_op = False)

                    # Now, find the mean MHHW
                    MHHW_depth_mask_uri = \
                        os.path.join(args['intermediate_dir'], \
                            'MHHW_depth_mask.tif')

                    # Find the filename
                    MHHW_uri = ''
                    for raster_uri in in_raster_list:
                        if 'mhhw' in raster_uri:
                            MHHW_uri = raster_uri
                            break
                    if not MHHW_uri:
                        print("Can't find MHHW raster in raster list")
                        print('Files are:', in_raster_list)
                        assert MHHW_uri, "Didn't find MHHW raster from landmass shapefile."

                    # Compute the average MHHW
                    MHHW_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(MHHW_uri)
                    MHHW_raster = gdal.Open(MHHW_uri)
                    MHHW_band = MHHW_raster.GetRasterBand(1)
                    (MHHW_min, MHHW_max) = MHHW_band.ComputeRasterMinMax()
                    MHHW_band = None
                    MHHW_raster = None

                    mean_MHHW = (MHHW_min + MHHW_max) / 2.

                    assert mean_MHHW > 0, "MHHW can't be negative"

                    # Scale depths to mean_MHHW
                    pygeoprocessing.geoprocessing.vectorize_datasets( \
                        [MHHW_depth_mask_uri, args['bathymetry_raster_uri']], \
                        lambda x, y: numpy.where(x==0, y / mean_MHHW, 0.), \
                        constraint_uri, gdal.GDT_Float32, bathymetry_nodata, cell_size, \
                        'intersection', vectorize_op = False)

                args['constraints_type']['MHHW'] = constraint_uri


            # Detect a mean low water-related depth constraint
            if 'MLLW' in habitat_constraints:
                constraint_uri = os.path.join(args['intermediate_dir'], \
                    'MLLW_depth_map.tif')

                # Create the constraint raster if it doesn't exist already
                if not os.path.isfile(constraint_uri):

                    MLLW_depth_mask_uri = \
                        os.path.join(args['intermediate_dir'], \
                            'MLLW_depth_mask.tif')

                    pygeoprocessing.geoprocessing.vectorize_datasets( \
                        [args['bathymetry_raster_uri'], args['aoi_raster_uri']], \
                        keep_water, MLLW_depth_mask_uri, gdal.GDT_Float32, \
                        bathymetry_nodata, cell_size, 'intersection', vectorize_op = False)

                    # Now, find the mean MLLW
                    MLLW_depth_mask_uri = \
                        os.path.join(args['intermediate_dir'], \
                            'MLLW_depth_mask.tif')

                    # Find the filename
                    MLLW_uri = ''
                    for raster_uri in in_raster_list:
                        if 'mllw' in raster_uri:
                            MLLW_uri = raster_uri
                            break
                    assert MLLW_uri, "Didn't find MLLW raster from landmass shapefile."

                    # Compute the average MLLW
                    MLLW_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(MLLW_uri)
                    MLLW_raster = gdal.Open(MLLW_uri)
                    MLLW_band = MLLW_raster.GetRasterBand(1)
                    (MLLW_min, MLLW_max) = MLLW_band.ComputeRasterMinMax()
                    MLLW_band = None
                    MLLW_raster = None

                    mean_MLLW = (MLLW_min + MLLW_max) / 2.

                    assert mean_MLLW < 0, "MLLW can't be positive"

                    # Scale depths to mean_MLLW
                    pygeoprocessing.geoprocessing.vectorize_datasets( \
                        [MLLW_depth_mask_uri, args['bathymetry_raster_uri']], \
                        lambda x, y: numpy.where(x==0, y / mean_MLLW, 0.), \
                        constraint_uri, gdal.GDT_Float32, bathymetry_nodata, cell_size, \
                        'intersection', vectorize_op = False)

                args['constraints_type']['MLLW'] = constraint_uri



def combine_habitat_contraints(args):
    """Build maps that will be used to constrain where habitats can be placed:

        Every constraint can be expressed as a mask that follows the same
        pattern:
            1- Compute masks with limits dictated by the threshold's sign
            2- Take the masks intersection

        Inputs:
            -args['habitat_information']: List of Habitath information, like:
                ('kelp',
                {'shapefile type':'seagrass', 'type':1},
                {'constraints':{'land':0.}})

            -args['shapefile types']['natural habitats']: dictionary of all
                the habitat shapefile types:

                args['shapefile types'][category][basename] = shapefile_type

                where:
                    -category is 'natural habitats'
                    -basename is the habitat filename on disk (no path)
                    -shapefile_type: keys of args['shapefile_required_fields']
                        examples are 'seagrass', 'coral reef', 'beach',
                        'marsh', 'mangroves', 'underwater structures' etc.

            -args['constraints_type']: Dictionary of the constraint URIs.
                Example keys are 'land', 'water', MHHW', 'MLLW'.

                args['constraints_type']['land'] = constraint_raster_uri

                where constraint_raster_uri is a raster that has been scaled
                so thet the constraint threshold ('value' in the code) can
                directly be applied to get the mask.


        Appends the constraints masks URI to each constrained habitat in
        args['habitat_information']. For instance, the entry

            ('kelp',
            {'shapefile type':'seagrass', 'type':1},
            {'constraints':{'land':0.}})

        could be updated witht he following information:

            ('kelp',
            {'shapefile type':'seagrass', 'type':1},
            {'constraints':{'land':0.},
            'constraints_uri':'full_path_to_constraint_raster'})


        Returns nothing.
    """
    # Compute the constraints for each habitat
    def combine_constraints(*x):
        result = x[0]
        for i in range(1, len(x)):
            result = numpy.logical_or(result, x[i])
        return result


    def set_threshold_and_nodata(T, nodata):
        """Closure to store the threshold and nodata used to compute the mask."""


        def apply_positive_threshold(x):
            """Apply the threshold to the vector x"""

            # First step: value between 0 and T
            result = numpy.logical_and(x > 0, x <= T)

            # Second step: mask out nodata
            result = numpy.logical_and(result, x != nodata)

            return result

        def apply_negative_threshold(x):
            """Apply the threshold to the vector x"""

            # Mask out for T and nodata in a single step
            result = numpy.logical_and(x < T, x != nodata)

            return result


        # Determine what function to return depending on T's sign
        if T >= 0:
            function = apply_positive_threshold
        else:
            function = apply_positive_threshold

        return function



    cell_size = args['cell_size']

    # Loop through the habitats
    for habitat_info in args['habitat_information']:
        habitat_name = habitat_info[0]

        current_habitat_was_detected = False

        for detected_habitat in args['shapefile types']['natural habitats']:

            if habitat_name ==  \
                args['shapefile types']['natural habitats'][detected_habitat]:
                current_habitat_was_detected = True
                continue

        if not current_habitat_was_detected:
            continue

        # -For each constraint:
        constraint_uri_list = []
        for constraint_type in habitat_info[2]['constraints'].keys():
            # Retreive constraint URI
            constraint_uri = args['constraints_type'][constraint_type]
            constraint_nodata = \
                pygeoprocessing.geoprocessing.get_nodata_from_uri(constraint_uri)

            # -Extract the constraint value
            value = habitat_info[2]['constraints'][constraint_type]

            # Get the masking function:
            mask_function = set_threshold_and_nodata(value, constraint_nodata)

            # Where the mask raster will be stored
            output_uri = pygeoprocessing.geoprocessing.temporary_filename()

            # Computer the mask raster
            pygeoprocessing.geoprocessing.vectorize_datasets( \
                [constraint_uri], \
                mask_function, \
                output_uri, \
                gdal.GDT_Float32, \
                constraint_nodata, \
                cell_size, \
                'intersection', \
                vectorize_op = False)

            # -Add constraint to constraint_list
            constraint_uri_list.append(output_uri)

        # Where the habitat constraint mask raster will be stored
        habitat_info[2]['constraint_uri'] = \
            os.path.join(args['intermediate_dir'], \
                habitat_name + '_constraint_mask.tif')

        # Combine individual constraints for this natural habitat
        pygeoprocessing.geoprocessing.vectorize_datasets(constraint_uri_list, \
            combine_constraints, habitat_info[2]['constraint_uri'], \
            gdal.GDT_Float32, constraint_nodata, cell_size, \
            'intersection', vectorize_op = False)



def uniformize_input_rasters(args):
    """Uniformize the input rsters so they are size-compatible

        Inputs:
            -args['in_raster_list']: list of rasters which size is to be uniformized
            -args['landmass_raster_uri']: raster URI for the landmass
            -args['bathymetry_uri']: raster URI for the bathymetry

        Returns nothing.
    """

    LOGGER.debug('Uniformizing the input raster sizes...')

    # Need to uniformize the size of land and bathymetry rasters
    in_raster_list = args['in_raster_list']

    in_raster_list.append(args['landmass_raster_uri'])
    in_raster_list.append(args['bathymetry_raster_uri'])


    # For every input raster, create a corresponding output raster
    out_raster_list = []
    for uri in in_raster_list:
        out_raster_list.append(pygeoprocessing.geoprocessing.temporary_filename())

    # Gather info for aligning rasters properly
    cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(args['landmass_raster_uri'])
    resample_method_list = ['bilinear'] * len(out_raster_list)
    out_pixel_size = cell_size
    mode = 'intersection'
    dataset_to_align_index = 0
    dataset_to_bound_index = 0

    # Invoke raster alignment function
    pygeoprocessing.geoprocessing.align_dataset_list( \
        in_raster_list, out_raster_list, resample_method_list,
        out_pixel_size, mode, dataset_to_align_index, dataset_to_bound_index)

    # Now copy the result back to the original files
    for in_uri, out_uri in zip(in_raster_list, out_raster_list):
        os.remove(in_uri)
        os.rename(out_uri, in_uri)

    # Quick sanity test with shape just to make sure
    landmass_raster_shape = \
        pygeoprocessing.geoprocessing.get_row_col_from_uri(args['landmass_raster_uri'])
    bathymetry_raster_shape = \
        pygeoprocessing.geoprocessing.get_row_col_from_uri(args['bathymetry_raster_uri'])
    assert landmass_raster_shape == bathymetry_raster_shape

    LOGGER.debug('Done')


def assign_shapefile_field_index(args):
    """ Assign a positional index to every field of the known shapefiles

        Inputs:
            -args: the argument dictionary where the field indices are going
                to be stored.

            -args['shapefile_required_fields']: a dictionary of all the required fields
                stored for each known shapefile

            -args['habitat_shapefiles']: the set of habitat shapefiles

            -args['habitat_priority']: priority value for every habitat.
                args['habitat_priority'][habitat_name] = 1 (kelp), 2 (seagrass),...


        Creates a 'field_index' entry in 'args' and stores the shapefile fields
            per category and per shapefile name, e.g.:

        ID = args['field_index']['natural habitats']['coral reef']['SLRKeepUp']
    """

    args['field_index'] = {}
    for shapefile in args['shapefile_required_fields']:

        # Group natural habitats together, keep the other fields
        # in their own category

        # Found a known habitat => put it in 'natural habitats' category
        if shapefile in args['habitat_shapefiles']:
            # Start by creating a habitat if necessary
            if 'natural habitats' not in args['field_index']:
                args['field_index']['natural habitats'] = {}

            destination = args['field_index']['natural habitats']

        # Otherwise => put the shapefile in its own category
        else:
            args['field_index'][shapefile] = {}

            destination = args['field_index'][shapefile]

        # Create priority keys (numerical) for natural habitats:
        if shapefile in args['habitat_shapefiles']:

            # Find the natural habitat in habitat information
            for habitat_name in args['habitat_priority']:
                if habitat_name == shapefile:
                    priority = args['habitat_priority'][habitat_name]

                    destination[priority] = {}

                    destination = destination[priority]

                    destination['name'] = shapefile

                    break


        # Gather all the fields for this shapefile type
        required_fields = args['shapefile_required_fields'][shapefile]

        # And create an entry when they will be stored
        destination['fields'] = {}

        field_id = 0
        for field_name in required_fields:

            # Only add if field is not 'type'
            if field_name.lower() != 'type':

                # Field name is set to its index in the required fields array
                destination['fields'][field_name.lower()] = field_id

                field_id += 1

#This part is for command line invocation and allows json objects to be passed
#as the argument dictionary
if __name__ == '__main__':

    modulename, json_args = sys.argv
    args = json.loads(json_args)
    execute(args)
