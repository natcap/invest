"""Module for the execution of the biophysical component of the InVEST Nutrient
Retention model."""

import logging
import os
import csv

from osgeo import gdal
from osgeo import ogr
import numpy

import pygeoprocessing.geoprocessing
import pygeoprocessing.routing
import pygeoprocessing.routing.routing_core
import natcap.invest.hydropower.hydropower_water_yield


LOGGER = logging.getLogger('natcap.invest.nutrient.nutrient')
logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')


def execute(args):
    """
    A high level wrapper for the InVEST nutrient model that first calls
    through to the InVEST water yield function.  This is a historical
    separation that used to make sense when we manually required users
    to pass the water yield pixel raster to the nutrient output.

    Args:
        workspace_dir (string): directory in which the output and intermediate
            files will be saved
        results_suffix (string): This text will be appended to the end of the
            output files to help separate multiple runs.
        dem_uri (string): A GIS raster dataset with an elevation value for
            each cell.
        precipitation_uri (string): A GIS raster dataset with a non-zero value
            for average annual precipitation for each cell.
        eto_uri (string): A GIS raster dataset, with an annual average
            evapotranspiration value for each cell.
        depth_to_root_rest_layer_uri (string): GIS raster dataset with an
            average root restricting layer depth value for each cell.
        pawc_uri (string): A GIS raster dataset with a plant available water
            content value for each cell.
        lulc_uri (string): A GIS raster dataset, with an integer LULC code for
            each cell.
        seasonality_constant (float): Floating point value on the order of 1
            to 20 corresponding to the seasonal distribution of precipitation
        watersheds_uri (string): This is a layer of watersheds such that each
            watershed contributes to a point of interest where water quality
            will be analyzed.
        biophysical_table_uri (string): A table containing model information
            corresponding to each of the land use classes in the LULC raster
            input.
        calc_p (boolean): Select to calculate phosphorous export.
        calc_n (boolean): Select to calculate nitrogen export.
        accum_threshold (int): The number of upstream cells that must flow
            into a cell before it's considered part of a stream such that
            retention stops and the remaining export is exported to the stream.
        water_purification_threshold_table_uri (string): A table containing
            annual nutrient load threshold information.
        valuation_enabled (boolean): indicates whether the model will include
            valuation
        water_purification_valuation_table_uri (string): A table containing
            valuation information

    Example Args Dictionary::

        {
            'workspace_dir': 'path/to/workspace_dir',
            'results_suffix': '_results',
            'dem_uri': 'path/to/',
            'precipitation_uri': 'path/to/raster',
            'eto_uri': 'path/to/raster',
            'depth_to_root_rest_layer_uri': 'path/to/raster',
            'pawc_uri': 'path/to/raster',
            'lulc_uri': 'path/to/raster',
            'seasonality_constant': 5,
            'watersheds_uri': 'path/to/shapefile',
            'biophysical_table_uri': 'path/to/csv',
            'calc_p': True,
            'calc_n': True,
            'accum_threshold': 1000,
            'water_purification_threshold_table_uri': 'path/to/csv',
            'valuation_enabled': True,
            'water_purification_valuation_table_uri': 'path/to/csv',

        }

    """

    def _validate_inputs(
        nutrients_to_process, lucode_to_parameters, threshold_lookup,
        valuation_lookup):

        """Validation helper method to check that table headers are included
            that are necessary depending on the nutrient type requested by
            the user"""

        #Make sure all the nutrient inputs are good
        if len(nutrients_to_process) == 0:
            raise ValueError("Neither phosphorous nor nitrogen was selected"
                             " to be processed.  Choose at least one.")

        #Build up a list that'll let us iterate through all the input tables
        #and check for the required rows, and report errors if something
        #is missing.
        row_header_table_list = []

        lu_parameter_row = lucode_to_parameters.values()[0]
        row_header_table_list.append(
            (lu_parameter_row, ['load_', 'eff_'],
             args['biophysical_table_uri']))

        threshold_row = threshold_lookup.values()[0]
        row_header_table_list.append(
            (threshold_row, ['thresh_'],
             args['water_purification_threshold_table_uri']))

        if valuation_lookup is not None:
            valuation_row = valuation_lookup.values()[0]
            row_header_table_list.append(
                (valuation_row, ['cost_', 'time_span_', 'discount_'],
                 args['biophysical_table_uri']))

        missing_headers = []
        for row, header_prefixes, table_type in row_header_table_list:
            for nutrient_id in nutrients_to_process:
                for header_prefix in header_prefixes:
                    header = header_prefix + nutrient_id
                    if header not in row:
                        missing_headers.append(
                            "Missing header %s from %s" % (header, table_type))

        if len(missing_headers) > 0:
            raise ValueError('\n'.join(missing_headers))

        #make sure values are in range
        csv_dict_reader = csv.DictReader(open(args['biophysical_table_uri'], 'rU'))
        biophysical_table = {}
        for row in csv_dict_reader:
            biophysical_table[int(row['lucode'])] = row

        value_headers_to_check = [('Kc', (0.0, 1.5))]
        for nutrient_id in nutrients_to_process:
            value_headers_to_check.append(('eff_' + nutrient_id, (0.0, 1.0)))

        #Test to see if c or p values are outside of 0..1
        for table_key, (low_range, upper_range) in value_headers_to_check:
            for (lulc_code, table) in biophysical_table.iteritems():
                try:
                    float_value = float(table[table_key])
                    #print lulc_code, table_key, table_key, low_range, upper_range, float_value
                    if not (low_range <= float_value <= upper_range):
                        raise Exception(
                            'Value should be within range %f..%f offending value '
                            'table %s, lulc_code %s, value %s' % (
                                low_range, upper_range, table_key, str(lulc_code),
                                str(float_value)))
                except ValueError as e:
                    raise Exception(
                        'Value is not a floating point value within range %f..%f '
                        'offending value table %s, lulc_code %s, value %s' % (
                            low_range, upper_range, table_key, str(lulc_code),
                            table[table_key]))

    if not args['calc_p'] and not args['calc_n']:
        raise Exception('Neither "Calculate Nitrogen" nor "Calculate Phosporus" is selected.  At least one must be selected.')

    nutrients_to_process = []
    for nutrient_id in ['n', 'p']:
        if args['calc_' + nutrient_id]:
            nutrients_to_process.append(nutrient_id)
    lucode_to_parameters = pygeoprocessing.geoprocessing.get_lookup_from_csv(
        args['biophysical_table_uri'], 'lucode')

    threshold_table = pygeoprocessing.geoprocessing.get_lookup_from_csv(
        args['water_purification_threshold_table_uri'], 'ws_id')
    valuation_lookup = None
    if args['valuation_enabled']:
        valuation_lookup = pygeoprocessing.geoprocessing.get_lookup_from_csv(
            args['water_purification_valuation_table_uri'], 'ws_id')
    _validate_inputs(nutrients_to_process, lucode_to_parameters,
        threshold_table, valuation_lookup)

    #Set up the water yield arguments that might be a little different than
    #nutrient retention
    water_yield_args = args.copy()
    water_yield_args['workspace_dir'] = os.path.join(
        args['workspace_dir'], 'water_yield_workspace')
    if 'results_suffix' in args:
        water_yield_args['results_suffix'] = args['results_suffix']
    natcap.invest.hydropower.hydropower_water_yield.execute(water_yield_args)

    #Get the pixel output of hydropower to plug into nutrient retention.
    #Tricky because the water yield output might have a different suffix.
    try:
        file_suffix = args['results_suffix']
        if file_suffix != "" and not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''
    args['pixel_yield_uri'] = os.path.join(
        water_yield_args['workspace_dir'], 'output', 'per_pixel',
        'wyield%s.tif' % file_suffix)
    _execute_nutrient(args)


def _execute_nutrient(args):
    """File opening layer for the InVEST nutrient retention model.

        args - a python dictionary with the following entries:
            'workspace_dir' - a string uri pointing to the current workspace.
            'dem_uri' - a string uri pointing to the Digital Elevation Map
                (DEM), a GDAL raster on disk.
            'pixel_yield_uri' - a string uri pointing to the water yield raster
                output from the InVEST Water Yield model.
            'lulc_uri' - a string uri pointing to the landcover GDAL raster.
            'watersheds_uri' - a string uri pointing to an OGR shapefile on
                disk representing the user's watersheds.
            'biophysical_table_uri' - a string uri to a supported table on disk
                containing nutrient retention values. (SAY WHAT VALUES ARE)
            'soil_depth_uri' - a uri to an input raster describing the
                average soil depth value for each cell (mm) (required)
            'precipitation_uri' - a uri to an input raster describing the
                average annual precipitation value for each cell (mm) (required)
            'pawc_uri' - a uri to an input raster describing the
                plant available water content value for each cell. Plant Available
                Water Content fraction (PAWC) is the fraction of water that can be
                stored in the soil profile that is available for plants' use.
                PAWC is a fraction from 0 to 1 (required)
            'eto_uri' - a uri to an input raster describing the
                annual average evapotranspiration value for each cell. Potential
                evapotranspiration is the potential loss of water from soil by
                both evaporation from the soil and transpiration by healthy Alfalfa
                (or grass) if sufficient water is available (mm) (required)
            'seasonality_constant' - floating point value between 1 and 10
                corresponding to the seasonal distribution of precipitation
                (required)
            'calc_p' - True if phosphorous is meant to be modeled, if True then
                biophyscial table and threshold table and valuation table must
                have p fields in them.
            'calc_n' - True if nitrogen is meant to be modeled, if True then
                biophyscial table and threshold table and valuation table must
                have n fields in them.
            'results_suffix' - (optional) a text field to append to all output files.
            'water_purification_threshold_table_uri' - a string uri to a
                csv table containing water purification details.
            'nutrient_type' - a string, either 'nitrogen' or 'phosphorus'
            'accum_threshold' - a number representing the flow accumulation.
            'water_purification_valuation_table_uri' - (optional) a uri to a
                csv used for valuation

        returns nothing.
    """

    #Load all the tables for preprocessing
    workspace = args['workspace_dir']
    output_dir = os.path.join(workspace, 'output')
    intermediate_dir = os.path.join(workspace, 'intermediate')

    try:
        file_suffix = args['results_suffix']
        if not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''

    for folder in [workspace, output_dir, intermediate_dir]:
        if not os.path.exists(folder):
            LOGGER.debug('Making folder %s', folder)
            os.makedirs(folder)

    #Build up a list of nutrients to process based on what's checked on
    nutrients_to_process = []
    for nutrient_id in ['n', 'p']:
        if args['calc_' + nutrient_id]:
            nutrients_to_process.append(nutrient_id)
    lucode_to_parameters = pygeoprocessing.geoprocessing.get_lookup_from_csv(
        args['biophysical_table_uri'], 'lucode')

    threshold_table = pygeoprocessing.geoprocessing.get_lookup_from_csv(
        args['water_purification_threshold_table_uri'], 'ws_id')
    valuation_lookup = None
    if args['valuation_enabled']:
        valuation_lookup = pygeoprocessing.geoprocessing.get_lookup_from_csv(
            args['water_purification_valuation_table_uri'], 'ws_id')

    #This one is tricky, we want to make a dictionary that indexes by nutrient
    #id and yields a dicitonary indexed by ws_id to the threshold amount of
    #that type.  The get_lookup_from_csv only gives us a flat table, so this
    #processing is working around that.
    threshold_lookup = {}
    for nutrient_id in nutrients_to_process:
        threshold_lookup[nutrient_id] = {}
        for ws_id, value in threshold_table.iteritems():
            threshold_lookup[nutrient_id][ws_id] = (
                value['thresh_%s' % (nutrient_id)])

    dem_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
        args['dem_uri'])
    #Pixel size is in m^2, so square and divide by 10000 to get cell size in Ha
    cell_area_ha = dem_pixel_size ** 2 / 10000.0
    out_pixel_size = dem_pixel_size

    #Align all the input rasters
    dem_uri = pygeoprocessing.geoprocessing.temporary_filename()
    water_yield_uri = pygeoprocessing.geoprocessing.temporary_filename()
    lulc_uri = pygeoprocessing.geoprocessing.temporary_filename()
    pygeoprocessing.geoprocessing.align_dataset_list(
        [args['dem_uri'], args['pixel_yield_uri'], args['lulc_uri']],
        [dem_uri, water_yield_uri, lulc_uri], ['nearest'] * 3,
        out_pixel_size, 'intersection', dataset_to_align_index=0,
        aoi_uri=args['watersheds_uri'])

    nodata_landuse = pygeoprocessing.geoprocessing.get_nodata_from_uri(lulc_uri)
    nodata_load = -1.0

    #Calculate flow accumulation
    LOGGER.info("calculating flow accumulation")
    flow_accumulation_uri = os.path.join(
        intermediate_dir, 'flow_accumulation%s.tif' % file_suffix)
    flow_direction_uri = os.path.join(
        intermediate_dir, 'flow_direction%s.tif' % file_suffix)

    pygeoprocessing.routing.flow_direction_d_inf(dem_uri, flow_direction_uri)
    pygeoprocessing.routing.flow_accumulation(
        flow_direction_uri, dem_uri, flow_accumulation_uri)

    #classify streams from the flow accumulation raster
    LOGGER.info("Classifying streams from flow accumulation raster")

    #Make the streams
    stream_uri = os.path.join(intermediate_dir, 'stream%s.tif' % file_suffix)
    pygeoprocessing.routing.stream_threshold(
        flow_accumulation_uri, float(args['accum_threshold']), stream_uri)
    nodata_stream = pygeoprocessing.geoprocessing.get_nodata_from_uri(stream_uri)

    def map_load_function(load_type):
        """Function generator to map arbitrary nutrient type"""
        def map_load(lucode):
            """converts unit load to total load & handles nodata"""
            if lucode == nodata_landuse:
                return nodata_load
            return lucode_to_parameters[lucode][load_type] * cell_area_ha
        return map_load
    def map_eff_function(load_type):
        """Function generator to map arbitrary efficiency type"""
        def map_load(lucode, stream):
            """maps efficiencies from lulcs, handles nodata, and is aware that
                streams have no retention"""
            if lucode == nodata_landuse or stream == nodata_stream:
                return nodata_load
            #Retention efficiency is 0 when there's a stream.
            return lucode_to_parameters[lucode][load_type] * (1 - stream)
        return map_load

    #Build up the load and efficiency rasters from the landcover map
    load_uri = {}
    eff_uri = {}
    for nutrient in nutrients_to_process:
        load_uri[nutrient] = os.path.join(
            intermediate_dir, 'load_%s%s.tif' % (nutrient, file_suffix))
        LOGGER.debug(lulc_uri)
        pygeoprocessing.geoprocessing.vectorize_datasets(
            [lulc_uri], map_load_function('load_%s' % nutrient),
            load_uri[nutrient], gdal.GDT_Float32, nodata_load, out_pixel_size,
            "intersection")
        eff_uri[nutrient] = os.path.join(
            intermediate_dir, 'eff_%s%s.tif' % (nutrient, file_suffix))
        pygeoprocessing.geoprocessing.vectorize_datasets(
            [lulc_uri, stream_uri], map_eff_function('eff_%s' % nutrient),
            eff_uri[nutrient], gdal.GDT_Float32, nodata_load, out_pixel_size,
            "intersection")

    #Calcualte the sum of water yield pixels
    upstream_water_yield_uri = os.path.join(
        intermediate_dir, 'upstream_water_yield%s.tif' % file_suffix)
    water_loss_uri = pygeoprocessing.geoprocessing.temporary_filename()
    zero_raster_uri = pygeoprocessing.geoprocessing.temporary_filename()
    pygeoprocessing.geoprocessing.make_constant_raster_from_base_uri(
        dem_uri, 0.0, zero_raster_uri)

    pygeoprocessing.routing.route_flux(
        flow_direction_uri, dem_uri, water_yield_uri, zero_raster_uri,
        water_loss_uri, upstream_water_yield_uri, 'flux_only',
        aoi_uri=args['watersheds_uri'], stream_uri=stream_uri)

    #Calculate the 'log' of the upstream_water_yield raster
    runoff_index_uri = os.path.join(
        intermediate_dir, 'runoff_index%s.tif' % file_suffix)
    nodata_upstream = pygeoprocessing.geoprocessing.get_nodata_from_uri(upstream_water_yield_uri)
    def nodata_log(value):
        """Calculates the log value whiel handling nodata values correctly"""
        nodata_mask = value == nodata_upstream
        result = numpy.log(value)
        result[result < 0] = 0.0
        return numpy.where(
            nodata_mask, nodata_upstream, result)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [upstream_water_yield_uri], nodata_log, runoff_index_uri,
        gdal.GDT_Float32, nodata_upstream, out_pixel_size, "intersection",
        vectorize_op=False)

    field_summaries = {
        'mn_run_ind': pygeoprocessing.geoprocessing.aggregate_raster_values_uri(
            runoff_index_uri, args['watersheds_uri'], 'ws_id').pixel_mean
        }
    field_header_order = ['mn_run_ind']

    watershed_output_datasource_uri = os.path.join(
        output_dir, 'watershed_results_nutrient%s.shp' % file_suffix)
    #If there is already an existing shapefile with the same name and path,
    #delete it then copy the input shapefile into the designated output folder
    if os.path.isfile(watershed_output_datasource_uri):
        os.remove(watershed_output_datasource_uri)
    esri_driver = ogr.GetDriverByName('ESRI Shapefile')
    original_datasource = ogr.Open(args['watersheds_uri'])
    output_datasource = esri_driver.CopyDataSource(
        original_datasource, watershed_output_datasource_uri)
    output_layer = output_datasource.GetLayer()

    add_fields_to_shapefile('ws_id', field_summaries, output_layer, field_header_order)
    field_header_order = []

    #Burn the mean runoff values to a raster that matches the watersheds
    upstream_water_yield_dataset = gdal.Open(upstream_water_yield_uri)
    mean_runoff_index_uri = os.path.join(
        intermediate_dir, 'mean_runoff_index%s.tif' % file_suffix)
    mean_runoff_dataset = pygeoprocessing.geoprocessing.new_raster_from_base(
        upstream_water_yield_dataset, mean_runoff_index_uri, 'GTiff', -1.0,
        gdal.GDT_Float32, -1.0)
    upstream_water_yield_dataset = None
    gdal.RasterizeLayer(
        mean_runoff_dataset, [1], output_layer,
        options=['ATTRIBUTE=mn_run_ind'])
    mean_runoff_dataset = None

    def alv_calculation(load, runoff_index, mean_runoff_index, stream):
        """Calculates the adjusted loading value index"""
        if nodata_load in [load, runoff_index, mean_runoff_index] or \
                stream == nodata_stream or mean_runoff_index == 0.0:
            return nodata_load
        return load * runoff_index / mean_runoff_index * (1 - stream)
    alv_uri = {}
    retention_uri = {}
    export_uri = {}
    field_summaries = {}
    pts_uri = {}  # percent_to_stream uri
    for nutrient in nutrients_to_process:
        alv_uri[nutrient] = os.path.join(
            intermediate_dir, 'alv_%s%s.tif' % (nutrient, file_suffix))
        pygeoprocessing.geoprocessing.vectorize_datasets(
            [load_uri[nutrient], runoff_index_uri, mean_runoff_index_uri,
             stream_uri],  alv_calculation, alv_uri[nutrient], gdal.GDT_Float32,
            nodata_load, out_pixel_size, "intersection")

        #The retention calculation is only interesting to see where nutrient
        # retains on the landscape
        retention_uri[nutrient] = os.path.join(
            output_dir, '%s_retention%s.tif' % (nutrient, file_suffix))
        tmp_flux_uri = pygeoprocessing.geoprocessing.temporary_filename()
        pygeoprocessing.routing.route_flux(
            flow_direction_uri, dem_uri, alv_uri[nutrient], eff_uri[nutrient],
            retention_uri[nutrient], tmp_flux_uri, 'flux_only',
            aoi_uri=args['watersheds_uri'], stream_uri=stream_uri)

        export_uri[nutrient] = os.path.join(
            output_dir, '%s_export%s.tif' % (nutrient, file_suffix))
        pts_uri[nutrient] = os.path.join(intermediate_dir,
            '%s_percent_to_stream%s.tif' % (nutrient, file_suffix))
        pygeoprocessing.routing.pixel_amount_exported(
            flow_direction_uri, dem_uri, stream_uri, eff_uri[nutrient], alv_uri[nutrient],
            export_uri[nutrient], aoi_uri=args['watersheds_uri'],
            percent_to_stream_uri=pts_uri[nutrient])

        #Summarize the results in terms of watershed:
        LOGGER.info("Summarizing the results of nutrient %s" % nutrient)
        alv_tot = pygeoprocessing.geoprocessing.aggregate_raster_values_uri(
            alv_uri[nutrient], args['watersheds_uri'], 'ws_id').total
        export_tot = pygeoprocessing.geoprocessing.aggregate_raster_values_uri(
            export_uri[nutrient], args['watersheds_uri'], 'ws_id').total

        #Retention is alv-export
        retention_tot = {}
        for ws_id in alv_tot:
            retention_tot[ws_id] = alv_tot[ws_id] - export_tot[ws_id]

        #Threshold export is export - threshold
        threshold_retention_tot = {}
        for ws_id in alv_tot:
            threshold_retention_tot[ws_id] = (
                retention_tot[ws_id] - threshold_lookup[nutrient][ws_id])

        field_summaries['%s_alv_tot' % nutrient] = alv_tot
        field_summaries['%s_ret_tot' % nutrient] = retention_tot
        field_summaries['%s_ret_adj' % nutrient] = threshold_retention_tot
        field_summaries['%s_exp_tot' % nutrient] = export_tot
        field_header_order = (
            map(lambda(x): x % nutrient, ['%s_alv_tot', '%s_ret_tot', '%s_ret_adj', '%s_exp_tot']) + field_header_order)
        #Do valuation if necessary
        if valuation_lookup is not None:
            field_summaries['value_%s' % nutrient] = {}
            for ws_id, value in \
                    field_summaries['%s_ret_tot' % nutrient].iteritems():
                discount = disc(
                    valuation_lookup[ws_id]['time_span_%s' % nutrient],
                    valuation_lookup[ws_id]['discount_%s' % nutrient])
                field_summaries['value_%s' % nutrient][ws_id] = (
                    field_summaries['%s_ret_tot' % nutrient][ws_id] *
                    valuation_lookup[ws_id]['cost_%s' % nutrient] * discount)
            field_header_order.append('value_%s' % nutrient)
    LOGGER.info('Writing summaries to output shapefile')
    add_fields_to_shapefile('ws_id', field_summaries, output_layer, field_header_order)


def disc(years, percent_rate):
    """Calculate discount rate for a given number of years

        years - an integer number of years
        percent_rate - a discount rate in percent

        returns the discount rate for the number of years to use in
            a calculation like yearly_cost * disc(years, percent_rate)"""

    discount = 0.0
    for time_index in range(int(years) - 1):
        discount += 1.0 / (1.0 + percent_rate / 100.0) ** time_index
    return discount


def add_fields_to_shapefile(key_field, field_summaries, output_layer,
    field_header_order=None):
    """Adds fields and their values indexed by key fields to an OGR
        layer open for writing.

        key_field - name of the key field in the output_layer that
            uniquely identifies each polygon.
        field_summaries - a dictionary indexed by the desired field
            name to place in the polygon that indexes to another
            dictionary indexed by key_field value to map to that
            particular polygon.  ex {'field_name_1': {key_val1: value,
            key_val2: value}, 'field_name_2': {key_val1: value, etc.
        output_layer - an open writable OGR layer
        field_header_order - a list of field headers in the order we
            wish them to appear in the output table, if None then
            random key order in field summaries is used.

        returns nothing"""
    if field_header_order == None:
        field_header_order = field_summaries.keys()

    for field_name in field_header_order:
        field_def = ogr.FieldDefn(field_name, ogr.OFTReal)
        output_layer.CreateField(field_def)

    #Initialize each feature field to 0.0
    for feature_id in xrange(output_layer.GetFeatureCount()):
        feature = output_layer.GetFeature(feature_id)
        for field_name in field_header_order:
            try:
                ws_id = feature.GetFieldAsInteger(key_field)
                feature.SetField(
                    field_name, float(field_summaries[field_name][ws_id]))
            except KeyError:
                LOGGER.warning('unknown field %s' % field_name)
                feature.SetField(field_name, 0.0)
        #Save back to datasource
        output_layer.SetFeature(feature)
