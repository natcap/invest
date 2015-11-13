"""InVEST Sediment Delivery Ratio (SDR) module"""

import os
import csv
import logging

from osgeo import gdal
from osgeo import ogr
import numpy

import pygeoprocessing.geoprocessing
import pygeoprocessing.routing
import pygeoprocessing.routing.routing_core

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.sdr.sdr')


def execute(args):
    """This function invokes the SDR model given
        URI inputs of files. It may write log, warning, or error messages to
        stdout.

        args - a python dictionary with at the following possible entries:
        args['workspace_dir'] - a uri to the directory that will write output
            and other temporary files during calculation. (required)
        args['results_suffix'] - a string to append to any output file name (optional)
        args['dem_uri'] - a uri to a digital elevation raster file (required)
        args['erosivity_uri'] - a uri to an input raster describing the
            rainfall eroisivity index (required)
        args['erodibility_uri'] - a uri to an input raster describing soil
            erodibility (required)
        args['lulc_uri'] - a uri to a land use/land cover raster whose
            LULC indexes correspond to indexs in the biophysical table input.
            Used for determining soil retention and other biophysical
            properties of the landscape.  (required)
        args['watersheds_uri'] - a uri to an input shapefile of the watersheds
            of interest as polygons. (required)
        args['biophysical_table_uri'] - a uri to an input CSV file with
            biophysical information about each of the land use classes.
        args['threshold_flow_accumulation'] - an integer describing the number
            of upstream cells that must flow int a cell before it's considered
            part of a stream.  required if 'stream_uri' is not provided.
        args['k_param'] - k calibration parameter (see user's guide for values)
        args['sdr_max'] - the max value the SDR can be
        args['ic_0_param'] - ic_0 calibration parameter (see user's guide for
            values)
        args['drainage_uri'] - An optional GIS raster dataset mask, that
            indicates areas that drain to the watershed.  Format is that 1's
            indicate drainage areas and 0's or nodata indicate areas with no
            additional drainage.  This model is most accurate when the drainage
            raster aligns with the DEM.
        args['_prepare'] - (optional) The preprocessed set of data created by the
            sdr._prepare call.  This argument could be used in cases where the
            call to this function is scripted and can save a significant amount
            of runtime.

        returns nothing."""

    #append a _ to the suffix if it's not empty and doens't already have one
    try:
        file_suffix = args['results_suffix']
        if file_suffix != "" and not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''

    csv_dict_reader = csv.DictReader(open(args['biophysical_table_uri'], 'rU'))
    biophysical_table = {}
    for row in csv_dict_reader:
        try:
            biophysical_table[int(row['lucode'])] = row
        except ValueError:
            if row['lucode'] == '':
                # this can happen in some CSV files where there are a bunch
                # of blank lines.  This pass lets us tolerate that and
                # ultimately the model will crash later if there's missing
                # lucodes or something like that
                pass
            else:
                # it's something other than a blank line, best to crash
                raise

    #Test to see if c or p values are outside of 0..1
    for table_key in ['usle_c', 'usle_p']:
        for (lulc_code, table) in biophysical_table.iteritems():
            try:
                float_value = float(table[table_key])
                if float_value < 0 or float_value > 1:
                    raise Exception(
                        'Value should be within range 0..1 offending value '
                        'table %s, lulc_code %s, value %s' % (
                            table_key, str(lulc_code), str(float_value)))
            except ValueError as e:
                raise Exception(
                    'Value is not a floating point value within range 0..1 '
                    'offending value table %s, lulc_code %s, value %s' % (
                        table_key, str(lulc_code), table[table_key]))

    intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')
    output_dir = os.path.join(args['workspace_dir'], 'output')

    #Sets up the intermediate and output directory structure for the workspace
    pygeoprocessing.geoprocessing.create_directories([output_dir, intermediate_dir])


    #check if we've already prepared the DEM
    if '_prepare' in args:
        preprocessed_data = args['_prepare']
    else:
        preprocessed_data = _prepare(**args)

    aligned_dem_uri = preprocessed_data['aligned_dem_uri']
    aligned_erosivity_uri = preprocessed_data['aligned_erosivity_uri']
    aligned_erodibility_uri = preprocessed_data['aligned_erodibility_uri']
    thresholded_slope_uri = preprocessed_data['thresholded_slope_uri']
    flow_accumulation_uri = preprocessed_data['flow_accumulation_uri']
    flow_direction_uri = preprocessed_data['flow_direction_uri']
    ls_uri = preprocessed_data['ls_uri']

    #this section is to align the lulc with the prepared data, we need to make
    #a garbage tempoary dem to conform to the align_dataset_list API that
    #requires as many outputs as inputs
    aligned_lulc_uri = os.path.join(intermediate_dir, 'aligned_lulc.tif')
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
        preprocessed_data['aligned_dem_uri'])
    tmp_dem_uri = pygeoprocessing.geoprocessing.temporary_filename()
    pygeoprocessing.geoprocessing.align_dataset_list(
        [aligned_dem_uri, args['lulc_uri']], [tmp_dem_uri, aligned_lulc_uri],
        ['nearest'] * 2, out_pixel_size, 'dataset',
        0, dataset_to_bound_index=0, aoi_uri=args['watersheds_uri'])
    os.remove(tmp_dem_uri)

    #classify streams from the flow accumulation raster
    LOGGER.info("Classifying streams from flow accumulation raster")
    stream_uri = os.path.join(intermediate_dir, 'stream%s.tif' % file_suffix)

    pygeoprocessing.routing.stream_threshold(flow_accumulation_uri,
        float(args['threshold_flow_accumulation']), stream_uri)
    stream_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(stream_uri)

    dem_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(args['dem_uri'])

    if 'drainage_uri' in args and args['drainage_uri'] != '':
        def add_drainage(stream, drainage):
            return numpy.where(drainage == 1, 1, stream)

        stream_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(stream_uri)
        #add additional drainage to the stream
        drainage_uri = os.path.join(output_dir, 'drainage%s.tif' % file_suffix)

        pygeoprocessing.geoprocessing.vectorize_datasets(
            [stream_uri, args['drainage_uri']], add_drainage, drainage_uri,
            gdal.GDT_Byte, stream_nodata, out_pixel_size, "intersection",
            dataset_to_align_index=0, vectorize_op=False)
        stream_uri = drainage_uri

    #Calculate the W factor
    LOGGER.info('calculate per pixel W')
    original_w_factor_uri = os.path.join(
        intermediate_dir, 'w_factor%s.tif' % file_suffix)
    thresholded_w_factor_uri = os.path.join(
        intermediate_dir, 'thresholded_w_factor%s.tif' % file_suffix)
    #map lulc to biophysical table
    lulc_to_c = dict(
        [(lulc_code, float(table['usle_c'])) for
        (lulc_code, table) in biophysical_table.items()])
    lulc_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(aligned_lulc_uri)
    w_nodata = -1.0

    pygeoprocessing.geoprocessing.reclassify_dataset_uri(
        aligned_lulc_uri, lulc_to_c, original_w_factor_uri, gdal.GDT_Float64,
        w_nodata, exception_flag='values_required')
    def threshold_w(w_val):
        '''Threshold w to 0.001'''
        w_val_copy = w_val.copy()
        nodata_mask = w_val == w_nodata
        w_val_copy[w_val < 0.001] = 0.001
        w_val_copy[nodata_mask] = w_nodata
        return w_val_copy
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [original_w_factor_uri], threshold_w, thresholded_w_factor_uri,
        gdal.GDT_Float64, w_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    cp_factor_uri = os.path.join(
        intermediate_dir, 'cp_factor%s.tif' % file_suffix)

    lulc_to_cp = dict(
        [(lulc_code, float(table['usle_c']) * float(table['usle_p'])) for
         (lulc_code, table) in biophysical_table.items()])

    cp_nodata = -1.0
    pygeoprocessing.geoprocessing.reclassify_dataset_uri(
        aligned_lulc_uri, lulc_to_cp, cp_factor_uri, gdal.GDT_Float64,
        cp_nodata, exception_flag='values_required')

    LOGGER.info('calculating rkls')
    rkls_uri = os.path.join(output_dir, 'rkls%s.tif' % file_suffix)
    calculate_rkls(
        ls_uri, aligned_erosivity_uri, aligned_erodibility_uri,
        stream_uri, rkls_uri)

    LOGGER.info('calculating USLE')
    usle_uri = os.path.join(output_dir, 'usle%s.tif' % file_suffix)
    nodata_rkls = pygeoprocessing.geoprocessing.get_nodata_from_uri(rkls_uri)
    nodata_cp = pygeoprocessing.geoprocessing.get_nodata_from_uri(cp_factor_uri)
    nodata_usle = -1.0
    def mult_rkls_cp(rkls, cp_factor, stream):
        return numpy.where((rkls == nodata_rkls) | (cp_factor == nodata_cp),
            nodata_usle, rkls * cp_factor * (1 - stream))
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [rkls_uri, cp_factor_uri, stream_uri], mult_rkls_cp, usle_uri,
        gdal.GDT_Float64, nodata_usle, out_pixel_size, "intersection",
        dataset_to_align_index=0, aoi_uri=args['watersheds_uri'],
        vectorize_op=False)

    #calculate W_bar
    zero_absorption_source_uri = pygeoprocessing.geoprocessing.temporary_filename()
    loss_uri = pygeoprocessing.geoprocessing.temporary_filename()
    #need this for low level route_flux function
    pygeoprocessing.geoprocessing.make_constant_raster_from_base_uri(
        aligned_dem_uri, 0.0, zero_absorption_source_uri)

    flow_accumulation_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
        flow_accumulation_uri)

    w_accumulation_uri = os.path.join(
        intermediate_dir, 'w_accumulation%s.tif' % file_suffix)
    s_accumulation_uri = os.path.join(
        intermediate_dir, 's_accumulation%s.tif' % file_suffix)

    for factor_uri, accumulation_uri in [
            (thresholded_w_factor_uri, w_accumulation_uri),
            (thresholded_slope_uri, s_accumulation_uri)]:
        LOGGER.info("calculating %s", accumulation_uri)
        pygeoprocessing.routing.route_flux(
            flow_direction_uri, aligned_dem_uri, factor_uri,
            zero_absorption_source_uri, loss_uri, accumulation_uri, 'flux_only',
            aoi_uri=args['watersheds_uri'])

    LOGGER.info("calculating w_bar")

    w_bar_uri = os.path.join(intermediate_dir, 'w_bar%s.tif' % file_suffix)
    w_bar_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(w_accumulation_uri)
    s_bar_uri = os.path.join(intermediate_dir, 's_bar%s.tif' % file_suffix)
    s_bar_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(s_accumulation_uri)
    for bar_nodata, accumulation_uri, bar_uri in [
            (w_bar_nodata, w_accumulation_uri, w_bar_uri),
            (s_bar_nodata, s_accumulation_uri, s_bar_uri)]:
        LOGGER.info("calculating %s", accumulation_uri)
        def bar_op(base_accumulation, flow_accumulation):
            return numpy.where(
                (base_accumulation != bar_nodata) & (flow_accumulation != flow_accumulation_nodata),
                base_accumulation / flow_accumulation, bar_nodata)
        pygeoprocessing.geoprocessing.vectorize_datasets(
            [accumulation_uri, flow_accumulation_uri], bar_op, bar_uri,
            gdal.GDT_Float32, bar_nodata, out_pixel_size, "intersection",
            dataset_to_align_index=0, vectorize_op=False)

    LOGGER.info('calculating d_up')
    d_up_uri = os.path.join(intermediate_dir, 'd_up%s.tif' % file_suffix)
    cell_area = out_pixel_size ** 2
    d_up_nodata = -1.0
    def d_up(w_bar, s_bar, flow_accumulation):
        """Calculate the d_up index
            w_bar * s_bar * sqrt(upstream area) """
        d_up_array = w_bar * s_bar * numpy.sqrt(flow_accumulation * cell_area)
        return numpy.where(
            (w_bar != w_bar_nodata) & (s_bar != s_bar_nodata) &
            (flow_accumulation != flow_accumulation_nodata), d_up_array,
            d_up_nodata)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [w_bar_uri, s_bar_uri, flow_accumulation_uri], d_up, d_up_uri,
        gdal.GDT_Float32, d_up_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    LOGGER.info('calculate WS factor')
    ws_factor_inverse_uri = os.path.join(
        intermediate_dir, 'ws_factor_inverse%s.tif' % file_suffix)
    ws_nodata = -1.0
    slope_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
        preprocessed_data['thresholded_slope_uri'])

    def ws_op(w_factor, s_factor):
        #calculating the inverse so we can use the distance to stream factor function
        return numpy.where(
            (w_factor != w_nodata) & (s_factor != slope_nodata),
            1.0 / (w_factor * s_factor), ws_nodata)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [thresholded_w_factor_uri, thresholded_slope_uri], ws_op, ws_factor_inverse_uri,
        gdal.GDT_Float32, ws_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    LOGGER.info('calculating d_dn')
    d_dn_uri = os.path.join(intermediate_dir, 'd_dn%s.tif' % file_suffix)
    pygeoprocessing.routing.routing_core.distance_to_stream(
        flow_direction_uri, stream_uri, d_dn_uri, factor_uri=ws_factor_inverse_uri)

    LOGGER.info('calculate ic')
    ic_factor_uri = os.path.join(intermediate_dir, 'ic_factor%s.tif' % file_suffix)
    ic_nodata = -9999.0
    d_up_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(d_up_uri)
    d_dn_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(d_dn_uri)
    def ic_op(d_up, d_dn):
        nodata_mask = (d_up == d_up_nodata) | (d_dn == d_dn_nodata)
        return numpy.where(
            nodata_mask, ic_nodata, numpy.log10(d_up/d_dn))
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [d_up_uri, d_dn_uri], ic_op, ic_factor_uri,
        gdal.GDT_Float32, ic_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    LOGGER.info('calculate sdr')
    sdr_factor_uri = os.path.join(intermediate_dir, 'sdr_factor%s.tif' % file_suffix)
    sdr_nodata = -9999.0
    k = float(args['k_param'])
    ic_0 = float(args['ic_0_param'])
    sdr_max = float(args['sdr_max'])
    def sdr_op(ic_factor, stream):
        nodata_mask = (ic_factor == ic_nodata)
        sdr = numpy.where(
            nodata_mask, sdr_nodata, sdr_max/(1+numpy.exp((ic_0-ic_factor)/k)))
        #mask out the stream layer
        return numpy.where(stream == 1, 0.0, sdr)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [ic_factor_uri, stream_uri], sdr_op, sdr_factor_uri,
        gdal.GDT_Float32, sdr_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    LOGGER.info('calculate sed export')
    sed_export_uri = os.path.join(output_dir, 'sed_export%s.tif' % file_suffix)
    sed_export_nodata = -1.0
    def sed_export_op(usle, sdr):
        nodata_mask = (usle == nodata_usle) | (sdr == sdr_nodata)
        return numpy.where(
            nodata_mask, sed_export_nodata, usle * sdr)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [usle_uri, sdr_factor_uri], sed_export_op, sed_export_uri,
        gdal.GDT_Float32, sed_export_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    LOGGER.info('calculate sediment retention index')
    def sediment_index_op(rkls, usle, sdr_factor):
        nodata_mask = (rkls == nodata_rkls) | (usle == nodata_usle) | (sdr_factor == sdr_nodata)
        return numpy.where(
            nodata_mask, nodata_sed_retention_index, (rkls - usle) * sdr_factor / sdr_max)

    nodata_sed_retention_index = -1
    sed_retention_index_uri = os.path.join(
        output_dir, 'sed_retention_index%s.tif' % file_suffix)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [rkls_uri, usle_uri, sdr_factor_uri], sediment_index_op, sed_retention_index_uri,
        gdal.GDT_Float32, nodata_sed_retention_index, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    LOGGER.info('calculate sediment retention')
    d_up_bare_soil_uri = os.path.join(intermediate_dir, 'd_up_bare_soil%s.tif' % file_suffix)
    d_up_nodata = -1.0
    def d_up_bare_soil_op(s_bar, flow_accumulation):
        """Calculate the d_up index for bare soil
            1.0 * s_bar * sqrt(upstream area) """
        d_up_array = s_bar * numpy.sqrt(flow_accumulation * cell_area)
        return numpy.where(
            (s_bar != s_bar_nodata) &
            (flow_accumulation != flow_accumulation_nodata), d_up_array,
            d_up_nodata)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [s_bar_uri, flow_accumulation_uri], d_up_bare_soil_op, d_up_bare_soil_uri,
        gdal.GDT_Float32, d_up_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    #when calculating d_dn_bare the c factors are all 1,
    #so we invert just s, then accumulate it downstream
    s_factor_inverse_uri = os.path.join(
        intermediate_dir, 's_factor_inverse%s.tif' % file_suffix)
    s_nodata = -1.0
    def s_op(s_factor):
        #calculating the inverse so we can use the distance to stream factor function
        return numpy.where(s_factor != slope_nodata, 1.0 / s_factor, s_nodata)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [thresholded_slope_uri], s_op, s_factor_inverse_uri,
        gdal.GDT_Float32, s_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)
    d_dn_bare_soil_uri = os.path.join(intermediate_dir, 'd_dn_bare_soil%s.tif' % file_suffix)
    d_up_nodata = -1.0
    pygeoprocessing.routing.routing_core.distance_to_stream(
        flow_direction_uri, stream_uri, d_dn_bare_soil_uri, factor_uri=s_factor_inverse_uri)

    ic_factor_bare_soil_uri = os.path.join(
        intermediate_dir, 'ic_factor_bare_soil%s.tif' % file_suffix)
    ic_bare_soil_nodata = -9999.0
    def ic_bare_soil_op(d_up_bare_soil, d_dn_bare_soil):
        nodata_mask = (d_up_bare_soil == d_up_nodata) | (d_dn_bare_soil == d_dn_nodata)
        return numpy.where(
            nodata_mask, ic_nodata, numpy.log10(d_up_bare_soil/d_dn_bare_soil))
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [d_up_bare_soil_uri, d_dn_bare_soil_uri], ic_bare_soil_op, ic_factor_bare_soil_uri,
        gdal.GDT_Float32, ic_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    sdr_factor_bare_soil_uri = os.path.join(intermediate_dir, 'sdr_factor_bare_soil%s.tif' % file_suffix)
    def sdr_bare_soil_op(ic_bare_soil_factor, stream):
        nodata_mask = (ic_bare_soil_factor == ic_nodata)
        sdr_bare_soil = numpy.where(
            nodata_mask, sdr_nodata, sdr_max/(1+numpy.exp((ic_0-ic_bare_soil_factor)/k)))
        #mask out the stream layer
        return numpy.where(stream == 1, 0.0, sdr_bare_soil)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [ic_factor_bare_soil_uri, stream_uri], sdr_bare_soil_op, sdr_factor_bare_soil_uri,
        gdal.GDT_Float32, sdr_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    def sediment_retention_bare_soil_op(rkls, usle, stream_factor, sdr_factor, sdr_factor_bare_soil):
        nodata_mask = (
            (rkls == nodata_rkls) | (usle == nodata_usle) |
            (stream_factor == stream_nodata) | (sdr_factor == sdr_nodata) |
            (sdr_factor_bare_soil == sdr_nodata))
        return numpy.where(
            nodata_mask, nodata_sediment_retention,
            (rkls * sdr_factor_bare_soil - usle * sdr_factor) * (1 - stream_factor))

    nodata_sediment_retention = -1
    sed_retention_bare_soil_uri = os.path.join(
        intermediate_dir, 'sed_retention%s.tif' % file_suffix)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [rkls_uri, usle_uri, stream_uri, sdr_factor_uri, sdr_factor_bare_soil_uri],
        sediment_retention_bare_soil_op, sed_retention_bare_soil_uri,
        gdal.GDT_Float32, nodata_sediment_retention, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)


    LOGGER.info('generating report')
    esri_driver = ogr.GetDriverByName('ESRI Shapefile')

    field_summaries = {
        'usle_tot': pygeoprocessing.geoprocessing.aggregate_raster_values_uri(usle_uri, args['watersheds_uri'], 'ws_id').total,
        'sed_export': pygeoprocessing.geoprocessing.aggregate_raster_values_uri(sed_export_uri, args['watersheds_uri'], 'ws_id').total,
        'sed_retent': pygeoprocessing.geoprocessing.aggregate_raster_values_uri(sed_retention_bare_soil_uri, args['watersheds_uri'], 'ws_id').total,
        }

    original_datasource = ogr.Open(args['watersheds_uri'])
    watershed_output_datasource_uri = os.path.join(output_dir, 'watershed_results_sdr%s.shp' % file_suffix)
    #If there is already an existing shapefile with the same name and path, delete it
    #Copy the input shapefile into the designated output folder
    if os.path.isfile(watershed_output_datasource_uri):
        os.remove(watershed_output_datasource_uri)
    datasource_copy = esri_driver.CopyDataSource(original_datasource, watershed_output_datasource_uri)
    layer = datasource_copy.GetLayer()

    for field_name in field_summaries:
        field_def = ogr.FieldDefn(field_name, ogr.OFTReal)
        layer.CreateField(field_def)

    #Initialize each feature field to 0.0
    for feature_id in xrange(layer.GetFeatureCount()):
        feature = layer.GetFeature(feature_id)
        for field_name in field_summaries:
            try:
                ws_id = feature.GetFieldAsInteger('ws_id')
                feature.SetField(field_name, float(field_summaries[field_name][ws_id]))
            except KeyError:
                LOGGER.warning('unknown field %s' % field_name)
                feature.SetField(field_name, 0.0)
        #Save back to datasource
        layer.SetFeature(feature)

    original_datasource.Destroy()
    datasource_copy.Destroy()

    for ds_uri in [zero_absorption_source_uri, loss_uri]:
        try:
            os.remove(ds_uri)
        except OSError as e:
            LOGGER.warn("couldn't remove %s because it's still open", ds_uri)
            LOGGER.warn(e)


def calculate_ls_factor(
    flow_accumulation_uri, slope_uri, aspect_uri, ls_factor_uri, ls_nodata):
    """Calculates the LS factor as Equation 3 from "Extension and validation
        of a geographic information system-based method for calculating the
        Revised Universal Soil Loss Equation length-slope factor for erosion
        risk assessments in large watersheds"

        (Required that all raster inputs are same dimensions and projections
        and have square cells)
        flow_accumulation_uri - a uri to a  single band raster of type float that
            indicates the contributing area at the inlet of a grid cell
        slope_uri - a uri to a single band raster of type float that indicates
            the slope at a pixel given as a percent
        aspect_uri - a uri to a single band raster of type float that indicates the
            direction that slopes are facing in terms of radians east and
            increase clockwise: pi/2 is north, pi is west, 3pi/2, south and
            0 or 2pi is east.
        ls_factor_uri - (input) a string to the path where the LS raster will
            be written

        returns nothing"""

    flow_accumulation_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
        flow_accumulation_uri)
    slope_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(slope_uri)
    aspect_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(aspect_uri)

    #Assumes that cells are square
    cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(flow_accumulation_uri)
    cell_area = cell_size ** 2

    def ls_factor_function(aspect_angle, percent_slope, flow_accumulation):
        """Calculate the ls factor

            aspect_angle - flow direction in radians
            percent_slope - slope in terms of percent
            flow_accumulation - upstream pixels at this point

            returns the ls_factor calculation for this point"""

        #Skip the calculation if any of the inputs are nodata
        nodata_mask = (
            (aspect_angle == aspect_nodata) | (percent_slope == slope_nodata) |
            (flow_accumulation == flow_accumulation_nodata))

        #Here the aspect direction can range from 0 to 2PI, but the purpose
        #of the term is to determine the length of the flow path on the
        #pixel, thus we take the absolute value of each trigonometric
        #function to keep the computation in the first quadrant
        xij = (numpy.abs(numpy.sin(aspect_angle)) +
            numpy.abs(numpy.cos(aspect_angle)))

        contributing_area = (flow_accumulation-1) * cell_area

        #To convert to radians, we need to divide the percent_slope by 100 since
        #it's a percent.
        slope_in_radians = numpy.arctan(percent_slope / 100.0)

        #From Equation 4 in "Extension and validation of a geographic
        #information system ..."
        slope_factor = numpy.where(percent_slope < 9.0,
            10.8 * numpy.sin(slope_in_radians) + 0.03,
            16.8 * numpy.sin(slope_in_radians) - 0.5)

        #Set the m value to the lookup table that's Table 1 in
        #InVEST Sediment Model_modifications_10-01-2012_RS.docx in the
        #FT Team dropbox
        beta = ((numpy.sin(slope_in_radians) / 0.0896) /
            (3 * numpy.sin(slope_in_radians)**0.8 + 0.56))

        #slope table in percent
        slope_table = [1., 3.5, 5., 9.]
        exponent_table = [0.2, 0.3, 0.4, 0.5]
        #Look up the correct m value from the table
        m_exp = beta/(1+beta)
        for i in range(4):
            m_exp[percent_slope <= slope_table[i]] = exponent_table[i]

        #The length part of the ls_factor:
        l_factor = (
            ((contributing_area + cell_area)**(m_exp+1) -
             contributing_area ** (m_exp+1)) /
            ((cell_size ** (m_exp + 2)) * (xij**m_exp) * (22.13**m_exp)))

        #From the McCool paper "as a final check against excessively long slope
        #length calculations ... cap of 333m"
        l_factor[l_factor > 333] = 333

        #This is the ls_factor
        return numpy.where(nodata_mask, ls_nodata, l_factor * slope_factor)

    #Call vectorize datasets to calculate the ls_factor
    dataset_uri_list = [aspect_uri, slope_uri, flow_accumulation_uri]
    pygeoprocessing.geoprocessing.vectorize_datasets(
        dataset_uri_list, ls_factor_function, ls_factor_uri, gdal.GDT_Float32,
        ls_nodata, cell_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)

    base_directory = os.path.dirname(ls_factor_uri)
    xi_uri = os.path.join(base_directory, "xi.tif")
    s_factor_uri = os.path.join(base_directory, "slope_factor.tif")
    beta_uri = os.path.join(base_directory, "beta.tif")
    m_uri = os.path.join(base_directory, "m.tif")


    def m_op(aspect_angle, percent_slope, flow_accumulation):
        slope_in_radians = numpy.arctan(percent_slope / 100.0)

        beta = ((numpy.sin(slope_in_radians) / 0.0896) /
            (3 * numpy.sin(slope_in_radians)**0.8 + 0.56))

        #slope table in percent
        slope_table = [1., 3.5, 5., 9.]
        exponent_table = [0.2, 0.3, 0.4, 0.5]
        #Look up the correct m value from the table
        m_exp = beta/(1+beta)
        for i in range(4):
            m_exp[percent_slope <= slope_table[i]] = exponent_table[i]

        return m_exp

    pygeoprocessing.geoprocessing.vectorize_datasets(
        dataset_uri_list, m_op, m_uri, gdal.GDT_Float32,
        ls_nodata, cell_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)


    def beta_op(aspect_angle, percent_slope, flow_accumulation):
        slope_in_radians = numpy.arctan(percent_slope / 100.0)

        #Set the m value to the lookup table that's Table 1 in
        #InVEST Sediment Model_modifications_10-01-2012_RS.docx in the
        #FT Team dropbox
        return ((numpy.sin(slope_in_radians) / 0.0896) /
            (3 * numpy.sin(slope_in_radians)**0.8 + 0.56))

    pygeoprocessing.geoprocessing.vectorize_datasets(
        dataset_uri_list, beta_op, beta_uri, gdal.GDT_Float32,
        ls_nodata, cell_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)

    def s_factor_op(aspect_angle, percent_slope, flow_accumulation):
        slope_in_radians = numpy.arctan(percent_slope / 100.0)

        #From Equation 4 in "Extension and validation of a geographic
        #information system ..."
        return numpy.where(percent_slope < 9.0,
            10.8 * numpy.sin(slope_in_radians) + 0.03,
            16.8 * numpy.sin(slope_in_radians) - 0.5)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        dataset_uri_list, s_factor_op, s_factor_uri, gdal.GDT_Float32,
        ls_nodata, cell_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)

    def xi_op(aspect_angle, percent_slope, flow_accumulation):
        return (numpy.abs(numpy.sin(aspect_angle)) +
            numpy.abs(numpy.cos(aspect_angle)))
    pygeoprocessing.geoprocessing.vectorize_datasets(
        dataset_uri_list, xi_op, xi_uri, gdal.GDT_Float32,
        ls_nodata, cell_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)


def calculate_rkls(
    ls_factor_uri, erosivity_uri, erodibility_uri, stream_uri,
    rkls_uri):

    """Calculates per-pixel potential soil loss using the RKLS (revised
        universial soil loss equation with no C or P).

        ls_factor_uri - GDAL uri with the LS factor pre-calculated
        erosivity_uri - GDAL uri with per pixel erosivity
        erodibility_uri - GDAL uri with per pixel erodibility
        stream_uri - GDAL uri indicating locations with streams
            (0 is no stream, 1 stream)
        rkls_uri - string input indicating the path to disk
            for the resulting potential soil loss raster

        returns nothing"""

    ls_factor_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(ls_factor_uri)
    erosivity_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(erosivity_uri)
    erodibility_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(erodibility_uri)
    stream_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(stream_uri)
    usle_nodata = -1.0

    cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(ls_factor_uri)
    cell_area_ha = cell_size ** 2 / 10000.0

    def rkls_function(ls_factor, erosivity, erodibility, stream):
        """Calculates the USLE equation

        ls_factor - length/slope factor
        erosivity - related to peak rainfall events
        erodibility - related to the potential for soil to erode
        stream - 1 or 0 depending if there is a stream there.  If so, no
            potential soil loss due to USLE

        returns ls_factor * erosivity * erodibility * usle_c_p if all arguments
            defined, nodata if some are not defined, 0 if in a stream
            (stream)"""

        rkls = numpy.where(
            stream == 1, 0.0,
            ls_factor * erosivity * erodibility * cell_area_ha)
        return numpy.where(
            (ls_factor == ls_factor_nodata) | (erosivity == erosivity_nodata) |
            (erodibility == erodibility_nodata) | (stream == stream_nodata),
            usle_nodata, rkls)

    dataset_uri_list = [
        ls_factor_uri, erosivity_uri, erodibility_uri, stream_uri]

    #Aligning with index 3 that's the stream and the most likely to be
    #aligned with LULCs
    pygeoprocessing.geoprocessing.vectorize_datasets(
        dataset_uri_list, rkls_function, rkls_uri, gdal.GDT_Float32,
        usle_nodata, cell_size, "intersection", dataset_to_align_index=3,
        vectorize_op=False)


def _prepare(**args):
    """A function to preprocess the static data that goes into the SDR model
        that is unlikely to change when running a batch process.

        args['dem_uri'] - dem layer
        args['erosivity_uri'] - erosivity data that will be used to align and
            precalculate rkls
        args['erodibility_uri'] - erodibility data that will be used to align
            and precalculate rkls
        args['workspace_dir'] - output directory for the generated rasters

        return a dictionary with the keys:
            'aligned_dem_uri' - input dem aligned with the rest of the inputs
            'aligned_erosivity_uri' - input erosivity aligned with the inputs
            'aligned_erodibility_uri' - input erodability aligned with the
                inputs
    """

    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(args['dem_uri'])
    intermediate_dir = os.path.join(args['workspace_dir'], 'prepared_data')

    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)

    tiled_dem_uri = os.path.join(intermediate_dir, 'tiled_dem.tif')
    pygeoprocessing.geoprocessing.tile_dataset_uri(args['dem_uri'], tiled_dem_uri, 256)
    aligned_dem_uri = os.path.join(intermediate_dir, 'aligned_dem.tif')
    aligned_erosivity_uri = os.path.join(
        intermediate_dir, 'aligned_erosivity.tif')
    aligned_erodibility_uri = os.path.join(
        intermediate_dir, 'aligned_erodibility.tif')

    input_list = [tiled_dem_uri, args['erosivity_uri'], args['erodibility_uri']]
    dataset_out_uri_list = [
        aligned_dem_uri, aligned_erosivity_uri, aligned_erodibility_uri]
    pygeoprocessing.geoprocessing.align_dataset_list(
        input_list, dataset_out_uri_list,
        ['nearest'] * len(dataset_out_uri_list), out_pixel_size, 'intersection',
        0, aoi_uri=args['watersheds_uri'])

    #Calculate slope
    LOGGER.info("Calculating slope")
    original_slope_uri = os.path.join(intermediate_dir, 'slope.tif')
    thresholded_slope_uri = os.path.join(
        intermediate_dir, 'thresholded_slope.tif')
    pygeoprocessing.geoprocessing.calculate_slope(aligned_dem_uri, original_slope_uri)
    slope_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(original_slope_uri)
    def threshold_slope(slope):
        '''Convert slope to m/m and clamp at 0.005 and 1.0 as
            desribed in Cavalli et al., 2013. '''
        slope_copy = slope / 100
        nodata_mask = slope == slope_nodata
        slope_copy[slope_copy < 0.005] = 0.005
        slope_copy[slope_copy > 1.0] = 1.0
        slope_copy[nodata_mask] = slope_nodata
        return slope_copy
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [original_slope_uri], threshold_slope, thresholded_slope_uri,
        gdal.GDT_Float64, slope_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    #Calculate flow accumulation
    LOGGER.info("calculating flow accumulation")
    flow_accumulation_uri = os.path.join(
        intermediate_dir, 'flow_accumulation.tif')
    flow_direction_uri = os.path.join(
        intermediate_dir, 'flow_direction.tif')

    pygeoprocessing.routing.flow_direction_d_inf(aligned_dem_uri, flow_direction_uri)
    pygeoprocessing.routing.flow_accumulation(
        flow_direction_uri, aligned_dem_uri, flow_accumulation_uri)

    #Calculate LS term
    LOGGER.info('calculate ls term')
    ls_uri = os.path.join(intermediate_dir, 'ls.tif')
    ls_nodata = -1.0
    calculate_ls_factor(
        flow_accumulation_uri, original_slope_uri, flow_direction_uri, ls_uri,
        ls_nodata)

    return {
        'aligned_dem_uri': aligned_dem_uri,
        'aligned_erosivity_uri': aligned_erosivity_uri,
        'aligned_erodibility_uri': aligned_erodibility_uri,
        'thresholded_slope_uri': thresholded_slope_uri,
        'flow_accumulation_uri': flow_accumulation_uri,
        'flow_direction_uri': flow_direction_uri,
        'ls_uri': ls_uri,
        }
