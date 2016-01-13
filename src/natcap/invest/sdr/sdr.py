"""InVEST Sediment Delivery Ratio (SDR) module."""

import os
import logging

from osgeo import gdal
from osgeo import ogr
import numpy

import pygeoprocessing.geoprocessing
import pygeoprocessing.routing
import pygeoprocessing.routing.routing_core
import natcap.invest.utils

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.sdr.sdr')

_OUTPUT_BASE_FILES = {
    'rkls_path': 'rkls.tif',
    'sed_export_path': 'sed_export.tif',
    'stream_path': 'stream.tif',
    'usle_path': 'usle.tif',
    'sed_retention_index_path': 'sed_retention_index.tif',
    'watershed_results_sdr_path': 'watershed_results_sdr.shp',
    }

_INTERMEDIATE_BASE_FILES = {
    'dem_offset_path': 'dem_offset.tif',
    'slope_path': 'slope.tif',
    'thresholded_slope_path': 'thresholded_slope.tif',
    'flow_direction_path': 'flow_direction.tif',
    'flow_accumulation_path': 'flow_accumulation.tif',
    'ls_path': 'ls.tif',
    'w_bar_path': 'w_bar.tif',
    's_bar_path': 's_bar.tif',
    'd_up_path': 'd_up.tif',
    'ws_factor_path': 'ws_factor.tif',
    'd_dn_path': 'd_dn.tif',
    'ic_factor_path': 'ic_factor.tif',
    'sdr_factor_path': 'sdr_factor.tif',
    }

_TMP_BASE_FILES = {
    'aligned_dem_path': 'aligned_dem.tif',
    'aligned_lulc_path': 'aligned_lulc.tif',
    'aligned_erosivity_path': 'aligned_erosivity.tif',
    'aligned_erodibility_path': 'aligned_erodibility.tif',
    'aligned_watersheds_path': 'aligned_watersheds_path.tif',
    'aligned_drainage_path': 'aligned_drainage.tif',
    }


def execute(args):
    """This function invokes the SDR model given
        URI inputs of files. It may write log, warning, or error messages to
        stdout.

        args - a python dictionary with at the following possible entries:
        args['workspace_dir'] - a uri to the directory that will write output
            and other temporary files during calculation. (required)
        args['results_suffix'] - a string to append to any output file name (optional)
        args['dem_path'] - a uri to a digital elevation raster file (required)
        args['erosivity_path'] - a uri to an input raster describing the
            rainfall eroisivity index (required)
        args['erodibility_path'] - a uri to an input raster describing soil
            erodibility (required)
        args['lulc_path'] - a uri to a land use/land cover raster whose
            LULC indexes correspond to indexs in the biophysical table input.
            Used for determining soil retention and other biophysical
            properties of the landscape.  (required)
        args['watersheds_path'] - a uri to an input shapefile of the watersheds
            of interest as polygons. (required)
        args['biophysical_table_path'] - a uri to an input CSV file with
            biophysical information about each of the land use classes.
        args['threshold_flow_accumulation'] - an integer describing the number
            of upstream cells that must flow int a cell before it's considered
            part of a stream.  required if 'stream_path' is not provided.
        args['k_param'] - k calibration parameter (see user's guide for values)
        args['sdr_max'] - the max value the SDR can be
        args['ic_0_param'] - ic_0 calibration parameter (see user's guide for
            values)
        args['drainage_path'] - An optional GIS raster dataset mask, that
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
    file_suffix = natcap.invest.utils.make_suffix_string(
        args, 'results_suffix')

    biophysical_table = pygeoprocessing.get_lookup_from_csv(
        args['biophysical_table_path'], 'lucode')

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
            except ValueError:
                raise Exception(
                    'Value is not a floating point value within range 0..1 '
                    'offending value table %s, lulc_code %s, value %s' % (
                        table_key, str(lulc_code), table[table_key]))

    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate')
    output_dir = os.path.join(args['workspace_dir'])
    pygeoprocessing.geoprocessing.create_directories(
        [output_dir, intermediate_output_dir])

    file_registry = _build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir),
         (_TMP_BASE_FILES, output_dir)], file_suffix)

    base_list = []
    aligned_list = []
    for file_key in ['lulc', 'dem', 'erosivity', 'erodibility']:
        base_list.append(file_key + "_path")
        aligned_list.append("aligned_" + file_key + "_path")

    out_pixel_size = pygeoprocessing.get_cell_size_from_uri(
        args['lulc_path'])
    pygeoprocessing.geoprocessing.align_dataset_list(
        base_list, aligned_list, ['nearest'] * len(base_list), out_pixel_size,
        'intersection', 0, aoi_uri=args['watersheds_path'])

    # do DEM processing here
    _process_dem(*[file_registry[key] for key in [
        'aligned_dem_path', 'slope_path', 'thresholded_slope_path',
        'flow_direction_path', 'flow_accumulation_path', 'ls_path']])

    #classify streams from the flow accumulation raster
    LOGGER.info("Classifying streams from flow accumulation raster")
    stream_path = os.path.join(intermediate_dir, 'stream%s.tif' % file_suffix)

    pygeoprocessing.routing.stream_threshold(flow_accumulation_path,
        float(args['threshold_flow_accumulation']), stream_path)
    stream_nodata = pygeoprocessing.geoprocessing.get_nodata_from_path(stream_path)

    dem_nodata = pygeoprocessing.geoprocessing.get_nodata_from_path(args['dem_path'])

    if 'drainage_path' in args and args['drainage_path'] != '':
        def add_drainage(stream, drainage):
            return numpy.where(drainage == 1, 1, stream)

        stream_nodata = pygeoprocessing.geoprocessing.get_nodata_from_path(stream_path)
        #add additional drainage to the stream
        drainage_path = os.path.join(output_dir, 'drainage%s.tif' % file_suffix)

        pygeoprocessing.geoprocessing.vectorize_datasets(
            [stream_path, args['drainage_path']], add_drainage, drainage_path,
            gdal.GDT_Byte, stream_nodata, out_pixel_size, "intersection",
            dataset_to_align_index=0, vectorize_op=False)
        stream_path = drainage_path

    #Calculate the W factor
    LOGGER.info('calculate per pixel W')
    original_w_factor_path = os.path.join(
        intermediate_dir, 'w_factor%s.tif' % file_suffix)
    thresholded_w_factor_path = os.path.join(
        intermediate_dir, 'thresholded_w_factor%s.tif' % file_suffix)
    #map lulc to biophysical table
    lulc_to_c = dict(
        [(lulc_code, float(table['usle_c'])) for
        (lulc_code, table) in biophysical_table.items()])
    lulc_nodata = pygeoprocessing.geoprocessing.get_nodata_from_path(aligned_lulc_path)
    w_nodata = -1.0

    pygeoprocessing.geoprocessing.reclassify_dataset_path(
        aligned_lulc_path, lulc_to_c, original_w_factor_path, gdal.GDT_Float64,
        w_nodata, exception_flag='values_required')
    def threshold_w(w_val):
        '''Threshold w to 0.001'''
        w_val_copy = w_val.copy()
        nodata_mask = w_val == w_nodata
        w_val_copy[w_val < 0.001] = 0.001
        w_val_copy[nodata_mask] = w_nodata
        return w_val_copy
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [original_w_factor_path], threshold_w, thresholded_w_factor_path,
        gdal.GDT_Float64, w_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    cp_factor_path = os.path.join(
        intermediate_dir, 'cp_factor%s.tif' % file_suffix)

    lulc_to_cp = dict(
        [(lulc_code, float(table['usle_c']) * float(table['usle_p'])) for
         (lulc_code, table) in biophysical_table.items()])

    cp_nodata = -1.0
    pygeoprocessing.geoprocessing.reclassify_dataset_path(
        aligned_lulc_path, lulc_to_cp, cp_factor_path, gdal.GDT_Float64,
        cp_nodata, exception_flag='values_required')

    LOGGER.info('calculating rkls')
    rkls_path = os.path.join(output_dir, 'rkls%s.tif' % file_suffix)
    calculate_rkls(
        ls_path, aligned_erosivity_path, aligned_erodibility_path,
        stream_path, rkls_path)

    LOGGER.info('calculating USLE')
    usle_path = os.path.join(output_dir, 'usle%s.tif' % file_suffix)
    nodata_rkls = pygeoprocessing.geoprocessing.get_nodata_from_path(rkls_path)
    nodata_cp = pygeoprocessing.geoprocessing.get_nodata_from_path(cp_factor_path)
    nodata_usle = -1.0
    def mult_rkls_cp(rkls, cp_factor, stream):
        return numpy.where((rkls == nodata_rkls) | (cp_factor == nodata_cp),
            nodata_usle, rkls * cp_factor * (1 - stream))
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [rkls_path, cp_factor_path, stream_path], mult_rkls_cp, usle_path,
        gdal.GDT_Float64, nodata_usle, out_pixel_size, "intersection",
        dataset_to_align_index=0, aoi_path=args['watersheds_path'],
        vectorize_op=False)

    #calculate W_bar
    zero_absorption_source_path = pygeoprocessing.geoprocessing.temporary_filename()
    loss_path = pygeoprocessing.geoprocessing.temporary_filename()
    #need this for low level route_flux function
    pygeoprocessing.geoprocessing.make_constant_raster_from_base_path(
        aligned_dem_path, 0.0, zero_absorption_source_path)

    flow_accumulation_nodata = pygeoprocessing.geoprocessing.get_nodata_from_path(
        flow_accumulation_path)

    w_accumulation_path = os.path.join(
        intermediate_dir, 'w_accumulation%s.tif' % file_suffix)
    s_accumulation_path = os.path.join(
        intermediate_dir, 's_accumulation%s.tif' % file_suffix)

    for factor_path, accumulation_path in [
            (thresholded_w_factor_path, w_accumulation_path),
            (thresholded_slope_path, s_accumulation_path)]:
        LOGGER.info("calculating %s", accumulation_path)
        pygeoprocessing.routing.route_flux(
            flow_direction_path, aligned_dem_path, factor_path,
            zero_absorption_source_path, loss_path, accumulation_path, 'flux_only',
            aoi_path=args['watersheds_path'])

    LOGGER.info("calculating w_bar")

    w_bar_path = os.path.join(intermediate_dir, 'w_bar%s.tif' % file_suffix)
    w_bar_nodata = pygeoprocessing.geoprocessing.get_nodata_from_path(w_accumulation_path)
    s_bar_path = os.path.join(intermediate_dir, 's_bar%s.tif' % file_suffix)
    s_bar_nodata = pygeoprocessing.geoprocessing.get_nodata_from_path(s_accumulation_path)
    for bar_nodata, accumulation_path, bar_path in [
            (w_bar_nodata, w_accumulation_path, w_bar_path),
            (s_bar_nodata, s_accumulation_path, s_bar_path)]:
        LOGGER.info("calculating %s", accumulation_path)
        def bar_op(base_accumulation, flow_accumulation):
            return numpy.where(
                (base_accumulation != bar_nodata) & (flow_accumulation != flow_accumulation_nodata),
                base_accumulation / flow_accumulation, bar_nodata)
        pygeoprocessing.geoprocessing.vectorize_datasets(
            [accumulation_path, flow_accumulation_path], bar_op, bar_path,
            gdal.GDT_Float32, bar_nodata, out_pixel_size, "intersection",
            dataset_to_align_index=0, vectorize_op=False)

    LOGGER.info('calculating d_up')
    d_up_path = os.path.join(intermediate_dir, 'd_up%s.tif' % file_suffix)
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
        [w_bar_path, s_bar_path, flow_accumulation_path], d_up, d_up_path,
        gdal.GDT_Float32, d_up_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    LOGGER.info('calculate WS factor')
    ws_factor_inverse_path = os.path.join(
        intermediate_dir, 'ws_factor_inverse%s.tif' % file_suffix)
    ws_nodata = -1.0
    slope_nodata = pygeoprocessing.geoprocessing.get_nodata_from_path(
        preprocessed_data['thresholded_slope_path'])

    def ws_op(w_factor, s_factor):
        #calculating the inverse so we can use the distance to stream factor function
        return numpy.where(
            (w_factor != w_nodata) & (s_factor != slope_nodata),
            1.0 / (w_factor * s_factor), ws_nodata)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [thresholded_w_factor_path, thresholded_slope_path], ws_op, ws_factor_inverse_path,
        gdal.GDT_Float32, ws_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    LOGGER.info('calculating d_dn')
    d_dn_path = os.path.join(intermediate_dir, 'd_dn%s.tif' % file_suffix)
    pygeoprocessing.routing.routing_core.distance_to_stream(
        flow_direction_path, stream_path, d_dn_path, factor_path=ws_factor_inverse_path)

    LOGGER.info('calculate ic')
    ic_factor_path = os.path.join(intermediate_dir, 'ic_factor%s.tif' % file_suffix)
    ic_nodata = -9999.0
    d_up_nodata = pygeoprocessing.geoprocessing.get_nodata_from_path(d_up_path)
    d_dn_nodata = pygeoprocessing.geoprocessing.get_nodata_from_path(d_dn_path)
    def ic_op(d_up, d_dn):
        nodata_mask = (d_up == d_up_nodata) | (d_dn == d_dn_nodata)
        return numpy.where(
            nodata_mask, ic_nodata, numpy.log10(d_up/d_dn))
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [d_up_path, d_dn_path], ic_op, ic_factor_path,
        gdal.GDT_Float32, ic_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    LOGGER.info('calculate sdr')
    sdr_factor_path = os.path.join(intermediate_dir, 'sdr_factor%s.tif' % file_suffix)
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
        [ic_factor_path, stream_path], sdr_op, sdr_factor_path,
        gdal.GDT_Float32, sdr_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    LOGGER.info('calculate sed export')
    sed_export_path = os.path.join(output_dir, 'sed_export%s.tif' % file_suffix)
    sed_export_nodata = -1.0
    def sed_export_op(usle, sdr):
        nodata_mask = (usle == nodata_usle) | (sdr == sdr_nodata)
        return numpy.where(
            nodata_mask, sed_export_nodata, usle * sdr)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [usle_path, sdr_factor_path], sed_export_op, sed_export_path,
        gdal.GDT_Float32, sed_export_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    LOGGER.info('calculate sediment retention index')
    def sediment_index_op(rkls, usle, sdr_factor):
        nodata_mask = (rkls == nodata_rkls) | (usle == nodata_usle) | (sdr_factor == sdr_nodata)
        return numpy.where(
            nodata_mask, nodata_sed_retention_index, (rkls - usle) * sdr_factor / sdr_max)

    nodata_sed_retention_index = -1
    sed_retention_index_path = os.path.join(
        output_dir, 'sed_retention_index%s.tif' % file_suffix)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [rkls_path, usle_path, sdr_factor_path], sediment_index_op, sed_retention_index_path,
        gdal.GDT_Float32, nodata_sed_retention_index, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    LOGGER.info('calculate sediment retention')
    d_up_bare_soil_path = os.path.join(intermediate_dir, 'd_up_bare_soil%s.tif' % file_suffix)
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
        [s_bar_path, flow_accumulation_path], d_up_bare_soil_op, d_up_bare_soil_path,
        gdal.GDT_Float32, d_up_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    #when calculating d_dn_bare the c factors are all 1,
    #so we invert just s, then accumulate it downstream
    s_factor_inverse_path = os.path.join(
        intermediate_dir, 's_factor_inverse%s.tif' % file_suffix)
    s_nodata = -1.0
    def s_op(s_factor):
        #calculating the inverse so we can use the distance to stream factor function
        return numpy.where(s_factor != slope_nodata, 1.0 / s_factor, s_nodata)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [thresholded_slope_path], s_op, s_factor_inverse_path,
        gdal.GDT_Float32, s_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)
    d_dn_bare_soil_path = os.path.join(intermediate_dir, 'd_dn_bare_soil%s.tif' % file_suffix)
    d_up_nodata = -1.0
    pygeoprocessing.routing.routing_core.distance_to_stream(
        flow_direction_path, stream_path, d_dn_bare_soil_path, factor_path=s_factor_inverse_path)

    ic_factor_bare_soil_path = os.path.join(
        intermediate_dir, 'ic_factor_bare_soil%s.tif' % file_suffix)
    ic_bare_soil_nodata = -9999.0
    def ic_bare_soil_op(d_up_bare_soil, d_dn_bare_soil):
        nodata_mask = (d_up_bare_soil == d_up_nodata) | (d_dn_bare_soil == d_dn_nodata)
        return numpy.where(
            nodata_mask, ic_nodata, numpy.log10(d_up_bare_soil/d_dn_bare_soil))
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [d_up_bare_soil_path, d_dn_bare_soil_path], ic_bare_soil_op, ic_factor_bare_soil_path,
        gdal.GDT_Float32, ic_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    sdr_factor_bare_soil_path = os.path.join(intermediate_dir, 'sdr_factor_bare_soil%s.tif' % file_suffix)
    def sdr_bare_soil_op(ic_bare_soil_factor, stream):
        nodata_mask = (ic_bare_soil_factor == ic_nodata)
        sdr_bare_soil = numpy.where(
            nodata_mask, sdr_nodata, sdr_max/(1+numpy.exp((ic_0-ic_bare_soil_factor)/k)))
        #mask out the stream layer
        return numpy.where(stream == 1, 0.0, sdr_bare_soil)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [ic_factor_bare_soil_path, stream_path], sdr_bare_soil_op, sdr_factor_bare_soil_path,
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
    sed_retention_bare_soil_path = os.path.join(
        intermediate_dir, 'sed_retention%s.tif' % file_suffix)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [rkls_path, usle_path, stream_path, sdr_factor_path, sdr_factor_bare_soil_path],
        sediment_retention_bare_soil_op, sed_retention_bare_soil_path,
        gdal.GDT_Float32, nodata_sediment_retention, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)


    LOGGER.info('generating report')
    esri_driver = ogr.GetDriverByName('ESRI Shapefile')

    field_summaries = {
        'usle_tot': pygeoprocessing.geoprocessing.aggregate_raster_values_path(usle_path, args['watersheds_path'], 'ws_id').total,
        'sed_export': pygeoprocessing.geoprocessing.aggregate_raster_values_path(sed_export_path, args['watersheds_path'], 'ws_id').total,
        'sed_retent': pygeoprocessing.geoprocessing.aggregate_raster_values_path(sed_retention_bare_soil_path, args['watersheds_path'], 'ws_id').total,
        }

    original_datasource = ogr.Open(args['watersheds_path'])
    watershed_output_datasource_path = os.path.join(output_dir, 'watershed_results_sdr%s.shp' % file_suffix)
    #If there is already an existing shapefile with the same name and path, delete it
    #Copy the input shapefile into the designated output folder
    if os.path.isfile(watershed_output_datasource_path):
        os.remove(watershed_output_datasource_path)
    datasource_copy = esri_driver.CopyDataSource(original_datasource, watershed_output_datasource_path)
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

    for ds_path in [zero_absorption_source_path, loss_path]:
        try:
            os.remove(ds_path)
        except OSError as e:
            LOGGER.warn("couldn't remove %s because it's still open", ds_path)
            LOGGER.warn(e)


def calculate_ls_factor(
    flow_accumulation_path, slope_path, aspect_path, ls_factor_path, ls_nodata):
    """Calculates the LS factor as Equation 3 from "Extension and validation
        of a geographic information system-based method for calculating the
        Revised Universal Soil Loss Equation length-slope factor for erosion
        risk assessments in large watersheds"

        (Required that all raster inputs are same dimensions and projections
        and have square cells)
        flow_accumulation_path - a uri to a  single band raster of type float that
            indicates the contributing area at the inlet of a grid cell
        slope_path - a uri to a single band raster of type float that indicates
            the slope at a pixel given as a percent
        aspect_path - a uri to a single band raster of type float that indicates the
            direction that slopes are facing in terms of radians east and
            increase clockwise: pi/2 is north, pi is west, 3pi/2, south and
            0 or 2pi is east.
        ls_factor_path - (input) a string to the path where the LS raster will
            be written

        returns nothing"""

    flow_accumulation_nodata = pygeoprocessing.geoprocessing.get_nodata_from_path(
        flow_accumulation_path)
    slope_nodata = pygeoprocessing.geoprocessing.get_nodata_from_path(slope_path)
    aspect_nodata = pygeoprocessing.geoprocessing.get_nodata_from_path(aspect_path)

    #Assumes that cells are square
    cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_path(flow_accumulation_path)
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
    dataset_path_list = [aspect_path, slope_path, flow_accumulation_path]
    pygeoprocessing.geoprocessing.vectorize_datasets(
        dataset_path_list, ls_factor_function, ls_factor_path, gdal.GDT_Float32,
        ls_nodata, cell_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)

    base_directory = os.path.dirname(ls_factor_path)
    xi_path = os.path.join(base_directory, "xi.tif")
    s_factor_path = os.path.join(base_directory, "slope_factor.tif")
    beta_path = os.path.join(base_directory, "beta.tif")
    m_path = os.path.join(base_directory, "m.tif")


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
        dataset_path_list, m_op, m_path, gdal.GDT_Float32,
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
        dataset_path_list, beta_op, beta_path, gdal.GDT_Float32,
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
        dataset_path_list, s_factor_op, s_factor_path, gdal.GDT_Float32,
        ls_nodata, cell_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)

    def xi_op(aspect_angle, percent_slope, flow_accumulation):
        return (numpy.abs(numpy.sin(aspect_angle)) +
            numpy.abs(numpy.cos(aspect_angle)))
    pygeoprocessing.geoprocessing.vectorize_datasets(
        dataset_path_list, xi_op, xi_path, gdal.GDT_Float32,
        ls_nodata, cell_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)


def calculate_rkls(
    ls_factor_path, erosivity_path, erodibility_path, stream_path,
    rkls_path):

    """Calculates per-pixel potential soil loss using the RKLS (revised
        universial soil loss equation with no C or P).

        ls_factor_path - GDAL uri with the LS factor pre-calculated
        erosivity_path - GDAL uri with per pixel erosivity
        erodibility_path - GDAL uri with per pixel erodibility
        stream_path - GDAL uri indicating locations with streams
            (0 is no stream, 1 stream)
        rkls_path - string input indicating the path to disk
            for the resulting potential soil loss raster

        returns nothing"""

    ls_factor_nodata = pygeoprocessing.geoprocessing.get_nodata_from_path(ls_factor_path)
    erosivity_nodata = pygeoprocessing.geoprocessing.get_nodata_from_path(erosivity_path)
    erodibility_nodata = pygeoprocessing.geoprocessing.get_nodata_from_path(erodibility_path)
    stream_nodata = pygeoprocessing.geoprocessing.get_nodata_from_path(stream_path)
    usle_nodata = -1.0

    cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_path(ls_factor_path)
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

    dataset_path_list = [
        ls_factor_path, erosivity_path, erodibility_path, stream_path]

    #Aligning with index 3 that's the stream and the most likely to be
    #aligned with LULCs
    pygeoprocessing.geoprocessing.vectorize_datasets(
        dataset_path_list, rkls_function, rkls_path, gdal.GDT_Float32,
        usle_nodata, cell_size, "intersection", dataset_to_align_index=3,
        vectorize_op=False)


def _process_dem(
        dem_path, slope_path, thresholded_slope_path, flow_direction_path,
        flow_accumulation_path, ls_path):
    """Process the DEM related operations such as slope and flow accumulation.

    """
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_path(
        dem_path)

    #Calculate slope
    LOGGER.info("Calculating slope")
    pygeoprocessing.geoprocessing.calculate_slope(dem_path, slope_path)
    slope_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
        slope_path)
    def threshold_slope(slope):
        """Convert slope to m/m and clamp at 0.005 and 1.0.

        As desribed in Cavalli et al., 2013.
        """
        slope_copy = slope / 100
        nodata_mask = slope == slope_nodata
        slope_copy[slope_copy < 0.005] = 0.005
        slope_copy[slope_copy > 1.0] = 1.0
        slope_copy[nodata_mask] = slope_nodata
        return slope_copy

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [slope_path], threshold_slope, thresholded_slope_path,
        gdal.GDT_Float64, slope_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    #Calculate flow accumulation
    LOGGER.info("calculating flow accumulation")
    pygeoprocessing.routing.flow_direction_d_inf(
        dem_path, flow_direction_path)
    pygeoprocessing.routing.flow_accumulation(
        flow_direction_path, dem_path, flow_accumulation_path)

    #Calculate LS term
    LOGGER.info('calculate ls term')
    ls_nodata = -1.0
    calculate_ls_factor(
        flow_accumulation_path, slope_path, flow_direction_path, ls_path,
        ls_nodata)


def _build_file_registry(base_file_path_list, file_suffix):
    """Combine file suffixes with key names, base filenames, and directories.

    Parameters:
        base_file_tuple_list (list): a list of (dict, path) tuples where
            the dictionaries have a 'file_key': 'basefilename' pair, or
            'file_key': list of 'basefilename's.  'path'
            indicates the file directory path to prepend to the basefile name.
        file_suffix (string): a string to append to every filename, can be
            empty string

    Returns:
        dictionary of 'file_keys' from the dictionaries in
        `base_file_tuple_list` mapping to full file paths with suffixes or
        lists of file paths with suffixes depending on the original type of
        the 'basefilename' pair.

    Raises:
        ValueError if there are duplicate file keys or duplicate file paths.
    """
    all_paths = set()
    duplicate_keys = set()
    duplicate_paths = set()
    file_registry = {}

    def _build_path(base_filename, path):
        """Internal helper to avoid code duplication."""
        pre, post = os.path.splitext(base_filename)
        full_path = os.path.join(path, pre+file_suffix+post)

        # Check for duplicate keys or paths
        if full_path in all_paths:
            duplicate_paths.add(full_path)
        else:
            all_paths.add(full_path)
        return full_path

    for base_file_dict, path in base_file_path_list:
        for file_key, file_payload in base_file_dict.iteritems():
            # check for duplicate keys
            if file_key in file_registry:
                duplicate_keys.add(file_key)
            else:
                # handle the case whether it's a filename or a list of strings
                if isinstance(file_payload, basestring):
                    full_path = _build_path(file_payload, path)
                    file_registry[file_key] = full_path
                elif isinstance(file_payload, list):
                    file_registry[file_key] = []
                    for filename in file_payload:
                        full_path = _build_path(filename, path)
                        file_registry[file_key].append(full_path)

    if len(duplicate_paths) > 0 or len(duplicate_keys):
        raise ValueError(
            "Cannot consolidate because of duplicate paths or keys: "
            "duplicate_keys: %s duplicate_paths: %s" % (
                duplicate_keys, duplicate_paths))

    return file_registry
