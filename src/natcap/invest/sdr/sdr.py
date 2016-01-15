"""InVEST Sediment Delivery Ratio (SDR) module."""
import os
import logging

from osgeo import gdal
from osgeo import ogr
import numpy

import pygeoprocessing
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
    'sed_retention_path': 'sed_retention.tif',
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
    'd_dn_path': 'd_dn.tif',
    'd_dn_bare_soil_path': 'd_dn_bare_soil.tif',
    'd_up_bare_soil_path': 'd_up_bare_soil.tif',
    'ws_factor_path': 'ws_factor.tif',
    'ic_path': 'ic.tif',
    'ic_bare_soil_path': 'ic_bare_soil.tif',
    'sdr_path': 'sdr_factor.tif',
    'sdr_bare_soil_path': 'sdr_bare_soil.tif',
    'stream_and_drainage_path': 'stream_and_drainage.tif',
    'w_path': 'w.tif',
    'thresholded_w_path': 'w_threshold.tif',
    'ws_inverse_path': 'ws_inverse.tif',
    's_inverse_path': 's_inverse.tif',
    'cp_factor_path': 'cp.tif',
    }

_TMP_BASE_FILES = {
    'aligned_dem_path': 'aligned_dem.tif',
    'aligned_lulc_path': 'aligned_lulc.tif',
    'aligned_erosivity_path': 'aligned_erosivity.tif',
    'aligned_erodibility_path': 'aligned_erodibility.tif',
    'aligned_watersheds_path': 'aligned_watersheds_path.tif',
    'aligned_drainage_path': 'aligned_drainage.tif',
    'zero_absorption_source_path': 'zero_absorption_source.tif',
    'loss_path': 'loss.tif',
    'w_accumulation_path': 'w_accumulation.tif',
    's_accumulation_path': 's_accumulation.tif',
    }

NODATA_USLE = -1.0


def execute(args):
    """InvEST SDR model.

    This function calculates the sediment export and retention of a landscape
    using the sediment delivery ratio model described in the InVEST user's
    guide.

    Parameters:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output file names
        args['dem_path'] (string): path to a digital elevation raster
        args['erosivity_path'] (string): path to rainfall erosivity index
            raster
        args['erodibility_path'] (string): a path to soil erodibility raster
        args['lulc_path'] (string): path to land use/land cover raster
        args['watersheds_path'] (string): path to vector of the watersheds
        args['biophysical_table_path'] (string): path to CSV file with
            biophysical information of each land use classes.  contain the
            fields 'usle_c' and 'usle_p'
        args['threshold_flow_accumulation'] (number): number of upstream pixels
            on the dem to threshold to a stream.
        args['k_param'] (number): k calibration parameter
        args['sdr_max'] (number): max value the SDR
        args['ic_0_param'] (number): ic_0 calibration parameter
        args['drainage_path'] (string): (optional) path to drainage raster that
            is used to add additional drainage areas to the internally
            calculated stream layer

    Returns:
        None.
    """
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
    pygeoprocessing.create_directories(
        [output_dir, intermediate_output_dir])

    f_reg = _build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir),
         (_TMP_BASE_FILES, output_dir)], file_suffix)

    base_list = []
    aligned_list = []
    for file_key in ['lulc', 'dem', 'erosivity', 'erodibility']:
        base_list.append(args[file_key + "_path"])
        aligned_list.append(f_reg["aligned_" + file_key + "_path"])

    drainage_present = False
    if 'drainage_path' in args and args['drainage_path'] != '':
        drainage_present = True
        base_list.append(args['drainage_path'])
        aligned_list.append(f_reg['aligned_drainage_path'])

    out_pixel_size = pygeoprocessing.get_cell_size_from_uri(
        args['lulc_path'])
    pygeoprocessing.align_dataset_list(
        base_list, aligned_list, ['nearest'] * len(base_list), out_pixel_size,
        'intersection', 0, aoi_uri=args['watersheds_path'])

    # do DEM processing here
    _process_dem(*[f_reg[key] for key in [
        'aligned_dem_path', 'slope_path', 'thresholded_slope_path',
        'flow_direction_path', 'flow_accumulation_path', 'ls_path']])

    #classify streams from the flow accumulation raster
    LOGGER.info("Classifying streams from flow accumulation raster")
    pygeoprocessing.routing.stream_threshold(
        f_reg['flow_accumulation_path'],
        float(args['threshold_flow_accumulation']),
        f_reg['stream_path'])
    stream_nodata = pygeoprocessing.get_nodata_from_uri(
        f_reg['stream_path'])

    if drainage_present:
        _add_drainage(
            f_reg['stream_path'],
            f_reg['aligned_drainage_path'],
            f_reg['stream_and_drainage_path'])
        f_reg['drainage_raster_path'] = (
            f_reg['stream_and_drainage_path'])
    else:
        f_reg['drainage_raster_path'] = (
            f_reg['stream_path'])

    #Calculate the W factor
    LOGGER.info('calculate per pixel W')
    _calculate_w(biophysical_table, *[f_reg[key] for key in [
        'aligned_lulc_path', 'w_path', 'thresholded_w_path']])

    LOGGER.info('calculate CP raster')
    _calculate_cp(
        biophysical_table, f_reg['aligned_lulc_path'],
        f_reg['cp_factor_path'])

    LOGGER.info('calculating RKLS')
    _calculate_rkls(*[f_reg[key] for key in [
        'ls_path', 'aligned_erosivity_path', 'aligned_erodibility_path',
        'stream_path', 'rkls_path']])

    LOGGER.info('calculating USLE')
    _calculate_usle(*[f_reg[key] for key in [
        'rkls_path', 'cp_factor_path', 'drainage_raster_path', 'usle_path']])

    #calculate W_bar
    LOGGER.info('calculating W_bar')
    for factor_path, accumulation_path, out_bar_path in [
            (f_reg['thresholded_w_path'], f_reg['w_accumulation_path'],
             f_reg['w_bar_path']),
            (f_reg['thresholded_slope_path'], f_reg['s_accumulation_path'],
             f_reg['s_bar_path'])]:
        _calculate_bar_factor(
            f_reg['aligned_dem_path'], factor_path,
            f_reg['flow_accumulation_path'], f_reg['flow_direction_path'],
            f_reg['zero_absorption_source_path'], f_reg['loss_path'],
            accumulation_path, out_bar_path)

    LOGGER.info('calculating d_up')
    _calculate_d_up(
        *[f_reg[key] for key in [
            'w_bar_path', 's_bar_path', 'flow_accumulation_path',
            'd_up_path']])

    LOGGER.info('calculate WS factor')
    _calculate_inverse_ws_factor(
        f_reg['thresholded_slope_path'], f_reg['thresholded_w_path'],
        f_reg['ws_inverse_path'])

    LOGGER.info('calculating d_dn')
    pygeoprocessing.routing.routing_core.distance_to_stream(
        f_reg['flow_direction_path'], f_reg['stream_path'],
        f_reg['d_dn_path'], factor_uri=f_reg['ws_inverse_path'])

    LOGGER.info('calculate ic')
    _calculate_ic(
        f_reg['d_up_path'], f_reg['d_dn_path'], f_reg['ic_path'])

    LOGGER.info('calculate sdr')
    _calculate_sdr(
        float(args['k_param']), float(args['ic_0_param']),
        float(args['sdr_max']), f_reg['ic_path'], f_reg['stream_path'],
        f_reg['sdr_path'])

    LOGGER.info('calculate sed export')
    _calculate_sed_export(
        f_reg['usle_path'], f_reg['sdr_path'], f_reg['sed_export_path'])

    LOGGER.info('calculate sediment retention index')
    _calculate_sed_retention_index(
        f_reg['rkls_path'], f_reg['usle_path'], f_reg['sdr_path'],
        float(args['sdr_max']), f_reg['sed_retention_index_path'])

    LOGGER.info('calculate sediment retention')
    # calculate inverse s factor (not ws)
    LOGGER.info('calculate S factor')
    _calculate_inverse_s_factor(
        f_reg['thresholded_slope_path'], f_reg['s_inverse_path'])
    # calculate d_dn_bare_soil_path
    LOGGER.info('calculating d_dn bare soil')
    pygeoprocessing.routing.routing_core.distance_to_stream(
        f_reg['flow_direction_path'], f_reg['stream_path'],
        f_reg['d_dn_bare_soil_path'], factor_uri=f_reg['s_inverse_path'])

    # calculate d_up_bare_soil_path
    LOGGER.info('calculating d_up bare soil')
    _calculate_d_up_bare(
        f_reg['s_bar_path'], f_reg['flow_accumulation_path'],
        f_reg['d_up_bare_soil_path'])

    # calculate ic_factor_bare_soil_path
    LOGGER.info('calculate ic')
    _calculate_ic(
        f_reg['d_up_bare_soil_path'], f_reg['d_dn_bare_soil_path'],
        f_reg['ic_bare_soil_path'])
    # calculate sdr_factor_bare_soil_path
    _calculate_sdr(
        float(args['k_param']), float(args['ic_0_param']),
        float(args['sdr_max']), f_reg['ic_bare_soil_path'],
        f_reg['stream_path'], f_reg['sdr_bare_soil_path'])

    # sed_retention_bare_soil_path
    _calculate_sed_retention(
        f_reg['rkls_path'], f_reg['usle_path'], f_reg['stream_path'],
        f_reg['sdr_path'], f_reg['sdr_bare_soil_path'],
        f_reg['sed_retention_path'])

    LOGGER.info('generating report')
    _generate_report(
        args['watersheds_path'], f_reg['usle_path'],
        f_reg['sed_export_path'], f_reg['sed_retention_path'],
        f_reg['watershed_results_sdr_path'])


def calculate_ls_factor(
        flow_accumulation_path, slope_path, aspect_path, ls_factor_path,
        ls_nodata):
    """Calculates the LS factor.

    LS factor as Equation 3 from "Extension and validation
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

    Returns:
        None
    """
    flow_accumulation_nodata = pygeoprocessing.get_nodata_from_uri(
        flow_accumulation_path)
    slope_nodata = pygeoprocessing.get_nodata_from_uri(slope_path)
    aspect_nodata = pygeoprocessing.get_nodata_from_uri(aspect_path)

    #Assumes that cells are square
    cell_size = pygeoprocessing.get_cell_size_from_uri(flow_accumulation_path)
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
    pygeoprocessing.vectorize_datasets(
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

    pygeoprocessing.vectorize_datasets(
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

    pygeoprocessing.vectorize_datasets(
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
    pygeoprocessing.vectorize_datasets(
        dataset_path_list, s_factor_op, s_factor_path, gdal.GDT_Float32,
        ls_nodata, cell_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)

    def xi_op(aspect_angle, percent_slope, flow_accumulation):
        return (numpy.abs(numpy.sin(aspect_angle)) +
            numpy.abs(numpy.cos(aspect_angle)))
    pygeoprocessing.vectorize_datasets(
        dataset_path_list, xi_op, xi_path, gdal.GDT_Float32,
        ls_nodata, cell_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)


def _calculate_rkls(
        ls_factor_path, erosivity_path, erodibility_path, stream_path,
        rkls_path):
    """Calculate per-pixel potential soil loss using the RKLS.

    (revised universial soil loss equation with no C or P).

    Parameters:
        ls_factor_path (string): path to LS raster
        erosivity_path (string): path to per pixel erosivity raster
        erodibility_path (string): path to erodibility raster
        stream_path (string): path to drainage raster
            (1 is drainage, 0 is not)
        rkls_path (string): path to RKLS raster

    Returns:
        None
    """
    ls_factor_nodata = pygeoprocessing.get_nodata_from_uri(ls_factor_path)
    erosivity_nodata = pygeoprocessing.get_nodata_from_uri(erosivity_path)
    erodibility_nodata = pygeoprocessing.get_nodata_from_uri(erodibility_path)
    stream_nodata = pygeoprocessing.get_nodata_from_uri(stream_path)
    usle_nodata = -1.0

    cell_size = pygeoprocessing.get_cell_size_from_uri(ls_factor_path)
    cell_area_ha = cell_size ** 2 / 10000.0

    def rkls_function(ls_factor, erosivity, erodibility, stream):
        """Calculates the RKLS equation.

        ls_factor - length/slope factor
        erosivity - related to peak rainfall events
        erodibility - related to the potential for soil to erode
        stream - 1 or 0 depending if there is a stream there.  If so, no
            potential soil loss due to USLE

        returns ls_factor * erosivity * erodibility * usle_c_p if all arguments
            defined, nodata if some are not defined, 0 if in a stream
            (stream)"""

        rkls = numpy.empty(ls_factor.shape, dtype=numpy.float32)
        nodata_mask = (
            (ls_factor != ls_factor_nodata) & (erosivity != erosivity_nodata) &
            (erodibility != erodibility_nodata) & (stream != stream_nodata))
        valid_mask = nodata_mask & (stream == 0)
        rkls[:] = usle_nodata

        rkls[valid_mask] = (
            ls_factor[valid_mask] * erosivity[valid_mask] *
            erodibility[valid_mask] * cell_area_ha)

        # rkls is 1 on the stream
        rkls[nodata_mask & (stream == 1)] = 1
        return rkls

    dataset_path_list = [
        ls_factor_path, erosivity_path, erodibility_path, stream_path]

    #Aligning with index 3 that's the stream and the most likely to be
    #aligned with LULCs
    pygeoprocessing.vectorize_datasets(
        dataset_path_list, rkls_function, rkls_path, gdal.GDT_Float32,
        usle_nodata, cell_size, "intersection", dataset_to_align_index=3,
        vectorize_op=False)


def _process_dem(
        dem_path, slope_path, thresholded_slope_path, flow_direction_path,
        flow_accumulation_path, ls_path):
    """Process the DEM related operations such as slope and flow accumulation.

    """
    out_pixel_size = pygeoprocessing.get_cell_size_from_uri(
        dem_path)

    #Calculate slope
    LOGGER.info("Calculating slope")
    pygeoprocessing.calculate_slope(dem_path, slope_path)
    slope_nodata = pygeoprocessing.get_nodata_from_uri(
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

    pygeoprocessing.vectorize_datasets(
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
    f_reg = {}

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
            if file_key in f_reg:
                duplicate_keys.add(file_key)
            else:
                # handle the case whether it's a filename or a list of strings
                if isinstance(file_payload, basestring):
                    full_path = _build_path(file_payload, path)
                    f_reg[file_key] = full_path
                elif isinstance(file_payload, list):
                    f_reg[file_key] = []
                    for filename in file_payload:
                        full_path = _build_path(filename, path)
                        f_reg[file_key].append(full_path)

    if len(duplicate_paths) > 0 or len(duplicate_keys):
        raise ValueError(
            "Cannot consolidate because of duplicate paths or keys: "
            "duplicate_keys: %s duplicate_paths: %s" % (
                duplicate_keys, duplicate_paths))

    return f_reg


def _add_drainage(stream_path, drainage_path, out_stream_and_drainage_path):
    """Add drainage layer to the stream path."""
    def add_drainage_op(stream, drainage):
        """Add drainage mask to stream layer."""
        return numpy.where(drainage == 1, 1, stream)

    stream_nodata = pygeoprocessing.get_nodata_from_uri(
        stream_path)

    out_pixel_size = pygeoprocessing.get_cell_size_from_uri(stream_path)
    pygeoprocessing.vectorize_datasets(
        [stream_path, drainage_path], add_drainage_op,
        out_stream_and_drainage_path, gdal.GDT_Byte, stream_nodata,
        out_pixel_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)


def _calculate_w(
        biophysical_table, lulc_path, w_factor_path,
        thresholded_w_factor_path):
    """Calculate W factor."""
    #map lulc to biophysical table
    lulc_to_c = dict(
        [(lulc_code, float(table['usle_c'])) for
         (lulc_code, table) in biophysical_table.items()])
    w_nodata = -1.0

    pygeoprocessing.reclassify_dataset_uri(
        lulc_path, lulc_to_c, w_factor_path, gdal.GDT_Float64,
        w_nodata, exception_flag='values_required')
    def threshold_w(w_val):
        """Threshold w to 0.001."""
        w_val_copy = w_val.copy()
        nodata_mask = w_val == w_nodata
        w_val_copy[w_val < 0.001] = 0.001
        w_val_copy[nodata_mask] = w_nodata
        return w_val_copy

    out_pixel_size = pygeoprocessing.get_cell_size_from_uri(lulc_path)
    pygeoprocessing.vectorize_datasets(
        [w_factor_path], threshold_w, thresholded_w_factor_path,
        gdal.GDT_Float64, w_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)


def _calculate_cp(biophysical_table, lulc_path, cp_factor_path):
    """Reclass landcover to CP values."""
    lulc_to_cp = dict(
        [(lulc_code, float(table['usle_c']) * float(table['usle_p'])) for
         (lulc_code, table) in biophysical_table.items()])
    cp_nodata = -1.0
    pygeoprocessing.reclassify_dataset_uri(
        lulc_path, lulc_to_cp, cp_factor_path, gdal.GDT_Float64,
        cp_nodata, exception_flag='values_required')


def _calculate_usle(
        rkls_path, cp_factor_path, drainage_raster_path, out_usle_path):
    """Calculate USLE."""
    nodata_rkls = pygeoprocessing.get_nodata_from_uri(rkls_path)
    nodata_cp = pygeoprocessing.get_nodata_from_uri(
        cp_factor_path)

    def usle_op(rkls, cp_factor, drainage):
        """Calculate USLE."""
        result = numpy.empty(rkls.shape)
        result[:] = NODATA_USLE
        valid_mask = (rkls != nodata_rkls) & (cp_factor != nodata_cp)
        result[valid_mask] = rkls[valid_mask] * cp_factor[valid_mask] * (
            1 - drainage[valid_mask])
        return result

    out_pixel_size = pygeoprocessing.get_cell_size_from_uri(rkls_path)
    LOGGER.debug(
        "%s\n%s\n%s", rkls_path, cp_factor_path, drainage_raster_path)
    LOGGER.debug("%s", out_usle_path)
    pygeoprocessing.vectorize_datasets(
        [rkls_path, cp_factor_path, drainage_raster_path], usle_op,
        out_usle_path, gdal.GDT_Float64, NODATA_USLE, out_pixel_size,
        "intersection", dataset_to_align_index=0, vectorize_op=False)


def _calculate_bar_factor(
        dem_path, factor_path, flow_accumulation_path, flow_direction_path,
        zero_absorption_source_path, loss_path, accumulation_path,
        out_bar_path):
    """Calculate a bar factor, likely W or S."""
    #need this for low level route_flux function
    pygeoprocessing.make_constant_raster_from_base_uri(
        dem_path, 0.0, zero_absorption_source_path)

    flow_accumulation_nodata = pygeoprocessing.get_nodata_from_uri(
        flow_accumulation_path)

    LOGGER.info("calculating %s", accumulation_path)
    pygeoprocessing.routing.route_flux(
        flow_direction_path, dem_path, factor_path,
        zero_absorption_source_path, loss_path, accumulation_path,
        'flux_only')

    LOGGER.info("calculating bar factor")
    bar_nodata = pygeoprocessing.get_nodata_from_uri(accumulation_path)
    out_pixel_size = pygeoprocessing.get_cell_size_from_uri(dem_path)
    LOGGER.info("calculating %s", accumulation_path)
    def bar_op(base_accumulation, flow_accumulation):
        """Aggreegate accumulation from base divided by the flow accum."""
        result = numpy.empty(base_accumulation.shape)
        valid_mask = (
            (base_accumulation != bar_nodata) &
            (flow_accumulation != flow_accumulation_nodata))
        result[:] = bar_nodata
        result[valid_mask] = (
            base_accumulation[valid_mask] / flow_accumulation[valid_mask])
        return result
    pygeoprocessing.vectorize_datasets(
        [accumulation_path, flow_accumulation_path], bar_op, out_bar_path,
        gdal.GDT_Float32, bar_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)


def _calculate_d_up(
        w_bar_path, s_bar_path, flow_accumulation_path, out_d_up_path):
    """Calculate D_up."""
    out_pixel_size = pygeoprocessing.get_cell_size_from_uri(w_bar_path)
    cell_area = out_pixel_size ** 2
    d_up_nodata = -1.0
    w_bar_nodata = pygeoprocessing.get_nodata_from_uri(w_bar_path)
    s_bar_nodata = pygeoprocessing.get_nodata_from_uri(s_bar_path)
    flow_accumulation_nodata = pygeoprocessing.get_nodata_from_uri(
        flow_accumulation_path)

    def d_up(w_bar, s_bar, flow_accumulation):
        """Calculate the d_up index.

        w_bar * s_bar * sqrt(upstream area)

        """
        valid_mask = (
            (w_bar != w_bar_nodata) & (s_bar != s_bar_nodata) &
            (flow_accumulation != flow_accumulation_nodata))
        d_up_array = numpy.empty(valid_mask.shape)
        d_up_array[:] = d_up_nodata
        d_up_array[valid_mask] = (
            w_bar[valid_mask] * s_bar[valid_mask] * numpy.sqrt(
                flow_accumulation[valid_mask] * cell_area))
        return d_up_array

    pygeoprocessing.vectorize_datasets(
        [w_bar_path, s_bar_path, flow_accumulation_path], d_up, out_d_up_path,
        gdal.GDT_Float32, d_up_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)


def _calculate_d_up_bare(
        s_bar_path, flow_accumulation_path, out_d_up_bare_path):
    """Calculate D_up."""
    out_pixel_size = pygeoprocessing.get_cell_size_from_uri(s_bar_path)
    cell_area = out_pixel_size ** 2
    d_up_nodata = -1.0
    s_bar_nodata = pygeoprocessing.get_nodata_from_uri(s_bar_path)
    flow_accumulation_nodata = pygeoprocessing.get_nodata_from_uri(
        flow_accumulation_path)

    def d_up(s_bar, flow_accumulation):
        """Calculate the d_up index.

        w_bar * s_bar * sqrt(upstream area)

        """
        valid_mask = (
            (flow_accumulation != flow_accumulation_nodata) &
            (s_bar != s_bar_nodata))
        d_up_array = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        d_up_array[:] = d_up_nodata
        d_up_array[valid_mask] = (
            numpy.sqrt(flow_accumulation[valid_mask] * cell_area) *
            s_bar[valid_mask])
        return d_up_array

    pygeoprocessing.vectorize_datasets(
        [s_bar_path, flow_accumulation_path], d_up, out_d_up_bare_path,
        gdal.GDT_Float32, d_up_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)


def _calculate_inverse_ws_factor(
        thresholded_slope_path, thresholded_w_factor_path,
        out_ws_factor_inverse_path):
    ws_nodata = -1.0
    slope_nodata = pygeoprocessing.get_nodata_from_uri(thresholded_slope_path)
    w_nodata = pygeoprocessing.get_nodata_from_uri(thresholded_w_factor_path)
    out_pixel_size = pygeoprocessing.get_cell_size_from_uri(
        thresholded_slope_path)

    def ws_op(w_factor, s_factor):
        """Calculate the inverse ws factor."""
        return numpy.where(
            (w_factor != w_nodata) & (s_factor != slope_nodata),
            1.0 / (w_factor * s_factor), ws_nodata)

    pygeoprocessing.vectorize_datasets(
        [thresholded_w_factor_path, thresholded_slope_path], ws_op,
        out_ws_factor_inverse_path, gdal.GDT_Float32, ws_nodata,
        out_pixel_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)


def _calculate_inverse_s_factor(
        thresholded_slope_path, out_s_factor_inverse_path):
    s_nodata = -1.0
    slope_nodata = pygeoprocessing.get_nodata_from_uri(thresholded_slope_path)
    out_pixel_size = pygeoprocessing.get_cell_size_from_uri(
        thresholded_slope_path)

    def s_op(s_factor):
        """Calculate the inverse s factor."""
        return numpy.where(
            (s_factor != slope_nodata), 1.0 / s_factor, s_nodata)

    pygeoprocessing.vectorize_datasets(
        [thresholded_slope_path], s_op,
        out_s_factor_inverse_path, gdal.GDT_Float32, s_nodata,
        out_pixel_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)


def _calculate_ic(d_up_path, d_dn_path, out_ic_factor_path):
    ic_nodata = -9999.0
    d_up_nodata = pygeoprocessing.get_nodata_from_uri(d_up_path)
    d_dn_nodata = pygeoprocessing.get_nodata_from_uri(d_dn_path)
    out_pixel_size = pygeoprocessing.get_cell_size_from_uri(d_up_path)

    def ic_op(d_up, d_dn):
        """Calculate IC factor."""
        valid_mask = (
            (d_up != d_up_nodata) & (d_dn != d_dn_nodata) & (d_dn != 0) &
            (d_up != 0))
        ic_array = numpy.empty(valid_mask.shape)
        ic_array[:] = ic_nodata
        ic_array[valid_mask] = numpy.log10(d_up[valid_mask] / d_dn[valid_mask])
        return ic_array

    pygeoprocessing.vectorize_datasets(
        [d_up_path, d_dn_path], ic_op, out_ic_factor_path,
        gdal.GDT_Float32, ic_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)


def _calculate_sdr(
        k_factor, ic_0, sdr_max, ic_path, stream_path, out_sdr_path):
    sdr_nodata = -9999.0
    out_pixel_size = pygeoprocessing.get_cell_size_from_uri(stream_path)
    ic_nodata = pygeoprocessing.get_nodata_from_uri(ic_path)

    def sdr_op(ic_factor, stream):
        """Calculate SDR factor."""
        nodata_mask = (ic_factor == ic_nodata)
        sdr = numpy.where(
            nodata_mask, sdr_nodata, sdr_max/(1+numpy.exp((ic_0-ic_factor)/k_factor)))
        #mask out the stream layer
        return numpy.where(stream == 1, 0.0, sdr)

    pygeoprocessing.vectorize_datasets(
        [ic_path, stream_path], sdr_op, out_sdr_path,
        gdal.GDT_Float32, sdr_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)


def _calculate_sed_export(usle_path, sdr_path, out_sed_export_path):
    sed_export_nodata = -1.0
    out_pixel_size = pygeoprocessing.get_cell_size_from_uri(usle_path)
    sdr_nodata = pygeoprocessing.get_nodata_from_uri(sdr_path)
    usle_nodata = pygeoprocessing.get_nodata_from_uri(usle_path)

    def sed_export_op(usle, sdr):
        """Sediment export."""
        nodata_mask = (usle == usle_nodata) | (sdr == sdr_nodata)
        return numpy.where(
            nodata_mask, sed_export_nodata, usle * sdr)

    pygeoprocessing.vectorize_datasets(
        [usle_path, sdr_path], sed_export_op, out_sed_export_path,
        gdal.GDT_Float32, sed_export_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)


def _calculate_sed_retention_index(
        rkls_path, usle_path, sdr_path, sdr_max,
        out_sed_retention_index_path):

    rkls_nodata = pygeoprocessing.get_nodata_from_uri(rkls_path)
    usle_nodata = pygeoprocessing.get_nodata_from_uri(usle_path)
    sdr_nodata = pygeoprocessing.get_nodata_from_uri(sdr_path)

    def sediment_index_op(rkls, usle, sdr_factor):
        """Calculate sediment retention index."""
        nodata_mask = (
            (rkls == rkls_nodata) | (usle == usle_nodata) |
            (sdr_factor == sdr_nodata))
        return numpy.where(
            nodata_mask,
            nodata_sed_retention_index, (rkls - usle) * sdr_factor / sdr_max)

    nodata_sed_retention_index = -1
    out_pixel_size = pygeoprocessing.get_cell_size_from_uri(rkls_path)

    pygeoprocessing.vectorize_datasets(
        [rkls_path, usle_path, sdr_path], sediment_index_op,
        out_sed_retention_index_path, gdal.GDT_Float32,
        nodata_sed_retention_index, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)


def _calculate_sed_retention(
        rkls_path, usle_path, stream_path, sdr_path, sdr_bare_soil_path,
        out_sed_ret_bare_soil_path):

    rkls_nodata = pygeoprocessing.get_nodata_from_uri(rkls_path)
    usle_nodata = pygeoprocessing.get_nodata_from_uri(usle_path)
    stream_nodata = pygeoprocessing.get_nodata_from_uri(stream_path)
    sdr_nodata = pygeoprocessing.get_nodata_from_uri(sdr_path)

    def sediment_retention_bare_soil_op(
            rkls, usle, stream_factor, sdr_factor, sdr_factor_bare_soil):
        """Subract bare soil export from real landcover."""
        nodata_mask = (
            (rkls == rkls_nodata) | (usle == usle_nodata) |
            (stream_factor == stream_nodata) | (sdr_factor == sdr_nodata) |
            (sdr_factor_bare_soil == sdr_nodata))
        return numpy.where(
            nodata_mask, nodata_sediment_retention,
            (rkls * sdr_factor_bare_soil - usle * sdr_factor) * (
                1 - stream_factor))

    nodata_sediment_retention = -1

    out_pixel_size = pygeoprocessing.get_cell_size_from_uri(rkls_path)
    pygeoprocessing.vectorize_datasets(
        [rkls_path, usle_path, stream_path, sdr_path, sdr_bare_soil_path],
        sediment_retention_bare_soil_op, out_sed_ret_bare_soil_path,
        gdal.GDT_Float32, nodata_sediment_retention, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)


def _generate_report(
        watersheds_path, usle_path, sed_export_path, sed_retention_path,
        watershed_results_sdr_path):

    esri_driver = ogr.GetDriverByName('ESRI Shapefile')

    field_summaries = {
        'usle_tot': pygeoprocessing.aggregate_raster_values_uri(
            usle_path, watersheds_path, 'ws_id').total,
        'sed_export': pygeoprocessing.aggregate_raster_values_uri(
            sed_export_path, watersheds_path, 'ws_id').total,
        'sed_retent': pygeoprocessing.aggregate_raster_values_uri(
            sed_retention_path, watersheds_path, 'ws_id').total,
        }

    original_datasource = ogr.Open(watersheds_path)
    # Delete if existing shapefile with the same name and path
    if os.path.isfile(watershed_results_sdr_path):
        os.remove(watershed_results_sdr_path)
    datasource_copy = esri_driver.CopyDataSource(
        original_datasource, watershed_results_sdr_path)
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
                feature.SetField(
                    field_name, float(field_summaries[field_name][ws_id]))
            except KeyError:
                LOGGER.warning('unknown field %s', field_name)
                feature.SetField(field_name, 0.0)
        #Save back to datasource
        layer.SetFeature(feature)
    original_datasource.Destroy()
    datasource_copy.Destroy()
