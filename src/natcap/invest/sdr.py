"""InVEST Sediment Delivery Ratio (SDR) module.

The SDR method in this model is based on:
    Winchell, M. F., et al. "Extension and validation of a geographic
    information system-based method for calculating the Revised Universal
    Soil Loss Equation length-slope factor for erosion risk assessments in
    large watersheds." Journal of Soil and Water Conservation 63.3 (2008):
    105-111.
"""
from __future__ import absolute_import

import os
import logging

from osgeo import gdal
from osgeo import ogr
import numpy

import pygeoprocessing
import natcap.invest.pygeoprocessing_0_3_3.routing
import natcap.invest.pygeoprocessing_0_3_3.routing.routing_core
from . import utils
from . import validation

LOGGER = logging.getLogger('natcap.invest.sdr')

_OUTPUT_BASE_FILES = {
    'rkls_path': 'rkls.tif',
    'sed_export_path': 'sed_export.tif',
    'stream_path': 'stream.tif',
    'usle_path': 'usle.tif',
    'sed_retention_index_path': 'sed_retention_index.tif',
    'sed_retention_path': 'sed_retention.tif',
    'watershed_results_sdr_path': 'watershed_results_sdr.shp',
    'stream_and_drainage_path': 'stream_and_drainage.tif',
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
    'ic_bare_soil_path': 'ic_bare_soil.tif',
    'sdr_bare_soil_path': 'sdr_bare_soil.tif',
    'ws_factor_path': 'ws_factor.tif',
    'ic_path': 'ic.tif',
    'sdr_path': 'sdr_factor.tif',
    'w_path': 'w.tif',
    }

_TMP_BASE_FILES = {
    'cp_factor_path': 'cp.tif',
    'aligned_dem_path': 'aligned_dem.tif',
    'aligned_lulc_path': 'aligned_lulc.tif',
    'aligned_erosivity_path': 'aligned_erosivity.tif',
    'aligned_erodibility_path': 'aligned_erodibility.tif',
    'aligned_drainage_path': 'aligned_drainage.tif',
    'zero_absorption_source_path': 'zero_absorption_source.tif',
    'loss_path': 'loss.tif',
    'w_accumulation_path': 'w_accumulation.tif',
    's_accumulation_path': 's_accumulation.tif',
    'thresholded_w_path': 'w_threshold.tif',
    'ws_inverse_path': 'ws_inverse.tif',
    's_inverse_path': 's_inverse.tif',
    }

# Target nodata is for general rasters that are positive, and _IC_NODATA are
# for rasters that are any range
_TARGET_NODATA = -1.0
_IC_NODATA = numpy.finfo('float32').min


def execute(args):
    """Sediment Delivery Ratio.

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
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    biophysical_table = utils.build_lookup_from_csv(
        args['biophysical_table_path'], 'lucode')

    # Test to see if c or p values are outside of 0..1
    for table_key in ['usle_c', 'usle_p']:
        for (lulc_code, table) in biophysical_table.iteritems():
            try:
                float_value = float(table[table_key])
                if float_value < 0 or float_value > 1:
                    raise ValueError(
                        'Value should be within range 0..1 offending value '
                        'table %s, lulc_code %s, value %s' % (
                            table_key, str(lulc_code), str(float_value)))
            except ValueError:
                raise ValueError(
                    'Value is not a floating point value within range 0..1 '
                    'offending value table %s, lulc_code %s, value %s' % (
                        table_key, str(lulc_code), table[table_key]))

    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    output_dir = os.path.join(args['workspace_dir'])
    utils.make_directories([output_dir, intermediate_output_dir])

    f_reg = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir),
         (_TMP_BASE_FILES, output_dir)], file_suffix)

    base_list = []
    aligned_list = []
    for file_key in ['dem', 'lulc', 'erosivity', 'erodibility']:
        base_list.append(args[file_key + "_path"])
        aligned_list.append(f_reg["aligned_" + file_key + "_path"])

    drainage_present = False
    if 'drainage_path' in args and args['drainage_path'] != '':
        drainage_present = True
        base_list.append(args['drainage_path'])
        aligned_list.append(f_reg['aligned_drainage_path'])

    dem_pixel_size = pygeoprocessing.get_raster_info(
        args['dem_path'])['pixel_size']
    pygeoprocessing.align_and_resize_raster_stack(
        base_list, aligned_list, ['nearest'] * len(base_list),
        dem_pixel_size, 'intersection',
        base_vector_path_list=[args['watersheds_path']],
        raster_align_index=0)

    LOGGER.info("calculating slope")
    natcap.invest.pygeoprocessing_0_3_3.calculate_slope(
        f_reg['aligned_dem_path'], f_reg['slope_path'])
    _threshold_slope(f_reg['slope_path'], f_reg['thresholded_slope_path'])

    LOGGER.info("calculating flow direction")
    natcap.invest.pygeoprocessing_0_3_3.routing.flow_direction_d_inf(
        f_reg['aligned_dem_path'], f_reg['flow_direction_path'])

    LOGGER.info("calculating flow accumulation")
    natcap.invest.pygeoprocessing_0_3_3.routing.flow_accumulation(
        f_reg['flow_direction_path'], f_reg['aligned_dem_path'],
        f_reg['flow_accumulation_path'])

    LOGGER.info('calculate ls term')

    _calculate_ls_factor(
        f_reg['flow_accumulation_path'], f_reg['slope_path'],
        f_reg['flow_direction_path'], f_reg['ls_path'])

    LOGGER.info("classifying streams from flow accumulation raster")
    natcap.invest.pygeoprocessing_0_3_3.routing.stream_threshold(
        f_reg['flow_accumulation_path'],
        float(args['threshold_flow_accumulation']),
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

    LOGGER.info('calculate per pixel W')
    _calculate_w(
        biophysical_table, f_reg['aligned_lulc_path'], f_reg['w_path'],
        f_reg['thresholded_w_path'])

    LOGGER.info('calculate CP raster')
    _calculate_cp(
        biophysical_table, f_reg['aligned_lulc_path'],
        f_reg['cp_factor_path'])

    LOGGER.info('calculating RKLS')
    _calculate_rkls(*[f_reg[key] for key in [
        'ls_path', 'aligned_erosivity_path', 'aligned_erodibility_path',
        'drainage_raster_path', 'rkls_path']])

    LOGGER.info('calculating USLE')
    _calculate_usle(*[f_reg[key] for key in [
        'rkls_path', 'cp_factor_path', 'drainage_raster_path', 'usle_path']])

    LOGGER.info('calculating w_bar')
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
    natcap.invest.pygeoprocessing_0_3_3.routing.routing_core.distance_to_stream(
        f_reg['flow_direction_path'], f_reg['drainage_raster_path'],
        f_reg['d_dn_path'], factor_uri=f_reg['ws_inverse_path'])

    LOGGER.info('calculate ic')
    _calculate_ic(
        f_reg['d_up_path'], f_reg['d_dn_path'], f_reg['ic_path'])

    LOGGER.info('calculate sdr')
    _calculate_sdr(
        float(args['k_param']), float(args['ic_0_param']),
        float(args['sdr_max']), f_reg['ic_path'],
        f_reg['drainage_raster_path'], f_reg['sdr_path'])

    LOGGER.info('calculate sed export')
    _calculate_sed_export(
        f_reg['usle_path'], f_reg['sdr_path'], f_reg['sed_export_path'])

    LOGGER.info('calculate sediment retention index')
    _calculate_sed_retention_index(
        f_reg['rkls_path'], f_reg['usle_path'], f_reg['sdr_path'],
        float(args['sdr_max']), f_reg['sed_retention_index_path'])

    LOGGER.info('calculate sediment retention')
    LOGGER.info('calculate S factor')
    _calculate_inverse_s_factor(
        f_reg['thresholded_slope_path'], f_reg['s_inverse_path'])

    LOGGER.info('calculating d_dn bare soil')
    natcap.invest.pygeoprocessing_0_3_3.routing.routing_core.distance_to_stream(
        f_reg['flow_direction_path'], f_reg['drainage_raster_path'],
        f_reg['d_dn_bare_soil_path'], factor_uri=f_reg['s_inverse_path'])

    LOGGER.info('calculating d_up bare soil')
    _calculate_d_up_bare(
        f_reg['s_bar_path'], f_reg['flow_accumulation_path'],
        f_reg['d_up_bare_soil_path'])

    LOGGER.info('calculate ic')
    _calculate_ic(
        f_reg['d_up_bare_soil_path'], f_reg['d_dn_bare_soil_path'],
        f_reg['ic_bare_soil_path'])

    _calculate_sdr(
        float(args['k_param']), float(args['ic_0_param']),
        float(args['sdr_max']), f_reg['ic_bare_soil_path'],
        f_reg['drainage_raster_path'], f_reg['sdr_bare_soil_path'])

    _calculate_sed_retention(
        f_reg['rkls_path'], f_reg['usle_path'], f_reg['drainage_raster_path'],
        f_reg['sdr_path'], f_reg['sdr_bare_soil_path'],
        f_reg['sed_retention_path'])

    LOGGER.info('generating report')
    _generate_report(
        args['watersheds_path'], f_reg['usle_path'],
        f_reg['sed_export_path'], f_reg['sed_retention_path'],
        f_reg['watershed_results_sdr_path'])

    for tmp_filename_key in _TMP_BASE_FILES:
        if os.path.exists(f_reg[tmp_filename_key]):
            os.remove(f_reg[tmp_filename_key])


def _calculate_ls_factor(
        flow_accumulation_path, slope_path, aspect_path, out_ls_factor_path):
    """Calculate LS factor.

    LS factor as Equation 3 from "Extension and validation
    of a geographic information system-based method for calculating the
    Revised Universal Soil Loss Equation length-slope factor for erosion
    risk assessments in large watersheds"

    Parameters:
        flow_accumulation_path (string): path to raster, pixel values are the
            contributing upstream area at that cell
        slope_path (string): path to slope raster as a percent
        aspect_path string): path to raster flow direction raster in radians
        out_ls_factor_path (string): path to output ls_factor raster

    Returns:
        None
    """
    slope_nodata = pygeoprocessing.get_raster_info(slope_path)['nodata'][0]
    aspect_nodata = pygeoprocessing.get_raster_info(aspect_path)['nodata'][0]

    flow_accumulation_info = pygeoprocessing.get_raster_info(
        flow_accumulation_path)
    flow_accumulation_nodata = flow_accumulation_info['nodata'][0]
    cell_size = flow_accumulation_info['mean_pixel_size']
    cell_area = cell_size ** 2

    def ls_factor_function(aspect_angle, percent_slope, flow_accumulation):
        """Calculate the LS factor.

        Parameters:
            aspect_angle (numpy.ndarray): flow direction in radians
            percent_slope (numpy.ndarray): slope in percent
            flow_accumulation (numpy.ndarray): upstream pixels
        Returns:
            ls_factor
        """
        valid_mask = (
            (aspect_angle != aspect_nodata) &
            (percent_slope != slope_nodata) &
            (flow_accumulation != flow_accumulation_nodata))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA

        # Determine the length of the flow path on the pixel
        xij = (numpy.abs(numpy.sin(aspect_angle[valid_mask])) +
               numpy.abs(numpy.cos(aspect_angle[valid_mask])))

        contributing_area = (flow_accumulation[valid_mask]-1) * cell_area
        slope_in_radians = numpy.arctan(percent_slope[valid_mask] / 100.0)

        # From Equation 4 in "Extension and validation of a geographic
        # information system ..."
        slope_factor = numpy.where(
            percent_slope[valid_mask] < 9.0,
            10.8 * numpy.sin(slope_in_radians) + 0.03,
            16.8 * numpy.sin(slope_in_radians) - 0.5)

        beta = (
            (numpy.sin(slope_in_radians) / 0.0986) /
            (3 * numpy.sin(slope_in_radians)**0.8 + 0.56))

        # Set m value via lookup table: Table 1 in
        # InVEST Sediment Model_modifications_10-01-2012_RS.docx
        # note slope_table in percent
        slope_table = numpy.array([1., 3.5, 5., 9.])
        m_table = numpy.array([0.2, 0.3, 0.4, 0.5])
        # mask where slopes are larger than lookup table
        big_slope_mask = percent_slope[valid_mask] > slope_table[-1]
        m_indexes = numpy.digitize(
            percent_slope[valid_mask][~big_slope_mask], slope_table,
            right=True)
        m_exp = numpy.empty(big_slope_mask.shape, dtype=numpy.float32)
        m_exp[big_slope_mask] = (
            beta[big_slope_mask] / (1 + beta[big_slope_mask]))
        m_exp[~big_slope_mask] = m_table[m_indexes]

        l_factor = (
            ((contributing_area + cell_area)**(m_exp+1) -
             contributing_area ** (m_exp+1)) /
            ((cell_size ** (m_exp + 2)) * (xij**m_exp) * (22.13**m_exp)))

        # from McCool paper: "as a final check against excessively long slope
        # length calculations ... cap of 333m"
        l_factor[l_factor > 333] = 333

        result[valid_mask] = l_factor * slope_factor
        return result

    # call vectorize datasets to calculate the ls_factor
    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [
            aspect_path, slope_path, flow_accumulation_path]],
        ls_factor_function, out_ls_factor_path, gdal.GDT_Float32,
        _TARGET_NODATA)


def _calculate_rkls(
        ls_factor_path, erosivity_path, erodibility_path, stream_path,
        rkls_path):
    """Calculate per-pixel potential soil loss using the RKLS.

    (revised universal soil loss equation with no C or P).

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
    erosivity_nodata = pygeoprocessing.get_raster_info(
        erosivity_path)['nodata'][0]
    erodibility_nodata = pygeoprocessing.get_raster_info(
        erodibility_path)['nodata'][0]
    stream_nodata = pygeoprocessing.get_raster_info(
        stream_path)['nodata'][0]

    cell_size = pygeoprocessing.get_raster_info(
        ls_factor_path)['mean_pixel_size']
    cell_area_ha = cell_size ** 2 / 10000.0

    def rkls_function(ls_factor, erosivity, erodibility, stream):
        """Calculate the RKLS equation.

        Parameters:
            ls_factor (numpy.ndarray): length/slope factor
        erosivity (numpy.ndarray): related to peak rainfall events
        erodibility (numpy.ndarray): related to the potential for soil to
            erode
        stream (numpy.ndarray): stream mask (1 stream, 0 no stream)

        Returns:
            ls_factor * erosivity * erodibility * usle_c_p or nodata if
            any values are nodata themselves.
        """
        rkls = numpy.empty(ls_factor.shape, dtype=numpy.float32)
        nodata_mask = (
            (ls_factor != _TARGET_NODATA) &
            (erosivity != erosivity_nodata) &
            (erodibility != erodibility_nodata) & (stream != stream_nodata))
        valid_mask = nodata_mask & (stream == 0)
        rkls[:] = _TARGET_NODATA

        rkls[valid_mask] = (
            ls_factor[valid_mask] * erosivity[valid_mask] *
            erodibility[valid_mask] * cell_area_ha)

        # rkls is 1 on the stream
        rkls[nodata_mask & (stream == 1)] = 1
        return rkls

    # aligning with index 3 that's the stream and the most likely to be
    # aligned with LULCs
    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [
            ls_factor_path, erosivity_path, erodibility_path, stream_path]],
        rkls_function, rkls_path, gdal.GDT_Float32, _TARGET_NODATA)


def _threshold_slope(slope_path, out_thresholded_slope_path):
    """Threshold the slope between 0.005 and 1.0.

    Parameters:
        slope_path (string): path to a raster of slope in percent
        out_thresholded_slope_path (string): path to output raster of
            thresholded slope between 0.005 and 1.0

    Returns:
        None
    """
    slope_nodata = pygeoprocessing.get_raster_info(slope_path)['nodata'][0]

    def threshold_slope(slope):
        """Convert slope to m/m and clamp at 0.005 and 1.0.

        As desribed in Cavalli et al., 2013.
        """
        valid_slope = slope != slope_nodata
        slope_m = slope[valid_slope] / 100.0
        slope_m[slope_m < 0.005] = 0.005
        slope_m[slope_m > 1.0] = 1.0
        result = numpy.empty(valid_slope.shape)
        result[:] = slope_nodata
        result[valid_slope] = slope_m
        return result

    pygeoprocessing.raster_calculator(
        [(slope_path, 1)], threshold_slope, out_thresholded_slope_path,
        gdal.GDT_Float32, slope_nodata)


def _add_drainage(stream_path, drainage_path, out_stream_and_drainage_path):
    """Combine stream and drainage masks into one raster mask.

    Parameters:
        stream_path (string): path to stream raster mask where 1 indicates
            a stream, and 0 is a valid landscape pixel but not a stream.
        drainage_raster_path (string): path to 1/0 mask of drainage areas.
            1 indicates any water reaching that pixel drains to a stream.
        out_stream_and_drainage_path (string): output raster of a logical
            OR of stream and drainage inputs

    Returns:
        None
    """
    def add_drainage_op(stream, drainage):
        """Add drainage mask to stream layer."""
        return numpy.where(drainage == 1, 1, stream)

    stream_nodata = pygeoprocessing.get_raster_info(stream_path)['nodata'][0]
    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [stream_path, drainage_path]], add_drainage_op,
        out_stream_and_drainage_path, gdal.GDT_Byte, stream_nodata)


def _calculate_w(
        biophysical_table, lulc_path, w_factor_path,
        out_thresholded_w_factor_path):
    """W factor: map C values from LULC and lower threshold to 0.001.

    W is a factor in calculating d_up accumulation for SDR.

    Parameters:
        biophysical_table (dict): map of LULC codes to dictionaries that
            contain at least a 'usle_c' field
        lulc_path (string): path to LULC raster
        w_factor_path (string): path to outputed raw W factor
        out_thresholded_w_factor_path (string): W factor from `w_factor_path`
            thresholded to be no less than 0.001.

    Returns:
        None
    """
    lulc_to_c = dict(
        [(lulc_code, float(table['usle_c'])) for
         (lulc_code, table) in biophysical_table.items()])

    pygeoprocessing.reclassify_raster(
        (lulc_path, 1), lulc_to_c, w_factor_path, gdal.GDT_Float32,
        _TARGET_NODATA, values_required=True)

    def threshold_w(w_val):
        """Threshold w to 0.001."""
        w_val_copy = w_val.copy()
        nodata_mask = w_val == _TARGET_NODATA
        w_val_copy[w_val < 0.001] = 0.001
        w_val_copy[nodata_mask] = _TARGET_NODATA
        return w_val_copy

    pygeoprocessing.raster_calculator(
        [(w_factor_path, 1)], threshold_w, out_thresholded_w_factor_path,
        gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_cp(biophysical_table, lulc_path, cp_factor_path):
    """Map LULC to C*P value.

    Parameters:
        biophysical_table (dict): map of lulc codes to dictionaries that
            contain at least the entry 'usle_c" and 'usle_p' corresponding to
            those USLE components.
        lulc_path (string): path to LULC raster
        cp_factor_path (string): path to output raster of LULC mapped to C*P
            values

    Returns:
        None
    """
    lulc_to_cp = dict(
        [(lulc_code, float(table['usle_c']) * float(table['usle_p'])) for
         (lulc_code, table) in biophysical_table.items()])
    pygeoprocessing.reclassify_raster(
        (lulc_path, 1), lulc_to_cp, cp_factor_path, gdal.GDT_Float32,
        _TARGET_NODATA, values_required=True)


def _calculate_usle(
        rkls_path, cp_factor_path, drainage_raster_path, out_usle_path):
    """Calculate USLE, multiply RKLS by CP and set to 1 on drains."""

    def usle_op(rkls, cp_factor, drainage):
        """Calculate USLE."""
        result = numpy.empty(rkls.shape)
        result[:] = _TARGET_NODATA
        valid_mask = (rkls != _TARGET_NODATA) & (cp_factor != _TARGET_NODATA)
        result[valid_mask] = rkls[valid_mask] * cp_factor[valid_mask] * (
            1 - drainage[valid_mask])
        return result

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [
            rkls_path, cp_factor_path, drainage_raster_path]], usle_op,
        out_usle_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_bar_factor(
        dem_path, factor_path, flow_accumulation_path, flow_direction_path,
        zero_absorption_source_path, loss_path, accumulation_path,
        out_bar_path):
    """Route user defined source across DEM.

    Used for calcualting S and W bar in the SDR operation.

    Parameters:
        dem_path (string): path to DEM raster
        factor_path (string): path to arbitrary factor raster
        flow_accumulation_path (string): path to flow accumulation raster
        flow_direction_path (string): path to flow direction path (in radians)
        zero_absorption_source_path (string): path to a raster that is all
            0s and same size as `dem_path`.  Temporary file.
        loss_path (string): path to a raster that can save the loss raster
            from routing.  Temporary file.
        accumulation_path (string): path to a raster that can be used to
            save the accumulation of the factor.  Temporary file.
        out_bar_path (string): path to output raster that is the result of
            the factor accumulation raster divided by the flow accumulation
            raster.

    Returns:
        None.
    """
    pygeoprocessing.new_raster_from_base(
        dem_path, zero_absorption_source_path, gdal.GDT_Float32,
        [_TARGET_NODATA], fill_value_list=[0.0])

    flow_accumulation_nodata = pygeoprocessing.get_raster_info(
        flow_accumulation_path)['nodata'][0]

    natcap.invest.pygeoprocessing_0_3_3.routing.route_flux(
        flow_direction_path, dem_path, factor_path,
        zero_absorption_source_path, loss_path, accumulation_path,
        'flux_only')

    def bar_op(base_accumulation, flow_accumulation):
        """Aggregate accumulation from base divided by the flow accum."""
        result = numpy.empty(base_accumulation.shape)
        valid_mask = (
            (base_accumulation != _TARGET_NODATA) &
            (flow_accumulation != flow_accumulation_nodata))
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            base_accumulation[valid_mask] / flow_accumulation[valid_mask])
        return result
    pygeoprocessing.raster_calculator(
        [(accumulation_path, 1), (flow_accumulation_path, 1)], bar_op,
        out_bar_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_d_up(
        w_bar_path, s_bar_path, flow_accumulation_path, out_d_up_path):
    """Calculate w_bar * s_bar * sqrt(flow accumulation * cell area)."""
    cell_area = pygeoprocessing.get_raster_info(
        w_bar_path)['mean_pixel_size'] ** 2
    flow_accumulation_nodata = pygeoprocessing.get_raster_info(
        flow_accumulation_path)['nodata'][0]

    def d_up_op(w_bar, s_bar, flow_accumulation):
        """Calculate the d_up index.

        w_bar * s_bar * sqrt(upstream area)

        """
        valid_mask = (
            (w_bar != _TARGET_NODATA) & (s_bar != _TARGET_NODATA) &
            (flow_accumulation != flow_accumulation_nodata))
        d_up_array = numpy.empty(valid_mask.shape)
        d_up_array[:] = _TARGET_NODATA
        d_up_array[valid_mask] = (
            w_bar[valid_mask] * s_bar[valid_mask] * numpy.sqrt(
                flow_accumulation[valid_mask] * cell_area))
        return d_up_array

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [
            w_bar_path, s_bar_path, flow_accumulation_path]], d_up_op,
        out_d_up_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_d_up_bare(
        s_bar_path, flow_accumulation_path, out_d_up_bare_path):
    """Calculate s_bar * sqrt(flow accumulation * cell area)."""
    cell_area = pygeoprocessing.get_raster_info(
        s_bar_path)['mean_pixel_size'] ** 2
    flow_accumulation_nodata = pygeoprocessing.get_raster_info(
        flow_accumulation_path)['nodata'][0]

    def d_up_op(s_bar, flow_accumulation):
        """Calculate the bare d_up index.

        s_bar * sqrt(upstream area)

        """
        valid_mask = (
            (flow_accumulation != flow_accumulation_nodata) &
            (s_bar != _TARGET_NODATA))
        d_up_array = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        d_up_array[:] = _TARGET_NODATA
        d_up_array[valid_mask] = (
            numpy.sqrt(flow_accumulation[valid_mask] * cell_area) *
            s_bar[valid_mask])
        return d_up_array

    pygeoprocessing.raster_calculator(
        [(s_bar_path, 1), (flow_accumulation_path, 1)], d_up_op,
        out_d_up_bare_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_inverse_ws_factor(
        thresholded_slope_path, thresholded_w_factor_path,
        out_ws_factor_inverse_path):
    """Calculate 1/(w*s)."""
    slope_nodata = pygeoprocessing.get_raster_info(
        thresholded_slope_path)['nodata'][0]

    def ws_op(w_factor, s_factor):
        """Calculate the inverse ws factor."""
        valid_mask = (w_factor != _TARGET_NODATA) & (s_factor != slope_nodata)
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            1.0 / (w_factor[valid_mask] * s_factor[valid_mask]))
        return result

    pygeoprocessing.raster_calculator(
        [(thresholded_w_factor_path, 1), (thresholded_slope_path, 1)], ws_op,
        out_ws_factor_inverse_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_inverse_s_factor(
        thresholded_slope_path, out_s_factor_inverse_path):
    """Calculate 1/s."""

    slope_nodata = pygeoprocessing.get_raster_info(
        thresholded_slope_path)['nodata'][0]

    def s_op(s_factor):
        """Calculate the inverse s factor."""
        valid_mask = (s_factor != slope_nodata)
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = 1.0 / s_factor[valid_mask]
        return result

    pygeoprocessing.raster_calculator(
        [(thresholded_slope_path, 1)], s_op,
        out_s_factor_inverse_path, gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_ic(d_up_path, d_dn_path, out_ic_factor_path):
    """Calculate log10(d_up/d_dn)."""
    # ic can be positive or negative, so float.min is a reasonable nodata value
    d_dn_nodata = pygeoprocessing.get_raster_info(d_dn_path)['nodata'][0]

    def ic_op(d_up, d_dn):
        """Calculate IC factor."""
        valid_mask = (
            (d_up != _TARGET_NODATA) & (d_dn != d_dn_nodata) & (d_dn != 0) &
            (d_up != 0))
        ic_array = numpy.empty(valid_mask.shape)
        ic_array[:] = _IC_NODATA
        ic_array[valid_mask] = numpy.log10(
            d_up[valid_mask] / d_dn[valid_mask])
        return ic_array

    pygeoprocessing.raster_calculator(
        [(d_up_path, 1), (d_dn_path, 1)], ic_op, out_ic_factor_path,
        gdal.GDT_Float32, _IC_NODATA)


def _calculate_sdr(
        k_factor, ic_0, sdr_max, ic_path, stream_path, out_sdr_path):
    """Derive SDR from k, ic0, ic; 0 on the stream and clamped to sdr_max."""

    def sdr_op(ic_factor, stream):
        """Calculate SDR factor."""
        valid_mask = (
            (ic_factor != _IC_NODATA) & (stream != 1))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            sdr_max / (1+numpy.exp((ic_0-ic_factor[valid_mask])/k_factor)))
        result[stream == 1] = 0.0
        return result

    pygeoprocessing.raster_calculator(
        [(ic_path, 1), (stream_path, 1)], sdr_op, out_sdr_path,
        gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_sed_export(usle_path, sdr_path, out_sed_export_path):
    """Calculate USLE * SDR."""

    def sed_export_op(usle, sdr):
        """Sediment export."""
        valid_mask = (usle != _TARGET_NODATA) & (sdr != _TARGET_NODATA)
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = usle[valid_mask] * sdr[valid_mask]
        return result

    pygeoprocessing.raster_calculator(
        [(usle_path, 1), (sdr_path, 1)], sed_export_op, out_sed_export_path,
        gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_sed_retention_index(
        rkls_path, usle_path, sdr_path, sdr_max,
        out_sed_retention_index_path):
    """Calculate (rkls-usle) * sdr  / sdr_max."""

    def sediment_index_op(rkls, usle, sdr_factor):
        """Calculate sediment retention index."""
        valid_mask = (
            (rkls != _TARGET_NODATA) & (usle != _TARGET_NODATA) &
            (sdr_factor != _TARGET_NODATA))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            (rkls[valid_mask] - usle[valid_mask]) *
            sdr_factor[valid_mask] / sdr_max)
        return result

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [rkls_path, usle_path, sdr_path]],
        sediment_index_op, out_sed_retention_index_path, gdal.GDT_Float32,
        _TARGET_NODATA)


def _calculate_sed_retention(
        rkls_path, usle_path, stream_path, sdr_path, sdr_bare_soil_path,
        out_sed_ret_bare_soil_path):
    """Difference in exported sediments on basic and bare watershed.

    Calculates the difference of sediment export on the real landscape and
    a bare soil landscape given that SDR has been calculated for bare soil.
    Essentially:

        RKLS * SDR_bare - USLE * SDR

    Parameters:
        rkls_path (string): path to RKLS raster
        usle_path (string): path to USLE raster
        stream_path (string): path to stream/drainage mask
        sdr_path (string): path to SDR raster
        sdr_bare_soil_path (string): path to SDR raster calculated for a bare
            watershed
        out_sed_ret_bare_soil_path (string): path to output raster indicating
            where sediment is retained

    Returns:
        None
    """
    stream_nodata = pygeoprocessing.get_raster_info(stream_path)['nodata'][0]

    def sediment_retention_bare_soil_op(
            rkls, usle, stream_factor, sdr_factor, sdr_factor_bare_soil):
        """Subtract bare soil export from real landcover."""
        valid_mask = (
            (rkls != _TARGET_NODATA) & (usle != _TARGET_NODATA) &
            (stream_factor != stream_nodata) &
            (sdr_factor != _TARGET_NODATA) &
            (sdr_factor_bare_soil != _TARGET_NODATA))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            rkls[valid_mask] * sdr_factor_bare_soil[valid_mask] -
            usle[valid_mask] * sdr_factor[valid_mask]) * (
                1 - stream_factor[valid_mask])
        return result

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [
            rkls_path, usle_path, stream_path, sdr_path, sdr_bare_soil_path]],
        sediment_retention_bare_soil_op, out_sed_ret_bare_soil_path,
        gdal.GDT_Float32, _TARGET_NODATA)


def _generate_report(
        watersheds_path, usle_path, sed_export_path, sed_retention_path,
        watershed_results_sdr_path):
    """Create shapefile with USLE, sed export, and sed retention fields."""
    field_summaries = {
        'usle_tot': pygeoprocessing.zonal_statistics(
            (usle_path, 1), watersheds_path, 'ws_id'),
        'sed_export': pygeoprocessing.zonal_statistics(
            (sed_export_path, 1), watersheds_path, 'ws_id'),
        'sed_retent': pygeoprocessing.zonal_statistics(
            (sed_retention_path, 1), watersheds_path, 'ws_id'),
        }

    original_datasource = gdal.OpenEx(watersheds_path, gdal.OF_VECTOR)
    # Delete if existing shapefile with the same name and path
    if os.path.isfile(watershed_results_sdr_path):
        os.remove(watershed_results_sdr_path)
    driver = gdal.GetDriverByName('ESRI Shapefile')
    datasource_copy = driver.CreateCopy(
        watershed_results_sdr_path, original_datasource)
    layer = datasource_copy.GetLayer()

    for field_name in field_summaries:
        field_def = ogr.FieldDefn(field_name, ogr.OFTReal)
        field_def.SetWidth(24)
        field_def.SetPrecision(11)
        layer.CreateField(field_def)

    # initialize each feature field to 0.0
    for feature_id in xrange(layer.GetFeatureCount()):
        feature = layer.GetFeature(feature_id)
        for field_name in field_summaries:
            ws_id = feature.GetFieldAsInteger('ws_id')
            feature.SetField(
                field_name, float(field_summaries[field_name][ws_id]['sum']))
        layer.SetFeature(feature)


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to `execute`'s contract.

    Parameters:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in `execute` docstring.
        limit_to (str): (optional) if not None indicates that validation
            should only occur on the args[limit_to] value. The intent that
            individual key validation could be significantly less expensive
            than validating the entire `args` dictionary.

    Returns:
        list of ([invalid key_a, invalid_keyb, ...], 'warning/error message')
            tuples. Where an entry indicates that the invalid keys caused
            the error message in the second part of the tuple. This should
            be an empty list if validation succeeds.
    """
    missing_key_list = []
    no_value_list = []
    validation_error_list = []

    required_keys = [
        'workspace_dir',
        'dem_path',
        'erosivity_path',
        'erodibility_path',
        'lulc_path',
        'watersheds_path',
        'biophysical_table_path',
        'threshold_flow_accumulation',
        'k_param',
        'ic_0_param',
        'sdr_max']

    for key in required_keys:
        if limit_to is None or limit_to == key:
            if key not in args:
                missing_key_list.append(key)
            elif args[key] in ['', None]:
                no_value_list.append(key)

    if missing_key_list:
        # if there are missing keys, we have raise KeyError to stop hard
        raise KeyError(
            "The following keys were expected in `args` but were missing " +
            ', '.join(missing_key_list))

    if no_value_list:
        validation_error_list.append(
            (no_value_list, 'parameter has no value'))

    file_type_list = [
        ('dem_path', 'raster'),
        ('erosivity_path', 'raster'),
        ('erodibility_path', 'raster'),
        ('lulc_path', 'raster'),
        ('watersheds_path', 'vector'),
        ('biophysical_table_path', 'table')]

    if limit_to in ['drainage_path', None] and (
            'drainage_path' in args and
            args['drainage_path'] not in ['', None]):
        file_type_list.append(('drainage_path', 'raster'))

    # check that existing/optional files are the correct types
    with utils.capture_gdal_logging():
        for key, key_type in file_type_list:
            if (limit_to is None or limit_to == key) and key in args:
                if not os.path.exists(args[key]):
                    validation_error_list.append(
                        ([key], 'not found on disk'))
                    continue
                if key_type == 'raster':
                    raster = gdal.OpenEx(args[key])
                    if raster is None:
                        validation_error_list.append(
                            ([key], 'not a raster'))
                    del raster

        if limit_to in ['watersheds_path', None]:
            # checks that watersheds are a vector, that they have 'ws_id' and
            # that all their fields are defined
            if os.path.exists(args['watersheds_path']):
                try:
                    watersheds_vector = gdal.OpenEx(
                        args['watersheds_path'], gdal.OF_VECTOR)
                    if watersheds_vector is None:
                        validation_error_list.append(
                            (['watersheds_path'], 'not a vector'))
                    else:
                        watersheds_layer = watersheds_vector.GetLayer()
                        watersheds_defn = watersheds_layer.GetLayerDefn()
                        if watersheds_defn.GetFieldIndex('ws_id') == -1:
                            validation_error_list.append((
                                ['watersheds_path'],
                                'does not have a `ws_id` field defined.'))
                        else:
                            for feature in watersheds_layer:
                                try:
                                    value = feature.GetFieldAsString('ws_id')
                                    _ = int(value)  # value should be an integer
                                except ValueError:
                                    validation_error_list.append((
                                        ['watersheds_path'],
                                        'feature %s has an invalid value of '
                                        '"%s" in \'ws_id\' column, it should '
                                        'be an integer value' % (
                                            str(feature.GetFID()), value)))
                finally:
                    feature = None
                    watersheds_defn = None
                    watersheds_layer = None
                    watersheds_vector = None

    return validation_error_list
