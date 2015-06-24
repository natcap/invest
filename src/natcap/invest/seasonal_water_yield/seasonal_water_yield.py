"""InVEST Seasonal Water Yield Model"""

import os
import logging
import re
import fractions

import numpy
import gdal
import pygeoprocessing
import pygeoprocessing.routing



logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger(
    'natcap.invest.seasonal_water_yield.seasonal_water_yield')

N_MONTHS = 12

def execute(args):
    """This function invokes the seasonal water yield model given
        URI inputs of files. It may write log, warning, or error messages to
        stdout.
    """

    alpha_m = float(fractions.Fraction(args['alpha_m']))
    beta_i = float(fractions.Fraction(args['beta_i']))
    gamma = float(fractions.Fraction(args['gamma']))

    try:
        file_suffix = args['results_suffix']
        if file_suffix != "" and not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''

    precip_uri_list = []
    et0_uri_list = []

    et0_dir_list = [
        os.path.join(args['et0_dir'], f) for f in os.listdir(args['et0_dir'])]
    precip_dir_list = [
        os.path.join(args['precip_dir'], f) for f in os.listdir(
            args['precip_dir'])]

    for month_index in range(1, N_MONTHS + 1):
        month_file_match = re.compile(r'.*[^\d]%d\.[^.]+$' % month_index)

        for data_type, dir_list, uri_list in [
                ('et0', et0_dir_list, et0_uri_list),
                ('Precip', precip_dir_list, precip_uri_list)]:

            file_list = [x for x in dir_list if month_file_match.match(x)]
            if len(file_list) == 0:
                raise ValueError(
                    "No %s found for month %d" % (data_type, month_index))
            if len(file_list) > 1:
                raise ValueError(
                    "Ambiguous set of files found for month %d: %s" %
                    (month_index, file_list))
            uri_list.append(file_list[0])

    pygeoprocessing.geoprocessing.create_directories([args['workspace_dir']])

    qfi_uri = os.path.join(args['workspace_dir'], 'qf%s.tif' % file_suffix)
    cn_uri = os.path.join(args['workspace_dir'], 'cn%s.tif' % file_suffix)

    #pre align all the datasets
    precip_uri_aligned_list = [
        pygeoprocessing.geoprocessing.temporary_filename() for _ in
        range(len(precip_uri_list))]
    et0_uri_aligned_list = [
        pygeoprocessing.geoprocessing.temporary_filename() for _ in
        range(len(precip_uri_list))]

    lulc_uri_aligned = pygeoprocessing.geoprocessing.temporary_filename()
    dem_uri_aligned = pygeoprocessing.geoprocessing.temporary_filename()

    pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
        args['lulc_uri'])

    LOGGER.info('Aligning and clipping dataset list')
    pygeoprocessing.geoprocessing.align_dataset_list(
        precip_uri_list + et0_uri_list + [args['lulc_uri'], args['dem_uri']],
        precip_uri_aligned_list + et0_uri_aligned_list +
        [lulc_uri_aligned, dem_uri_aligned],
        ['nearest'] * (len(precip_uri_list) + len(et0_uri_aligned_list) + 2),
        pixel_size, 'intersection', 0, aoi_uri=args['aoi_uri'],
        assert_datasets_projected=True)

    qf_monthly_uri_list = []
    for m_index in range(1, N_MONTHS + 1):
        qf_monthly_uri_list.append(
            os.path.join(
                args['workspace_dir'], 'qf_%d%s.tif' % (m_index, file_suffix)))

    flow_dir_uri = os.path.join(
        args['workspace_dir'], 'flow_dir%s.tif' % file_suffix)
    pygeoprocessing.routing.flow_direction_d_inf(dem_uri_aligned, flow_dir_uri)

    flow_accum_uri = os.path.join(
        args['workspace_dir'], 'flow_accum%s.tif' % file_suffix)
    pygeoprocessing.routing.flow_accumulation(
        flow_dir_uri, dem_uri_aligned, flow_accum_uri)
    stream_uri = os.path.join(
        args['workspace_dir'], 'stream%s.tif' % file_suffix)
    threshold_flow_accumulation = 1000
    pygeoprocessing.routing.stream_threshold(
        flow_accum_uri, threshold_flow_accumulation, stream_uri)

    calculate_quick_flow(
        precip_uri_aligned_list, args['rain_events_table_uri'],
        lulc_uri_aligned, args['cn_table_uri'], cn_uri, stream_uri, qfi_uri,
        qf_monthly_uri_list, args['workspace_dir'])

    biophysical_table = pygeoprocessing.geoprocessing.get_lookup_from_table(
        args['biophysical_table_uri'], 'lucode')

    kc_lookup = dict([
        (lucode, biophysical_table[lucode]['kc']) for lucode in
        biophysical_table])

    recharge_uri = os.path.join(
        args['workspace_dir'], 'recharge%s.tif' % file_suffix)
    recharge_avail_uri = os.path.join(
        args['workspace_dir'], 'recharge_avail%s.tif' % file_suffix)
    r_sum_avail_uri = os.path.join(
        args['workspace_dir'], 'r_sum_avail%s.tif' % file_suffix)
    vri_uri = os.path.join(args['workspace_dir'], 'vri%s.tif' % file_suffix)
    aet_uri = os.path.join(args['workspace_dir'], 'aet%s.tif' % file_suffix)

    calculate_slow_flow(
        args['aoi_uri'], precip_uri_aligned_list, et0_uri_aligned_list,
        flow_dir_uri, dem_uri_aligned, lulc_uri_aligned,
        kc_lookup, alpha_m, beta_i, gamma, qfi_uri, recharge_uri,
        recharge_avail_uri, r_sum_avail_uri, aet_uri, vri_uri, stream_uri)


def calculate_quick_flow(
        precip_uri_list, rain_events_table_uri, lulc_uri,
        cn_table_uri, cn_uri, stream_uri, qfi_uri, qf_monthly_uri_list,
        intermediate_dir):
    """Calculates quick flow """

    LOGGER.info('loading number of monthly events')
    rain_events_lookup = pygeoprocessing.geoprocessing.get_lookup_from_table(
        rain_events_table_uri, 'month')
    n_events = dict([
        (month, rain_events_lookup[month]['events'])
        for month in rain_events_lookup])

    LOGGER.info('calculating curve number')
    cn_lookup = pygeoprocessing.geoprocessing.get_lookup_from_table(
        cn_table_uri, 'lucode')
    cn_lookup = dict([
        (lucode, cn_lookup[lucode]['cn']) for lucode in cn_lookup])

    cn_nodata = -1
    pygeoprocessing.geoprocessing.reclassify_dataset_uri(
        lulc_uri, cn_lookup, cn_uri, gdal.GDT_Float32, cn_nodata,
        exception_flag='values_required')

    si_nodata = -1
    def si_op(ci_array, stream_array):
        """potential maximum retention"""
        si_array = 1000.0 / ci_array - 10
        si_array = numpy.where(ci_array != cn_nodata, si_array, si_nodata)
        si_array[stream_array == 1] = 0
        return si_array

    LOGGER.info('calculating Si')
    si_uri = os.path.join(intermediate_dir, 'si.tif')
    pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(lulc_uri)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [cn_uri, stream_uri], si_op, si_uri, gdal.GDT_Float32,
        si_nodata, pixel_size, 'intersection', vectorize_op=False,
        datasets_are_pre_aligned=True)

    qf_nodata = -1
    p_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
        precip_uri_list[0])

    for m_index in range(1, N_MONTHS + 1):

        def qf_op(pm_array, s_array, stream_array):
            """calculate quickflow"""
            inner_value = pm_array/float(n_events[m_index])/25.4 - 0.2 * s_array
            numerator = numpy.where(
                inner_value > 0,
                25.4 * n_events[m_index] * inner_value**2, 0.0)

            denominator = pm_array/float(n_events[m_index])/25.4 +0.8 * s_array
            quickflow = numpy.where(
                (pm_array == p_nodata) | (s_array == si_nodata) |
                (denominator == 0), qf_nodata, numerator / denominator)
            quickflow[stream_array == 1] = pm_array[stream_array == 1]
            return quickflow

        LOGGER.info('calculating QFi_%d of %d', m_index, N_MONTHS)
        pygeoprocessing.geoprocessing.vectorize_datasets(
            [precip_uri_list[m_index-1], si_uri, stream_uri], qf_op,
            qf_monthly_uri_list[m_index-1], gdal.GDT_Float32, qf_nodata,
            pixel_size, 'intersection', vectorize_op=False,
            datasets_are_pre_aligned=True)

    LOGGER.info('calculating QFi')
    def qfi_sum(*qf_values):
        """sum the monthly qfis"""
        qf_sum = qf_values[0].copy()
        for index in range(1, len(qf_values)):
            qf_sum += qf_values[index]
        qf_sum[qf_values[0] == qf_nodata] = qf_nodata
        return qf_sum
    pygeoprocessing.geoprocessing.vectorize_datasets(
        qf_monthly_uri_list, qfi_sum, qfi_uri,
        gdal.GDT_Float32, qf_nodata, pixel_size, 'intersection',
        vectorize_op=False, datasets_are_pre_aligned=True)


def calculate_slow_flow(
        aoi_uri, precip_uri_list, et0_uri_list, flow_dir_uri, dem_uri, lulc_uri,
        kc_lookup, alpha_m, beta_i, gamma, qfi_uri,
        recharge_uri, recharge_avail_uri, r_sum_avail_uri, aet_uri, vri_uri,
        stream_uri):
    """calculate slow flow index"""

    import seasonal_water_yield_core
    seasonal_water_yield_core.calculate_recharge(
        precip_uri_list, et0_uri_list, flow_dir_uri, dem_uri, lulc_uri,
        kc_lookup, alpha_m, beta_i, gamma, qfi_uri, stream_uri, recharge_uri,
        recharge_avail_uri, r_sum_avail_uri, aet_uri, vri_uri)

    #calcualte Qb as the sum of recharge_avail over the aoi
    qb_results = pygeoprocessing.geoprocessing.aggregate_raster_values_uri(
        recharge_avail_uri, aoi_uri)

    qb = qb_results.total[9999] / qb_results.n_pixels[9999]
    #9999 is the value used to index fields if no shapefile ID is provided
    LOGGER.info("Qb = %f", qb)

    pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
        recharge_uri)
    ri_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(recharge_uri)

    def vri_op(ri):
        """calc vri index"""
        return numpy.where(ri != ri_nodata, ri / qb, ri_nodata)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [recharge_uri], vri_op, vri_uri,
        gdal.GDT_Float32, ri_nodata, pixel_size, 'intersection',
        vectorize_op=False, datasets_are_pre_aligned=True)

    out_dir = os.path.dirname(r_sum_avail_uri)
    r_sum_avail_pour_uri = os.path.join(out_dir, 'r_sum_avail_pour.tif')

    seasonal_water_yield_core.calculate_r_sum_avail_pour(
        r_sum_avail_uri, flow_dir_uri, r_sum_avail_pour_uri)

    sf_uri = os.path.join(out_dir, 'sf.tif')
    sf_down_uri = os.path.join(out_dir, 'sf_down.tif')

    outflow_weights_uri = os.path.join(out_dir, 'outflow_weights.tif')
    outflow_direction_uri = os.path.join(out_dir, 'outflow_direction.tif')

    LOGGER.info('calculating flow weights')
    seasonal_water_yield_core.calculate_flow_weights(
        flow_dir_uri, outflow_weights_uri, outflow_direction_uri)

    LOGGER.info('calculating slow flow')
    seasonal_water_yield_core.route_sf(
        dem_uri, recharge_avail_uri, r_sum_avail_uri, r_sum_avail_pour_uri,
        outflow_direction_uri, outflow_weights_uri, stream_uri, sf_uri,
        sf_down_uri)
