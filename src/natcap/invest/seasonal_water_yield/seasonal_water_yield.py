"""InVEST Seasonal Water Yield Model"""
import warnings
warnings.filterwarnings('error')

import os
import logging
import re
import fractions

import scipy.special
import numpy
import gdal
import pygeoprocessing
import pygeoprocessing.routing

import seasonal_water_yield_core

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger(
    'natcap.invest.seasonal_water_yield.seasonal_water_yield')

N_MONTHS = 12


def execute(args):
    """This function invokes the InVEST seasonal water yield model described in
    "Spatial attribution of baseflow generation at the parcel level for
    ecosystem-service valuation", Guswa, et. al (under review in Water
    "Resources Research")

    Parameters:
        output_dir (string): output directory for intermediate,
        temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output files
        args['threshold_flow_accumulation'] (number): used when classifying
            stream pixels from the DEM by thresholding the number of upstream
            cells that must flow int a cell before it's considered
            part of a stream.
        args['et0_dir'] (string): required if
            args['user_defined_local_recharge'] is False.  Path to a directory
            that contains rasters of monthly reference evapotranspiration;
            units in mm.
        args['precip_dir'] (string): required if
            args['user_defined_local_recharge'] is False. A path to a directory
            that contains rasters of monthly precipitation; units in mm.
        args['dem_path'] (string): a path to a digital elevation raster
        args['lulc_path'] (string): a path to a land cover raster used to
            classify biophysical properties of pixels.
        args['soil_group_path'] (string): required if
            args['user_defined_local_recharge'] is  False. A path to a raster
            indicating SCS soil groups where integer values are mapped to soil
            types:
                1: A
                2: B
                3: C
                4: D
        args['aoi_path'] (string): path to a vector that indicates the area
            over which the model should be run, as well as the area in which to
            aggregate over when calculating the output Qb.
        args['biophysical_table_path'] (string): path to a CSV table that maps
            landcover codes paired with soil group types to curve numbers as
            well as Kc values.  Headers must be 'lucode', 'CN_A', 'CN_B',
            'CN_C', 'CN_D', and 'Kc'.
        args['rain_events_table_path'] (string): Required if
            args['user_defined_local_recharge'] is  False. Path to a CSV table
            that has headers 'month' (1-12) and 'events' (int >= 0) that
            indicates the number of rain events per month
        args['alpha_m'] (float or string): proportion of upslope annual
            available local recharge that is available in month m.
        args['beta_i'] (float or string): is the fraction of the upgradient
            subsidy that is available for downgradient evapotranspiration.
        args['gamma'] (float or string): is the fraction of pixel local
            recharge that is available to downgradient pixels.
        args['user_defined_local_recharge'] (boolean): if True, indicates user
            will provide pre-defined local recharge raster layer
        args['l_path'] (string): required if
            args['user_defined_local_recharge'] is True.  If provided pixels
            indicate the amount of local recharge; units in mm.
    """

    # prepare and test inputs for common errors
    alpha_m = float(fractions.Fraction(args['alpha_m']))
    beta_i = float(fractions.Fraction(args['beta_i']))
    gamma = float(fractions.Fraction(args['gamma']))
    threshold_flow_accumulation = float(args['threshold_flow_accumulation'])
    biophysical_table = pygeoprocessing.geoprocessing.get_lookup_from_table(
        args['biophysical_table_path'], 'lucode')
    LOGGER.debug(biophysical_table)
    missing_headers = []
    for header_id in ['kc', 'cn_a', 'cn_b', 'cn_c', 'cn_d']:
        field = biophysical_table.itervalues().next()
        if header_id not in field:
            missing_headers.append(header_id)
    if len(missing_headers) > 0:
        raise ValueError(
            "biophysical table missing the following headers: %s" %
            str(missing_headers))

    try:
        file_suffix = args['results_suffix']
        if file_suffix != "" and not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''

    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    output_dir = args['workspace_dir']
    pygeoprocessing.geoprocessing.create_directories(
        [intermediate_output_dir, output_dir])

    #TODO: Change all the output file tags to be called the variables in the UG
    output_file_registry = {
        'aet_path': os.path.join(intermediate_output_dir, 'aet.tif'),
        'aetm_path_list': [None] * N_MONTHS,
        'cn_path': os.path.join(output_dir, 'CN.tif'),
        'flow_dir_path': os.path.join(intermediate_output_dir, 'flow_dir.tif'),
        'kc_path': os.path.join(intermediate_output_dir, 'kc.tif'),
        'l_avail_path': os.path.join(output_dir, 'L_avail.tif'),
        'l_path': None,  # might be predefined, use as placeholder
        'l_sum_path': os.path.join(output_dir, 'L_sum.tif'),
        'l_sum_avail_path': os.path.join(output_dir, 'L_sum_avail.tif'),
        'outflow_direction_path': os.path.join(intermediate_output_dir, 'outflow_direction.tif'),
        'outflow_weights_path': os.path.join(intermediate_output_dir, 'outflow_weights.tif'),
        'qb_out_path': os.path.join(output_dir, 'Qb.txt'),
        'qf_path': os.path.join(output_dir, 'QF.tif'),
        'qfm_path_list': [None] * N_MONTHS,
        'b_sum_path': os.path.join(output_dir, 'B_sum.tif'),
        'b_path': os.path.join(output_dir, 'B.tif'),
        'si_path': os.path.join(intermediate_output_dir, 'si.tif'),
        'stream_path': os.path.join(intermediate_output_dir, 'stream.tif'),
        'vri_path': os.path.join(output_dir, 'Vri.tif'),
        }

    # add a suffix to all the output files
    for file_id in output_file_registry:
        if isinstance(output_file_registry[file_id], basestring):
            output_file_registry[file_id] = file_suffix.join(
                os.path.splitext(output_file_registry[file_id]))

    # add the monthly quick flow rasters
    for m_index in range(N_MONTHS):
        output_file_registry['qfm_path_list'][m_index] = (
            os.path.join(intermediate_output_dir, 'qf_%d%s.tif' % (
                m_index+1, file_suffix)))
        output_file_registry['aetm_path_list'][m_index] = (
            os.path.join(intermediate_output_dir, 'aetm_%d%s.tif' % (
                m_index+1, file_suffix)))

    # this variable is only needed if there is not a predefined recharge file
    if not args['user_defined_local_recharge']:
        output_file_registry['l_path'] = os.path.join(
            output_dir, 'L%s.tif' % file_suffix)
        output_file_registry['l_avail_path'] = os.path.join(
            output_dir, 'L_avail%s.tif' % file_suffix)
        output_file_registry['annual_precip_path'] = os.path.join(
            output_dir, 'P_i%s.tif' % file_suffix)

    temporary_file_registry = {
        'lulc_aligned_path': pygeoprocessing.temporary_filename(),
        'dem_aligned_path': pygeoprocessing.temporary_filename(),
        'loss_path': pygeoprocessing.geoprocessing.temporary_filename(),
        'zero_absorption_source_path': (
            pygeoprocessing.geoprocessing.temporary_filename()),
        'soil_group_aligned_path': pygeoprocessing.temporary_filename(),
        'flow_accum_path': pygeoprocessing.temporary_filename(),
    }

    if args['user_defined_local_recharge']:
        temporary_file_registry['local_recharge_aligned_path'] = (
            pygeoprocessing.geoprocessing.temporary_filename())

    pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
        args['lulc_path'])

    #TODO: put align into helper function
    LOGGER.info('Aligning and clipping dataset list')
    input_align_list = [args['lulc_path'], args['dem_path']]
    output_align_list = [
        temporary_file_registry['lulc_aligned_path'],
        temporary_file_registry['dem_aligned_path'],
        ]

    if not args['user_defined_local_recharge']:
        precip_path_list = []
        et0_path_list = []

        et0_dir_list = [
            os.path.join(args['et0_dir'], f) for f in os.listdir(
                args['et0_dir'])]
        precip_dir_list = [
            os.path.join(args['precip_dir'], f) for f in os.listdir(
                args['precip_dir'])]

        for month_index in range(1, N_MONTHS + 1):
            month_file_match = re.compile(r'.*[^\d]%d\.[^.]+$' % month_index)

            for data_type, dir_list, path_list in [
                    ('et0', et0_dir_list, et0_path_list),
                    ('Precip', precip_dir_list, precip_path_list)]:

                file_list = [x for x in dir_list if month_file_match.match(x)]
                if len(file_list) == 0:
                    raise ValueError(
                        "No %s found for month %d" % (data_type, month_index))
                if len(file_list) > 1:
                    raise ValueError(
                        "Ambiguous set of files found for month %d: %s" %
                        (month_index, file_list))
                path_list.append(file_list[0])

        #pre align all the datasets
        precip_path_aligned_list = [
            pygeoprocessing.geoprocessing.temporary_filename() for _ in
            range(len(precip_path_list))]
        et0_path_aligned_list = [
            pygeoprocessing.geoprocessing.temporary_filename() for _ in
            range(len(precip_path_list))]
        input_align_list = (
            precip_path_list + [args['soil_group_path']] + et0_path_list +
            input_align_list)
        output_align_list = (
            precip_path_aligned_list +
            [temporary_file_registry['soil_group_aligned_path']] +
            et0_path_aligned_list + output_align_list)

    interpolate_list = ['nearest'] * len(input_align_list)
    align_index = 0
    if args['user_defined_local_recharge']:
        input_align_list.append(args['l_path'])
        output_align_list.append(
            temporary_file_registry['local_recharge_aligned_path'])
        interpolate_list.append('nearest')
        align_index = len(interpolate_list) - 1

    pygeoprocessing.geoprocessing.align_dataset_list(
        input_align_list, output_align_list, interpolate_list, pixel_size,
        'intersection', align_index, aoi_uri=args['aoi_path'],
        assert_datasets_projected=True)

    LOGGER.info('flow direction')
    pygeoprocessing.routing.flow_direction_d_inf(
        temporary_file_registry['dem_aligned_path'],
        output_file_registry['flow_dir_path'])

    LOGGER.info('flow accumulation')
    pygeoprocessing.routing.flow_accumulation(
        output_file_registry['flow_dir_path'],
        temporary_file_registry['dem_aligned_path'],
        temporary_file_registry['flow_accum_path'])

    LOGGER.info('stream thresholding')
    pygeoprocessing.routing.stream_threshold(
        temporary_file_registry['flow_accum_path'],
        threshold_flow_accumulation,
        output_file_registry['stream_path'])

    LOGGER.info('quick flow')
    if args['user_defined_local_recharge']:
        output_file_registry['l_path'] = (
            temporary_file_registry['local_recharge_aligned_path'])
    else:
        # user didn't predefine local recharge, calculate it
        LOGGER.info('loading number of monthly events')
        rain_events_lookup = (
            pygeoprocessing.geoprocessing.get_lookup_from_table(
                args['rain_events_table_path'], 'month'))
        n_events = dict([
            (month, rain_events_lookup[month]['events'])
            for month in rain_events_lookup])

        LOGGER.info('curve number')
        _calculate_curve_number_raster(
            temporary_file_registry['lulc_aligned_path'],
            temporary_file_registry['soil_group_aligned_path'],
            biophysical_table, pixel_size, output_file_registry['cn_path'])

        LOGGER.info('Si raster')
        _calculate_si_raster(
            output_file_registry['cn_path'],
            output_file_registry['si_path'],
            output_file_registry['stream_path'])

        for month_index in xrange(N_MONTHS):
            LOGGER.info('calculate quick flow for month %d', month_index+1)
            _calculate_monthly_quick_flow(
                precip_path_aligned_list[month_index],
                temporary_file_registry['lulc_aligned_path'],
                output_file_registry['cn_path'], n_events[month_index+1],
                output_file_registry['stream_path'],
                output_file_registry['qfm_path_list'][month_index],
                output_file_registry['si_path'])

        qf_nodata = -1
        LOGGER.info('calculating QFi')

        def qfi_sum_op(*qf_values):
            """sum the monthly qfis"""
            qf_sum = numpy.empty(qf_values[0].shape)
            valid_mask = qf_values[0] != qf_nodata
            for index in range(len(qf_values)):
                qf_sum[valid_mask] += qf_values[index][valid_mask]
            qf_sum[~valid_mask] = qf_nodata
            return qf_sum
        pygeoprocessing.geoprocessing.vectorize_datasets(
            output_file_registry['qfm_path_list'], qfi_sum_op,
            output_file_registry['qf_path'], gdal.GDT_Float32, qf_nodata,
            pixel_size, 'intersection', vectorize_op=False,
            datasets_are_pre_aligned=True)

        LOGGER.info('calculate local recharge')
        kc_lookup = dict([
            (lucode, biophysical_table[lucode]['kc']) for lucode in
            biophysical_table])

        LOGGER.info('flow weights')
        seasonal_water_yield_core.calculate_flow_weights(
            output_file_registry['flow_dir_path'],
            output_file_registry['outflow_weights_path'],
            output_file_registry['outflow_direction_path'])

        LOGGER.info('classifying kc')
        pygeoprocessing.geoprocessing.reclassify_dataset_uri(
            temporary_file_registry['lulc_aligned_path'], kc_lookup,
            output_file_registry['kc_path'], gdal.GDT_Float32, -1)

        # call through to a cython function that does the necessary routing
        # between AET and L.sum.avail in equation [7], [4], and [3]
        seasonal_water_yield_core.calculate_local_recharge(
            precip_path_aligned_list, et0_path_aligned_list,
            output_file_registry['qfm_path_list'],
            output_file_registry['flow_dir_path'],
            output_file_registry['outflow_weights_path'],
            output_file_registry['outflow_direction_path'],
            temporary_file_registry['dem_aligned_path'],
            temporary_file_registry['lulc_aligned_path'], kc_lookup, alpha_m,
            beta_i, gamma, output_file_registry['stream_path'],
            output_file_registry['l_path'],
            output_file_registry['l_avail_path'],
            output_file_registry['l_sum_avail_path'],
            output_file_registry['aet_path'], output_file_registry['kc_path'])

    #calculate Qb as the sum of local_recharge_avail over the AOI, Eq [9]
    qb_results = pygeoprocessing.geoprocessing.aggregate_raster_values_uri(
        output_file_registry['l_path'], args['aoi_path'])
    qb_result = qb_results.total[9999] / qb_results.n_pixels[9999]
    #9999 is the value used to index fields if no shapefile ID is provided
    qb_file = open(output_file_registry['qb_out_path'], 'w')
    qb_file.write("%f\n" % qb_result)
    qb_file.close()
    LOGGER.info("Qb = %f", qb_result)

    pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
        output_file_registry['l_path'])
    ri_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
        output_file_registry['l_path'])

    def vri_op(ri_array):
        """calc vri index Eq 10"""
        return numpy.where(
            ri_array != ri_nodata,
            ri_array / qb_result / qb_results.n_pixels[9999], ri_nodata)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [output_file_registry['l_path']], vri_op,
        output_file_registry['vri_path'], gdal.GDT_Float32, ri_nodata,
        pixel_size, 'intersection', vectorize_op=False,
        datasets_are_pre_aligned=True)

    LOGGER.info('calculate L_sum')  # Eq. [12]
    temporary_file_registry['zero_absorption_source_path'] = (
        pygeoprocessing.temporary_filename())
    temporary_file_registry['loss_path'] = pygeoprocessing.temporary_filename()
    pygeoprocessing.make_constant_raster_from_base_uri(
        temporary_file_registry['dem_aligned_path'], 0.0,
        temporary_file_registry['zero_absorption_source_path'])
    pygeoprocessing.routing.route_flux(
        output_file_registry['flow_dir_path'],
        temporary_file_registry['dem_aligned_path'],
        output_file_registry['l_path'],
        temporary_file_registry['zero_absorption_source_path'],
        temporary_file_registry['loss_path'],
        output_file_registry['l_sum_path'], 'flux_only',
        aoi_uri=args['aoi_path'])

    LOGGER.info('calculating base flow')
    seasonal_water_yield_core.route_baseflow(
        temporary_file_registry['dem_aligned_path'],
        output_file_registry['l_path'],
        output_file_registry['l_avail_path'],
        output_file_registry['l_sum_avail_path'],
        output_file_registry['outflow_direction_path'],
        output_file_registry['outflow_weights_path'],
        output_file_registry['stream_path'],
        output_file_registry['b_path'])

    sys.exit(1)
    LOGGER.info('calculating l_sum_avail_pour')
    seasonal_water_yield_core.calculate_r_sum_avail_pour(
        output_file_registry['l_sum_avail_path'],
        output_file_registry['outflow_weights_path'],
        output_file_registry['outflow_direction_path'],
        output_file_registry['l_sum_avail_pour_path'])


    LOGGER.info('route slow flow')
    seasonal_water_yield_core.route_sf(
        temporary_file_registry['dem_aligned_path'],
        output_file_registry['l_avail_path'],
        output_file_registry['l_sum_avail_path'],
        output_file_registry['l_sum_avail_pour_path'],
        output_file_registry['outflow_direction_path'],
        output_file_registry['outflow_weights_path'],
        output_file_registry['stream_path'],
        output_file_registry['b_path'],
        output_file_registry['b_sum_path'])

    LOGGER.info('  (\\w/)  SWY Complete!')
    LOGGER.info('  (..  \\ ')
    LOGGER.info(' _/  )  \\______')
    LOGGER.info('(oo /\'\\        )`,')
    LOGGER.info(' `--\' (v  __( / ||')
    LOGGER.info('       |||  ||| ||')
    LOGGER.info('      //_| //_|')


def _calculate_monthly_quick_flow(
        precip_path, lulc_path, cn_path, n_events, stream_path,
        qf_monthly_path, si_path):
    """Calculates quick flow for a month

    Parameters:
        precip_path (string): path to file that correspond to monthly
            precipitation
        lulc_path (string): path to landcover raster
        cn_path (string): path to curve number raster
        n_events (dict of int -> int): maps the number of rain events per month
            where the index to n_events is the calendar month starting at 1.
        stream_path (string): path to stream mask raster where 1 indicates a
            stream pixel, 0 is a non-stream but otherwise valid area from the
            original DEM, and nodata indicates areas outside the valid DEM.
        qf_monthly_path_list (list of string): list of paths to output monthly
            rasters.
        si_path (string): list to output raster for potential maximum retention
        """

    si_nodata = -1
    cn_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(cn_path)

    def si_op(ci_array, stream_array):
        """potential maximum retention"""
        si_array = 1000.0 / ci_array - 10
        si_array = numpy.where(ci_array != cn_nodata, si_array, si_nodata)
        si_array[stream_array == 1] = 0
        return si_array

    pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
        lulc_path)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [cn_path, stream_path], si_op, si_path, gdal.GDT_Float32,
        si_nodata, pixel_size, 'intersection', vectorize_op=False,
        datasets_are_pre_aligned=True)

    qf_nodata = -1
    p_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(precip_path)

    def qf_op(p_im, s_i, stream_array):
        """Calculate quick flow as in Eq [1] in user's guide

        Parameters:
            p_im (numpy.array): precipitation at pixel i on month m
            s_i (numpy.array): factor that is 1000/CN_i - 10
                (Equation 1b from user's guide)
            stream_mask (numpy.array): 1 if stream, otherwise not a stream
                pixel.

        Returns:
            quick flow (numpy.array)"""

        valid_mask = (
            (p_im != p_nodata) & (s_i != si_nodata) & (p_im != 0.0) &
            (stream_array != 1))
        # a_im is the mean rain depth on a rainy day at pixel i on month m
        # the 25.4 converts inches to mm since Si is in inches
        a_im = p_im[valid_mask] / n_events / 25.4
        valid_si = s_i[valid_mask]
        qf_im = numpy.empty(p_im.shape)
        qf_im[:] = qf_nodata

        # qf_im is the quickflow at pixel i on month m Eq. [1]
        qf_im[valid_mask] = (25.4 * n_events * (
            (a_im - valid_si) * numpy.exp(-0.2 * valid_si / a_im) +
            valid_si ** 2 / a_im * numpy.exp((0.8 * valid_si) / a_im) *
            scipy.special.expn(1, valid_si / a_im)))

        # if precip is 0, then QF should be zero
        qf_im[p_im == 0] = 0.0
        # if we're on a stream, set quickflow to the precipitation
        qf_im[stream_array == 1] = p_im[stream_array == 1]
        return qf_im

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [precip_path, si_path, stream_path], qf_op, qf_monthly_path,
        gdal.GDT_Float32, qf_nodata, pixel_size, 'intersection',
        vectorize_op=False, datasets_are_pre_aligned=True)


def _calculate_curve_number_raster(
        lulc_path, soil_group_path, biophysical_table, pixel_size, cn_path):
    """Calculate the CN raster from the landcover and soil group rasters"""

    soil_nodata = pygeoprocessing.get_nodata_from_uri(soil_group_path)
    map_soil_type_to_header = {
        1: 'cn_a',
        2: 'cn_b',
        3: 'cn_c',
        4: 'cn_d',
    }
    cn_nodata = -1
    lulc_to_soil = {}
    lulc_nodata = pygeoprocessing.get_nodata_from_uri(lulc_path)
    for soil_id, soil_column in map_soil_type_to_header.iteritems():
        lulc_to_soil[soil_id] = {
            'lulc_values': [],
            'cn_values': []
        }
        for lucode in sorted(biophysical_table.keys() + [lulc_nodata]):
            try:
                lulc_to_soil[soil_id]['cn_values'].append(
                    biophysical_table[lucode][soil_column])
                lulc_to_soil[soil_id]['lulc_values'].append(lucode)
            except KeyError:
                if lucode == lulc_nodata:
                    lulc_to_soil[soil_id]['lulc_values'].append(lucode)
                    lulc_to_soil[soil_id]['cn_values'].append(cn_nodata)
                else:
                    raise
        lulc_to_soil[soil_id]['lulc_values'] = (
            numpy.array(lulc_to_soil[soil_id]['lulc_values'],
                        dtype=numpy.int32))
        lulc_to_soil[soil_id]['cn_values'] = (
            numpy.array(lulc_to_soil[soil_id]['cn_values'],
                        dtype=numpy.float32))

    def cn_op(lulc_array, soil_group_array):
        """map lulc code and soil to a curve number"""
        cn_result = numpy.empty(lulc_array.shape)
        cn_result[:] = cn_nodata
        for soil_group_id in numpy.unique(soil_group_array):
            if soil_group_id == soil_nodata:
                continue
            current_soil_mask = (soil_group_array == soil_group_id)
            index = numpy.digitize(
                lulc_array.ravel(),
                lulc_to_soil[soil_group_id]['lulc_values'], right=True)
            cn_values = (
                lulc_to_soil[soil_group_id]['cn_values'][index]).reshape(
                    lulc_array.shape)
            cn_result[current_soil_mask] = cn_values[current_soil_mask]
        return cn_result

    cn_nodata = -1
    pygeoprocessing.vectorize_datasets(
        [lulc_path, soil_group_path], cn_op, cn_path, gdal.GDT_Float32,
        cn_nodata, pixel_size, 'intersection', vectorize_op=False,
        datasets_are_pre_aligned=True)


def _calculate_si_raster(cn_path, si_path, stream_path):
    """Calculates the S factor of the SCS Runoff equation also known as the
    potential maximum retention.

    Parameters:
        cn_path (string): path to curve number raster
        lulc_path (string): path to landcover raster
        si_path (string): path to output s_i raster

    Returns:
        None
    """

    si_nodata = -1
    cn_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(cn_path)

    def si_op(ci_factor, stream_mask):
        """calculate si factor"""
        valid_mask = (ci_factor != cn_nodata) & (ci_factor > 0)
        si_array = numpy.empty(ci_factor.shape)
        si_array[:] = si_nodata
        # multiply by the stream mask != 1 so we get 0s on the stream and
        # unaffected results everywhere else
        si_array[valid_mask] = (
            (1000.0 / ci_factor[valid_mask] - 10) * (
                stream_mask[valid_mask] != 1))
        return si_array

    pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(cn_path)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [cn_path, stream_path], si_op, si_path, gdal.GDT_Float32,
        si_nodata, pixel_size, 'intersection', vectorize_op=False,
        datasets_are_pre_aligned=True)
