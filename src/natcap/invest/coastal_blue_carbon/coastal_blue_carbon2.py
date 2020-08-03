import os
import logging

from osgeo import gdal
import taskgraph
import pygeoprocessing
import pandas
import numpy
import scipy.sparse

from .. import utils

LOGGER = logging.getLogger(__name__)


NODATA_FLOAT32 = float(numpy.finfo(numpy.float32).min)
NODATA_UINT16 = int(numpy.iinfo(numpy.uint16).max)


def execute(args):
    suffix = utils.make_suffix_string(args, 'results_suffix')
    output_dir = os.path.join(args['workspace_dir'], 'outputs')
    intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')
    taskgraph_cache_dir = os.path.join(intermediate_dir, 'task_cache')

    utils.make_directories([output_dir, intermediate_dir, taskgraph_cache_dir])

    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous mode.
    task_graph = taskgraph.TaskGraph(
        taskgraph_cache_dir, n_workers, reporting_interval=5.0)

    if 'transitions_csv' in args and args['transitions_csv'] not in ('', None):
        transitions = _extract_transitions_from_table(args['transitions_csv'])
    else:
        transitions = {}

    # Phase 1: alignment and preparation of inputs
    baseline_lulc_info = pygeoprocessing.get_raster_info(
        args['baseline_lulc_path'])
    target_sr_wkt = baseline_lulc_info['projection_wkt']
    min_pixel_size = numpy.min(numpy.abs(baseline_lulc_info['pixel_size']))
    target_pixel_size = (min_pixel_size, -min_pixel_size)

    transition_years = set()
    try:
        baseline_lulc_year = int(args['baseline_lulc_year'])
    except (KeyError, ValueError, TypeError):
        LOGGER.error('The baseline_lulc_year is required but not provided.')
        raise ValueError('Baseline lulc year is required.')

    try:
        # TODO: validate that args['analysis_year'] > max(transition_years)
        analysis_year = int(args['analysis_year'])
    except (KeyError, ValueError, TypeError):
        analysis_year = None

    base_paths = [args['baseline_lulc_path']]
    aligned_lulc_paths = {}
    aligned_paths = [os.path.join(
        intermediate_dir,
        f'aligned_lulc_baseline_{baseline_lulc_year}{suffix}.tif')]
    aligned_lulc_paths[int(args['baseline_lulc_year'])] = aligned_paths[0]
    for transition_year in transitions:
        base_paths.append(transitions[transition_year])
        transition_years.add(transition_year)
        aligned_paths.append(
            os.path.join(
                intermediate_dir,
                f'aligned_lulc_transition_{transition_year}{suffix}.tif'))
        aligned_lulc_paths[transition_year] = aligned_paths[-1]

    alignment_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(base_paths, aligned_paths, ['nearest']*len(base_paths),
              target_pixel_size, 'intersection'),
        kwargs={
            'target_projection_wkt': target_sr_wkt,
            'raster_align_index': 0,
        },
        hash_algorithm='md5',
        copy_duplicate_artifact=True,
        target_path_list=aligned_paths,
        task_name='Align input landcover rasters.')

    # We're assuming that the LULC initial variables and the carbon pool
    # transient table are combined into a single lookup table.
    biophysical_parameters = utils.build_lookup_from_csv(
        args['biophysical_table_path'], 'code')

    stock_rasters = {}
    disturbance_rasters = {}
    halflife_rasters = {}
    yearly_accum_rasters = {}
    net_sequestration_rasters = {}
    year_of_disturbance_rasters = {}
    emissions_rasters = {}
    total_carbon_rasters = {}

    # TODO: make sure we're also returning a sparse matrix representing the
    # 'accum' values from the table.  This should then be used when creating
    # the accumulation raster, and only those pixel values with an 'accum'
    # transition should actually accumulate.
    biomass_disturb_matrix, soil_disturb_matrix = _read_transition_matrix(
        args['landcover_transitions_table'], biophysical_parameters)
    disturbance_matrices = {}
    disturbance_matrices['soil'] = soil_disturb_matrix
    disturbance_matrices['biomass'] = biomass_disturb_matrix

    # Phase 2: do the timeseries analysis, with a few special cases when we're
    # at a transition year.
    if analysis_year:
        final_timestep = analysis_year
    else:
        final_timestep = max(max(transition_years), analysis_year)

    first_transition_year = min(transition_years)

    years_with_lulc_rasters = set(aligned_lulc_paths.keys())

    # Defining these here so that the linter doesn't complain about the
    # variable not existing.  These are actually defined at the end of the
    # timeseries loop.
    prior_stock_tasks = None
    prior_net_sequestration_tasks = None

    current_accumulation_tasks = {}
    current_halflife_tasks = {}
    current_stock_tasks = {}
    current_year_of_disturbance_tasks = {}
    current_disturbance_tasks = {}
    current_net_sequestration_tasks = {}
    current_emissions_tasks = {}

    current_transition_year = baseline_lulc_year
    prior_transition_year = None
    for year in range(baseline_lulc_year, final_timestep+1):
        stock_rasters[year] = {}
        halflife_rasters[year] = {}
        yearly_accum_rasters[year] = {}
        net_sequestration_rasters[year] = {}
        year_of_disturbance_rasters[year] = {}
        emissions_rasters[year] = {}
        total_carbon_rasters[year] = {}

        if year in transition_years:
            prior_transition_year = current_transition_year
            current_transition_year = year

        # Stocks are only reclassified from inputs in the baseline year.
        # In all other years, stocks are based on the previous year's
        # calculations.
        if year == baseline_lulc_year:
            for pool in ('soil', 'biomass'):
                stock_rasters[year][pool] = os.path.join(
                    intermediate_dir, f'stocks-{pool}-{year}{suffix}.tif')
                current_stock_tasks[pool] = task_graph.add_task(
                    func=pygeoprocessing.reclassify_raster,
                    args=(
                        (aligned_lulc_paths[year], 1),
                        {lucode: values[f'{pool}-initial'] for (lucode, values)
                            in biophysical_parameters.items()},
                        stock_rasters[year][pool],
                        gdal.GDT_Float32,
                        NODATA_FLOAT32),
                    dependent_task_list=[alignment_task],
                    target_path_list=[stock_rasters[year][pool]],
                    task_name=f'Mapping initial {pool} carbon stocks')
        else:
            # In every year after the baseline, stocks are calculated as
            # Stock[pool][thisyear] = stock[pool][lastyear] +
            #       netsequestration[pool][lastyear]
            for pool in ('soil', 'biomass'):
                stock_rasters[year][pool] = os.path.join(
                    intermediate_dir, f'stocks-{pool}-{year}{suffix}.tif')
                current_stock_tasks[pool] = task_graph.add_task(
                    func=_sum_n_rasters,
                    args=([stock_rasters[year-1][pool],
                           net_sequestration_rasters[year-1][pool]],
                          stock_rasters[year][pool]),
                    dependent_task_list=[
                        prior_stock_tasks[pool],
                        prior_net_sequestration_tasks[pool]],
                    target_path_list=[stock_rasters[year][pool]],
                    task_name=f'Calculating {pool} carbon stock for {year}')

        # These variables happen on ALL transition years, including baseline.
        if year in years_with_lulc_rasters:
            disturbance_rasters[current_transition_year] = {}
            for pool in ('soil', 'biomass'):
                yearly_accum_rasters[year][pool] = os.path.join(
                    intermediate_dir,
                    f'accumulation-{pool}-{year}{suffix}.tif')
                current_accumulation_tasks[pool] = task_graph.add_task(
                        func=pygeoprocessing.reclassify_raster,
                        args=(
                            (aligned_lulc_paths[year], 1),
                            {lucode: values[f'{pool}-yearly-accumulation']
                                for (lucode, values)
                                in biophysical_parameters.items()},
                            yearly_accum_rasters[year][pool],
                            gdal.GDT_Float32,
                            NODATA_FLOAT32),
                        dependent_task_list=[alignment_task],
                        target_path_list=[
                            yearly_accum_rasters[year][pool]],
                        task_name=(
                            f'Mapping {pool} carbon accumulation for {year}'))

                halflife_rasters[year][pool] = os.path.join(
                    intermediate_dir,
                    f'halflife-{pool}-{year}{suffix}.tif')
                current_halflife_tasks[pool] = task_graph.add_task(
                        func=pygeoprocessing.reclassify_raster,
                        args=(
                            (aligned_lulc_paths[year], 1),
                            {lucode: values[f'{pool}-initial']
                                for (lucode, values)
                                in biophysical_parameters.items()},
                            halflife_rasters[year][pool],
                            gdal.GDT_Float32,
                            NODATA_FLOAT32),
                        dependent_task_list=[alignment_task],
                        target_path_list=[halflife_rasters[year][pool]],
                        task_name=f'Mapping {pool} half-life for {year}')

                # Disturbances only happen during transition years, not during
                # the baseline year.
                # disturbances only affect soil and biomass carbon, not litter.
                if year != baseline_lulc_year and pool != 'litter':
                    # Need to enforce that the alignment and prior stocks tasks
                    # are complete before we fetch nodata values.
                    alignment_task.join()
                    prior_stock_tasks[pool].join()
                    prior_transition_nodata = pygeoprocessing.get_raster_info(
                        aligned_lulc_paths[prior_transition_year])['nodata'][0]
                    current_transition_nodata = pygeoprocessing.get_raster_info(
                        aligned_lulc_paths[current_transition_year])['nodata'][0]
                    stock_nodata = pygeoprocessing.get_raster_info(
                        stock_rasters[year-1][pool])['nodata'][0]

                    disturbance_rasters[current_transition_year][pool] = os.path.join(
                        intermediate_dir,
                        f'disturbance-{pool}-{current_transition_year}{suffix}.tif')
                    current_disturbance_tasks[pool] = task_graph.add_task(
                        func=pygeoprocessing.raster_calculator,
                        args=(
                            [(aligned_lulc_paths[prior_transition_year], 1),
                             (aligned_lulc_paths[current_transition_year], 1),
                             (stock_rasters[year-1][pool], 1),
                             (disturbance_matrices[pool], 'raw'),
                             (prior_transition_nodata, 'raw'),
                             (current_transition_nodata, 'raw'),
                             (stock_nodata, 'raw')],
                            _reclassify_transition,
                            disturbance_rasters[current_transition_year][pool],
                            gdal.GDT_Float32,
                            NODATA_FLOAT32),
                        dependent_task_list=[
                            prior_stock_tasks[pool], alignment_task],
                        target_path_list=[
                            disturbance_rasters[current_transition_year][pool]],
                        task_name=(
                            f'Mapping {pool} carbon volume disturbed in {year}'))

                    year_of_disturbance_rasters[current_transition_year][pool] = os.path.join(
                        intermediate_dir,
                        (f'year-of-latest-disturbance-{pool}-'
                            f'{current_transition_year}{suffix}.tif'))

                    # If there's a prior disturbance, use that raster.
                    # Otherwise, pass None to indicate no disturbance
                    # available.
                    if prior_transition_year not in transitions:
                        prior_year_of_disturbance_raster_tuple = (
                            None, 'raw')
                    else:
                        prior_year_of_disturbance_raster_tuple = (
                            year_of_disturbance_rasters[
                                prior_transition_year][pool], 1)
                    current_year_of_disturbance_tasks[pool] = (
                        task_graph.add_task(
                            func=pygeoprocessing.raster_calculator,
                            args=(
                                [(disturbance_rasters[current_transition_year][pool], 1),
                                 prior_year_of_disturbance_raster_tuple,
                                 (current_transition_year, 'raw'),
                                 (NODATA_FLOAT32, 'raw'),
                                 (NODATA_UINT16, 'raw')],
                                _track_latest_transition_year,
                                year_of_disturbance_rasters[current_transition_year][pool],
                                gdal.GDT_UInt16,
                                NODATA_UINT16),
                            dependent_task_list=[current_disturbance_tasks[pool]],
                            target_path_list=[
                                year_of_disturbance_rasters[current_transition_year][pool]],
                            task_name=(
                                f'Tracking the year of latest {pool} carbon '
                                f'disturbance as of {current_transition_year}')))

        for pool in ('soil', 'biomass'):
            if year >= first_transition_year:
                # calculate emissions for this year
                emissions_rasters[year][pool] = os.path.join(
                    intermediate_dir, f'emissions-{pool}-{year}.tif')
                current_emissions_tasks[pool] = task_graph.add_task(
                    func=pygeoprocessing.raster_calculator,
                    args=(
                        [(disturbance_rasters[current_transition_year][pool], 1),
                         (year_of_disturbance_rasters[current_transition_year][pool], 1),  # TODO
                         (halflife_rasters[current_transition_year][pool], 1),
                         (year, 'raw')],
                        _calculate_emissions,
                        emissions_rasters[year][pool],
                        gdal.GDT_Float32,
                        NODATA_FLOAT32),
                    dependent_task_list=[
                        current_disturbance_tasks[pool],
                        current_year_of_disturbance_tasks[pool],
                        current_halflife_tasks[pool]],
                    target_path_list=[emissions_rasters[year][pool]],
                    task_name=f'Mapping {pool} carbon emissions in {year}')

                # calculate net sequestration for this timestep
                # Net sequestration = A - E per pool
                net_sequestration_rasters[year][pool] = os.path.join(
                    intermediate_dir,
                    f'net-sequestration-{pool}-{year}{suffix}.tif')
                current_net_sequestration_tasks[pool] = task_graph.add_task(
                    func=_subtract_rasters,
                    args=(yearly_accum_rasters[current_transition_year][pool],
                          emissions_rasters[year][pool],
                          net_sequestration_rasters[year][pool]),
                    dependent_task_list=[
                        current_accumulation_tasks[pool],
                        current_emissions_tasks[pool]],
                    target_path_list=[net_sequestration_rasters[year][pool]],
                    task_name=(
                        f'Calculating net sequestration for {pool} in {year}'))
            else:
                # There are no emissions until the first timestep.  Therefore,
                # since net sequestration = accumulation - emissions, net
                # sequestration is equal to accumulation.
                net_sequestration_rasters[year][pool] = (
                    yearly_accum_rasters[current_transition_year][pool])
                current_net_sequestration_tasks[pool] = (
                    current_accumulation_tasks[pool])

        # Calculate total carbon stocks, T
        total_carbon_rasters[year] = os.path.join(
            intermediate_dir, f'total-carbon-stocks-{year}{suffix}.tif')
        _ = task_graph.add_task(
            func=_sum_n_rasters,
            args=([stock_rasters[year]['soil'],
                   stock_rasters[year]['biomass']],
                  total_carbon_rasters[year]),
            dependent_task_list=[
                current_stock_tasks['soil'],
                current_stock_tasks['biomass']],
            target_path_list=[total_carbon_rasters[year]],
            task_name=f'Calculating total carbon stocks in {year}')

        # If we're doing valuation, calculate the value of net sequestered
        # carbon.
        if ('do_economic_analysis' in args and
                args['do_economic_analysis'] not in (None, '')):
            # Need to verify the math on this, I'm unsure if the current
            # implementation is correct or makes sense:
            #    (N_biomass + N_soil[baseline]) * price[this year]
            # TODO!!!
            pass

        # These are the few sets of tasks that we care about referring to from the
        # prior year.
        prior_stock_tasks = current_stock_tasks
        prior_net_sequestration_tasks = current_net_sequestration_tasks

    # Final phase:
    # Sum timeseries rasters (A, E, N, T currently summed in the model)



def _track_latest_transition_year(
        current_disturbance_volume_matrix, known_transition_years_matrix,
        current_transition_year, current_disturbance_nodata,
        known_transition_years_nodata):

    target_matrix = numpy.empty(
        current_disturbance_volume_matrix.shape, dtype=numpy.uint16)
    target_matrix[:] = NODATA_UINT16

    if known_transition_years_matrix is not None:
        # Keep any years that are already known to be disturbed.
        pixels_previously_disturbed = ~numpy.isclose(
            known_transition_years_matrix, known_transition_years_nodata)
        target_matrix[pixels_previously_disturbed] = (
            known_transition_years_matrix[pixels_previously_disturbed])

    # Track any pixels that are known to be disturbed in this current
    # transition year.
    newly_disturbed_pixels = ~numpy.isclose(
        current_disturbance_volume_matrix, current_disturbance_nodata)
    target_matrix[newly_disturbed_pixels] = current_transition_year

    return target_matrix


def _subtract_rasters(raster_a_path, raster_b_path, target_raster_path):
    raster_a_nodata = pygeoprocessing.get_raster_info(
        raster_a_path)['nodata'][0]
    raster_b_nodata = pygeoprocessing.get_raster_info(
        raster_b_path)['nodata'][0]

    def _subtract(matrix_a, matrix_b):
        target_matrix = numpy.empty(matrix_a.shape, dtype=numpy.float32)
        target_matrix[:] = NODATA_FLOAT32

        valid_pixels = (
            ~numpy.isclose(matrix_a, raster_a_nodata) &
            ~numpy.isclose(matrix_b, raster_b_nodata))

        target_matrix[valid_pixels] = (
            matrix_a[valid_pixels] - matrix_b[valid_pixels])
        return target_matrix

    pygeoprocessing.raster_calculator(
        [(raster_a_path, 1), (raster_b_path, 1)], _subtract,
        target_raster_path, gdal.GDT_Float32, NODATA_FLOAT32)


def _calculate_emissions(
        carbon_disturbed_matrix, year_of_last_disturbance_matrix,
        carbon_half_life_matrix, current_year):
    # carbon_disturbed_matrix - the volume of carbon disturbed in the most
    # recent disturbance event AND any prior events.
    #
    # year_of_last_disturbance_matrix - a numpy matrix with pixel values of the
    # integer (uint16) years of the last transition.
    #
    # carbon_half_life_matrix - the halflife of the carbon in this pool,
    # spatially distributed.  Float32.
    #
    # Current timestep (integer), the current timestep year.
    #
    # Returns: A float32 matrix with the volume of carbon emissions THIS YEAR.
    emissions_matrix = numpy.empty(
        carbon_disturbed_matrix.shape, dtype=numpy.float32)
    emissions_matrix[:] = NODATA_FLOAT32

    valid_pixels = (
        ~numpy.isclose(carbon_disturbed_matrix, NODATA_FLOAT32) &
        (year_of_last_disturbance_matrix != NODATA_UINT16))

    n_years_elapsed = (
        current_year - year_of_last_disturbance_matrix[valid_pixels])
    valid_half_life_pixels = carbon_half_life_matrix[valid_pixels]

    # TODO: Verify this math is correct based on what's in the UG!
    emissions_matrix[valid_pixels] = (
        carbon_disturbed_matrix[valid_pixels] * (
            0.5**((n_years_elapsed-1) / valid_half_life_pixels) -
            0.5**(n_years_elapsed / valid_half_life_pixels)))

    return emissions_matrix


def _sum_n_rasters(raster_path_list, target_raster_path):
    LOGGER.info('Summing %s rasters to %s', len(raster_path_list),
                target_raster_path)
    pygeoprocessing.new_raster_from_base(
        raster_path_list[0], target_raster_path, gdal.GDT_Float32,
        [NODATA_FLOAT32])

    target_raster = gdal.OpenEx(
        target_raster_path, gdal.GA_Update | gdal.OF_RASTER)
    target_band = target_raster.GetRasterBand(1)
    for block_info in pygeoprocessing.iterblocks(
            (raster_path_list[0], 1), offset_only=True):

        sum_array = numpy.empty(
            (block_info['win_ysize'], block_info['win_xsize']),
            dtype=numpy.float32)
        sum_array[:] = 0.0

        # Assume everything is valid until proven otherwise
        valid_pixels = numpy.ones(sum_array.shape, dtype=numpy.bool)
        for raster_path in raster_path_list:
            raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
            band = raster.GetRasterBand(1)
            band_nodata = band.GetNoDataValue()

            array = band.ReadAsArray(**block_info).astype(numpy.float32)

            if band_nodata is not None:
                valid_pixels &= (~numpy.isclose(array, band_nodata))

            sum_array[valid_pixels] += array[valid_pixels]

        sum_array[~valid_pixels] = NODATA_FLOAT32

        target_band.WriteArray(
            sum_array, block_info['xoff'], block_info['yoff'])

    target_band = None
    target_raster = None


def _read_transition_matrix(transition_csv_path, biophysical_dict):
    encoding = None
    if utils.has_utf8_bom(transition_csv_path):
        encoding = 'utf-8-sig'

    table = pandas.read_csv(
        transition_csv_path, sep=None, index_col=False, engine='python',
        encoding=encoding)

    lulc_class_to_lucode = {
        values['lulc-class']: lucode for (lucode, values) in
        biophysical_dict.items()}

    # Load up a sparse matrix with the transitions to save on memory usage.
    n_rows = len(table.index)
    soil_disturbance_matrix = scipy.sparse.dok_matrix(
        (n_rows, n_rows), dtype=numpy.float32)
    biomass_disturbance_matrix = scipy.sparse.dok_matrix(
        (n_rows, n_rows), dtype=numpy.float32)

    # TODO: I don't actually know if this is any better than the dict-based
    # approach we had before since that, too, was basically sparse.
    # If we really wanted to save memory, we wouldn't duplicate the float32
    # values here and instead use the transitions to index into the various
    # biophysical values when reclassifying. That way we rely on python's
    # assumption that ints<2000 or so are singletons and thus use less memory.
    # Even so, the RIGHT way to do this is to have the user provide their own
    # maps of the following values PER TRANSITION:
    #  * {soil,biomass} disturbance values
    #  * {soil,biomass} halflife values
    #  * {soil,biomass} yearly accumulation
    #  * litter
    #  --> maybe some others, too?
    for index, row in table.iterrows():
        from_lucode = lulc_class_to_lucode[row['lulc-class'].lower()]

        for colname, field_value in row.items():
            if colname == 'lulc-class':
                continue

            to_lucode = lulc_class_to_lucode[colname.lower()]

            # Only set values where the transition HAS a value.
            # Takes advantage of the sparse characteristic of the model.
            if (isinstance(field_value, float) and
                    numpy.isnan(field_value)):
                continue

            if field_value.endswith('disturb'):
                soil_disturbance_matrix[from_lucode, to_lucode] = (
                    biophysical_dict[from_lucode][f'soil-{field_value}'])
                biomass_disturbance_matrix[from_lucode, to_lucode] = (
                    biophysical_dict[from_lucode][f'biomass-{field_value}'])

    return biomass_disturbance_matrix, soil_disturbance_matrix


def _reclassify_transition(
        landuse_transition_from_matrix, landuse_transition_to_matrix,
        carbon_storage_matrix, disturbance_magnitude_matrix, from_nodata,
        to_nodata, storage_nodata):
    """Calculate the volume of carbon disturbed.

    This function calculates the volume of disturbed carbon for each
    landcover transitioning from one landcover type to a disturbance type.
    The magnitude of the disturbance is in ``disturbance_magnitude_matrix`` and
    the existing carbon storage is found in ``carbon_storage_matrix``.

    The volume of carbon disturbed is calculated according to:

        carbon_disturbed = disturbance_magnitude * carbon_storage

    Args:
        landuse_transition_from_matrix (numpy.ndarray): An integer landcover
            array representing landcover codes that we are transitioning FROM.
        landuse_transition_to_matrix (numpy.ndarray): An integer landcover
            array representing landcover codes that we are transitioning TO.
        disturbance_magnitude_matrix (scipy.sparse.dok_matrix): A sparse matrix
            where axis 0 represents the integer landcover codes being
            transitioned from and axis 1 represents the integer landcover codes
            being transitioned to.  The values at the intersection of these
            coordinate pairs are ``numpy.float32`` values representing the
            magnitude of the disturbance in a given carbon stock during this
            transition.
        carbon_storage_matrix(numpy.ndarray): A ``numpy.float32`` matrix of
            values representing carbon storage in some pool of carbon.
        from_nodata (number or None): The nodata value of the
            ``landuse_transition_from_matrix``, or ``None`` if no nodata value
            is defined.
        to_nodata (number or None): The nodata value of the
            ``landuse_transition_to_matrix``, or ``None`` if no nodata value
            is defined.
        storage_nodata (number or None): The nodata value of the
            ``carbon_storage_matrix``, or ``None`` if no nodata value
            is defined.


    Returns:
        A ``numpy.array`` of dtype ``numpy.float32`` with the volume of
        disturbed carbon for this transition.
    """
    output_matrix = numpy.empty(landuse_transition_from_matrix.shape,
                                dtype=numpy.float32)
    output_matrix[:] = NODATA_FLOAT32

    valid_pixels = numpy.ones(landuse_transition_from_matrix.shape,
                              dtype=numpy.bool)
    if from_nodata is not None:
        valid_pixels &= (landuse_transition_from_matrix != from_nodata)

    if to_nodata is not None:
        valid_pixels &= (landuse_transition_to_matrix != to_nodata)

    if storage_nodata is not None:
        valid_pixels &= (~numpy.isclose(carbon_storage_matrix, storage_nodata))

    output_matrix[valid_pixels] = (
        carbon_storage_matrix[valid_pixels] * disturbance_magnitude_matrix[
            landuse_transition_from_matrix[valid_pixels],
            landuse_transition_to_matrix[valid_pixels]].toarray().flatten())

    return output_matrix


def _extract_transitions_from_table(csv_path):
    encoding = None
    if utils.has_utf8_bom(csv_path):
        encoding = 'utf-8-sig'

    table = pandas.read_csv(
        csv_path, sep=None, index_col=False, engine='python',
        encoding=encoding)
    table.columns = table.columns.str.lower()

    output_dict = {}
    table.set_index('transition_year', drop=False, inplace=True)
    for index, row in table.iterrows():
        output_dict[int(index)] = row['raster_path']

    return output_dict
