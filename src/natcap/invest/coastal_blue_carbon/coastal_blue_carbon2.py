import itertools
import logging
import os

import numpy
import pandas
import pygeoprocessing
import scipy.sparse
import taskgraph
from osgeo import gdal
from pygeoprocessing.symbolic import evaluate_raster_calculator_expression

from .. import utils

LOGGER = logging.getLogger(__name__)


POOL_SOIL = 'soil'
POOL_BIOMASS = 'biomass'
POOL_LITTER = 'litter'
NODATA_FLOAT32 = float(numpy.finfo(numpy.float32).min)
NODATA_UINT16 = int(numpy.iinfo(numpy.uint16).max)

STOCKS_RASTER_PATTERN = 'stocks-{pool}-{year}{suffix}.tif'
DISTURBANCE_VOL_RASTER_PATTERN = 'disturbance-volume-{pool}-{year}{suffix}.tif'
DISTURBANCE_MAGNITUDE_RASTER_PATTERN = (
    'disturbance-magnitude-{pool}-{year}{suffix}.tif')
EMISSIONS_RASTER_PATTERN = 'emissions-{pool}-{year}{suffix}.tif'
YEAR_OF_DIST_RASTER_PATTERN = (
    'year-of-latest-disturbance-{pool}-{year}{suffix}.tif')
NET_SEQUESTRATION_RASTER_PATTERN = (
    'net-sequestration-{pool}-{year}{suffix}.tif')
TOTAL_STOCKS_RASTER_PATTERN = 'total-carbon-stocks-{year}{suffix}.tif'
VALUE_RASTER_PATTERN = 'valuation-{year}{suffix}.tif'
EMISSIONS_SINCE_TRANSITION_RASTER_PATTERN = (
    'carbon-emissions-between-{start_year}-and-{end_year}{suffix}.tif')
ACCUMULATION_SINCE_TRANSITION_RASTER_PATTERN = (
    'carbon-accumulation-between-{start_year}-and-{end_year}{suffix}.tif')
TOTAL_NET_SEQ_SINCE_TRANSITION_RASTER_PATTERN = (
    'total-net-carbon-sequestration-between-{start_year}-and-'
    '{end_year}{suffix}.tif')


INTERMEDIATE_DIR_NAME = 'intermediate'
TASKGRAPH_CACHE_DIR_NAME = 'task_cache'
OUTPUT_DIR_NAME = 'output'


# TODO: restructure to separate initial from baseline from transitions.
#  Phase 0: Table reading and raster alignment.
#  Phase 1: Map initial variables (reclassify initial biomass, soil, litter)
#  Phase 2: Accumulate everything up to the first transition or analysis year.
#  Phase 3: Do the transition timeseries analysis.
#
# This structure should nicely separate things out so that the looping logic is
# substantially simpler and easier to maintain in the long run.
#
# Besides, net sequestration until the first transition is always just a
# multiple of the accumulation per pool, then summed.  So why loop over piles
# of rasters when we can just multiply?


def execute_transition_analysis(args):
    # If a graph already exists, use that.  Otherwise, create one.
    try:
        task_graph = args['task_graph']
    except KeyError:
        taskgraph_cache_dir = os.path.join(
            args['workspace_dir'], TASKGRAPH_CACHE_DIR_NAME)
        task_graph = taskgraph.TaskGraph(
            taskgraph_cache_dir,
            int(args['n_workers']))

    suffix = args.get('suffix', '')
    intermediate_dir = os.path.join(
        args['workspace_dir'], INTERMEDIATE_DIR_NAME)
    output_dir = os.path.join(
        args['workspace_dir'], OUTPUT_DIR_NAME)

    transition_years = set(args['transition_years'])
    disturbance_magnitude_rasters = args['disturbance_magnitude_rasters']
    half_life_rasters = args['half_life_rasters']
    yearly_accum_rasters = args['annual_rate_of_accumulation_rasters']
    prices = args['carbon_prices_per_year']

    # ASSUMPTIONS
    #
    # This rebuild assumes that this timeseries analysis is ONLY taking place
    # for the transitions at hand.  Everything that happens between the
    # baseline year and the first transition isn't all that interesting since
    # the only thing that can happen is accumulation. That can be modeled with
    # a few raster calculator operations.  Everything within this loop is way
    # more interesting and tricky to get right, hence the need for an extra
    # function.
    stock_rasters = {}
    net_sequestration_rasters = {}
    disturbance_vol_rasters = {}
    emissions_rasters = {}
    year_of_disturbance_rasters = {}
    total_carbon_rasters = {}
    prior_transition_year = None
    current_transition_year = None

    current_disturbance_vol_tasks = {}
    prior_stock_tasks = {}
    current_year_of_disturbance_tasks = {}
    current_emissions_tasks = {}
    prior_net_sequestration_tasks = {}
    current_net_sequestration_tasks = {}
    current_accumulation_tasks = {}

    summary_net_sequestration_tasks = []
    summary_net_sequestration_raster_paths = []

    for year in range(args['first_transition_year'], args['final_year']+1):
        current_stock_tasks = {}
        net_sequestration_rasters[year] = {}
        stock_rasters[year] = {}
        disturbance_vol_rasters[year] = {}
        emissions_rasters[year] = {}
        year_of_disturbance_rasters[year] = {}

        for pool in (POOL_SOIL, POOL_BIOMASS, POOL_LITTER):
            # Calculate stocks from last year's stock plus last year's net
            # sequestration.
            stock_rasters[year][pool] = os.path.join(
                intermediate_dir,
                STOCKS_RASTER_PATTERN.format(
                    year=year, pool=pool, suffix=suffix))
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

            # Calculate disturbance volume if we're at a transition year.
            #    TODO: provide disturbance magnitude as a raster input
            if year in transition_years:
                prior_transition_year = current_transition_year
                current_transition_year = year
                disturbance_vol_rasters[year][pool] = os.path.join(
                    intermediate_dir,
                    DISTURBANCE_VOL_RASTER_PATTERN.format(
                        pool=pool, year=year, suffix=suffix))
                current_disturbance_vol_tasks[pool] = task_graph.add_task(
                    func=evaluate_raster_calculator_expression,
                    args=("magnitude*stocks",
                          {"magnitude": (
                              disturbance_magnitude_rasters[year][pool]),
                           "stocks": stock_rasters[year-1][pool]},
                          NODATA_FLOAT32,
                          disturbance_vol_rasters[year][pool]),
                    dependent_task_list=[prior_stock_tasks[pool]],
                    target_path_list=[
                        disturbance_vol_rasters[year][pool]],
                    task_name=(
                        f'Mapping {pool} carbon volume disturbed in {year}'))

                # Year-of-disturbance rasters track the year of the most recent
                # disturbance.  This is important because a disturbance could
                # span multiple transition years.  This raster is derived from
                # the incoming landcover rasters and is not something that is
                # defined by the user.
                year_of_disturbance_rasters[year][pool] = os.path.join(
                    intermediate_dir, YEAR_OF_DIST_RASTER_PATTERN.format(
                        pool=pool, year=year, suffix=suffix))
                current_year_of_disturbance_tasks[pool] = task_graph.add_task(
                    func=_track_latest_transition_year,
                    args=(disturbance_vol_rasters[year][pool],
                          year_of_disturbance_rasters.get(
                              prior_transition_year, None),
                          year_of_disturbance_rasters[year][pool]),
                    dependent_task_list=[
                        current_disturbance_vol_tasks[pool]],
                    target_path_list=[
                        year_of_disturbance_rasters[year][pool]],
                    task_name=(
                        f'Track year of latest {pool} carbon disturbance as '
                        f'of {year}'))

            # Calculate emissions (all years after 1st transition)
            # Emissions in this context are a function of:
            #  * stocks at the disturbance year
            #  * disturbance magnitude
            #  * halflife
            emissions_rasters[year][pool] = os.path.join(
                intermediate_dir, EMISSIONS_RASTER_PATTERN.format(
                    pool=pool, year=year, suffix=suffix))
            current_emissions_tasks[pool] = task_graph.add_task(
                func=pygeoprocessing.raster_calculator,
                args=(
                    [(disturbance_vol_rasters[pool], 1),
                     (year_of_disturbance_rasters[
                          current_transition_year][pool], 1),
                     (half_life_rasters[current_transition_year][pool], 1),
                     (year, 'raw')],
                    _calculate_emissions,
                    emissions_rasters[year][pool],
                    gdal.GDT_Float32,
                    NODATA_FLOAT32),
                dependent_task_list=[
                    current_disturbance_vol_tasks[pool],
                    current_year_of_disturbance_tasks[pool]],
                target_path_list=[
                    emissions_rasters[year][pool]],
                task_name=f'Mapping {pool} carbon emissions in {year}')

            # Calculate net sequestration (all years after 1st transition)
            #   * Where pixels are accumulating, accumulate.
            #   * Where pixels are emitting, emit.
            net_sequestration_rasters[year][pool] = os.path.join(
                intermediate_dir, NET_SEQUESTRATION_RASTER_PATTERN.format(
                    pool=pool, year=year, suffix=suffix))
            current_net_sequestration_tasks[pool] = task_graph.add_task(
                func=_calculate_net_sequestration,
                args=(yearly_accum_rasters[current_transition_year][pool],
                      emissions_rasters[year][pool],
                      net_sequestration_rasters[year][pool]),
                dependent_task_list=[
                    current_accumulation_tasks[pool],
                    current_emissions_tasks[pool]],
                target_path_list=[net_sequestration_rasters[year][pool]],
                task_name=(
                    f'Calculating net sequestration for {pool} in {year}'))

        # Calculate total carbon stocks (sum stocks across all 3 pools)
        total_carbon_rasters[year] = os.path.join(
            intermediate_dir, TOTAL_STOCKS_RASTER_PATTERN.format(
                year=year, suffix=suffix))
        _ = task_graph.add_task(
            func=_sum_n_rasters,
            args=([stock_rasters[year][POOL_SOIL],
                   stock_rasters[year][POOL_BIOMASS],
                   stock_rasters[year][POOL_LITTER]],
                  total_carbon_rasters[year]),
            dependent_task_list=[
                current_stock_tasks[POOL_SOIL],
                current_stock_tasks[POOL_BIOMASS],
                current_stock_tasks[POOL_LITTER]],
            target_path_list=[total_carbon_rasters[year]],
            task_name=f'Calculating total carbon stocks in {year}')

        # Calculate valuation if we're doing valuation (requires Net Seq.)
        if ('do_economic_analysis' in args and
                args['do_economic_analysis'] not in (None, '')):
            # TODO: Need to verify the math on this, I'm unsure if the current
            # implementation is correct or makes sense:
            #    (N_biomass + N_soil[baseline]) * price[this year]
            valuation_raster = os.path.join(
                intermediate_dir, VALUE_RASTER_PATTERN.format(
                    year=year, suffix=suffix))
            _ = task_graph.add_task(
                func=pygeoprocessing.raster_calculator,
                args=([(net_sequestration_rasters[year][POOL_BIOMASS], 1),
                       (net_sequestration_rasters[year][POOL_SOIL], 1),
                       (prices[year], 'raw')],
                      _calculate_valuation,
                      valuation_raster,
                      gdal.GDT_Float32,
                      NODATA_FLOAT32),
                dependent_task_list=[
                    current_net_sequestration_tasks[POOL_BIOMASS],
                    current_net_sequestration_tasks[POOL_SOIL]],
                target_path_list=[valuation_raster],
                task_name=f'Calculating the value of carbon for {year}')

        # If in the last year before a transition:
        #  * sum emissions since last transition
        #  * sum accumulation since last transition
        #  * sum net sequestration since last transition
        if (year + 1) in transition_years:
            emissions_rasters_since_transition = []
            net_seq_rasters_since_transition = []
            for _year in range(current_transition_year, year + 1):
                emissions_rasters_since_transition.extend(
                    list(emissions_rasters[year].values()))
                net_seq_rasters_since_transition.extend(
                    list(net_sequestration_rasters[year].values()))

            emissions_since_last_transition_raster = os.path.join(
                output_dir, EMISSIONS_SINCE_TRANSITION_RASTER_PATTERN.format(
                    start_year=current_transition_year, end_year=(year + 1),
                    suffix=suffix))
            _ = task_graph.add_task(
                func=_sum_n_rasters,
                args=(emissions_rasters_since_transition,
                      emissions_since_last_transition_raster),
                dependent_task_list=[current_emissions_tasks[year]],
                target_path_list=[emissions_since_last_transition_raster],
                task_name=(
                    f'Sum emissions between {current_transition_year} '
                    f'and {year}'))

            accumulation_since_last_transition = os.path.join(
                output_dir,
                ACCUMULATION_SINCE_TRANSITION_RASTER_PATTERN.format(
                    start_year=current_transition_year, end_year=(year + 1),
                    suffix=suffix))
            _ = task_graph.add_task(
                func=pygeoprocessing.raster_calculator,
                args=(
                    [(yearly_accum_rasters[
                        current_transition_year][POOL_SOIL], 1),
                     (yearly_accum_rasters[
                         current_transition_year][POOL_BIOMASS], 1),
                     (((year + 1) - current_transition_year), 'raw')],
                    _calculate_accumulation_over_time,
                    accumulation_since_last_transition,
                    gdal.GDT_Float32,
                    NODATA_FLOAT32),
                dependent_task_list=[current_accumulation_tasks],
                target_path_list=[accumulation_since_last_transition],
                task_name=(
                    f'Summing accumulation between {current_transition_year} '
                    f'and {year+1}'))

            net_carbon_sequestration_since_last_transition = os.path.join(
                output_dir,
                TOTAL_NET_SEQ_SINCE_TRANSITION_RASTER_PATTERN.format(
                    start_year=current_transition_year, end_year=(year + 1),
                    suffix=suffix))
            summary_net_sequestration_tasks.append(task_graph.add_task(
                func=_sum_n_rasters,
                args=(net_seq_rasters_since_transition,
                      net_carbon_sequestration_since_last_transition),
                dependent_task_list=list(
                    current_net_sequestration_tasks.values()),
                target_path_list=[
                    net_carbon_sequestration_since_last_transition],
                task_name=(
                    f'Summing sequestration between {current_transition_year} '
                    f'and {year}')))
            summary_net_sequestration_raster_paths.append(
                net_carbon_sequestration_since_last_transition)

        # These are the few sets of tasks that we care about referring to from
        # the prior year.
        prior_stock_tasks = current_stock_tasks
        prior_net_sequestration_tasks = current_net_sequestration_tasks

    # Calculate total net sequestration.


def execute(args):
    suffix = utils.make_suffix_string(args, 'results_suffix')
    output_dir = os.path.join(args['workspace_dir'], 'output')
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

    # TODO: check that the years in the price table match the years in the
    # range of the timesteps we're running.
    if ('do_economic_analysis' in args and
            args['do_economic_analysis'] not in ('', None)):
        if args.get('do_price_table', False):
            prices = {
                year: values['price'] for (year, values) in
                utils.build_lookup_from_csv(
                    args['price_table'], 'year').items()}
        else:
            inflation_rate = float(args['inflation_rate']) * 0.01
            annual_price = float(args['price'])
            max_year = max(transition_years.keys()).union(set([analysis_year]))

            prices = {}
            for timestep_index, year in enumerate(
                    range(baseline_lulc_year, max_year + 1)):
                prices[year] = (
                    ((1 + inflation_rate) ** timestep_index) *
                    annual_price)

        discount_rate = float(args['discount_rate']) * 0.01
        for year, price in prices.items():
            n_years_elapsed = year - baseline_lulc_year
            prices[year] /= (1 + discount_rate) ** n_years_elapsed

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
    (biomass_disturb_matrix, soil_disturb_matrix,
        biomass_accum_matrix, soil_accum_matrix) = _read_transition_matrix(
        args['landcover_transitions_table'], biophysical_parameters)
    disturbance_matrices = {
        'soil': soil_disturb_matrix,
        'biomass': biomass_disturb_matrix
    }
    accumulation_matrices = {
        'soil': soil_accum_matrix,
        'biomass': biomass_accum_matrix,
    }

    # Phase 2: do the timeseries analysis, with a few special cases when we're
    # at a transition year.
    if analysis_year:
        final_timestep = analysis_year
    else:
        final_timestep = max(transition_years)

    first_transition_year = min(transition_years)
    LOGGER.info('Transition years: %s', sorted(transition_years))

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

    summary_net_sequestration_tasks = []
    summary_net_sequestration_raster_paths = []

    emissions_tasks_since_transition = []
    net_sequestration_tasks_since_transition = []
    accumulation_tasks_since_transition = []

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

            disturbance_rasters[current_transition_year] = {}
            emissions_tasks_since_transition = []
            net_sequestration_tasks_since_transition = []
            accumulation_tasks_since_transition = []

        # Stocks are only reclassified from inputs in the baseline year.
        # In all other years, stocks are based on the previous year's
        # calculations.
        for pool in (POOL_SOIL, POOL_BIOMASS, POOL_LITTER):
            if year == baseline_lulc_year:
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

                # Initial accumulation values are a simple reclassification
                # rather than a mapping by the transition.
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
                accumulation_tasks_since_transition.append(
                    current_accumulation_tasks[pool])
            else:
                # In every year after the baseline, stocks are calculated as
                # Stock[pool][thisyear] = stock[pool][lastyear] +
                #       netsequestration[pool][lastyear]
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

            # These variables happen on ALL transition years, including
            # baseline.
            # TODO: allow accumulation to be spatially defined and not just
            # classified.
            if year in years_with_lulc_rasters and year != baseline_lulc_year:
                if prior_transition_year is None:
                    prior_transition_year = baseline_lulc_year
                yearly_accum_rasters[year][pool] = os.path.join(
                    intermediate_dir,
                    f'accumulation-{pool}-{year}{suffix}.tif')

                # Litter accumulation is a bit of a special case in that it
                # we don't care about how it transitions from one landcover
                # type to another. We only care about how it accumulates.
                if pool == POOL_LITTER:
                    current_accumulation_tasks[pool] = task_graph.add_task(
                        func=pygeoprocessing.reclassify_raster,
                        args=((aligned_lulc_paths[year], 1),
                              {lucode: values[f'{pool}-yearly-accumulation']
                               for (lucode, values) in
                               biophysical_parameters.items()},
                              yearly_accum_rasters[year][pool],
                              gdal.GDT_Float32,
                              NODATA_FLOAT32),
                        dependent_task_list=[alignment_task],
                        target_path_list=[yearly_accum_rasters[year][pool]],
                        task_name=f'Mapping litter accumulation for {year}')
                else:
                    current_accumulation_tasks[pool] = task_graph.add_task(
                        func=_reclassify_accumulation_transition,
                        args=(aligned_lulc_paths[prior_transition_year],
                              aligned_lulc_paths[current_transition_year],
                              accumulation_matrices[pool],
                              yearly_accum_rasters[year][pool]),
                        dependent_task_list=[alignment_task],
                        target_path_list=[
                            yearly_accum_rasters[year][pool]],
                        task_name=(
                            f'Mapping {pool} carbon accumulation for {year}'))
                accumulation_tasks_since_transition.append(
                    current_accumulation_tasks[pool])

            if year in years_with_lulc_rasters and pool != POOL_LITTER:
                # Halflife reflect last transition's state
                # Emissions happen when we're transitioning AWAY from CBC
                # landcovers, and only in the soil and biomass pools.
                if prior_transition_year is None:
                    halflife_source_year = year
                else:
                    halflife_source_year = prior_transition_year

                halflife_rasters[year][pool] = os.path.join(
                    intermediate_dir,
                    f'halflife-{pool}-{year}{suffix}.tif')
                current_halflife_tasks[pool] = task_graph.add_task(
                    func=pygeoprocessing.reclassify_raster,
                    args=(
                        (aligned_lulc_paths[halflife_source_year], 1),
                        {lucode: values[f'{pool}-half-life']
                            for (lucode, values)
                            in biophysical_parameters.items()},
                        halflife_rasters[year][pool],
                        gdal.GDT_Float32,
                        NODATA_FLOAT32),
                    dependent_task_list=[alignment_task],
                    target_path_list=[halflife_rasters[year][pool]],
                    task_name=f'Mapping {pool} half-life for {year}')

                # Disturbances only happen during transition years, not during
                # the baseline year.  Disturbances only affect soil, biomass
                # pools.
                if year != baseline_lulc_year:
                    disturbance_rasters[current_transition_year][pool] = (
                        os.path.join(
                            intermediate_dir,
                            (f'disturbance-{pool}-{current_transition_year}'
                                f'{suffix}.tif')))
                    current_disturbance_tasks[pool] = task_graph.add_task(
                        func=_reclassify_disturbance_transition,
                        args=(aligned_lulc_paths[prior_transition_year],
                              aligned_lulc_paths[current_transition_year],
                              stock_rasters[year-1][pool],
                              disturbance_matrices[pool],
                              disturbance_rasters[
                                current_transition_year][pool]),
                        dependent_task_list=[
                            prior_stock_tasks[pool], alignment_task],
                        target_path_list=[
                            disturbance_rasters[
                                current_transition_year][pool]],
                        task_name=(f'Mapping {pool} carbon volume disturbed '
                                   f'in {year}'))

                    year_of_disturbance_rasters[
                        current_transition_year][pool] = os.path.join(
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
                                [(disturbance_rasters[
                                    current_transition_year][pool], 1),
                                 prior_year_of_disturbance_raster_tuple,
                                 (current_transition_year, 'raw'),
                                 (NODATA_FLOAT32, 'raw'),
                                 (NODATA_UINT16, 'raw')],
                                _track_latest_transition_year,
                                year_of_disturbance_rasters[
                                    current_transition_year][pool],
                                gdal.GDT_UInt16,
                                NODATA_UINT16),
                            dependent_task_list=[
                                current_disturbance_tasks[pool]],
                            target_path_list=[
                                year_of_disturbance_rasters[
                                    current_transition_year][pool]],
                            task_name=(
                                f'Tracking the year of latest {pool} carbon '
                                f'disturbance as of {current_transition_year}')
                        ))

            if year >= first_transition_year and pool != POOL_LITTER:
                # calculate emissions for this year
                emissions_rasters[year][pool] = os.path.join(
                    intermediate_dir, f'emissions-{pool}-{year}{suffix}.tif')
                current_emissions_tasks[pool] = task_graph.add_task(
                    func=pygeoprocessing.raster_calculator,
                    args=(
                        [(disturbance_rasters[
                            current_transition_year][pool], 1),
                         (year_of_disturbance_rasters[
                             current_transition_year][pool], 1),
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
                emissions_tasks_since_transition.append(
                    current_emissions_tasks[pool])

                # calculate net sequestration for this timestep
                # Net sequestration = A - E per pool
                net_sequestration_rasters[year][pool] = os.path.join(
                    intermediate_dir,
                    f'net-sequestration-{pool}-{year}{suffix}.tif')
                current_net_sequestration_tasks[pool] = task_graph.add_task(
                    func=_calculate_net_sequestration,
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
                # The litter pool never has any disturbances, only no more
                # accumulation.  Therefore, net sequestration for litter is
                # just the rate of accumulation.
                net_sequestration_rasters[year][pool] = (
                    yearly_accum_rasters[current_transition_year][pool])
                current_net_sequestration_tasks[pool] = (
                    current_accumulation_tasks[pool])

            net_sequestration_tasks_since_transition.append(
                current_net_sequestration_tasks[pool])

        # Calculate total carbon stocks, T
        total_carbon_rasters[year] = os.path.join(
            intermediate_dir, f'total-carbon-stocks-{year}{suffix}.tif')
        _ = task_graph.add_task(
            func=_sum_n_rasters,
            args=([stock_rasters[year][POOL_SOIL],
                   stock_rasters[year][POOL_BIOMASS],
                   stock_rasters[year][POOL_LITTER]],
                  total_carbon_rasters[year]),
            dependent_task_list=[
                current_stock_tasks[POOL_SOIL],
                current_stock_tasks[POOL_BIOMASS],
                current_stock_tasks[POOL_LITTER]],
            target_path_list=[total_carbon_rasters[year]],
            task_name=f'Calculating total carbon stocks in {year}')

        # If we're doing valuation, calculate the value of net sequestered
        # carbon.
        if ('do_economic_analysis' in args and
                args['do_economic_analysis'] not in (None, '')):
            # Need to verify the math on this, I'm unsure if the current
            # implementation is correct or makes sense:
            #    (N_biomass + N_soil[baseline]) * price[this year]
            valuation_raster = os.path.join(
                intermediate_dir, f'valuation-{year}{suffix}.tif')
            _ = task_graph.add_task(
                func=pygeoprocessing.raster_calculator,
                args=([(net_sequestration_rasters[year][POOL_BIOMASS], 1),
                       (net_sequestration_rasters[year][POOL_SOIL], 1),
                       (prices[year], 'raw')],
                      _calculate_valuation,
                      valuation_raster,
                      gdal.GDT_Float32,
                      NODATA_FLOAT32),
                dependent_task_list=[
                    current_net_sequestration_tasks[POOL_BIOMASS],
                    current_net_sequestration_tasks[POOL_SOIL]],
                target_path_list=[valuation_raster],
                task_name=f'Calculating the value of carbon for {year}')

        # If we're in the last year before a transition, summarize the
        # appropriate rasters since the last transition until now.
        if year+1 in transition_years.union(set([final_timestep])):
            years_since_last_transition = list(
                range(current_transition_year, year+1))

            timestep_net_sequestration_rasters = list(itertools.chain(
                *[list(net_sequestration_rasters[target_year].values())
                  for target_year in years_since_last_transition]))

            # Emissions is only calculated after the first transition.
            if current_transition_year != baseline_lulc_year:
                timestep_emissions_rasters = list(itertools.chain(
                    *[list(emissions_rasters[target_year].values())
                      for target_year in years_since_last_transition]))

                emissions_since_last_transition = os.path.join(
                    output_dir, (
                        f'carbon_emissions_between_'
                        f'{current_transition_year}_and_{year+1}{suffix}.tif'))

                _ = task_graph.add_task(
                    func=_sum_n_rasters,
                    args=(timestep_emissions_rasters,
                          emissions_since_last_transition),
                    dependent_task_list=emissions_tasks_since_transition,
                    target_path_list=[emissions_since_last_transition],
                    task_name=(
                        f'Summing emissions between {current_transition_year} '
                        f'and {year}'))

            accumulation_since_last_transition = os.path.join(
                output_dir, (
                    f'carbon_accumulation_between_'
                    f'{current_transition_year}_and_{year+1}{suffix}.tif'))
            _ = task_graph.add_task(
                func=pygeoprocessing.raster_calculator,
                args=(
                    [(yearly_accum_rasters[
                        current_transition_year][POOL_SOIL], 1),
                     (yearly_accum_rasters[
                         current_transition_year][POOL_BIOMASS], 1),
                     (len(years_since_last_transition), 'raw')],
                    _calculate_accumulation_over_time,
                    accumulation_since_last_transition,
                    gdal.GDT_Float32,
                    NODATA_FLOAT32),
                dependent_task_list=accumulation_tasks_since_transition,
                target_path_list=[accumulation_since_last_transition],
                task_name=(
                    f'Summing accumulation between {current_transition_year} '
                    f'and {year}'))

            net_carbon_sequestration_since_last_transition = os.path.join(
                output_dir, (
                    f'total_net_carbon_sequestration_between_'
                    f'{current_transition_year}_and_{year+1}{suffix}.tif'))
            summary_net_sequestration_tasks.append(task_graph.add_task(
                func=_sum_n_rasters,
                args=(timestep_net_sequestration_rasters,
                      net_carbon_sequestration_since_last_transition),
                dependent_task_list=net_sequestration_tasks_since_transition,
                target_path_list=[
                    net_carbon_sequestration_since_last_transition],
                task_name=(
                    f'Summing sequestration between {current_transition_year} '
                    f'and {year}')))
            summary_net_sequestration_raster_paths.append(
                net_carbon_sequestration_since_last_transition)

        # These are the few sets of tasks that we care about referring to from
        # the prior year.
        prior_stock_tasks = current_stock_tasks
        prior_net_sequestration_tasks = current_net_sequestration_tasks

    total_net_sequestration_raster_path = os.path.join(
        output_dir, f'total_net_carbon_sequestration{suffix}.tif')
    _ = task_graph.add_task(
            func=_sum_n_rasters,
            args=(summary_net_sequestration_raster_paths,
                  total_net_sequestration_raster_path),
            kwargs={
                'allow_pixel_stacks_with_nodata': True,
            },
            dependent_task_list=summary_net_sequestration_tasks,
            target_path_list=[total_net_sequestration_raster_path],
            task_name=(
                'Calculate total net carbon sequestration across all years'))

    task_graph.close()
    task_graph.join()


def _calculate_disturbance_volume(
        disturbance_magnitude_raster_path, prior_stocks_raster_path,
        target_raster_path):
    disturbance_magnitude_nodata = pygeoprocessing.get_raster_info(
        disturbance_magnitude_raster_path)['nodata'][0]
    prior_stocks_nodata = pygeoprocessing.get_raster_info(
        prior_stocks_raster_path)['nodata'][0]

    def _calculate(disturbance_magnitude_matrix, prior_stocks_matrix):
        target_matrix = numpy.empty(disturbance_magnitude_matrix.shape,
                                    dtype=numpy.float32)
        target_matrix[:] = NODATA_FLOAT32

        valid_pixels = (
            (~numpy.isclose(disturbance_magnitude_matrix,
                            disturbance_magnitude_nodata)) &
            (~numpy.isclose(prior_stocks_matrix, prior_stocks_nodata)))

        target_matrix[valid_pixels] = (
            disturbance_magnitude_matrix[valid_pixels] *
            prior_stocks_matrix[valid_pixels])
        return target_matrix

    pygeoprocessing.raster_calculator(
        [(disturbance_magnitude_raster_path, 1),
         (prior_stocks_matrix, 1)], _calculate, target_raster_path,
        gdal.GDT_Float32, NODATA_FLOAT32)


def _calculate_accumulation_over_time(
        annual_biomass_matrix, annual_soil_matrix, n_years):
    target_matrix = numpy.empty(annual_biomass_matrix.shape,
                                dtype=numpy.float32)
    target_matrix[:] = NODATA_FLOAT32

    valid_pixels = (
        ~numpy.isclose(annual_biomass_matrix, NODATA_FLOAT32) &
        ~numpy.isclose(annual_soil_matrix, NODATA_FLOAT32))

    target_matrix[valid_pixels] = (
        (annual_biomass_matrix[valid_pixels] +
            annual_soil_matrix[valid_pixels]) * n_years)
    return target_matrix


def _calculate_valuation(
        biomass_sequestration_matrix, soil_sequestration_matrix,
        price):
    value_matrix = numpy.empty(
        biomass_sequestration_matrix.shape, dtype=numpy.float32)
    value_matrix[:] = NODATA_FLOAT32

    valid_pixels = (
        ~numpy.isclose(biomass_sequestration_matrix, NODATA_FLOAT32) &
        ~numpy.isclose(soil_sequestration_matrix, NODATA_FLOAT32))
    value_matrix[valid_pixels] = (
        (biomass_sequestration_matrix[valid_pixels] +
            soil_sequestration_matrix[valid_pixels]) * price)

    return value_matrix


def _track_latest_transition_year(
        current_disturbance_vol_raster_path,
        known_transition_years_raster_path,
        current_transition_year, target_path):
    current_disturbance_vol_nodata = pygeoprocessing.get_raster_info(
        current_disturbance_vol_raster_path)['nodata'][0]

    if known_transition_years_raster_path:
        known_transition_years_nodata = pygeoprocessing.get_raster_info(
            known_transition_years_raster_path)['nodata'][0]
        known_transition_years_tuple = (
            known_transition_years_raster_path, 1)
    else:
        known_transition_years_tuple = (None, 'raw')

    def _track_transition_year(
            current_disturbance_vol_matrix, known_transition_years_matrix):

        target_matrix = numpy.empty(
            current_disturbance_vol_matrix.shape, dtype=numpy.uint16)
        target_matrix[:] = NODATA_UINT16

        # If this is None, then we don't have any previously disturbed pixels
        # and everything disturbed in this timestep is newly disturbed.
        if known_transition_years_raster_path:
            # Keep any years that are already known to be disturbed.
            pixels_previously_disturbed = ~numpy.isclose(
                known_transition_years_matrix, known_transition_years_nodata)
            target_matrix[pixels_previously_disturbed] = (
                known_transition_years_matrix[pixels_previously_disturbed])

        # Track any pixels that are known to be disturbed in this current
        # transition year.
        # Exclude pixels that are nodata or effectively 0.
        newly_disturbed_pixels = (
            (~numpy.isclose(
                current_disturbance_vol_matrix,
                current_disturbance_vol_nodata)) &
            (~numpy.isclose(current_disturbance_vol_matrix, 0.0)))

        target_matrix[newly_disturbed_pixels] = current_transition_year

        return target_matrix

    pygeoprocessing.raster_calculator(
        [(current_disturbance_vol_raster_path, 1),
         known_transition_years_tuple], _track_transition_year, target_path,
        gdal.GDT_UInt16, NODATA_UINT16)


def _calculate_net_sequestration(
        accumulation_raster_path, emissions_raster_path, target_raster_path):
    accumulation_nodata = pygeoprocessing.get_raster_info(
        accumulation_raster_path)['nodata'][0]
    emissions_nodata = pygeoprocessing.get_raster_info(
        emissions_raster_path)['nodata'][0]

    def _record_sequestration(accumulation_matrix, emissions_matrix):
        target_matrix = numpy.zeros(
            accumulation_matrix.shape, dtype=numpy.float32)

        # A given cell can have either accumulation OR emissions, not both.
        # If there are pixel values on both matrices, emissions will take
        # precedent.  This is an arbitrary choice, but it'll be easier for the
        # user to provide a raster filled with some blanket accumulation value
        # and then assume that the Emissions raster has the extra spatial
        # nuances of the landscape (like nodata holes).
        valid_accumulation_pixels = numpy.ones(accumulation_matrix.shape,
                                               dtype=numpy.bool)
        if accumulation_nodata is not None:
            valid_accumulation_pixels &= (
                ~numpy.isclose(accumulation_matrix, accumulation_nodata))
        target_matrix[valid_accumulation_pixels] += (
            accumulation_matrix[valid_accumulation_pixels])

        valid_emissions_pixels = numpy.ones(emissions_matrix.shape,
                                            dtype=numpy.bool)
        if emissions_nodata is not None:
            valid_emissions_pixels &= (
                ~numpy.isclose(emissions_matrix, emissions_nodata))
        target_matrix[valid_emissions_pixels] = emissions_matrix[
            valid_emissions_pixels]

        valid_pixels = ~(valid_accumulation_pixels | valid_emissions_pixels)
        target_matrix[valid_pixels] = NODATA_FLOAT32
        return target_matrix

    pygeoprocessing.raster_calculator(
        [(accumulation_raster_path, 1), (emissions_raster_path, 1)],
        _record_sequestration, target_raster_path, gdal.GDT_Float32,
        NODATA_FLOAT32)


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

    # Landcovers with a carbon half-life of 0 will be assumed to have no
    # emissions.
    zero_half_life = numpy.isclose(carbon_half_life_matrix, 0.0)

    valid_pixels = (
        (~numpy.isclose(carbon_disturbed_matrix, NODATA_FLOAT32)) &
        (year_of_last_disturbance_matrix != NODATA_UINT16) &
        (~zero_half_life))

    n_years_elapsed = (
        current_year - year_of_last_disturbance_matrix[valid_pixels])

    valid_half_life_pixels = carbon_half_life_matrix[valid_pixels]

    # TODO: Verify this math is correct based on what's in the UG!
    # Note that `n_years_elapsed` can be 0, which maybe doesn't make sense, but
    # I'll need to check with someone to make sure of this.
    # TODO: should we be emitting carbon in the transition year?
    emissions_matrix[valid_pixels] = (
        carbon_disturbed_matrix[valid_pixels] * (
            0.5**((n_years_elapsed-1) / valid_half_life_pixels) -
            0.5**(n_years_elapsed / valid_half_life_pixels)))

    # See note above about a half-life of 0.0 representing no emissions.
    emissions_matrix[zero_half_life] = 0.0

    return emissions_matrix


def _sum_n_rasters(raster_path_list, target_raster_path,
        allow_pixel_stacks_with_nodata=False):
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
        pixels_touched = numpy.zeros(sum_array.shape, dtype=numpy.bool)
        for raster_path in raster_path_list:
            raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
            if raster is None:
                LOGGER.error('Could not open %s', raster_path)

            band = raster.GetRasterBand(1)
            band_nodata = band.GetNoDataValue()

            array = band.ReadAsArray(**block_info).astype(numpy.float32)

            if band_nodata is not None:
                valid_pixels &= (~numpy.isclose(array, band_nodata))

            sum_array[valid_pixels] += array[valid_pixels]
            pixels_touched[valid_pixels] = 1

        if allow_pixel_stacks_with_nodata:
            sum_array[~pixels_touched] = NODATA_FLOAT32
        else:
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

    lulc_class_to_lucode = {}
    max_lucode = 0
    for (lucode, values) in biophysical_dict.items():
        lulc_class_to_lucode[values['lulc-class']] = lucode
        max_lucode = max(max_lucode, lucode)

    # Load up a sparse matrix with the transitions to save on memory usage.
    # The number of possible rows/cols is the value of the maximum possible
    # lucode we're indexing with plus 1 (to account for 1-based counting).
    n_rows = max_lucode + 1
    soil_disturbance_matrix = scipy.sparse.dok_matrix(
        (n_rows, n_rows), dtype=numpy.float32)
    biomass_disturbance_matrix = scipy.sparse.dok_matrix(
        (n_rows, n_rows), dtype=numpy.float32)
    soil_accumulation_matrix = scipy.sparse.dok_matrix(
        (n_rows, n_rows), dtype=numpy.float32)
    biomass_accumulation_matrix = scipy.sparse.dok_matrix(
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
            elif field_value == 'accum':
                soil_accumulation_matrix[from_lucode, to_lucode] = (
                    biophysical_dict[from_lucode][
                        f'soil-yearly-accumulation'])
                biomass_accumulation_matrix[from_lucode, to_lucode] = (
                    biophysical_dict[from_lucode][
                        f'biomass-yearly-accumulation'])

    return (biomass_disturbance_matrix, soil_disturbance_matrix,
            biomass_accumulation_matrix, soil_accumulation_matrix)


def _reclassify_accumulation_transition(
        landuse_transition_from_raster, landuse_transition_to_raster,
        accumulation_rate_matrix, target_raster_path):

    from_nodata = pygeoprocessing.get_raster_info(
        landuse_transition_from_raster)['nodata'][0]
    to_nodata = pygeoprocessing.get_raster_info(
        landuse_transition_to_raster)['nodata'][0]

    def _reclassify_accumulation(
            landuse_transition_from_matrix, landuse_transition_to_matrix,
            accumulation_rate_matrix):
        output_matrix = numpy.empty(landuse_transition_from_matrix.shape,
                                    dtype=numpy.float32)
        output_matrix[:] = NODATA_FLOAT32

        valid_pixels = numpy.ones(landuse_transition_from_matrix.shape,
                                  dtype=numpy.bool)
        if from_nodata is not None:
            valid_pixels &= (landuse_transition_from_matrix != from_nodata)

        if to_nodata is not None:
            valid_pixels &= (landuse_transition_to_matrix != to_nodata)

        output_matrix[valid_pixels] = accumulation_rate_matrix[
                landuse_transition_from_matrix[valid_pixels],
                landuse_transition_to_matrix[valid_pixels]].toarray().flatten()
        return output_matrix

    pygeoprocessing.raster_calculator(
        [(landuse_transition_from_raster, 1),
            (landuse_transition_to_raster, 1),
            (accumulation_rate_matrix, 'raw')],
        _reclassify_accumulation, target_raster_path, gdal.GDT_Float32,
        NODATA_FLOAT32)


def _reclassify_disturbance_transition(
        landuse_transition_from_raster, landuse_transition_to_raster,
        carbon_storage_raster, disturbance_magnitude_matrix,
        target_raster_path):
    """Calculate the volume of carbon disturbed in a transition.

    This function calculates the volume of disturbed carbon for each
    landcover transitioning from one landcover type to a disturbance type.
    The magnitude of the disturbance is in ``disturbance_magnitude_matrix`` and
    the existing carbon storage is found in ``carbon_storage_matrix``.

    The volume of carbon disturbed is calculated according to:

        carbon_disturbed = disturbance_magnitude * carbon_storage

    Args:
        landuse_transition_from_raster (string): An integer landcover
            raster representing landcover codes that we are transitioning FROM.
        landuse_transition_to_raster (string): An integer landcover
            raster representing landcover codes that we are transitioning TO.
        disturbance_magnitude_matrix (scipy.sparse.dok_matrix): A sparse matrix
            where axis 0 represents the integer landcover codes being
            transitioned from and axis 1 represents the integer landcover codes
            being transitioned to.  The values at the intersection of these
            coordinate pairs are ``numpy.float32`` values representing the
            magnitude of the disturbance in a given carbon stock during this
            transition.
        carbon_storage_raster (string): A float32 raster of
            values representing carbon storage in some pool of carbon.

    Returns:
        ``None``
    """
    from_nodata = pygeoprocessing.get_raster_info(
        landuse_transition_from_raster)['nodata'][0]
    to_nodata = pygeoprocessing.get_raster_info(
        landuse_transition_to_raster)['nodata'][0]
    storage_nodata = pygeoprocessing.get_raster_info(
        carbon_storage_raster)['nodata'][0]

    def _reclassify_disturbance(
            landuse_transition_from_matrix, landuse_transition_to_matrix,
            carbon_storage_matrix):
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
            valid_pixels &= (
                ~numpy.isclose(carbon_storage_matrix, storage_nodata))

        disturbance_magnitude = disturbance_magnitude_matrix[
            landuse_transition_from_matrix[valid_pixels],
            landuse_transition_to_matrix[valid_pixels]].toarray().flatten()

        output_matrix[valid_pixels] = (
            carbon_storage_matrix[valid_pixels] * disturbance_magnitude)
        return output_matrix

    pygeoprocessing.raster_calculator(
        [(landuse_transition_from_raster, 1),
            (landuse_transition_to_raster, 1),
            (carbon_storage_raster, 1)], _reclassify_disturbance,
        target_raster_path, gdal.GDT_Float32, NODATA_FLOAT32)


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
