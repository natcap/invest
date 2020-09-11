import itertools
import logging
import os
import pprint

import numpy
import pandas
import pygeoprocessing
import scipy.sparse
import taskgraph
from osgeo import gdal
from pygeoprocessing.symbolic import evaluate_raster_calculator_expression

from .. import utils
from .. import validation

LOGGER = logging.getLogger(__name__)


POOL_SOIL = 'soil'
POOL_BIOMASS = 'biomass'
POOL_LITTER = 'litter'
NODATA_FLOAT32 = float(numpy.finfo(numpy.float32).min)
NODATA_UINT16 = int(numpy.iinfo(numpy.uint16).max)

STOCKS_RASTER_PATTERN = 'stocks-{pool}-{year}{suffix}.tif'
ACCUMULATION_RASTER_PATTERN = 'accumulation-{pool}-{year}{suffix}.tif'
HALF_LIFE_RASTER_PATTERN = 'halflife-{pool}-{year}{suffix}.tif'
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
TOTAL_NET_SEQ_ALL_YEARS_RASTER_PATTERN = (
    'total-net-carbon-sequestration{suffix}.tif')
NET_PRESENT_VALUE_RASTER_PATTERN = 'net-present-value-at-{year}{suffix}.tif'

INTERMEDIATE_DIR_NAME = 'intermediate'
TASKGRAPH_CACHE_DIR_NAME = 'task_cache'
OUTPUT_DIR_NAME = 'output'

ARGS_SPEC = {
    "model_name": "Coastal Blue Carbon",
    "module": __name__,
    "userguide_html": "coastal_blue_carbon.html",
    "args": {
        "workspace_dir": validation.WORKSPACE_SPEC,
        "results_suffix": validation.SUFFIX_SPEC,
        "n_workers": validation.N_WORKERS_SPEC,
        "baseline_lulc_path": {
            "type": "raster",
            "required": True,
            "validation_options": {
                "projected": True,
            },
            "about": (
                "The path to a land use/land cover raster representing the "
                "baseline scenario."),
            "name": "Baseline Land-Use/Land-Cover",
        },
        "transitions_csv": {
            "validation_options": {
                "required_fields": ["transition_year", "raster_path"],
            },
            "type": "csv",
            "required": False,
            "about": (
                "A CSV table where each row represents the year and path "
                "to a raster file on disk representing the landcover raster "
                "representing the state of the landscape in that year. "
                "Landcover codes match those in the biophysical table and in "
                "the landcover transitions table."
            ),
            "name": "Transitions Table",
        },
        "baseline_lulc_year": {
            "type": "number",
            "required": True,
            "name": "Year of Baseline LULC",
            "about": (
                "The year of the Baseline LULC raster.  Must predate the "
                "the first transition year."
            ),
        },
        "analysis_year": {
            "type": "number",
            "required": True,
            "name": "Analysis Year",
            "about": (
                "An analysis year extends the transient analysis "
                "beyond the transition years."),
        },
        "biophysical_table_path": {
            "name": "Biophysical Table",
            "type": "csv",
            "required": True,
            "validation_options": {
                "required_fields": [
                    "code",
                    "lulc-class",
                    "biomass-initial",
                    "soil-initial",
                    "litter-initial",
                    "biomass-half-life",
                    "biomass-low-impact-disturb",
                    "biomass-med-impact-disturb",
                    "biomass-high-impact-disturb",
                    "biomass-yearly-accumulation",
                    "soil-half-life",
                    "soil-low-impact-disturb",
                    "soil-med-impact-disturb",
                    "soil-high-impact-disturb",
                    "soil-yearly-accumulation",
                    "litter-yearly-accumulation",
                ],
            },
            "about": (
                "A table defining initial carbon stock values, low, medium "
                "and high-impact disturbance magnitudes (values between 0-1), "
                "and accumulation rates.  Initial values and accumulation "
                "rates are defined for soil, biomass and litter. "
                "Disturbance magnitudes are defined for soil and biomass only."
            ),
        },
        "landcover_transitions_table": {
            "name": "Landcover Transitions Table",
            "type": "csv",
            "validation_options": {
                "required_fields": ['lulc-class'],
            },
            "about": (
                "A transition matrix mapping the type of carbon action "
                "undergone when one landcover type transitions to another. "
                "The first column must have the fieldname 'lulc-class', and "
                "the field values of this must match the landcover class "
                "names in the biophysical table.  The remaining column "
                "headers must also match the landcover class names in the "
                "biophysical table.  The classes on the y axis represent "
                "the class we're transitioning from, the classes on the x "
                "axis represent the classes we're transitioning to. "
                "Field values within the transition matrix must have one "
                "of the following values: 'accum', representing a state of "
                "carbon accumulation, 'high-impact-disturb', "
                "'med-impact-disturb', 'low_impact_disturb', representing "
                "appropriate states of carbon disturbance rates, or 'NCC', "
                "representing no change to carbon.  Cells may also be empty, "
                "but only if this transition never takes place. "
                "The Coastal Blue Carbon preprocessor exists to help create "
                "this table for you."
            ),
        },
        "do_economic_analysis": {
            "name": "Calculate Net Present Value of Sequestered Carbon",
            "type": "boolean",
            "required": False,
            "about": (
                "A boolean value indicating whether the model should run an "
                "economic analysis."),
        },
        "use_price_table": {
            "name": "Use Price Table",
            "type": "boolean",
            "required": False,
            "about": (
                "boolean value indicating whether a price table is included "
                "in the arguments and to be used or a price and interest rate "
                "is provided and to be used instead."),
        },
        "price": {
            "name": "Price",
            "type": "number",
            "required": "do_economic_analysis and (not use_price_table)",
            "about": "The price per Megatonne CO2e at the base year.",
        },
        "inflation_rate": {
            "name": "Interest Rate (%)",
            "type": "number",
            "required": "do_economic_analysis and (not use_price_table)",
            "about": "Annual change in the price per unit of carbon",
        },
        "price_table_path": {
            "name": "Price Table",
            "type": "csv",
            "required": "use_price_table",
            "validation_options": {
                "required_fields": ["year", "price"],
            },
            "about": (
                "Can be used in place of price and interest rate "
                "inputs.  The provided CSV table contains the price "
                "per Megatonne CO2e sequestered for a given year, for "
                "all years from the original snapshot to the analysis "
                "year, if provided."),
        },
        "discount_rate": {
            "name": "Discount Rate (%)",
            "type": "number",
            "required": "do_economic_analysis",
            "about": (
                "The discount rate on future valuations of "
                "sequestered carbon, compounded yearly."),
        },
    }
}


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
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous mode.

    taskgraph_cache_dir = os.path.join(
        args['workspace_dir'], TASKGRAPH_CACHE_DIR_NAME)
    task_graph = taskgraph.TaskGraph(
        taskgraph_cache_dir, n_workers)

    suffix = args.get('suffix', '')
    intermediate_dir = os.path.join(
        args['workspace_dir'], INTERMEDIATE_DIR_NAME)
    output_dir = os.path.join(
        args['workspace_dir'], OUTPUT_DIR_NAME)

    transition_years = set([int(year) for year in args['transition_years']])
    disturbance_magnitude_rasters = args['disturbance_magnitude_rasters']
    half_life_rasters = args['half_life_rasters']
    yearly_accum_rasters = args['annual_rate_of_accumulation_rasters']

    prices = None
    if 'carbon_prices_per_year' in args and args['carbon_prices_per_year']:
        prices = {int(year) : float(price)
                  for (year, price) in args['carbon_prices_per_year'].items()}
    discount_rate = float(args['discount_rate'])
    baseline_lulc_year = int(args['baseline_lulc_year'])

    # ASSUMPTIONS
    #
    # This rebuild assumes that this timeseries analysis is ONLY taking place
    # for the transitions at hand.  Everything that happens between the
    # baseline year and the first transition isn't all that interesting since
    # the only thing that can happen is accumulation. That can be modeled with
    # a few raster calculator operations.  Everything within this loop is way
    # more interesting and tricky to get right, hence the need for an extra
    # function.
    stock_rasters = {
        (min(transition_years) - 1): {
            POOL_SOIL: args['stocks_at_first_transition'][POOL_SOIL],
            POOL_BIOMASS: args['stocks_at_first_transition'][POOL_BIOMASS],
            POOL_LITTER: args['stocks_at_first_transition'][POOL_LITTER],
        }
    }
    net_sequestration_rasters = {
        (min(transition_years) - 1): {
            POOL_SOIL: args['annual_rate_of_accumulation_rasters'][POOL_SOIL],
            POOL_BIOMASS: args['annual_rate_of_accumulation_rasters'][POOL_BIOMASS],
            POOL_LITTER: args['annual_rate_of_accumulation_rasters'][POOL_LITTER],
        }
    }
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
    valuation_tasks = {}
    net_present_value_paths = {}
    valuation_rasters = {}

    first_transition_year = min(transition_years)
    final_year = int(args['analysis_year'])

    summary_net_sequestration_tasks = []
    summary_net_sequestration_raster_paths = {
        first_transition_year: args['sequestration_since_baseline_raster']}

    for year in range(first_transition_year, final_year+1):
        current_stock_tasks = {}
        net_sequestration_rasters[year] = {}
        stock_rasters[year] = {}
        disturbance_vol_rasters[year] = {}
        emissions_rasters[year] = {}
        year_of_disturbance_rasters[year] = {}
        valuation_tasks[year] = {}
        net_present_value_paths = {}

        for pool in (POOL_SOIL, POOL_BIOMASS):
            # Calculate stocks from last year's stock plus last year's net
            # sequestration.
            stock_rasters[year][pool] = os.path.join(
                intermediate_dir,
                STOCKS_RASTER_PATTERN.format(
                    year=year, pool=pool, suffix=suffix))
            if year == first_transition_year:
                current_stock_dependent_tasks = []
                current_disturbance_vol_dependent_tasks = []
            else:
                current_stock_dependent_tasks = [
                    prior_stock_tasks[pool],
                    prior_net_sequestration_tasks[pool]]
                current_disturbance_vol_dependent_tasks = [
                    prior_stock_tasks[pool]]

            current_stock_tasks[pool] = task_graph.add_task(
                func=_sum_n_rasters,
                args=([stock_rasters[year-1][pool],
                       net_sequestration_rasters[year-1][pool]],
                      stock_rasters[year][pool]),
                dependent_task_list=current_stock_dependent_tasks,
                target_path_list=[stock_rasters[year][pool]],
                task_name=f'Calculating {pool} carbon stock for {year}')

            # Calculate disturbance volume if we're at a transition year.
            #    TODO: provide disturbance magnitude as a raster input
            if year in transition_years:
                # We should only switch around the transition years the first
                # time we encounter this, not for each pool.
                if current_transition_year != year:
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
                              disturbance_magnitude_rasters[year][pool], 1),
                           "stocks": (stock_rasters[year-1][pool], 1)},
                          NODATA_FLOAT32,
                          disturbance_vol_rasters[year][pool]),
                    dependent_task_list=current_disturbance_vol_dependent_tasks,
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
                if year == min(transition_years):
                    prior_transition_year_raster = None
                else:
                    prior_transition_year_raster = year_of_disturbance_rasters[
                        prior_transition_year][pool]
                current_year_of_disturbance_tasks[pool] = task_graph.add_task(
                    func=_track_latest_transition_year,
                    args=(disturbance_vol_rasters[year][pool],
                          prior_transition_year_raster,
                          year,
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
                    [(disturbance_vol_rasters[current_transition_year][pool], 1),
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
                dependent_task_list=[current_emissions_tasks[pool]],
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
                   yearly_accum_rasters[current_transition_year][POOL_LITTER]],
                  total_carbon_rasters[year]),
            dependent_task_list=[
                current_stock_tasks[POOL_SOIL],
                current_stock_tasks[POOL_BIOMASS]],
            target_path_list=[total_carbon_rasters[year]],
            task_name=f'Calculating total carbon stocks in {year}')

        # Calculate valuation if we're doing valuation (requires Net Seq.)
        if ('do_economic_analysis' in args and args['do_economic_analysis']):
            # TODO: Need to verify the math on this, I'm unsure if the current
            # implementation is correct or makes sense:
            #    (N_biomass + N_soil[baseline]) * price[this year]
            valuation_rasters[year] = os.path.join(
                intermediate_dir, VALUE_RASTER_PATTERN.format(
                    year=year, suffix=suffix))
            valuation_tasks[year][pool] = task_graph.add_task(
                func=pygeoprocessing.raster_calculator,
                args=([(net_sequestration_rasters[year][POOL_BIOMASS], 1),
                       (net_sequestration_rasters[year][POOL_SOIL], 1),
                       (prices[year], 'raw')],
                      _calculate_valuation,
                      valuation_rasters[year],
                      gdal.GDT_Float32,
                      NODATA_FLOAT32),
                dependent_task_list=[
                    current_net_sequestration_tasks[POOL_BIOMASS],
                    current_net_sequestration_tasks[POOL_SOIL]],
                target_path_list=[valuation_rasters[year]],
                task_name=f'Calculating the value of carbon for {year}')

        # If in the last year before a transition of the last year before the
        # final year of the analysis (which might not be a transition):
        #  * sum emissions since last transition
        #  * sum accumulation since last transition
        #  * sum net sequestration since last transition
        if (year + 1) in transition_years or (year + 1) == final_year:
            emissions_rasters_since_transition = []
            net_seq_rasters_since_transition = []
            for _year in range(current_transition_year, year + 1):
                emissions_rasters_since_transition.extend(
                    list(emissions_rasters[_year].values()))
                net_seq_rasters_since_transition.extend(
                    list(net_sequestration_rasters[_year].values()))

            emissions_since_last_transition_raster = os.path.join(
                output_dir, EMISSIONS_SINCE_TRANSITION_RASTER_PATTERN.format(
                    start_year=current_transition_year, end_year=(year + 1),
                    suffix=suffix))
            _ = task_graph.add_task(
                func=_sum_n_rasters,
                args=(emissions_rasters_since_transition,
                      emissions_since_last_transition_raster),
                dependent_task_list=[current_emissions_tasks[pool]],
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
                     (yearly_accum_rasters[
                         current_transition_year][POOL_LITTER], 1),
                     (((year + 1) - current_transition_year), 'raw')],
                    _calculate_accumulation_over_time,
                    accumulation_since_last_transition,
                    gdal.GDT_Float32,
                    NODATA_FLOAT32),
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
            summary_net_sequestration_raster_paths[year+1] = (
                net_carbon_sequestration_since_last_transition)

            if ('do_economic_analysis' in args and args['do_economic_analysis']):
                # Sum all of the valuation rasters leading up to and including
                # the current year.
                net_present_value_raster_path = os.path.join(
                    output_dir, NET_PRESENT_VALUE_RASTER_PATTERN.format(
                        year=(year + 1), suffix=suffix))
                _ = task_graph.add_task(
                    func=_sum_n_rasters,
                    args=([args['npv_since_baseline_raster']] +
                          list(raster_path for (year, raster_path) in
                               valuation_rasters.items() if year < year+1),
                          net_present_value_raster_path),
                    dependent_task_list=list(valuation_tasks[year].values()),
                    target_path_list=[net_present_value_raster_path],
                    task_name=f'Calculating net present value up to {year}')

        # These are the few sets of tasks that we care about referring to from
        # the prior year.
        prior_stock_tasks = current_stock_tasks
        prior_net_sequestration_tasks = current_net_sequestration_tasks

    if ('do_economic_analysis' in args and args['do_economic_analysis']):
        sorted_transition_years = sorted(transition_years) + [final_year]
        for transition_year, next_transition_year in zip(
                sorted_transition_years[:-1], sorted_transition_years[1:]):
            net_present_value_raster_path = os.path.join(
                output_dir, NET_PRESENT_VALUE_RASTER_PATTERN.format(
                    year=next_transition_year, suffix=suffix))


    # Calculate total net sequestration.
    total_net_sequestration_raster_path = os.path.join(
        output_dir, TOTAL_NET_SEQ_ALL_YEARS_RASTER_PATTERN.format(
            suffix=suffix))
    _ = task_graph.add_task(
        func=_sum_n_rasters,
        args=(list(summary_net_sequestration_raster_paths.values()),
              total_net_sequestration_raster_path),
        kwargs={
            'allow_pixel_stacks_with_nodata': True,
        },
        dependent_task_list=summary_net_sequestration_tasks,
        target_path_list=[total_net_sequestration_raster_path],
        task_name=(
             'Calculate total net carbon sequestration across all years'))

    # Calculate Net Present Value
    target_npv_paths = {}
    for transition_year in (
            sorted(set(transition_years).union(set([final_year])))[1:]):
        target_npv_paths[transition_year] = (
            NET_PRESENT_VALUE_RASTER_PATTERN.format(
                year=transition_year, suffix=suffix))
    _ = task_graph.add_task(
        func=_calculate_npv,
        args=(summary_net_sequestration_raster_paths,
              prices,
              discount_rate,
              baseline_lulc_year,
              target_npv_paths),
        dependent_task_list=summary_net_sequestration_tasks,
        target_path_list=list(target_npv_paths.values()),
        task_name=(
            'Calculate total net carbon sequestration across all years'))

    task_graph.close()
    task_graph.join()


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
    baseline_lulc_year = int(args['baseline_lulc_year'])
    analysis_year = int(args['analysis_year'])

    base_paths = [args['baseline_lulc_path']]
    aligned_lulc_paths = {}
    aligned_paths = [os.path.join(
        intermediate_dir,
        f'aligned_lulc_baseline_{baseline_lulc_year}{suffix}.tif')]
    aligned_lulc_paths[baseline_lulc_year] = aligned_paths[0]
    for transition_year in transitions:
        base_paths.append(transitions[transition_year])
        transition_years.add(transition_year)
        aligned_paths.append(
            os.path.join(
                intermediate_dir,
                f'aligned_lulc_transition_{transition_year}{suffix}.tif'))
        aligned_lulc_paths[transition_year] = aligned_paths[-1]

    # TODO: check that the years in the price table match the years in the
    # range of the timesteps we're running.
    prices = None
    if ('do_economic_analysis' in args and
            args['do_economic_analysis'] not in ('', None)):
        if args.get('use_price_table', False):
            prices = {
                year: values['price'] for (year, values) in
                utils.build_lookup_from_csv(
                    args['price_table_path'], 'year').items()}
        else:
            inflation_rate = float(args['inflation_rate']) * 0.01
            annual_price = float(args['price'])

            try:
                max_year = max(transition_years).union(set([analysis_year]))
            except ValueError:
                # When transition_years is an empty sequence.
                max_year = analysis_year

            prices = {}
            for timestep_index, year in enumerate(
                    range(baseline_lulc_year, max_year + 1)):
                prices[year] = (
                    ((1 + inflation_rate) ** timestep_index) *
                    annual_price)

        discount_rate = float(args['discount_rate']) * 0.01

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

    # Baseline stocks are simply reclassified.
    # Baseline accumulation are simply reclassified
    # There are no emissions, so net sequestration is only from accumulation.
    # Value can still be calculated from the net sequestration.
    end_of_baseline_period = baseline_lulc_year
    if transition_years:
        end_of_baseline_period = min(transition_years)
    else:
        end_of_baseline_period = analysis_year

    stock_rasters = {
        baseline_lulc_year: {},
        end_of_baseline_period-1: {},
    }
    baseline_stock_tasks = {}
    baseline_accum_tasks = {}
    yearly_accum_rasters = {}
    for pool in (POOL_BIOMASS, POOL_LITTER, POOL_SOIL):
        stock_rasters[baseline_lulc_year][pool] = os.path.join(
            intermediate_dir, STOCKS_RASTER_PATTERN.format(
                pool=pool, year=baseline_lulc_year, suffix=suffix))
        pool_stock_task = task_graph.add_task(
            func=pygeoprocessing.reclassify_raster,
            args=(
                (aligned_lulc_paths[baseline_lulc_year], 1),
                {lucode: values[f'{pool}-initial'] for (lucode, values)
                    in biophysical_parameters.items()},
                stock_rasters[baseline_lulc_year][pool],
                gdal.GDT_Float32,
                NODATA_FLOAT32),
            dependent_task_list=[alignment_task],
            target_path_list=[stock_rasters[baseline_lulc_year][pool]],
            task_name=f'Mapping initial {pool} carbon stocks')

        # Initial accumulation values are a simple reclassification
        # rather than a mapping by the transition.
        yearly_accum_rasters[pool] = os.path.join(
            intermediate_dir, ACCUMULATION_RASTER_PATTERN.format(
                pool=pool, year=baseline_lulc_year, suffix=suffix))
        baseline_accum_tasks[pool] = task_graph.add_task(
            func=pygeoprocessing.reclassify_raster,
            args=(
                (aligned_lulc_paths[baseline_lulc_year], 1),
                {lucode: values[f'{pool}-yearly-accumulation']
                    for (lucode, values)
                    in biophysical_parameters.items()},
                yearly_accum_rasters[pool],
                gdal.GDT_Float32,
                NODATA_FLOAT32),
            dependent_task_list=[alignment_task],
            target_path_list=[yearly_accum_rasters[pool]],
            task_name=(
                f'Mapping {pool} carbon accumulation for {baseline_lulc_year}'))

        if end_of_baseline_period != baseline_lulc_year:
            # The total stocks between baseline and the first year of interest is
            # just a sum-and-multiply for each pool.
            stock_rasters[end_of_baseline_period-1][pool] = os.path.join(
                STOCKS_RASTER_PATTERN.format(
                    pool=pool, year=end_of_baseline_period-1, suffix=suffix))
            baseline_stock_tasks[pool] = task_graph.add_task(
                func=_calculate_accumulation_from_baseline,
                args=(stock_rasters[baseline_lulc_year][pool],
                      yearly_accum_rasters[pool],
                      (end_of_baseline_period - baseline_lulc_year),
                      stock_rasters[end_of_baseline_period-1][pool]),
                dependent_task_list=[
                    baseline_accum_tasks[pool], pool_stock_task],
                target_path_list=[
                    stock_rasters[end_of_baseline_period-1][pool]],
                task_name=(
                    f'Calculating {pool} stocks before the first transition or '
                    'the analysis year'))

    total_net_sequestration_for_baseline_period = (
        os.path.join(
            output_dir, TOTAL_NET_SEQ_SINCE_TRANSITION_RASTER_PATTERN.format(
                start_year=baseline_lulc_year, end_year=end_of_baseline_period,
                suffix=suffix)))
    baseline_net_seq_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(yearly_accum_rasters[POOL_BIOMASS], 1),
               (yearly_accum_rasters[POOL_SOIL], 1),
               (yearly_accum_rasters[POOL_LITTER], 1),
               (end_of_baseline_period - baseline_lulc_year, 'raw')],
              _calculate_accumulation_over_time,
              total_net_sequestration_for_baseline_period,
              gdal.GDT_Float32,
              NODATA_FLOAT32),
        target_path_list=[total_net_sequestration_for_baseline_period],
        task_name=(
            f'Calculate accumulation between baseline year and final year'))

    # Reclassify transitions appropriately for each transition year.
    halflife_rasters = {}
    disturbance_magnitude_rasters = {}
    prior_transition_year = baseline_lulc_year
    for current_transition_year in sorted(transitions):
        yearly_accum_rasters[current_transition_year] = {}
        halflife_rasters[current_transition_year] = {}
        disturbance_magnitude_rasters[current_transition_year] = {}

        for pool in (POOL_BIOMASS, POOL_SOIL):
            # When carbon is emitted after a transition year, its halflife
            # actually comes from the carbon stores from the prior transition.
            # If Mangroves transition to a parking lot, we use the half-life of
            # the stored carbon from the mangroves.
            halflife_rasters[current_transition_year][pool] = os.path.join(
                intermediate_dir, HALF_LIFE_RASTER_PATTERN.format(
                    pool=pool, year=current_transition_year, suffix=suffix))
            _ = task_graph.add_task(
                func=pygeoprocessing.reclassify_raster,
                args=(
                    (aligned_lulc_paths[prior_transition_year], 1),
                    {lucode: values[f'{pool}-half-life']
                        for (lucode, values)
                        in biophysical_parameters.items()},
                    halflife_rasters[current_transition_year][pool],
                    gdal.GDT_Float32,
                    NODATA_FLOAT32),
                dependent_task_list=[alignment_task],
                target_path_list=[
                    halflife_rasters[current_transition_year][pool]],
                task_name=(
                    f'Mapping {pool} half-life for {current_transition_year}'))

            # Soil and biomass pools will only accumulate if the transition
            # table for this transition specifies accumulation.  We
            # can't assume that this will match a basic reclassification.
            yearly_accum_rasters[current_transition_year][pool] = os.path.join(
                intermediate_dir, ACCUMULATION_RASTER_PATTERN.format(
                    pool=pool, year=current_transition_year, suffix=suffix))
            _ = task_graph.add_task(
                func=_reclassify_accumulation_transition,
                args=(aligned_lulc_paths[prior_transition_year],
                      aligned_lulc_paths[current_transition_year],
                      accumulation_matrices[pool],
                      yearly_accum_rasters[current_transition_year][pool]),
                dependent_task_list=[alignment_task],
                target_path_list=[
                    yearly_accum_rasters[current_transition_year][pool]],
                task_name=(
                    f'Mapping {pool} carbon accumulation for '
                    f'{current_transition_year}'))

            disturbance_magnitude_rasters[
                current_transition_year][pool] = os.path.join(
                    intermediate_dir, DISTURBANCE_MAGNITUDE_RASTER_PATTERN.format(
                        pool=pool, year=current_transition_year,
                        suffix=suffix))
            # this is _actually_ the magnitude, not the magnitude multiplied by
            # the stocks.
            disturbance_magnitude_task = task_graph.add_task(
                func=_reclassify_disturbance_magnitude,
                args=(aligned_lulc_paths[prior_transition_year],
                      aligned_lulc_paths[current_transition_year],
                      disturbance_matrices[pool],
                      disturbance_magnitude_rasters[current_transition_year][pool]),
                dependent_task_list=[alignment_task],
                target_path_list=[
                    disturbance_magnitude_rasters[current_transition_year][pool]],
                task_name=(
                    f'map {pool} carbon disturbance {prior_transition_year} '
                    f'to {current_transition_year}'))

        # Litter accumulation is a simple reclassification because it really
        # isn't affected by transitions as soil and biomass carbon are.
        yearly_accum_rasters[
            current_transition_year][POOL_LITTER] = os.path.join(
            intermediate_dir, ACCUMULATION_RASTER_PATTERN.format(
                pool=POOL_LITTER, year=current_transition_year, suffix=suffix))
        _ = task_graph.add_task(
            func=pygeoprocessing.reclassify_raster,
            args=((aligned_lulc_paths[current_transition_year], 1),
                  {lucode: values[f'{POOL_LITTER}-yearly-accumulation']
                   for (lucode, values) in
                   biophysical_parameters.items()},
                  yearly_accum_rasters[current_transition_year][POOL_LITTER],
                  gdal.GDT_Float32,
                  NODATA_FLOAT32),
            dependent_task_list=[alignment_task],
            target_path_list=[
                yearly_accum_rasters[current_transition_year][pool]],
            task_name=(
                f'Mapping litter accumulation for {current_transition_year}'))

        prior_transition_year = current_transition_year

    transition_analysis_args = {
        'task_graph': task_graph,
        'workspace_dir': args['workspace_dir'],
        'suffix': suffix,
        'n_workers': n_workers,
        'transition_years': transition_years,
        'disturbance_magnitude_rasters': disturbance_magnitude_rasters,
        'half_life_rasters': halflife_rasters,
        'annual_rate_of_accumulation_rasters': yearly_accum_rasters,
        'carbon_prices_per_year': prices,
        'discount_rate': discount_rate,
        'analysis_year': analysis_year,
        'do_economic_analysis': args.get('do_economic_analysis', False),
        'baseline_lulc_raster': aligned_lulc_paths[baseline_lulc_year],
        'baseline_lulc_year': baseline_lulc_year,
        'sequestration_since_baseline_raster': (
            total_net_sequestration_for_baseline_period),
        'stocks_at_first_transition': {
            POOL_SOIL: stock_rasters[end_of_baseline_period-1][POOL_SOIL],
            POOL_BIOMASS: stock_rasters[
                end_of_baseline_period-1][POOL_BIOMASS],
            POOL_LITTER: stock_rasters[end_of_baseline_period-1][POOL_LITTER],
        }
    }

    if ('do_economic_analysis' in args and args['do_economic_analysis']):
        baseline_period_npv_raster = os.path.join(
            output_dir, NET_PRESENT_VALUE_RASTER_PATTERN.format(
                year=end_of_baseline_period, suffix=suffix))
        _ = task_graph.add_task(
            func=_calculate_npv,
            args=({end_of_baseline_period:
                   total_net_sequestration_for_baseline_period},
                  prices,
                  discount_rate,
                  baseline_lulc_year,
                  {end_of_baseline_period: baseline_period_npv_raster}),
            dependent_task_list=[baseline_net_seq_task],
            target_path_list=[baseline_period_npv_raster],
            task_name='baseline period NPV')
        transition_analysis_args['npv_since_baseline_raster'] = baseline_period_npv_raster

    task_graph.join()
    if transitions:
        execute_transition_analysis(transition_analysis_args)

    task_graph.close()
    task_graph.join()


def _calculate_npv(
        net_sequestration_rasters, prices_by_year, discount_rate, baseline_year,
        target_raster_years_and_paths):
    # net_sequestration_rasters = {end_of_transition_year: summary seq. path
    # target_raster_years_and_paths = does not include baseline period raster

    source_raster_path = list(net_sequestration_rasters.values())[0]

    for target_raster_year, target_raster_path in (
            target_raster_years_and_paths.items()):

        # TODO: Refactor this loop to cache prior valuation factors.
        # Skip refactor if it adds too much complexity.
        valuation_factor = 0
        for years_since_baseline, year in enumerate(
                range(baseline_year, target_raster_year)):
            valuation_factor += (
                prices_by_year[year] / (
                    (1 + discount_rate) ** years_since_baseline))

        nodata = pygeoprocessing.get_raster_info(
            source_raster_path)['nodata'][0]

        def _npv(net_sequestration_matrix):
            npv = numpy.empty(net_sequestration_matrix.shape,
                              dtype=numpy.float32)
            npv[:] = NODATA_FLOAT32
            valid_pixels = ~numpy.isclose(net_sequestration_matrix,
                                          nodata)
            npv[valid_pixels] = (
                net_sequestration_matrix[valid_pixels] * valuation_factor)
            return npv

        pygeoprocessing.raster_calculator(
            [(net_sequestration_rasters[target_raster_year], 1)],
            _npv, target_raster_path, gdal.GDT_Float32, NODATA_FLOAT32)


def _calculate_accumulation_from_baseline(
        baseline_stock_raster_path, yearly_accumulation_raster_path, n_years,
        target_raster_path):
    # Both of these values are assumed to be defined from earlier in the
    # model's execution.
    baseline_nodata = pygeoprocessing.get_raster_info(
        baseline_stock_raster_path)['nodata'][0]
    accum_nodata = pygeoprocessing.get_raster_info(
        yearly_accumulation_raster_path)['nodata'][0]

    def _calculate_accumulation_over_years(baseline_matrix, accum_matrix):
        target_matrix = numpy.empty(baseline_matrix.shape, dtype=numpy.float32)
        target_matrix[:] = NODATA_FLOAT32

        valid_pixels = (~numpy.isclose(baseline_matrix, baseline_nodata) &
                       (~numpy.isclose(accum_matrix, accum_nodata)))

        target_matrix[valid_pixels] = (
            baseline_matrix[valid_pixels] + (
                accum_matrix[valid_pixels] * n_years))

        return target_matrix

    pygeoprocessing.raster_calculator(
        [(baseline_stock_raster_path, 1),
         (yearly_accumulation_raster_path, 1)],
        _calculate_accumulation_over_years, target_raster_path,
        gdal.GDT_Float32, NODATA_FLOAT32)


def _calculate_accumulation_over_time(
        annual_biomass_matrix, annual_soil_matrix,
        annual_litter_matrix, n_years):
    target_matrix = numpy.empty(annual_biomass_matrix.shape,
                                dtype=numpy.float32)
    target_matrix[:] = NODATA_FLOAT32

    valid_pixels = (
        ~numpy.isclose(annual_biomass_matrix, NODATA_FLOAT32) &
        ~numpy.isclose(annual_soil_matrix, NODATA_FLOAT32) &
        ~numpy.isclose(annual_litter_matrix, NODATA_FLOAT32))

    target_matrix[valid_pixels] = (
        (annual_biomass_matrix[valid_pixels] +
            annual_soil_matrix[valid_pixels] +
            annual_litter_matrix[valid_pixels]) * n_years)
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
        current_transition_year,
        target_path):
    """Track the year of latest disturbance in a raster.

    Args:
        current_disturbance_vol_raster_path (string): The path to a raster on
            disk representing the volume of carbon disturbed in the most recent
            transition.  This raster must be a 32-bit floating-point raster.
        known_transition_years_raster_path (string or None): If a string, the
            path to the most recent raster of known transition years.  If
            ``None``, we assume that this is the first year for which
            transitions are being tracked.  This raster must be an unsigned
            16-bit integer raster.
        current_transition_year (int): The year of the transition that we are
            tracking.
        target_path (string): The path to a raster on disk where where the year
            of the latest transition values will be tracked.

    Returns:
        ``None``.

    """
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
        """Raster_calculator op for tracking the latest transition year."""
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
    """Calculate net sequestration for a given timestep and pool.

    Sequestration is the per-pixel tallying of carbon accumulated or
    emitted in this timestep and for this pool.  This model assume that a given
    pixel may be either in a state of accumulation or a state of emissions, but
    not both.  It is possible to have a state where a pixel might have both,
    since a user is able to provide rasters mapping the spatial distribution of
    carbon accumulated for a transition year.  If this happens, we assume that
    emissions takes precedence.

    Args:
        accumulation_raster_path (string): A string path to a raster located on
            disk.  Pixel values represent the annual rate of accumulation for
            this carbon pool.
        emissions_raster_path (string): A string path to a raster located on
            disk.  Pixel values represent the volume of this pool's carbon
            emitted in this timestep.
        target_raster_path (string): A string path to where the target raster
            should be written.

    Returns:
        ``None``.

    """
    accumulation_nodata = pygeoprocessing.get_raster_info(
        accumulation_raster_path)['nodata'][0]
    emissions_nodata = pygeoprocessing.get_raster_info(
        emissions_raster_path)['nodata'][0]

    def _record_sequestration(accumulation_matrix, emissions_matrix):
        """Given accumulation and emissions, calculate sequestration."""
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

        valid_emissions_pixels = ~numpy.isclose(emissions_matrix, 0.0)
        if emissions_nodata is not None:
            valid_emissions_pixels &= (
                ~numpy.isclose(emissions_matrix, emissions_nodata))

        target_matrix[valid_emissions_pixels] = emissions_matrix[
            valid_emissions_pixels] * -1

        invalid_pixels = ~(valid_accumulation_pixels | valid_emissions_pixels)
        target_matrix[invalid_pixels] = NODATA_FLOAT32
        return target_matrix

    pygeoprocessing.raster_calculator(
        [(accumulation_raster_path, 1), (emissions_raster_path, 1)],
        _record_sequestration, target_raster_path, gdal.GDT_Float32,
        NODATA_FLOAT32)


def _calculate_emissions(
        carbon_disturbed_matrix, year_of_last_disturbance_matrix,
        carbon_half_life_matrix, current_year):
    """Calculate emissions.

    Args:
        carbon_disturbed_matrix (numpy.array): The volume of carbon disturbed in
            the most recent transition year as time approaches infinity.
        year_of_last_disturbance_matrix (numpy.array): A matrix indicating the
            integer years of the most recent disturbance.
        carbon_half_life_matrux (numpy.array): A matrix indicating the spatial
            distribution of half-lives for this carbon pool.
        current_year (int): The current year for this timestep.

    Returns:
        A numpy array with the calculated emissions.
    """
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

    # Emissions happen immediately.
    # This means that if the transition happens in year 2020, the emissions in
    # 2020 will be that of the first year's worth of emissions.
    # Think of this as though the transition happens instantaneously and
    # completely at 12:01am on Jan. 1, 2020.  The year 2020 will have 1 full
    # year of emissions.
    n_years_elapsed = (
        current_year - year_of_last_disturbance_matrix[valid_pixels]) + 1

    valid_half_life_pixels = carbon_half_life_matrix[valid_pixels]

    emissions_matrix[valid_pixels] = (
        carbon_disturbed_matrix[valid_pixels] * (
            0.5**((n_years_elapsed-1) / valid_half_life_pixels) -
            0.5**(n_years_elapsed / valid_half_life_pixels)))

    # See note above about a half-life of 0.0 representing no emissions.
    emissions_matrix[zero_half_life] = 0.0

    return emissions_matrix


def _sum_n_rasters(raster_path_list, target_raster_path,
        allow_pixel_stacks_with_nodata=False):
    """Sum an arbitrarily-large list of rasters in a memory-efficient manner.

    Args:
        raster_path_list (list of strings): A list of string paths to rasters
            on disk.  All rasters in this list are assumed to be in the same
            projected coordinate system and to have identical dimensions.
        target_raster_path (string): The path to a raster on disk where the
            sum of rasters in ``raster_path_list`` will be stored.
        allow_pixel_stacks_with_nodata=False (bool): Whether to tolerate pixel
            stacks that contain nodata.  If ``True``, then the value of the sum
            of a given pixel stack will simply not include the numeric value of
            nodata for those pixels that match nodata.  If ``False``, the
            entire pixel stack will be excluded if any pixels in the stack are
            nodata.

    Returns:
        ``None``.

    """
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
    """Read a transition CSV table in to a series of sparse matrices.

    Args:
        transition_csv_path (string): The path to the transition CSV on disk.
            This CSV indicates the carbon actions taking place on the landscape
            when a landcover transitions from one landcover (the y axis on the
            table, under the column ``'lulc-class'``) to another landcover (the
            x axis, in the column headings).  Valid cell values may be one of:

                * ``'NCC'`` representing no change in carbon
                * ``'accum'`` representing a state of accumulation
                * ``'low-impact-disturb'`` indicating a low-impact disturbance
                * ``'medium-impact-disturb'`` indicating a medium-impact disturbance
                * ``'high-impact-disturb'`` indicating a high-impact disturbance
                * ``''`` (blank), which is equivalent to no carbon change.o
        biophysical_dict (dict): A ``dict`` mapping of integer landcover codes
            to biophysical values for disturbance and accumulation values for
            soil and biomass carbon pools.

    Returns:
        Four ``scipy.sparse.dok_matrix`` objects of type ``numpy.float32`` are
        returned:

            1. ``biomass_disturbance_matrix`` where nonzero floating-point
                values represent the disturbance magnitudes for the biomass
                carbon pool for this transition.
            2. ``soil_disturbance_matrix`` where nonzero floating-point
                values represent the disturbance magnitudes for the soil
                carbon pool for this transition.
            3. ``biomass_accumulation_matrix`` where nonzero floating-point
                values represent the rates of accumulation in the biomass
                carbon pool for this transition.
            4. ``soil_accumulation_matrix`` where nonzero floating-point values
                represent the rates of accumulation in the soil carbon pool for
                this transition.
    """
    table = utils.read_csv_to_dataframe(transition_csv_path, index_col=False)

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

    # I don't actually know if this uses less memory than the dict-based
    # approach we had before since that, too, was basically sparse.
    # At least this approach allows numpy-style indexing for easier 2D
    # reclassifications.
    # If we really wanted to save memory, we wouldn't duplicate the float32
    # values here and instead use the transitions to index into the various
    # biophysical values when reclassifying. That way we rely on python's
    # assumption that ints<2000 or so are singletons and thus use less memory.
    # Even so, the RIGHT way to do this is to have the user provide their own
    # maps of the spatial values per transition to the timeseries analysis
    # function.
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

            # When transition is a disturbance, we use the source landcover's
            # disturbance values.
            if field_value.endswith('disturb'):
                soil_disturbance_matrix[from_lucode, to_lucode] = (
                    biophysical_dict[from_lucode][f'soil-{field_value}'])
                biomass_disturbance_matrix[from_lucode, to_lucode] = (
                    biophysical_dict[from_lucode][f'biomass-{field_value}'])

            # When we're transitioning to a landcover that accumulates, use the
            # target landcover's accumulation value.
            elif field_value == 'accum':
                soil_accumulation_matrix[from_lucode, to_lucode] = (
                    biophysical_dict[to_lucode][
                        f'soil-yearly-accumulation'])
                biomass_accumulation_matrix[from_lucode, to_lucode] = (
                    biophysical_dict[to_lucode][
                        f'biomass-yearly-accumulation'])

    return (biomass_disturbance_matrix, soil_disturbance_matrix,
            biomass_accumulation_matrix, soil_accumulation_matrix)


def _reclassify_accumulation_transition(
        landuse_transition_from_raster, landuse_transition_to_raster,
        accumulation_rate_matrix, target_raster_path):
    """Determine rates of accumulation after a landcover transition.

    This function takes two landcover rasters and determines the rate of
    accumulation that will be taking place upon completion of the transition.
    Rates of accumulation are provided in ``accumulation_rate_matrix``, and the
    results are written to a raster at ``target_raster_path``.

    Args:
        landuse_transition_from_raster (string): An integer landcover raster
            representing landcover codes that we are transitioning FROM.
        landuse_transition_to_raster (string): An integer landcover raster
            representing landcover codes that we are transitioning TO.
        accumulation_rate_matrix (scipy.sparse.dok_matrix): A sparse matrix
            where axis 0 represents the integer landcover codes being
            transitioned from and axis 1 represents the integer landcover codes
            being transitioned to.  The values at the intersection of these
            coordinate pairs are ``numpy.float32`` values representing the
            magnitude of the disturbance in a given carbon stock during this
            transition.
        target_raster_path (string): The path to where the output raster should
            be stored on disk.

    Returns:
        ``None``.
    """
    from_nodata = pygeoprocessing.get_raster_info(
        landuse_transition_from_raster)['nodata'][0]
    to_nodata = pygeoprocessing.get_raster_info(
        landuse_transition_to_raster)['nodata'][0]

    def _reclassify_accumulation(
            landuse_transition_from_matrix, landuse_transition_to_matrix,
            accumulation_rate_matrix):
        """Pygeoprocessing op to reclassify accumulation."""
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


def _reclassify_disturbance_magnitude(
        landuse_transition_from_raster, landuse_transition_to_raster,
        disturbance_magnitude_matrix, target_raster_path):
    """Calculate the magnitude of carbon disturbed in a transition.

    This function calculates the magnitude of disturbed carbon for each
    landcover transitioning from one landcover type to a disturbance type,
    writing the output to a raster via ``raster_calculator``.
    The magnitude of the disturbance is in ``disturbance_magnitude_matrix``.

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
        target_raster_path (string): The path to where the output raster should
            be stored on disk.

    Returns:
        ``None``

    """
    from_nodata = pygeoprocessing.get_raster_info(
        landuse_transition_from_raster)['nodata'][0]
    to_nodata = pygeoprocessing.get_raster_info(
        landuse_transition_to_raster)['nodata'][0]

    def _reclassify_disturbance(
            landuse_transition_from_matrix, landuse_transition_to_matrix):
        """Pygeoprocessing op to reclassify disturbances."""
        output_matrix = numpy.empty(landuse_transition_from_matrix.shape,
                                    dtype=numpy.float32)
        output_matrix[:] = NODATA_FLOAT32

        valid_pixels = numpy.ones(landuse_transition_from_matrix.shape,
                                  dtype=numpy.bool)
        if from_nodata is not None:
            valid_pixels &= (landuse_transition_from_matrix != from_nodata)

        if to_nodata is not None:
            valid_pixels &= (landuse_transition_to_matrix != to_nodata)

        disturbance_magnitude = disturbance_magnitude_matrix[
            landuse_transition_from_matrix[valid_pixels],
            landuse_transition_to_matrix[valid_pixels]].toarray().flatten()

        output_matrix[valid_pixels] = disturbance_magnitude
        return output_matrix

    pygeoprocessing.raster_calculator(
        [(landuse_transition_from_raster, 1),
            (landuse_transition_to_raster, 1)], _reclassify_disturbance,
        target_raster_path, gdal.GDT_Float32, NODATA_FLOAT32)


def _extract_transitions_from_table(csv_path):
    """Extract the year/raster transition mapping from a CSV.

    No validation is performed on the years or raster paths.

    Parameters:
        csv_path (string): The path to a CSV on disk containing transition
            years and a corresponding transition raster path.  Transition years
            may be in any order in the CSV, but must be integers and no two
            years may be the same.  Transition raster paths must refer to a
            raster file located on disk representing the landcover at that
            transition.  If the path is absolute, the path will be used as
            given.  If the path is relative, the path will be interpreted as
            relative to the parent directory of this CSV file.

    Returns:
        A ``dict`` mapping int transition years to their corresponding raster
        paths.  These raster paths will be absolute paths.

    """
    table = utils.read_csv_to_dataframe(csv_path, index_col=False)
    table.columns = table.columns.str.lower()

    output_dict = {}
    table.set_index('transition_year', drop=False, inplace=True)
    for index, row in table.iterrows():
        raster_path = row['raster_path']
        if not os.path.isabs(raster_path):
            raster_path = os.path.relpath(
                raster_path, os.path.dirname(csv_path))
        output_dict[int(index)] = raster_path

    return output_dict


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate an input dictionary for Coastal Blue Carbon.

    Parameters:
        args (dict): The args dictionary.
        limit_to=None (str or None): If a string key, only this args parameter
            will be validated.  If ``None``, all args parameters will be
            validated.

    Returns:
        A list of tuples where tuple[0] is an iterable of keys that the error
        message applies to and tuple[1] is the string validation warning.
    """
    validation_warnings = validation.validate(
        args, ARGS_SPEC['args'])

    sufficient_keys = validation.get_sufficient_keys(args)
    invalid_keys = validation.get_invalid_keys(validation_warnings)

    if ("transitions_csv" not in invalid_keys and
            "transitions_csv" in sufficient_keys and
            "baseline_lulc_year" not in invalid_keys and
            "baseline_lulc_year" in sufficient_keys):
        transitions = _extract_transitions_from_table(args['transitions_csv'])

        for transition_year, transition_raster_path in transitions.items():
            raster_error_message = validation.check_raster(
                transition_raster_path)
            if raster_error_message:
                validation_warnings.append(
                    (['transitions_csv'], (
                        f"Transition raster for {transition_year} could not "
                        f"be validated: {raster_error_message}")))

        if min(set(transitions.keys())) < int(args['baseline_lulc_year']):
            validation_warnings.append(
                (['transitions_csv'], ("Transition years must predate the "
                                       "baseline lulc year.")))

        if ("analysis_year" not in invalid_keys
                and "analysis_year" in sufficient_keys):
            if max(set(transitions.keys())) > int(args['analysis_year']):
                validation_warnings.append(
                    (['transitions_csv'], ("Transition years must be <= the "
                                           "analysis year")))

    return validation_warnings
