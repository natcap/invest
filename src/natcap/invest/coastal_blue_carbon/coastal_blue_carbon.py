# -*- coding: utf-8 -*-
"""InVEST Coastal Blue Carbon: Main Model.

Implementation Notes
--------------------

Comparison with the Prior Implementation
========================================

This model is a timeseries analysis, where we iterate through each year in the
timeseries and track the state of carbon over time: the state of carbon
stocks, net sequestration, rates of accumulation, carbon emitted, etc, all for
each relevant carbon pool.  A lot of files are produced in both the
intermediate and output directories in the target workspace.

While some of these operations could be summarized per transition (which would
result in fewer files produced), this implementation avoids such summary.  This
is because the implementation that this file replaces did just that, and there
were a number of issues that arose from that implementation:

    1. **Unbounded memory consumption**
       This model has no upper bound on how many years the timeseries can
       operate over. So even if we were to have each task operating on only
       those years between transitions, even that number could be very large
       and exhaust available memory. We had many issues with memory usage in
       the prior model, and iterating over lots of files is the safest
       general-case solution for a timeseries model with an unbounded number of
       years.
    2. **In practice, the intermediate rasters are useful for analyses**
       The prior implementation wrote as few rasters out as it could, which
       ultimately ended up making it harder for some of our partner
       organizations to to the science they needed. By providing all these
       intermediate rasters, we allow our users to do their own analytics and
       summaries on how carbon stocks change over time, which would have been
       game-changing to our colleagues in analyses past.
    3. **Debugging multidimensional arrays is time-consuming and tricky**
       The prior implementation used 3D arrays (m x n x n_years) for its
       analysis, which was more concise to implement but so incredibly
       difficult to debug when there were issues. If we were to only output a
       few summary rasters per transition year, we would still have these 3D
       arrays, and debugging would be much trickier. Admittedly, there's
       probably a way to make all this much easier to debug than the prior
       implementation, but even if we were to do so, we'd still have the
       unbounded-array-size problem, mentioned in point 1.

The implementation as stated plays it safe, writing out individual rasters per
component per timestep.  There's a lot of bookkeeping as a result, but the
final model is consistent with development standards, correct (according to the
model's specification in the InVEST User's guide) and will be easier to
maintain and debug as issues arise.

"Advanced" Mode
===============

The Coastal Blue Carbon model as specified in the User's Guide is tied to the
landcover classifications provided by the user, and only offers an interesting
perspective on carbon stored when a landcover class transitions to another
landcover class.  This approach seriously hampers the ability to model spatial
carbon distributions.

In response to this shortcoming, the model now offers an 'advanced' entrypoint,
where a user can provide their own maps of spatial parameters, such as rates of
accumulation, half-lives of carbon, and other parameters for finer-grained
control.  See the docstring for ``execute_transition_analysis`` for more
information.

Units of Inputs and Outputs
===========================

The units of carbon in the model's inputs and outputs are, across the board,
density of CO2-equivalent per hectare.  Megatonnes of CO2-equivalent per Ha are
the units specified in the user's guide, but any such density could be used as
no conversion of units takes place in the model.

There has been some conversation within the Natural Capital Project staff of
whether to convert this carbon density per hectare to carbon density per pixel,
for consistency with the rest of InVEST (which often displays metrics per
pixel), and also for easier aggregation of spatial metrics such as total carbon
sequestered in the landscape.  The carbon density per hectare has been retained
here for several reasons:

    1. Many carbon scientists, including on the NatCap staff, prefer to work
       directly with carbon densities.
    2. Converting from CO2E/Ha to CO2E/pixel is an easy calculation should the
       conversion be desired, and can be done via a raster calculator call in
       any GIS software.
    3. Using rates of CO2E/Ha allows for the model to operate on landscapes
       where pixels might be non-square, or may vary in size over space such as
       in lat/long coordinate systems.  In such systems, converting to units
       per pixel would be nontrivial due to the area of a pixel varying across
       the raster.  Thus, using rates per hectare enables the model to be used
       across very large areas without modification.
"""
import logging
import os

import numpy
import pygeoprocessing
import scipy.sparse
import taskgraph
from osgeo import gdal

from .. import utils
from .. import validation

LOGGER = logging.getLogger(__name__)


POOL_SOIL = 'soil'
POOL_BIOMASS = 'biomass'
POOL_LITTER = 'litter'
NODATA_FLOAT32 = float(numpy.finfo(numpy.float32).min)
NODATA_UINT16 = int(numpy.iinfo(numpy.uint16).max)

# Rasters written to the intermediate directory
STOCKS_RASTER_PATTERN = 'stocks-{pool}-{year}{suffix}.tif'
ACCUMULATION_RASTER_PATTERN = 'accumulation-{pool}-{year}{suffix}.tif'
HALF_LIFE_RASTER_PATTERN = 'halflife-{pool}-{year}{suffix}.tif'
DISTURBANCE_VOL_RASTER_PATTERN = 'disturbance-volume-{pool}-{year}{suffix}.tif'
DISTURBANCE_MAGNITUDE_RASTER_PATTERN = (
    'disturbance-magnitude-{pool}-{year}{suffix}.tif')
EMISSIONS_RASTER_PATTERN = 'emissions-{pool}-{year}{suffix}.tif'
YEAR_OF_DIST_RASTER_PATTERN = (
    'year-of-latest-disturbance-{pool}-{year}{suffix}.tif')
ALIGNED_LULC_RASTER_PATTERN = (
    'aligned-lulc-{snapshot_type}-{year}{suffix}.tif')
NET_SEQUESTRATION_RASTER_PATTERN = (
    'net-sequestration-{pool}-{year}{suffix}.tif')
TOTAL_STOCKS_RASTER_PATTERN = 'total-carbon-stocks-{year}{suffix}.tif'

# Rasters written to the output directory
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
        "landcover_snapshot_csv": {
            "validation_options": {
                "required_fields": ["snapshot_year", "raster_path"],
            },
            "type": "csv",
            "required": False,
            "about": (
                "A CSV table where each row represents the year and path "
                "to a raster file on disk representing the landcover raster "
                "representing the state of the landscape in that year. "
                "Landcover codes match those in the biophysical table and in "
                "the landcover transitions table. All rasters represented in "
                "this table must be linearly projected in meters."
            ),
            "name": "Landcover Snapshots Table",
        },
        "analysis_year": {
            "type": "number",
            "required": False,
            "name": "Analysis Year",
            "about": (
                "An analysis year extends the transient analysis "
                "beyond the transition years. If not provided, the "
                "analysis will halt at the final transition year."
            ),
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
            "about": (
                "Annual change in the price per unit of carbon. A value of "
                "5 would represent a 5% inflation rate."),
        },
        "price_table_path": {
            "name": "Price Table",
            "type": "csv",
            "required": "use_price_table",
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
                "sequestered carbon, compounded yearly.  A "
                "value of 5 would represent a 5% discount, and -10 "
                "would represent a -10% discount."),
        },
    }
}


def execute(args):
    """Model Coastal Blue Carbon over a time series.

    Args:
        args['workspace_dir'] (string): the path to a workspace directory where
            outputs should be written.
        args['results_suffix'] (string): (optional) If provided, a string
            suffix that will be added to each output filename.
        args['n_workers'] (int): (optional) If provided, the number of workers
            to pass to ``taskgraph``.
        args['landcover_snapshot_csv'] (string): The path to a transitions
            CSV table containing transition years and the LULC rasters
            representing that year. Required for transition analysis.
        args['analysis_year'] (int): the year of the final analysis.
        args['do_economic_analysis'] (bool): Whether to do valuation.
        args['use_price_table'] (bool): Whether to use a table of annual carbon
            prices for valuation.  Defaults to ``False``.
        args['price_table_path'] (string): The path to a table of prices to use
            for valuation.  Required if ``args['use_price_table']`` is
            ``True``.
        args['inflation_rate'] (number): The rate of inflation.  The number
            provided is multiplied by ``0.01`` to compute the actual rate of
            inflation.  Required if ``args['use_price_table']`` is ``False``.
        args['price'] (number): The carbon price.  Required if
            ``args['use_price_table']`` is ``False``.
        args['discount_rate'] (number): The discount rate.  The number provided
            is multiplied by ``0.01`` to compute the actual discount rate.
            Required if ``args['do_economic_analysis']``.
        args['biophysical_table_path'] (string): The path to the biophysical
            table on disk.  This table has many required columns.  See
            ``ARGS_SPEC`` for the required columns.
        args['landcover_transitions_table'] (string): The path to the landcover
            transitions table, indicating the behavior of carbon when the
            landscape undergoes a transition.

    Returns:
        ``None``.

    """
    task_graph, n_workers, intermediate_dir, output_dir, suffix = (
        _set_up_workspace(args))

    snapshots = _extract_snapshots_from_table(args['landcover_snapshot_csv'])

    # Phase 1: alignment and preparation of inputs
    baseline_lulc_year = min(snapshots.keys())
    baseline_lulc_path = snapshots[baseline_lulc_year]
    baseline_lulc_info = pygeoprocessing.get_raster_info(
        baseline_lulc_path)
    min_pixel_size = numpy.min(numpy.abs(baseline_lulc_info['pixel_size']))
    target_pixel_size = (min_pixel_size, -min_pixel_size)

    try:
        analysis_year = int(args['analysis_year'])
    except (KeyError, TypeError, ValueError):
        # KeyError when not present in args
        # ValueError when an empty string.
        # TypeError when is None.
        analysis_year = max(snapshots.keys())

    # We're assuming that the LULC initial variables and the carbon pool
    # transient table are combined into a single lookup table.
    biophysical_parameters = utils.build_lookup_from_csv(
        args['biophysical_table_path'], 'code')

    # LULC Classnames are critical to the transition mapping, so they must be
    # unique.  This check is here in ``execute`` because it's possible that
    # someone might have a LOT of classes in their biophysical table.
    unique_lulc_classnames = set(
        params['lulc-class'] for params in biophysical_parameters.values())
    if len(unique_lulc_classnames) != len(biophysical_parameters):
        raise ValueError(
            "All values in `lulc-class` column must be unique, but "
            "duplicates were found.")

    aligned_lulc_paths = {
        baseline_lulc_year: os.path.join(
            intermediate_dir,
            ALIGNED_LULC_RASTER_PATTERN.format(
                snapshot_type='baseline', year=baseline_lulc_year,
                suffix=suffix))
    }

    for snapshot_year in snapshots:
        # We just created a baseline year, so don't re-create the path.
        if snapshot_year == baseline_lulc_year:
            continue

        aligned_path = os.path.join(
            intermediate_dir,
            ALIGNED_LULC_RASTER_PATTERN.format(
                snapshot_type='snapshot', year=snapshot_year,
                suffix=suffix))
        aligned_lulc_paths[snapshot_year] = aligned_path

    transition_years = set(aligned_lulc_paths.keys())
    transition_years.remove(baseline_lulc_year)

    alignment_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=([path for (year, path) in sorted(snapshots.items())],
              [path for (year, path) in sorted(aligned_lulc_paths.items())],
              ['near']*len(aligned_lulc_paths),
              target_pixel_size, 'intersection'),
        hash_algorithm='md5',
        copy_duplicate_artifact=True,
        target_path_list=aligned_lulc_paths.values(),
        task_name='Align input landcover rasters.')

    (disturbance_matrices, accumulation_matrices) = _read_transition_matrix(
        args['landcover_transitions_table'], biophysical_parameters)

    # Baseline stocks are simply reclassified.
    # Baseline accumulation are simply reclassified
    # There are no emissions, so net sequestration is only from accumulation.
    # Value can still be calculated from the net sequestration.
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
                f'Mapping {pool} carbon accumulation for '
                f'{baseline_lulc_year}'))

        if end_of_baseline_period != baseline_lulc_year:
            # The total stocks between baseline and the first year of interest
            # is just a sum-and-multiply for each pool.
            stock_rasters[end_of_baseline_period-1][pool] = os.path.join(
                STOCKS_RASTER_PATTERN.format(
                    pool=pool, year=end_of_baseline_period-1, suffix=suffix))
            baseline_stock_tasks[pool] = task_graph.add_task(
                func=_calculate_stocks_after_baseline_period,
                args=(stock_rasters[baseline_lulc_year][pool],
                      yearly_accum_rasters[pool],
                      (end_of_baseline_period - baseline_lulc_year),
                      stock_rasters[end_of_baseline_period-1][pool]),
                dependent_task_list=[
                    baseline_accum_tasks[pool], pool_stock_task],
                target_path_list=[
                    stock_rasters[end_of_baseline_period-1][pool]],
                task_name=(
                    f'Calculating {pool} stocks before the first transition '
                    'or the analysis year'))

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
            'Calculate accumulation between baseline year and final year'))

    # Reclassify transitions appropriately for each transition year.
    halflife_rasters = {}
    disturbance_magnitude_rasters = {}
    prior_transition_year = baseline_lulc_year
    for current_transition_year in sorted(transition_years):
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
                    intermediate_dir,
                    DISTURBANCE_MAGNITUDE_RASTER_PATTERN.format(
                        pool=pool, year=current_transition_year,
                        suffix=suffix))
            # this is _actually_ the magnitude, not the magnitude multiplied by
            # the stocks.
            _ = task_graph.add_task(
                func=_reclassify_disturbance_magnitude,
                args=(aligned_lulc_paths[prior_transition_year],
                      aligned_lulc_paths[current_transition_year],
                      disturbance_matrices[pool],
                      disturbance_magnitude_rasters[
                          current_transition_year][pool]),
                dependent_task_list=[alignment_task],
                target_path_list=[
                    disturbance_magnitude_rasters[
                        current_transition_year][pool]],
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
        'workspace_dir': args['workspace_dir'],
        'results_suffix': suffix,
        'n_workers': n_workers,
        'transition_years': transition_years,
        'disturbance_magnitude_rasters': disturbance_magnitude_rasters,
        'half_life_rasters': halflife_rasters,
        'annual_rate_of_accumulation_rasters': yearly_accum_rasters,
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

    prices = None
    if args.get('do_economic_analysis', False):  # Do if truthy
        if args.get('use_price_table', False):
            prices = {
                year: values['price'] for (year, values) in
                utils.build_lookup_from_csv(
                    args['price_table_path'], 'year').items()}
        else:
            inflation_rate = float(args['inflation_rate']) * 0.01
            annual_price = float(args['price'])

            if transition_years:
                max_year = max(transition_years.union(set([analysis_year])))
            else:
                max_year = analysis_year

            prices = {}
            for timestep_index, year in enumerate(
                    range(baseline_lulc_year, max_year + 1)):
                prices[year] = (
                    ((1 + inflation_rate) ** timestep_index) *
                    annual_price)
        discount_rate = float(args['discount_rate']) * 0.01

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

        transition_analysis_args.update({
            'npv_since_baseline_raster': baseline_period_npv_raster,
            'carbon_prices_per_year': prices,
            'discount_rate': discount_rate,
        })

    task_graph.join()
    if transition_years:
        execute_transition_analysis(transition_analysis_args)

    task_graph.close()
    task_graph.join()


def _set_up_workspace(args):
    """Set up the workspce for a Coastal Blue Carbon model run.

    Since the CBC model has two intended entrypoints, this allows for us to
    have consistent workspace layouts without duplicating the configuration
    between the two functions.

    Args:
        args (dict): A dict containing containing the necessary keys.
        args['workspace_dir'] (string): the path to a workspace directory where
            outputs should be written.
        args['results_suffix'] (string): Optional.  If provided, this string
            will be inserted into all filenames produced, just before the file
            extension.
        args['n_workers'] (int): (optional) If provided, the number of workers
            to pass to ``taskgraph``.

    Returns:
        A 5-element tuple containing:

            * ``task_graph`` - a ``taskgraph.TaskGraph`` object.
            * ``n_workers`` - the int ``n_workers`` parameter used
            * ``intermediate_dir`` - the path to the intermediate directory on
                disk.  This directory is created in this function if it does
                not already exist.
            * ``output_dir`` - the path to the output directory on disk.  This
                directory is created in this function if it does not already
                exist.
            * ``suffix`` - the suffix string, derived from the user-provided
                suffix, if it was provided.
    """
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
        taskgraph_cache_dir, n_workers, reporting_interval=5.0)

    suffix = utils.make_suffix_string(args, 'results_suffix')
    intermediate_dir = os.path.join(
        args['workspace_dir'], INTERMEDIATE_DIR_NAME)
    output_dir = os.path.join(
        args['workspace_dir'], OUTPUT_DIR_NAME)

    utils.make_directories([output_dir, intermediate_dir, taskgraph_cache_dir])

    return task_graph, n_workers, intermediate_dir, output_dir, suffix


def execute_transition_analysis(args):
    """Execute a transition analysis.

    The calculations required for an analysis centered around a baseline period
    are trivial and can be accomplished with a few calls to your
    ``raster_calculator`` of choice because of the linear rates of carbon
    accumulation that this model assumes.  By contrast, the raster calculations
    required for an analysis involving transitions are much more complex, and
    by providing this function separate from ``execute``, the adept modeller is
    able to provide spatially-explicit distributions of rates of accumulation,
    the magnitudes of disturbances, carbon half-lives and carbon stocks for
    each carbon pool as desired.

    There are certain constraints placed on some of these inputs:

        * The years listed in ``args['transition_years']`` must match the keys
            in ``args['disturbance_magnitude_rasters']``,
            ``args['half_life_rasters']``, and
            ``args['annual_rate_of_accumulation_rasters']``.
        * All rasters provided to this function must be in the same projected
            coordinate system and have identical dimensions and pixel sizes.
        * All rasters provided to this function are assumed to be 32-bit
            floating-point rasters with a nodata value matching this module's
            ``NODATA_FLOAT32`` attribute.
        * All data structures provided to this function that use pools as an
            index assume that the pools are strings matching the ``POOL_SOIL``,
            ``POOL_BIOMASS`` and ``POOL_LITTER`` attributes of this module.
        * Most dicts passed to this function are nested dictionaries indexed
            first by transition year, then by pool (see note about pool keys).
            These dicts are accessed like so::

                args['half_life_rasters'][transition_year][POOL_SOIL]

        * The "baseline period" is the range of years between the baseline year
            and the first transition, not including the year of the first
            transition.

    Args:
        args['workspace_dir'] (string): The path to a workspace directory where
            outputs should be written.
        args['results_suffix'] (string): If provided, a string suffix that will
            be added to each output filename. Optional.
        args['n_workers'] (int):  The number of workers that ``taskgraph`` may
            use.
        args['transition_years'] (set): A python set of int years in which a
            transition will take place.
        args['disturbance_magnitude_rasters'] (dict): A 2-level deep dict
            structure mapping int transition years to string pools to raster
            paths representing the magnitude (0-1, float32) of a disturbance.
        args['half_life_rasters'] (dict): A 2-level deep dict structure mapping
            int transition years to string pools to raster paths representing
            the half-life of the given carbon pool at that transition year.
        args['annual_rate_of_accumulation_rasters'] (dict): A 2-level deep dict
            structure mapping int transition years to string pools to raster
            paths representing the annual rate of accumulation for this pool in
            the given transition period.
        args['carbon_prices_per_year'] (dict): A dict mapping int years to the
            floating-point price of carbon in that year.  Every year between
            the baseline year and the analysis year (inclusive) must have a
            key-value pair defined.
        args['discount_rate'] (number): The discount rate for carbon
            valuation.
        args['analysis_year'] (int): The analysis year.  Must be greater than
            or equal to the final transition year.
        args['do_economic_analysis'] (bool): Whether to do valuation.
        args['baseline_lulc_raster'] (string): The path to the baseline lulc
            raster on disk.
        args['baseline_lulc_year'] (int): The year of the baseline scenario.
        args['sequestration_since_baseline_raster'] (string): The string path
            to a raster on disk representing the total carbon sequestration
            across all 3 carbon pools in the entire baseline period.
        args['stocks_at_first_transition'] (dict): A dict mapping pool strings
            (see above for the valid pool identifiers) to rasters representing
            the carbon stocks at the end of the baseline period.

    Returns:
        ``None``.

    """
    task_graph, n_workers, intermediate_dir, output_dir, suffix = (
        _set_up_workspace(args))

    transition_years = set([int(year) for year in args['transition_years']])
    disturbance_magnitude_rasters = args['disturbance_magnitude_rasters']
    half_life_rasters = args['half_life_rasters']
    yearly_accum_rasters = args['annual_rate_of_accumulation_rasters']

    prices = None
    if args.get('carbon_prices_per_year', None):
        prices = {int(year): float(price)
                  for (year, price) in args['carbon_prices_per_year'].items()}
    discount_rate = float(args['discount_rate'])
    baseline_lulc_year = int(args['baseline_lulc_year'])

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
            POOL_BIOMASS: (
                args['annual_rate_of_accumulation_rasters'][POOL_BIOMASS]),
            POOL_LITTER: (
                args['annual_rate_of_accumulation_rasters'][POOL_LITTER]),
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

        for pool in (POOL_SOIL, POOL_BIOMASS):
            # Calculate stocks from last year's stock plus last year's net
            # sequestration.
            # Stock rasters from ``year`` represent the carbon stocks present
            # at the very beginning of ``year``.
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
                    func=pygeoprocessing.raster_calculator,
                    args=([(disturbance_magnitude_rasters[year][pool], 1),
                           (stock_rasters[year][pool], 1)],
                          _calculate_disturbance_volume,
                          disturbance_vol_rasters[year][pool],
                          gdal.GDT_Float32,
                          NODATA_FLOAT32),
                    dependent_task_list=(
                        current_disturbance_vol_dependent_tasks),
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
                    [(disturbance_vol_rasters[
                        current_transition_year][pool], 1),
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

        # If in the last year before a transition or the last year before the
        # final year of the analysis (which might not be a transition):
        #  * sum emissions since last transition
        #  * sum accumulation since last transition
        #  * sum net sequestration since last transition
        if (year + 1) in transition_years or (year + 1) == final_year:
            emissions_rasters_since_transition = []
            net_seq_rasters_since_transition = []
            for year_after_transition in range(
                    current_transition_year, year + 1):
                emissions_rasters_since_transition.extend(
                    list(emissions_rasters[year_after_transition].values()))
                net_seq_rasters_since_transition.extend(
                    list(net_sequestration_rasters[
                        year_after_transition].values()))

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

        # These are the few sets of tasks that we care about referring to from
        # the prior year.
        prior_stock_tasks = current_stock_tasks
        prior_net_sequestration_tasks = current_net_sequestration_tasks

    # Calculate total net sequestration
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

    # Calculate Net Present Value for each of the transition years, relative to
    # the baseline.
    target_npv_paths = {}
    for transition_year in (
            sorted(set(transition_years).union(set([final_year])))[1:]):
        target_npv_paths[transition_year] = os.path.join(
            output_dir, NET_PRESENT_VALUE_RASTER_PATTERN.format(
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


def _calculate_disturbance_volume(disturbance_magnitude_matrix, stock_matrix):
    """Calculate disturbance volume.

    Args:
        disturbance_magnitude_matrix (numpy.array): A float32 matrix of the
            magnitude of a disturbance.  Nodata values must be NODATA_FLOAT32.
        stock_matrix (numpy.array): A float32 matrix of the stocks present at
            the time of disturbance.  Nodata values must be NODATA_FLOAT32.

    Returns:
        A numpy float32 matrix of disturbance volume.
    """
    disturbed_carbon_volume = numpy.empty(disturbance_magnitude_matrix.shape,
                                          dtype=numpy.float32)
    disturbed_carbon_volume[:] = NODATA_FLOAT32
    valid_pixels = (~numpy.isclose(disturbance_magnitude_matrix,
                                   NODATA_FLOAT32) &
                    ~numpy.isclose(stock_matrix, NODATA_FLOAT32))
    disturbed_carbon_volume[valid_pixels] = (
        disturbance_magnitude_matrix[valid_pixels] *
        stock_matrix[valid_pixels])
    return disturbed_carbon_volume


def _calculate_npv(
        net_sequestration_rasters, prices_by_year, discount_rate,
        baseline_year, target_raster_years_and_paths):
    """Calculate the net present value of carbon sequestered.

    Args:
        net_sequestration_rasters (dict): A dict mapping int years to string
            paths to sequestration rasters.  The year keys correspond to the
            years marking the end of a transition period.  All rasters must
            share the same projected CRS and have identical dimensions.
        prices_by_year (dict): A dict mapping int years between the baseline
            year and the final year.
        discount_rate (float): The discount rate.  A rate of 0.1 indicates 10%
            discount.
        baseline_year (int): The year of the baseline scenario.
        target_raster_years_and_paths (dict): A dict mapping int years to
            string paths indicating where the target rasters should be written.
            Year keys must match up with the keys of
            ``net_sequestration_rasters``.

    Returns:
        ``None``.

    """
    for target_raster_year, target_raster_path in sorted(
            target_raster_years_and_paths.items()):

        valuation_factor = 0
        for years_since_baseline, year in enumerate(
                range(baseline_year, target_raster_year)):
            valuation_factor += (
                prices_by_year[year] / (
                    (1 + discount_rate) ** years_since_baseline))

        def _npv(*sequestration_matrices):
            npv = numpy.empty(sequestration_matrices[0].shape,
                              dtype=numpy.float32)
            npv[:] = NODATA_FLOAT32

            matrix_sum = numpy.zeros(npv.shape, dtype=numpy.float32)
            valid_pixels = numpy.ones(npv.shape, dtype=numpy.bool)
            for matrix in sequestration_matrices:
                valid_pixels &= ~numpy.isclose(matrix, NODATA_FLOAT32)
                matrix_sum[valid_pixels] += matrix[valid_pixels]

            npv[valid_pixels] = (
                matrix_sum[valid_pixels] * valuation_factor)
            return npv

        raster_path_band_tuples = [
            (path, 1) for (year, path) in net_sequestration_rasters.items() if
            year <= target_raster_year]

        pygeoprocessing.raster_calculator(
            raster_path_band_tuples, _npv, target_raster_path,
            gdal.GDT_Float32, NODATA_FLOAT32)


def _calculate_stocks_after_baseline_period(
        baseline_stock_raster_path, yearly_accumulation_raster_path, n_years,
        target_raster_path):
    """Calculate the stocks after the baseline period.

    Stocks for the given pool at the end of the baseline period are a function
    of the initial baseline stocks, the yearly accumulation rate, and the
    number of years in the baseline period.

    Args:
        baseline_stock_raster_path (string): The string path to a GDAL raster
            on disk representing the initial carbon stocks for this pool at the
            baseline year.
        yearly_accumulation_raster_path (string): The string path to a GDAL
            raster on disk representing the yearly accumulation rate for this
            carbon pool.
        n_years (int): The number of years in the baseline period.
        target_raster_path (string): The path to where the calculated stocks
            raster should be written.

    Returns:
        ``None``.

    """
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
                        ~numpy.isclose(accum_matrix, accum_nodata))

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
    """Calculate the total accumulation over a period of years.

    This is a shortcut for adding up 3 rasters per year over n years.

    Args:
        annual_biomass_matrix (numpy.array): A float32 matrix of the annual
            rate of biomass accumulation.
        annual_soil_matrix (numpy.array): A float32 matrix of the annual
            rate of soil accumulation.
        annual_litter_matrix (numpy.array): A float32 matrix of the annual
            rate of litter accumulation.
        n_years (int): The number of years in the baseline period.

    Returns:
        A numpy array representing the sum of the 3 input matrices (excluding
        nodata pixels), multiplied by the number of years in the baseline
        period.

    """
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
        carbon_disturbed_matrix (numpy.array): The volume of carbon disturbed
            in the most recent transition year as time approaches infinity.
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


def _sum_n_rasters(
        raster_path_list, target_raster_path,
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
                * ``'medium-impact-disturb'`` indicating a
                    medium-impact disturbance
                * ``'high-impact-disturb'`` indicating a
                    high-impact disturbance
                * ``''`` (blank), which is equivalent to no carbon change.o
        biophysical_dict (dict): A ``dict`` mapping of integer landcover codes
            to biophysical values for disturbance and accumulation values for
            soil and biomass carbon pools.

    Returns:
        Two ``dict``s, each of which maps string keys "soil" and "disturbance"
        to ``scipy.sparse.dok_matrix`` objects of numpy.float32.  The first
        such dict contains disturbance magnitudes for the pool for each
        landcover transition, and the second contains accumulation rates for
        the pool for the landcover transition.
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
        # If the user is using the template, all rows have some sort of values
        # in them until the blank row before the legend.  If we find that row,
        # we can break out of the loop.
        if row.isnull().all():
            LOGGER.info(f"Halting transition table parsing on row {index}; "
                        "blank line encountered.")
            break

        from_lucode = lulc_class_to_lucode[str(row['lulc-class']).lower()]

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
                        'soil-yearly-accumulation'])
                biomass_accumulation_matrix[from_lucode, to_lucode] = (
                    biophysical_dict[to_lucode][
                        'biomass-yearly-accumulation'])

    disturbance_matrices = {
        'soil': soil_disturbance_matrix,
        'biomass': biomass_disturbance_matrix
    }
    accumulation_matrices = {
        'soil': soil_accumulation_matrix,
        'biomass': biomass_accumulation_matrix,
    }

    return (disturbance_matrices, accumulation_matrices)


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


def _extract_snapshots_from_table(csv_path):
    """Extract the year/raster snapshot mapping from a CSV.

    No validation is performed on the years or raster paths.

    Parameters:
        csv_path (string): The path to a CSV on disk containing snapshot
            years and a corresponding transition raster path.  Snapshot years
            may be in any order in the CSV, but must be integers and no two
            years may be the same.  Snapshot raster paths must refer to a
            raster file located on disk representing the landcover at that
            transition.  If the path is absolute, the path will be used as
            given.  If the path is relative, the path will be interpreted as
            relative to the parent directory of this CSV file.

    Returns:
        A ``dict`` mapping int snapshot years to their corresponding raster
        paths.  These raster paths will be absolute paths.

    """
    table = utils.read_csv_to_dataframe(csv_path, index_col=False)
    table.columns = table.columns.str.lower()

    output_dict = {}
    table.set_index("snapshot_year", drop=False, inplace=True)
    for index, row in table.iterrows():
        raster_path = row['raster_path']
        if not os.path.isabs(raster_path):
            raster_path = os.path.join(os.path.dirname(csv_path), raster_path)
        output_dict[int(index)] = os.path.abspath(raster_path)

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

    if ("landcover_snapshot_csv" not in invalid_keys and
            "landcover_snapshot_csv" in sufficient_keys):
        snapshots = _extract_snapshots_from_table(
            args['landcover_snapshot_csv'])

        for snapshot_year, snapshot_raster_path in snapshots.items():
            raster_error_message = validation.check_raster(
                snapshot_raster_path, projected=True, projection_units='m')
            if raster_error_message:
                validation_warnings.append(
                    (['landcover_snapshot_csv'], (
                        f"Raster for snapshot {snapshot_year} could not "
                        f"be validated: {raster_error_message}")))

        if ("analysis_year" not in invalid_keys
                and "analysis_year" in sufficient_keys):
            if max(set(snapshots.keys())) > int(args['analysis_year']):
                validation_warnings.append(
                    (['analysis_year'], (
                        f"Analysis year {args['analysis_year']} must be >= "
                        f"the latest snapshot year ({max(snapshots.keys())})"
                    )))

    return validation_warnings
