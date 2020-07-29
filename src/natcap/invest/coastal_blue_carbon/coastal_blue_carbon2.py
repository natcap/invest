import os
import logging
import collections

from osgeo import gdal
import taskgraph
import pygeoprocessing
from pygeoprocessing.symbolic import evaluate_raster_calculator_expression
import pandas
import numpy
import scipy.sparse

from .. import utils

LOGGER = logging.getLogger(__name__)


NODATA_FLOAT32 = float(numpy.finfo(numpy.float32).min)
TRANS_EMPTY = 0
TRANS_NO_CHANGE = 1
TRANS_ACCUM = 2
TRANS_LOW_IMPACT = 3
TRANS_MED_IMPACT = 4
TRANS_HIGH_IMPACT = 5


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
        transition_years.add(baseline_lulc_year)
    except (KeyError, ValueError, TypeError):
        LOGGER.error('The baseline_lulc_year is required but not provided.')
        raise ValueError('Baseline lulc year is required.')

    try:
        # TODO: validate that args['analysis_year'] > max(transition_years)
        transition_years.add(int(args['analysis_year']))
    except (KeyError, ValueError, TypeError):
        pass

    base_paths = [args['baseline_lulc_path']]
    aligned_paths_dict = {}
    aligned_paths = [os.path.join(
        intermediate_dir, f'aligned_lulc_baseline{suffix}.tif')]
    aligned_paths_dict[int(args['baseline_lulc_year'])] = aligned_paths[0]
    for transition_year in transitions:
        base_paths.append(transitions[transition_year])
        transition_years.add(transition_year)
        aligned_paths.append(
            os.path.join(
                intermediate_dir,
                f'aligned_lulc_transition_{transition_year}{suffix}.tif'))
        aligned_paths_dict[transition_year] = aligned_paths[-1]

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

    def _reclassify(
            source_raster_path, reclassification_map,
            target_raster_path):
        return task_graph.add_task(
            func=pygeoprocessing.reclassify_raster,
            args=(
                (source_raster_path, 1),
                reclassification_map,
                target_raster_path,
                gdal.GDT_Float32,
                NODATA_FLOAT32),
            dependent_task_list=[alignment_task],
            target_path_list=[target_raster_path],
            task_name=f'Reclassify {target_raster_path}')


    # Phase 2: Set up the spatial variables that change with each transitition
    disturbance_rasters = {}
    halflife_rasters = {}
    yearly_accum_rasters = {}

    # This dict of lists is so that we can use them effectively to schedule the
    # next waves of timeframe tasks in the main timeseries loop.
    transition_tasks = collections.defaultdict(list)
    for transition_year in sorted(transition_years):
        disturbance_rasters[transition_year] = {}
        halflife_rasters[transition_year] = {}
        yearly_accum_rasters[transition_year] = {}
        for pool in ('soil', 'biomass'):
            if transition_year == baseline_lulc_year:
                # There's no disturbance possible in the baseline because we're
                # not transitioning landcover classes.
                disturbance_rasters[transition_year][pool] = None
            else:
                # Reclassify disturbance{pool}
                # This is based on the transition from one landcover class to
                # another.
                disturbance_rasters[transition_year][pool] = os.path.join(
                    intermediate_dir,
                    f'disturbance-{pool}-{transition_year}{suffix}.tif')
                transition_tasks[transition_year].append(task_graph.add_task())

            # Reclassify halflife{pool}
            halflife_rasters[transition_year][pool] = os.path.join(
                intermediate_dir,
                f'halflife-{pool}-{transition_year}{suffix}.tif')
            transition_tasks[transition_year].append(
                task_graph.add_task(
                    func=pygeoprocessing.reclassify_raster,
                    args=(
                        (aligned_paths_dict[transition_year], 1),
                        {lucode: values[f'{pool}-initial'] for (lucode, values)
                            in biophysical_parameters.items()},
                        halflife_rasters[transition_year][pool],
                        gdal.GDT_Float32,
                        NODATA_FLOAT32),
                    dependent_task_list=[alignment_task],
                    target_path_list=[halflife_rasters[transition_year][pool]],
                    task_name=(
                        f'Mapping {pool} half-life for {transition_year}')))

            # Reclassify yearly-accumulation{pool}
            yearly_accum_rasters[transition_year][pool] = os.path.join(
                intermediate_dir,
                f'halflife-{pool}-{transition_year}{suffix}.tif')
            transition_tasks[transition_year].append(
                task_graph.add_task(
                    func=pygeoprocessing.reclassify_raster,
                    args=(
                        (aligned_paths_dict[transition_year], 1),
                        {lucode: values[f'{pool}-yearly-accumulation']
                            for (lucode, values) in
                            biophysical_parameters.items()},
                        yearly_accum_rasters[transition_year][pool],
                        gdal.GDT_Float32,
                        NODATA_FLOAT32),
                    dependent_task_list=[alignment_task],
                    target_path_list=[
                        yearly_accum_rasters[transition_year][pool]],
                    task_name=(
                        f'Mapping {pool} half-life for {transition_year}')))

    # Phase 3: do the timeseries analysis.
    for year in range(min(transition_years), max(transition_years)+1):
        if year in transitions:
            lulc_path = aligned_paths_dict[year]

        stocks_rasters = {}
        stocks_tasks = {}
        if year == min(transitions):
            # Reclassify soil, biomass stocks.
            for stock in ('biomass-initial', 'soil-initial'):
                stocks_rasters[stock] = os.path.join(
                    intermediate_dir, f'baseline-{stock}{suffix}.tif')
                stocks_tasks[stock] = task_graph.add_task(
                    func=pygeoprocessing.reclassify_raster,
                    args=(
                        (lulc_path, 1),
                        {lucode: values[stock] for (lucode, values) in
                            biophysical_parameters.items()},
                        stocks_rasters[stock],
                        gdal.GDT_Float32,
                        NODATA_FLOAT32),
                    dependent_task_list=[alignment_task],
                    target_path_list=[stocks_rasters[stock]],
                    task_name=f'Reclassify baseline {stock} raster')

            # Sum the biomass and stocks rasters for the base year.
            total_stock_raster_path = os.path.join(
                intermediate_dir, f'total_stock_{year}{suffix}.tif')
            total_stocks_task = task_graph.add_task(
                func=evaluate_raster_calculator_expression,
                args=(
                    'biomass + soil',
                    {'biomass': (stocks_rasters['biomass-initial'], 1),
                     'soil': (stocks_rasters['soil-initial'], 1)},
                    NODATA_FLOAT32,
                    total_stock_raster_path),
                dependent_task_list=list(stocks_tasks.values()),
                target_path_list=[total_stock_raster_path],
                task_name=f'Calculate total carbon stocks for {year}')

        if year in transitions:
            litter_raster_path = os.path.join(
                intermediate_dir, f'baseline-litter{suffix}.tif')
            task_graph.add_task(
                func=pygeoprocessing.reclassify_raster,
                args=(
                    (lulc_path, 1),
                    {lucode: values['litter'] for (lucode, values) in
                        biophysical_parameters.items()},
                    litter_raster_path,
                    gdal.GDT_Float32,
                    NODATA_FLOAT32),
                dependent_task_list=[alignment_task],
                target_path_list=[litter_raster_path],
                task_name=(
                    f'Reclassify litter raster for transition year {year}'))

            # Reclassify the transition LULC for this transition year into
            # the disturbance values.  This is a 2-dimensional
            # reclassification (so a raster calculator operation)
            # Disturbance reclassification happens for soil and biomass.
            disturbance_raster_path = os.path.join(
                intermediate_dir, f'disturbance-{year}{suffix}.tif')

            # Reclassify the transition LULC for soil and biomass halflife

            # Reclassify the transition LULC for soil and biomass annual
            # accumulation.

            # Calculate the total disturbed carbon (R_biomass and R_soil)

        # Transient analysis
        #
        # For each year from the baseline:
        #     * Calculate emissions for biomass and soil (uses R_biomass,
        #       R_soil from the most recent transition)
        #     * Calculate net sequestration (A - E, for each of {biomass,
        #       soil})
        #     * Calculate Total carbon stock for the year,
        #     * if we're doing economic analysis:
        #        * calculate the value (N * price_this_year)




















def _read_transition_matrix(transition_csv_path, biophysical_dict):
    encoding = None
    if utils.has_utf8_bom(csv_path):
        encoding = 'utf-8-sig'

    table = pandas.read_csv(
        transition_csv_path, sep=None, index_col=False, engine='python',
        encoding=encoding)

    # Load up a sparse matrix with the transitions to save on memory usage.
    n_rows = len(table.index)
    soil_disturbance_matrix = scipy.sparse.dok_matrix(
        (n_rows, n_rows), dtype=numpy.float32)
    biomass_disturbance_matrix = scipy.sparse.dok_matrix(
        (n_rows, n_rows), dtype=numpy.float32)
    transitions = {
        '': TRANS_EMPTY,
        'NCC': TRANS_NO_CHANGE,
        'accum': TRANS_ACCUM,
        'low-impact-disturb': TRANS_LOW_IMPACT,
        'med-impact-disturb': TRANS_MED_IMPACT,
        'high-impact-disturb': TRANS_HIGH_IMPACT,
    }

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
        for colname, col_value in row.items():
            # Only set values where the transition HAS a value.
            # Takes advantage of the sparse characteristic of the model.
            col_value = col_value.strip()
            if col_value.endswith('disturb'):
                soil_disturbance_matrix[index, colname] = (
                    biophysical_dict[f'soil-{col_value}'])
                biomass_disturbance_matrix[index, colname] = (
                    biophysical_dict[f'biomass-{col_value}'])

    return biomass_disturbance_matrix, soil_disturbance_matrix


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
