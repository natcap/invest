import os
import logging
import collections

from osgeo import gdal
import taskgraph
import pygeoprocessing
import pandas
import numpy
import scipy.sparse

from .. import utils

LOGGER = logging.getLogger(__name__)


NODATA_FLOAT32 = float(numpy.finfo(numpy.float32).min)


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

    # Phase 2: Set up the spatial variables that change with each transitition
    disturbance_rasters = {}
    halflife_rasters = {}
    yearly_accum_rasters = {}
    litter_rasters = {}

    biomass_disturb_matrix, soil_disturb_matrix = _read_transition_matrix(
        args['landcover_transitions_table'], biophysical_parameters)
    disturbance_matrices = {}
    disturbance_matrices['soil'] = soil_disturb_matrix
    disturbance_matrices['biomass'] = biomass_disturb_matrix

    # This dict of lists is so that we can use them effectively to schedule the
    # next waves of timeframe tasks in the main timeseries loop.
    transition_tasks = collections.defaultdict(list)
    for prior_transition_year, transition_year in zip(
            [None] + sorted(transition_years)[: -1],
            sorted(transition_years)):

        # TODO: should ``litter`` field be ``litter-initial``?
        litter_rasters[transition_year] = os.path.join(
            intermediate_dir, f'litter-{transition_year}{suffix}.tif')
        transition_tasks[transition_year].append(task_graph.add_task(
            func=pygeoprocessing.reclassify_raster,
            args=(
                (aligned_lulc_paths[transition_year], 1),
                {lucode: values['litter'] for (lucode, values) in
                    biophysical_parameters.items()},
                litter_rasters[transition_year],
                gdal.GDT_Float32,
                NODATA_FLOAT32),
            dependent_task_list=[alignment_task],
            target_path_list=[litter_rasters[transition_year]],
            task_name=(
                f'Reclassify litter raster for {transition_year}')))

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
                LOGGER.info(
                        f'Creating disturbance raster for transition '
                        f'{prior_transition_year} - {transition_year}')
                disturbance_rasters[transition_year][pool] = os.path.join(
                    intermediate_dir,
                    f'disturbance-{pool}-{transition_year}{suffix}.tif')
                prior_transition_nodata = pygeoprocessing.get_raster_info(
                    aligned_lulc_paths[prior_transition_year])['nodata'][0]
                current_transition_nodata = pygeoprocessing.get_raster_info(
                    aligned_lulc_paths[transition_year])['nodata'][0]
                transition_tasks[transition_year].append(task_graph.add_task(
                    func=pygeoprocessing.raster_calculator,
                    args=(
                        [(aligned_lulc_paths[prior_transition_year], 1),
                         (aligned_lulc_paths[transition_year], 1),
                         (disturbance_matrices[pool], 'raw'),
                         (prior_transition_nodata, 'raw'),
                         (current_transition_nodata, 'raw')],
                        _reclassify_transition,
                        disturbance_rasters[transition_year][pool],
                        gdal.GDT_Float32,
                        NODATA_FLOAT32)))

            # Reclassify halflife{pool}
            halflife_rasters[transition_year][pool] = os.path.join(
                intermediate_dir,
                f'halflife-{pool}-{transition_year}{suffix}.tif')
            transition_tasks[transition_year].append(
                task_graph.add_task(
                    func=pygeoprocessing.reclassify_raster,
                    args=(
                        (aligned_lulc_paths[transition_year], 1),
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
                        (aligned_lulc_paths[transition_year], 1),
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

    if transition_years == set([baseline_lulc_year]):
        LOGGER.info('No transition years provided.  Exiting.')
        task_graph.close()
        task_graph.join()
        return


    # Phase 3: do the timeseries analysis.
    if analysis_year:
        final_timestep = analysis_year
    else:
        final_timestep = max(transition_years)
    for year in range(min(transition_years), final_timestep+1):
        pass


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
    if utils.has_utf8_bom(transition_csv_path):
        encoding = 'utf-8-sig'

    table = pandas.read_csv(
        transition_csv_path, sep=None, index_col=False, engine='python',
        encoding=encoding)

    lulc_class_to_lucode = {
        values['lulc-class']: lucode for (lucode, values) in
        biophysical_dict.items()}
    import pprint; pprint.pprint(lulc_class_to_lucode)

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
