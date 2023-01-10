import logging
import math
import os
import re
import shutil
import tempfile

import numpy
import numpy.testing
import pandas
import pygeoprocessing
import shapely.ops
import shapely.wkb
import taskgraph
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from . import gettext
from . import spec_utils
from . import utils
from . import validation
from .model_metadata import MODEL_METADATA
from .ndr import ndr
from .spec_utils import u

LOGGER = logging.getLogger(__name__)
UINT32_NODATA = int(numpy.iinfo(numpy.uint32).max)
FLOAT32_NODATA = float(numpy.finfo(numpy.float32).min)
BYTE_NODATA = 255
KERNEL_LABEL_DICHOTOMY = 'dichotomy'
KERNEL_LABEL_EXPONENTIAL = 'exponential'
KERNEL_LABEL_GAUSSIAN = 'gaussian'
KERNEL_LABEL_DENSITY = 'density'
RADIUS_OPT_UNIFORM = 'radius_uniform'
RADIUS_OPT_GREENSPACE = 'radius_per_greenspace_class'
RADIUS_OPT_POP_GROUP = 'radius_per_pop_group'
POP_FIELD_REGEX = '^pop_'
ID_FIELDNAME = 'adm_unit_id'
ARGS_SPEC = {
    'model_name': MODEL_METADATA['urban_nature_access'].model_title,
    'pyname': MODEL_METADATA['urban_nature_access'].pyname,
    'userguide': MODEL_METADATA['urban_nature_access'].userguide,
    'args_with_spatial_overlap': {
        'spatial_keys': [
            'lulc_raster_path', 'population_raster_path',
            'aoi_vector_path'],
        'different_projections_ok': True,
    },
    'args': {
        'workspace_dir': spec_utils.WORKSPACE,
        'results_suffix': spec_utils.SUFFIX,
        'n_workers': spec_utils.N_WORKERS,
        'lulc_raster_path': {
            **spec_utils.LULC,
            'projected': True,
            'projection_units': u.meter,
            'about': (
                "A map of LULC codes. "
                "All values in this raster must have corresponding entries "
                "in the LULC attribute table."),
        },
        'lulc_attribute_table': {
            'name': 'LULC attribute table',
            'type': 'csv',
            'columns': {
                'lucode': {
                    'type': 'integer',
                    'about': (
                        "LULC code.  Every value in the LULC map must have a "
                        "corresponding entry in this column."),
                },
                'greenspace': {
                    'type': 'number',
                    'units': u.none,
                    'about': (
                        "1 if this landcover code represents greenspace, 0 "
                        "if not."
                    ),
                },
                'search_radius_m': {
                    'type': 'number',
                    'units': u.meter,
                    'required':
                        f'search_radius_mode == {RADIUS_OPT_GREENSPACE}',
                    'expression': 'value >= 0',
                    'about': (
                        'The distance in meters to use as the search radius '
                        'for this type of greenspace. Values must be >= 0.'
                    ),
                }
            },
            'about': (
                "A table identifying which LULC codes represent greenspace."
            ),
        },
        'population_raster_path': {
            'type': 'raster',
            'name': 'population raster',
            'bands': {
                1: {'type': 'number', 'units': u.none}
            },
            'projected': True,
            'projection_units': u.meter,
            'about': (
                "A raster representing the number of people who live in each "
                "pixel."
            ),
        },
        'aoi_vector_path': {
            'type': 'vector',
            'name': 'administrative boundaries',
            'geometries': spec_utils.POLYGONS,
            'fields': {
                "pop_[POP_GROUP]": {
                    "type": "ratio",
                    "about": gettext(
                        "The proportion of the population within this region "
                        "belonging to the identified population group "
                        "(POP_GROUP)."
                    ),
                }
            },
            'about': gettext(
                "Non-overlapping regions (typically administrative "
                "boundaries) within which population supply and demand are "
                "summarized."
            ),
        },
        'greenspace_demand': {
            'type': 'number',
            'name': 'greenspace demand per capita',
            'units': u.m**2,  # defined as m² per capita
            'expression': "value > 0",
            'about': gettext(
                "The amount of greenspace that each resident should have "
                "access to. This is often defined by local urban planning "
                "documents."
            )
        },
        'decay_function': {
            'name': 'decay function',
            'type': 'option_string',
            'options': {
                KERNEL_LABEL_DICHOTOMY: {
                    'display_name': 'Dichotomy',
                    'description': gettext(
                        'All pixels within the search radius contribute '
                        'equally to a greenspace pixel.'),
                },
                KERNEL_LABEL_EXPONENTIAL: {
                    'display_name': 'Exponential',
                    'description': gettext(
                        'Contributions to a greenspace pixel decrease '
                        'exponentially, where '
                        '"weight = e^(-pixel_dist / search_radius)"'),
                },
                KERNEL_LABEL_GAUSSIAN: {
                    'display_name': 'Gaussian',
                    'description': gettext(
                        'Contributions to a greenspace pixel decrease '
                        'according to a normal ("gaussian") distribution '
                        'with a sigma of 3.'),
                },
                KERNEL_LABEL_DENSITY: {
                    'display_name': 'Density',
                    'description': gettext(
                        'Contributions to a greenspace pixel decrease '
                        'faster as distances approach the search radius. '
                        'Weights are calculated by '
                        '"weight = 0.75 * (1-(pixel_dist / search_radius)^2)"'
                    ),
                },
            },
            'about': (
                'Pixels within the search radius of a greenspace pixel '
                'have a distance-weighted contribution to a greenspace '
                'pixel according to the selected decay function.'),
        },
        'search_radius_mode': {
            'name': 'search radius mode',
            'type': 'option_string',
            'required': True,
            'about': gettext(
                'The type of search radius to use.'
            ),
            'options': {
                RADIUS_OPT_UNIFORM: {
                    'display_name': 'Uniform radius',
                    'description': gettext(
                        'The search radius is the same for all greenspace '
                        'types.'),
                },
                RADIUS_OPT_GREENSPACE: {
                    'display_name': 'Radius defined per greenspace class',
                    'description': gettext(
                        'The search radius is defined for each distinct '
                        'greenspace LULC classification.'),
                },
                RADIUS_OPT_POP_GROUP: {
                    'display_name': 'Radius defined per population group',
                    'description': gettext(
                        'The search radius is defined for each distinct '
                        'population group.'),
                },
            },
        },
        'search_radius': {
            'type': 'number',
            'name': 'uniform search radius',
            'units': u.m,
            'expression': 'value > 0',
            'about': gettext(
                'The search radius to use when running the model under a '
                'uniform search radius'),
        },
        'population_group_radii_table': {
            'name': 'population group radii table',
            'type': 'csv',
            'columns': {
                "pop_group": {
                    "type": "ratio",
                    "required": False,
                    "about": gettext(
                        "The proportion of the population within this region "
                        "belonging to the identified population group. "
                        "Values in this column must match those population "
                        "groups in the administrative vector."
                    ),
                },
                'search_radius_m': {
                    'type': 'number',
                    'units': u.meter,
                    'required':
                        f'search_radius_mode == {RADIUS_OPT_POP_GROUP}',
                    'expression': 'value >= 0',
                    'about': gettext(
                        "The distance in meters to use as the search radius "
                        "for this population group.  Values must be >= 0."
                    ),
                },
            },
            'about': gettext('TBD'),
        }
    }
}


_OUTPUT_BASE_FILES = {
    'greenspace_supply': 'greenspace_supply.tif',
    'aois': 'aois.gpkg',
}
_INTERMEDIATE_BASE_FILES = {
    'attribute_table': 'attribute_table.csv',
    'aligned_population': 'aligned_population.tif',
    'aligned_lulc': 'aligned_lulc.tif',
    'greenspace_area': 'greenspace_area.tif',
    'greenspace_population_ratio': 'greenspace_population_ratio.tif',
    'convolved_population': 'convolved_population.tif',
    'greenspace_budget': 'greenspace_budget.tif',
    'greenspace_supply_demand_budget': 'greenspace_supply_demand_budget.tif',
    'undersupplied_population': 'undersupplied_population.tif',
    'oversupplied_population': 'oversupplied_population.tif',
    'reprojected_aois': 'reprojected_aois.gpkg',
    'aois_ids': 'aois_ids.tif',
}


def execute(args):
    """Urban Nature Access.

    Args:
        args['workspace_dir'] (string): (required) Output directory for
            intermediate, temporary and final files.
        args['results_suffix'] (string): (optional) String to append to any
            output file.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for executing the tasks of this model.  If omitted, computation
            will take place in the current process.
        args['lulc_raster_path'] (string): (required) A string path to a
            GDAL-compatible land-use/land-cover raster containing integer
            landcover codes.  Must be linearly projected in meters.
        args['lulc_attribute_table'] (string): (required) A string path to a
            CSV with the following columns:

            * ``lucode``: (required) the integer landcover code represented.
            * ``greenspace``: (required) ``0`` or ``1`` indicating whether
              this landcover code is (``1``) or is not (``0``) a greenspace
              pixel.
            * ``search_radius_m``: (required) the search radius for this
              greenspace landcover in meters. Required for all greenspace
              lucodes.

        args['population_raster_path'] (string): (required) A string path to a
            GDAL-compatible raster where pixels represent the population of
            that pixel.  Must be linearly projected in meters.
        args['aoi_vector_path'] (string): (required) A string path to a
            GDAL-compatible vector containing polygon areas of interest,
            typically administrative boundaries.  If this vector has any fields
            with fieldnames beginning with ``"pop_"``, these will be treated
            as representing the proportion of the population within an admin
            unit belonging to the given population group.  The name of the
            population group (everything other than a leading ``"pop_"``) must
            uniquely identify the group.
        args['greenspace_demand'] (number): (required) A positive, nonzero
            number indicating the required greenspace, in m² per capita.
        args['decay_function'] (string): (required) The selected kernel type.
            Must be one of the keys in ``KERNEL_TYPES``.
        args['population_group_radii_table'] (string): (optional) A table
            associating population groups with a search radius for that
            population group.

            TODO: must these group names match the admin unit vector
            groupnames?

        TODO: make sure the docstring matches actual requirements.

    Returns:
        ``None``
    """
    LOGGER.info('Starting Urban Nature Access Model')

    output_dir = os.path.join(args['workspace_dir'], 'output')
    intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')
    utils.make_directories([output_dir, intermediate_dir])

    suffix = utils.make_suffix_string(args, 'results_suffix')
    file_registry = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_dir)],
        suffix)

    work_token_dir = os.path.join(intermediate_dir, '_taskgraph_working_dir')
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous execution
    graph = taskgraph.TaskGraph(work_token_dir, n_workers)

    kernel_creation_functions = {
        KERNEL_LABEL_DICHOTOMY: dichotomous_decay_kernel_raster,
        # "exponential" is more consistent with other InVEST models'
        # terminology.  "Power function" is used in the design doc.
        KERNEL_LABEL_EXPONENTIAL: utils.exponential_decay_kernel_raster,
        KERNEL_LABEL_GAUSSIAN: utils.gaussian_decay_kernel_raster,
        KERNEL_LABEL_DENSITY: density_decay_kernel_raster,
    }
    # Since we have these keys defined in two places, I want to be super sure
    # that the labels match.
    assert sorted(kernel_creation_functions.keys()) == (
        sorted(ARGS_SPEC['args']['decay_function']['options']))

    decay_function = args['decay_function']
    LOGGER.info(f'Using decay function {decay_function}')

    # Align the population raster to the LULC.
    lulc_raster_info = pygeoprocessing.get_raster_info(
        args['lulc_raster_path'])

    squared_lulc_pixel_size = _square_off_pixels(args['lulc_raster_path'])
    lulc_alignment_task = graph.add_task(
        pygeoprocessing.warp_raster,
        kwargs={
            'base_raster_path': args['lulc_raster_path'],
            'target_pixel_size': squared_lulc_pixel_size,
            'target_bb': lulc_raster_info['bounding_box'],
            'target_raster_path': file_registry['aligned_lulc'],
            'resample_method': 'nearest',
        },
        target_path_list=[file_registry['aligned_lulc']],
        task_name='Resample LULC to have square pixels'
    )

    population_alignment_task = graph.add_task(
        _resample_population_raster,
        kwargs={
            'source_population_raster_path': args['population_raster_path'],
            'target_population_raster_path': file_registry[
                'aligned_population'],
            'lulc_pixel_size': squared_lulc_pixel_size,
            'lulc_bb': lulc_raster_info['bounding_box'],
            'lulc_projection_wkt': lulc_raster_info['projection_wkt'],
            'working_dir': intermediate_dir,
        },
        target_path_list=[file_registry['aligned_population']],
        task_name='Resample population to LULC resolution')

    aoi_reprojection_task = graph.add_task(
        _reproject_and_identify,
        kwargs={
            'base_vector_path': args['aoi_vector_path'],
            'target_projection_wkt': lulc_raster_info['projection_wkt'],
            'target_path': file_registry['reprojected_aois'],
            'driver_name': 'GPKG',
            'id_fieldname': ID_FIELDNAME,
        },
        task_name='Reproject admin units',
        target_path_list=[file_registry['reprojected_aois']],
        dependent_task_list=[lulc_alignment_task]
    )

    attr_table = utils.read_csv_to_dataframe(
        args['lulc_attribute_table'], to_lower=True)
    convolved_population_paths = {}  # search radius: convolved_population path
    convolved_population_tasks = {}  # search radius: convolved_population task
    kernel_paths = {}  # search_radius, kernel path
    kernel_tasks = {}  # search_radius, kernel task

    if args['search_radius_mode'] == RADIUS_OPT_UNIFORM:
        search_radii = set([float(args['search_radius'])])
    elif args['search_radius_mode'] == RADIUS_OPT_GREENSPACE:
        search_radii = set(attr_table[
            attr_table['greenspace'] == 1]['search_radius_m'].unique())
    elif args['search_radius_mode'] == RADIUS_OPT_POP_GROUP:
        pop_group_table = utils.read_csv_to_dataframe(
            args['population_group_radii_table'])
        search_radii = set(pop_group_table['search_radius_m'].unique())
    else:
        valid_options = ', '.join(
            ARGS_SPEC['args']['search_radius_mode']['options'].keys())
        raise ValueError(
            "Invalid search radius mode provided: "
            f"{args['search_radius_mode']}; must be one of {valid_options}")

    for search_radius_m in search_radii:
        search_radius_in_pixels = abs(
            search_radius_m / squared_lulc_pixel_size[0])
        kernel_path = os.path.join(
            intermediate_dir, f'kernel_{search_radius_m}{suffix}.tif')
        kernel_paths[search_radius_m] = kernel_path
        kernel_tasks[search_radius_m] = graph.add_task(
            # All kernel creation types have the same function signature
            kernel_creation_functions[decay_function],
            args=(search_radius_in_pixels, kernel_path),
            kwargs={'normalize': False},  # Model math calls for un-normalized
            task_name=(
                f'Create {decay_function} kernel - {search_radius_m}m'),
            target_path_list=[kernel_path]
        )

    if args['search_radius_mode'] == RADIUS_OPT_UNIFORM:
        search_radius_m = list(search_radii)[0]

        decayed_population_path = os.path.join(
            intermediate_dir,
            f'decayed_population_within_{search_radius_m}{suffix}.tif')
        decayed_population_task = graph.add_task(
            _convolve_and_set_lower_bounds_for_population,
            kwargs={
                'signal_path_band': (file_registry['aligned_population'], 1),
                'kernel_path_band': (kernel_paths[search_radius_m], 1),
                'target_path': convolved_population_paths[search_radius_m],
                'working_dir': intermediate_dir,
            },
            task_name=f'Convolve population - {search_radius_m}m',
            target_path_list=[decayed_population_path],
            dependent_task_list=[
                kernel_tasks[search_radius_m], population_alignment_task])

        greenspace_pixels_path = os.path.join(
            intermediate_dir, f'greenspace_area{suffix}.tif')
        greenspace_reclassification_task = graph.add_task(
            _reclassify_greenspace_area,
            kwargs={
                'lulc_raster_path': file_registry['aligned_lulc'],
                'lulc_attribute_table': args['lulc_attribute_table'],
                'target_raster_path': greenspace_pixels_path,
            },
            target_path_list=[greenspace_pixels_path],
            task_name='Identify greenspace areas',
            dependent_task_list=[lulc_alignment_task]
        )

        greenspace_population_ratio_path = os.path.join(
            intermediate_dir,
            ('greenspace_population_ratio{suffix}.tif'))
        greenspace_population_ratio_task = graph.add_task(
            _calculate_greenspace_population_ratio,
            args=(greenspace_pixels_path,
                  decayed_population_task,
                  greenspace_population_ratio_path),
            task_name=(
                '2SFCA: Calculate R_j greenspace/population ratio - '
                f'{search_radius_m}'),
            target_path_list=[greenspace_population_ratio_path],
            dependent_task_list=[
                greenspace_reclassification_task, decayed_population_task,
            ])

        greenspace_supply_path = os.path.join(
            intermediate_dir, f'greenspace_supply{suffix}.tif')
        greenspace_supply_task = graph.add_task(
            pygeoprocessing.convolve_2d,
            kwargs={
                'signal_path_band': (
                    greenspace_population_ratio_path, 1),
                'kernel_path_band': (kernel_path, 1),
                'target_path': greenspace_supply_path,
                'working_dir': intermediate_dir,
                # Insurance against future pygeoprocessing API changes.  The
                # target nodata right now is the minimum possible numpy float32
                # value, which is also what we use here as FLOAT32_NODATA.
                'target_nodata': FLOAT32_NODATA,
            },
            task_name=f'2SFCA - greenspace supply',
            target_path_list=[greenspace_supply_path],
            dependent_task_list=[
                kernel_task, greenspace_population_ratio_task])


    for search_radius_m in search_radii:
        search_radius_in_pixels = abs(
            search_radius_m / squared_lulc_pixel_size[0])
        kernel_paths[search_radius_m] = os.path.join(
            intermediate_dir, f'kernel_{search_radius_m}{suffix}.tif')
        kernel_tasks[search_radius_m] = graph.add_task(
            # All kernel creation types have the same function signature
            kernel_creation_functions[decay_function],
            args=(search_radius_in_pixels, kernel_paths[search_radius_m]),
            kwargs={'normalize': False},  # Model math calls for un-normalized
            task_name=(
                f'Create {decay_function} kernel - {search_radius_m}m'),
            target_path_list=[kernel_paths[search_radius_m]]
        )

        # Convolving the population within a non-normalized kernel gives us the
        # number of people (possibly weighted, depending on the kernel) within
        # the target search radius.
        convolved_population_paths[search_radius_m] = os.path.join(
            intermediate_dir,
            f'decayed_population_within_{search_radius_m}{suffix}.tif')
        convolved_population_tasks[search_radius_m] = graph.add_task(
            _convolve_and_set_lower_bounds_for_population,
            kwargs={
                'signal_path_band': (file_registry['aligned_population'], 1),
                'kernel_path_band': (kernel_paths[search_radius_m], 1),
                'target_path': convolved_population_paths[search_radius_m],
                'working_dir': intermediate_dir,
            },
            task_name=f'Convolve population - {search_radius_m}m',
            target_path_list=[convolved_population_paths[search_radius_m]],
            dependent_task_list=[
                kernel_tasks[search_radius_m],
                population_alignment_task,
            ])

    greenspace_attrs = attr_table[attr_table['greenspace'] == 1]
    search_radii = greenspace_attrs['search_radius_m'].unique()
    if len(search_radii) == 1:
        LOGGER.info(
            'Only 1 greenspace search radius found. '
            'Running 2SFCA in unified greenspace mode, calculating '
            'greenspace supply once for all LULC classes.')

        # If all of the search radii are the same, then we run 2SFCA with all
        # of the greenspace lucodes in one batch.
        radius_to_lucodes = [
            (search_radii[0], greenspace_attrs['lucode'].unique())]
    else:
        LOGGER.info(
            'Multiple search radii found. '
            'Running 2SFCA in split greenspace mode, calculating greenspace '
            'supply per LULC class.')

        # If there are varying search radii, we run 2SFCA once for each
        # radius:lucode combination so that modelers can see the contribution
        # of each landcover code to the total greenspace supply.  This comes at
        # the cost of more convolutions and more runtime (and disk space).
        radius_to_lucodes = [
            (search_radius_m, [lucode])
            for ((search_radius_m, lucode), _) in
            greenspace_attrs.groupby(['search_radius_m', 'lucode'])
        ]

    partial_greenspace_paths = []
    partial_greenspace_tasks = []
    for search_radius_m, lucodes in radius_to_lucodes:
        if len(lucodes) == 1:
            lucode_str = f'{lucodes[0]}'
        else:
            lucode_str = 'all'
        LOGGER.info(
            f'Using search radius {search_radius_m} for lucodes '
            f'{" ".join([str(c) for c in lucodes])}')

        # reclassify landcover to greenspace area needed for this kernel
        greenspace_pixels_path = os.path.join(
            intermediate_dir,
            f'greenspace_{lucode_str}_{search_radius_m}{suffix}.tif')
        greenspace_reclassification_task = graph.add_task(
            _reclassify_greenspace_area,
            kwargs={
                'lulc_raster_path': file_registry['aligned_lulc'],
                'lulc_attribute_table': args['lulc_attribute_table'],
                'target_raster_path': greenspace_pixels_path,
                'only_these_greenspace_codes': set(lucodes),
            },
            target_path_list=[greenspace_pixels_path],
            task_name='Identify greenspace areas',
            dependent_task_list=[lulc_alignment_task]
        )

        greenspace_population_ratio_path = os.path.join(
            intermediate_dir,
            ('greenspace_population_ratio_'
             f'{lucode_str}_{search_radius_m}{suffix}.tif'))
        greenspace_population_ratio_task = graph.add_task(
            _calculate_greenspace_population_ratio,
            args=(greenspace_pixels_path,
                  convolved_population_paths[search_radius_m],
                  greenspace_population_ratio_path),
            task_name=(
                '2SFCA: Calculate R_j greenspace/population ratio - '
                f'{search_radius_m}'),
            target_path_list=[greenspace_population_ratio_path],
            dependent_task_list=[
                greenspace_reclassification_task,
                convolved_population_tasks[search_radius_m],
            ])

        partial_greenspace_supply_path = os.path.join(
            intermediate_dir,
            f'greenspace_supply_{lucode_str}_{search_radius_m}{suffix}.tif')
        convolved_greenspace_population_ratio_task = graph.add_task(
            pygeoprocessing.convolve_2d,
            kwargs={
                'signal_path_band': (
                    greenspace_population_ratio_path, 1),
                'kernel_path_band': (kernel_paths[search_radius_m], 1),
                'target_path': partial_greenspace_supply_path,
                'working_dir': intermediate_dir,
                # Insurance against future pygeoprocessing API changes.  The
                # target nodata right now is the minimum possible numpy float32
                # value, which is also what we use here as FLOAT32_NODATA.
                'target_nodata': FLOAT32_NODATA,
            },
            task_name=(
                f'2SFCA - greenspace supply - {lucode_str}, {search_radius_m}m'
            ),
            target_path_list=[partial_greenspace_supply_path],
            dependent_task_list=[
                kernel_tasks[search_radius_m],
                greenspace_population_ratio_task,
            ])
        partial_greenspace_paths.append(partial_greenspace_supply_path)
        partial_greenspace_tasks.append(
            convolved_greenspace_population_ratio_task)

    greenspace_supply_task = graph.add_task(
        ndr._sum_rasters,
        kwargs={
            'raster_path_list': partial_greenspace_paths,
            'target_nodata': FLOAT32_NODATA,
            'target_result_path': file_registry['greenspace_supply'],
        },
        task_name='2SFCA - greenspace supply total',
        target_path_list=[file_registry['greenspace_supply']],
        dependent_task_list=partial_greenspace_tasks
    )

    # This is "SUP_DEMi_cap" from the user's guide
    per_capita_greenspace_budget_task = graph.add_task(
        pygeoprocessing.raster_calculator,
        kwargs={
            'base_raster_path_band_const_list': [
                (file_registry['greenspace_supply'], 1),
                (float(args['greenspace_demand']), 'raw')
            ],
            'local_op': _greenspace_budget_op,
            'target_raster_path': file_registry['greenspace_budget'],
            'datatype_target': gdal.GDT_Float32,
            'nodata_target': FLOAT32_NODATA
        },
        task_name='Calculate per-capita greenspace budget',
        target_path_list=[file_registry['greenspace_budget']],
        dependent_task_list=[
            greenspace_supply_task,
        ])

    # This is "SUP_DEMi" from the user's guide
    greenspace_supply_demand_task = graph.add_task(
        pygeoprocessing.raster_calculator,
        kwargs={
            'base_raster_path_band_const_list': [
                (file_registry['greenspace_budget'], 1),
                (file_registry['aligned_population'], 1)
            ],
            'local_op': _greenspace_supply_demand_op,
            'target_raster_path': (
                file_registry['greenspace_supply_demand_budget']),
            'datatype_target': gdal.GDT_Float32,
            'nodata_target': FLOAT32_NODATA
        },
        task_name='Calculate per-capita greenspace supply-demand',
        target_path_list=[file_registry['greenspace_supply_demand_budget']],
        dependent_task_list=[
             per_capita_greenspace_budget_task,
             population_alignment_task,
        ])

    supply_population_tasks = []
    for supply_type, op in [('under', numpy.less), ('over', numpy.greater)]:
        supply_population_tasks.append(graph.add_task(
            pygeoprocessing.raster_calculator,
            kwargs={
                'base_raster_path_band_const_list': [
                    (file_registry['aligned_population'], 1),
                    (file_registry['greenspace_budget'], 1),
                    (op, 'raw'),  # numpy element-wise comparator
                ],
                'local_op': _filter_population,
                'target_raster_path':
                    file_registry[f'{supply_type}supplied_population'],
                'datatype_target': gdal.GDT_Float32,
                'nodata_target': FLOAT32_NODATA,
            },
            task_name=f'Determine {supply_type}supplied populations',
            target_path_list=[
                file_registry[f'{supply_type}supplied_population']],
            dependent_task_list=[
                greenspace_supply_demand_task,
                population_alignment_task,
            ]))

    # only do split population if there are population group fields in the
    # AOIs/admin units vector
    aoi_reprojection_task.join()
    split_population_fields = list(
        filter(lambda x: re.match(POP_FIELD_REGEX, x),
               validation.load_fields_from_vector(
                   file_registry['reprojected_aois'])))

    if split_population_fields or pop_group_table:
        if _geometries_overlap(file_registry['reprojected_aois']):
            LOGGER.warning(
                "Some administrative boundaries overlap, which will affect "
                "the accuracy of supply rasters per population group. "
                "Summary statistics in "
                f"{os.path.basename(file_registry['aois'])} are unaffected.")

        aois_rasterization_task = graph.add_task(
            _rasterize_aois,
            kwargs={
                'base_raster_path': file_registry['aligned_lulc'],
                'aois_vector_path':
                    file_registry['reprojected_aois'],
                'target_raster_path': file_registry['aois_ids'],
                'id_fieldname': ID_FIELDNAME,
            },
            task_name='Rasterize the admin units vector',
            target_path_list=[file_registry['aois_ids']],
            dependent_task_list=[
                aoi_reprojection_task, lulc_alignment_task]
        )

    if split_population_fields:
        LOGGER.info(
            "Split population groups found: "
            f"{', '.join(split_population_fields)}, starting split "
            "population optional module.")

        for pop_field in split_population_fields:
            field_value_map = _read_field_from_vector(
                file_registry['reprojected_aois'],
                ID_FIELDNAME, pop_field)
            for supply_type, supply_filepath in (
                    ('under', file_registry['undersupplied_population']),
                    ('over', file_registry['oversupplied_population'])):
                groupname = re.sub(POP_FIELD_REGEX, '', pop_field)
                supply_to_group_filepath = os.path.join(
                    intermediate_dir,
                    f'{supply_type}supplied_population_{groupname}{suffix}.tif'
                )
                _ = graph.add_task(
                    _reclassify_and_multiply,
                    kwargs={
                        'aois_raster_path':
                            file_registry['aois_ids'],
                        'reclassification_map': field_value_map,
                        'supply_raster_path': supply_filepath,
                        'target_raster_path': supply_to_group_filepath,
                    },
                    task_name=(
                        f'Map proportion of {supply_type}supplied for'
                        f'{groupname}'),
                    target_path_list=[supply_filepath],
                    dependent_task_list=[aois_rasterization_task]
                )

    if pop_group_table:
        LOGGER.info("Search radii for population groups provided, "
                    "starting split population radii module.")

        for pop_group, pop_group_radius in pop_group_table[
                ['pop_group', 'search_radius_m']].itertuples(index=False):
            field_value_map = _read_field_from_vector(
                file_registry['reprojected_aois'],
                ID_FIELDNAME, pop_group)
            _ = graph.add_task(
                _reclassify_and_multiply,
                kwargs={
                    'aois_raster_path':
                        file_registry['aois_ids'],
                    'reclassification_map': field_value_map,
                    'supply_raster_path': supply_filepath,
                    'target_raster_path': supply_to_group_filepath,
                },
                task_name=(
                    f'Map proportion of {supply_type}supplied for'
                    f'{groupname}'),
                target_path_list=[supply_filepath],
                dependent_task_list=[aois_rasterization_task]
            )

            _ = graph.add_task(
                pygeoprocessing.raster_calculator,
                kwargs={
                    "base_raster_path_band_const_list": None,
                    "local_op": None,
                    "target_raster_path": None,
                    "datatype_target": None,
                    "nodata_target": None,
                    "calc_raster_stats": True,
                    "use_shared_memory": False,
                    "largest_block": 65536,
                    "max_timeout": 60.0,
                    "raster_driver_creation_tuple": ['GTIFF', ['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']],
                },
                task_name='',
                target_path_list=[],
                dependent_task_list=[]
            )

            convolve_2d()
            # JUST DO THE DAMN THING THAT NEEDS TO BE DONE
            # GET IT WORKING, OPTIMIZE LATER
            # * Multiply population raster by the pop group proportion
            # * convolve the population proportions with the pop group search
            #   radius

        # * sum all of the populations created in the pop group loop.
        # * calculate a new R_j (greenspace/population ratio) that uses
        #   per-popgroup search radii
        # * for each population group:
        #   * calculate Ai,gn (greenspace supply to the population group)
        # * Sum all Ai,gn*(pop_group proportion) to get a new Ai (total greenspace supply)
        # * Calculate new per-capita greenspace budget SUP_DEMi_cap (raster) from new Ai
        # * Calculate new per-capita greenspace budget for the group SUP_DEMi_cap,gn
        # * etc.

    _ = graph.add_task(
        _admin_level_supply_demand,
        kwargs={
            'greenspace_budget_path': file_registry[
                'greenspace_supply_demand_budget'],
            'population_path': file_registry['aligned_population'],
            'aoi_vector_path': file_registry['reprojected_aois'],
            'target_aoi_vector_path': file_registry['aois'],
            'undersupplied_populations_path': file_registry[
                'undersupplied_population'],
            'oversupplied_populations_path': file_registry[
                'oversupplied_population'],
        },
        task_name='Aggregate supply-demand to the admin units',
        target_path_list=[file_registry['aois']],
        dependent_task_list=[
            greenspace_supply_demand_task,
            population_alignment_task,
            *supply_population_tasks
        ])

    graph.close()
    graph.join()
    LOGGER.info('Finished Urban Nature Access Model')


def _geometries_overlap(vector_path):
    """Check if the geometries of the vector's first layer overlap.

    Args:
        vector_path (string): The path to a GDAL vector.

    Returns:
        bool: Whether there's numerically significant overlap between polygons
            in the first layer.

    """
    vector = gdal.OpenEx(vector_path)
    layer = vector.GetLayer()
    area_sum = 0
    geometries = []
    for feature in layer:
        ogr_geom = feature.GetGeometryRef()
        area_sum += ogr_geom.Area()
        shapely_geom = shapely.wkb.loads(bytes(ogr_geom.ExportToWkb()))
        geometries.append(shapely_geom)

    layer = None
    vector = None

    union_area = shapely.ops.unary_union(geometries).area
    if math.isclose(union_area, area_sum):
        return False
    return True


def _reproject_and_identify(base_vector_path, target_projection_wkt,
                            target_path, driver_name, id_fieldname):
    """Reproject a vector and add an ID field.

    Args:
        base_vector_path (string): The string path to the source vector.
        target_projection_wkt (string): The WKT of the target projection.
        target_path (string): The string path to where the new vector should be
            saved.
        driver_name (string): The GDAL driver name of the target vector.
        id_fieldname (string): The name of the ID field.  A new field with this
            name and an integer type will be created in the target vector.
            Each feature in the target vector will be assigned a unique integer
            ID.

    Returns:
        ``None``
    """
    pygeoprocessing.reproject_vector(
        base_vector_path, target_projection_wkt, target_path,
        driver_name=driver_name)

    vector = gdal.OpenEx(target_path, gdal.GA_Update)
    layer = vector.GetLayer()
    field = ogr.FieldDefn(id_fieldname, ogr.OFTInteger)
    layer.CreateField(field)

    layer.StartTransaction()
    for field_id, feature in enumerate(layer):
        feature.SetField(id_fieldname, field_id)
        layer.SetFeature(feature)
    layer.CommitTransaction()
    layer = None
    vector = None


def _reclassify_and_multiply(
        aois_raster_path, reclassification_map, supply_raster_path,
        target_raster_path):
    """Create a raster of greenspace supply given areas of interest.

    This is done by:

        1. Reclassifying AOI IDs to population group ratios and then
        2. Multiplying the population group ratios by the greenspace supply.

    Args:
        aois_raster_path (string): The path to a raster of integers
            identifying which admin unit a pixel belongs to.
        reclassification_map (dict): A dict mapping integer admin unit IDs to
            float population proportions (values 0-1) for a given population
            group.
        supply_raster_path (string): A string path to a raster of greenspace
            supply values for the total population.
        target_raster_path (string): The string path to where the resulting
            supply-to-group raster should be written.

    Returns:
        ``None``
    """
    pygeoprocessing.reclassify_raster(
        (aois_raster_path, 1), reclassification_map, target_raster_path,
        gdal.GDT_Float32, FLOAT32_NODATA)

    pop_group_raster = gdal.OpenEx(target_raster_path,
                                   gdal.GA_Update | gdal.OF_RASTER)
    pop_group_band = pop_group_raster.GetRasterBand(1)
    pop_group_nodata = pop_group_band.GetNoDataValue()
    supply_raster = gdal.OpenEx(supply_raster_path,
                                gdal.GA_ReadOnly | gdal.OF_RASTER)
    supply_band = supply_raster.GetRasterBand(1)
    supply_nodata = supply_band.GetNoDataValue()
    for block_info in pygeoprocessing.iterblocks((target_raster_path, 1),
                                                 offset_only=True):
        pop_group_proportion_block = pop_group_band.ReadAsArray(**block_info)
        supply_block = supply_band.ReadAsArray(**block_info)

        valid_mask = (
            ~utils.array_equals_nodata(
                pop_group_proportion_block, pop_group_nodata) &
            ~utils.array_equals_nodata(supply_block, supply_nodata))
        target_block = numpy.full(supply_block.shape, FLOAT32_NODATA,
                                  dtype=numpy.float32)
        target_block[valid_mask] = (
            pop_group_proportion_block[valid_mask] * supply_block[valid_mask])
        pop_group_band.WriteArray(
            target_block, xoff=block_info['xoff'], yoff=block_info['yoff'])

    pop_group_band = None
    pop_group_raster = None
    supply_band = None
    supply_raster = None


def _read_field_from_vector(vector_path, key_field, value_field):
    """Read a field from a vector's first layer.

    Args:
        vector_path (string): The string path to a vector.
        key_field (string): The string key field within the vector.
            ``key_field`` must exist within the vector at ``vector_path``.
            ``key_field`` is case-sensitive.
        value_field (string): The string value field within the vector.
            ``value_field`` must exist within the vector at ``vector_path``.
            ``value_field`` is case-sensitive.

    Returns:
        attribute_map (dict): A dict mapping each ``key_field`` key to
            the corresponding ``value_field`` value.
    """
    vector = gdal.OpenEx(vector_path)
    layer = vector.GetLayer()
    attribute_map = {}
    for feature in layer:
        attribute_map[
            feature.GetField(key_field)] = feature.GetField(value_field)
    return attribute_map


def _rasterize_aois(base_raster_path, aois_vector_path,
                    target_raster_path, id_fieldname):
    """Rasterize the admin units vector onto a new raster.

    Args:
        base_raster_path (string): The string path to a raster on disk to be
            used as a template raster.
        aois_vector_path (string): The path to a vector on disk of areas of
            interest, typically administrative units.  The ``id_fieldname``
            feature of the features in this vector will be rasterized onto a
            new raster.
        target_raster_path (string): The path to a new UInt32 raster created on
            disk with new values burned into it.
        id_fieldname (string): The fieldname of the ID field to rasterize.

    Returns:
        ``None``
    """
    pygeoprocessing.new_raster_from_base(
        base_raster_path, target_raster_path, gdal.GDT_UInt32,
        [UINT32_NODATA], [UINT32_NODATA])

    pygeoprocessing.rasterize(
        aois_vector_path, target_raster_path,
        option_list=[f"ATTRIBUTE={id_fieldname}"])


def _reclassify_greenspace_area(
        lulc_raster_path, lulc_attribute_table, target_raster_path,
        only_these_greenspace_codes=None):
    """Reclassify LULC pixels into the greenspace area they represent.

    After execution, greenspace pixels will have values representing the
    pixel's area, while non-greenspace pixels will have a pixel value of 0.
    Nodata values will propagate to the output raster.

    Args:
        lulc_raster_path (string): The path to a land-use/land-cover raster.
        lulc_attribute_table (string): The path to a CSV table representing
            LULC attributes.  Must have "lucode" and "greenspace" columns.
        target_raster_path (string): Where the reclassified greenspace raster
            should be written.
        only_these_greenspace_codes=None (iterable or None): If ``None``, all
            lucodes with a ``greenspace`` value of 1 will be reclassified to 1.
            If an iterable, must be an iterable of landuse codes matching codes
            in the lulc attribute table.  Only these landcover codes will have
            greenspace area classified in the target raster path.

    Returns:
        ``None``
    """
    attribute_table_dict = utils.build_lookup_from_csv(
        lulc_attribute_table, key_field='lucode')

    squared_pixel_area = abs(
        numpy.multiply(*_square_off_pixels(lulc_raster_path)))

    if only_these_greenspace_codes:
        valid_greenspace_codes = set(only_these_greenspace_codes)
    else:
        valid_greenspace_codes = set(
            lucode for lucode, attributes in attribute_table_dict.items()
            if (attributes['greenspace']) == 1)

    greenspace_area_map = {}
    for lucode, attributes in attribute_table_dict.items():
        greenspace_area = 0
        if lucode in valid_greenspace_codes:
            greenspace_area = squared_pixel_area
        greenspace_area_map[lucode] = greenspace_area

    lulc_raster_info = pygeoprocessing.get_raster_info(lulc_raster_path)
    greenspace_area_map[lulc_raster_info['nodata'][0]] = FLOAT32_NODATA

    utils.reclassify_raster(
        raster_path_band=(lulc_raster_path, 1),
        value_map=greenspace_area_map,
        target_raster_path=target_raster_path,
        target_datatype=gdal.GDT_Float32,
        target_nodata=FLOAT32_NODATA,
        error_details={
            'raster_name': ARGS_SPEC['args']['lulc_raster_path']['name'],
            'column_name': 'greenspace',
            'table_name': ARGS_SPEC['args']['lulc_attribute_table']['name'],
        }
    )


def _filter_population(population, greenspace_budget, numpy_filter_op):
    """Filter the population by a defined op and the greenspace budget.

    Note:
        The ``population`` and ``greenspace_budget`` inputs must have the same
        shape and must both use ``FLOAT32_NODATA`` as their nodata value.

    Args:
        population (numpy.array): A numpy array with population counts.
        greenspace_budget (numpy.array): A numpy array with the greenspace
            budget values.
        numpy_filter_op (callable): A function that takes a numpy array as
            parameter 1 and a scalar value as parameter 2.  This function must
            return a boolean numpy array of the same shape as parameter 1.

    Returns:
        A ``numpy.array`` with the population values where the
        ``greenspace_budget`` pixels match the ``numpy_filter_op``.
    """
    population_matching_filter = numpy.full(
        population.shape, FLOAT32_NODATA, dtype=numpy.float32)
    valid_pixels = (
        ~numpy.isclose(greenspace_budget, FLOAT32_NODATA) &
        ~numpy.isclose(population, FLOAT32_NODATA))

    population_matching_filter[valid_pixels] = numpy.where(
        numpy_filter_op(greenspace_budget[valid_pixels], 0),
        population[valid_pixels],  # If condition is true, use population
        0  # If condition is false, use 0
    )
    return population_matching_filter


def _admin_level_supply_demand(
        greenspace_budget_path, population_path, aoi_vector_path,
        target_aoi_vector_path, undersupplied_populations_path,
        oversupplied_populations_path):
    """Calculate average greenspace supply per admin unit.

    Note:
        The greenspace budget raster and population raster must align
        perfectly in terms of pixel sizes and raster dimensions, and must have
        the same projection.

    Args:
        greenspace_budget_path (string): The path to the greenspace budget
            raster on disk.
        population_path (string): The path to the population raster on disk.
        aoi_vector_path (string): The path to a vector of areas of interest,
            typically administrative units.
        target_aoi_vector_path (string): The path to where the target
            aoi/admin units vector will be created on disk.

    Returns:
        ``None``
    """
    source_vector = ogr.Open(aoi_vector_path)
    driver = ogr.GetDriverByName('GPKG')
    driver.CopyDataSource(source_vector, target_aoi_vector_path)
    source_vector = None

    target_vector = gdal.OpenEx(target_aoi_vector_path, gdal.GA_Update)
    target_layer = target_vector.GetLayer()

    supply_sum_fieldname = 'SUP_DEMadm_cap'
    undersupply_fieldname = 'Pund_adm'
    oversupply_fieldname = 'Povr_adm'
    target_fieldnames = []
    population_groupnames = []
    for defn in target_layer.schema:
        name = defn.GetName()
        if not re.match(POP_FIELD_REGEX, name):
            continue
        groupname = re.sub(POP_FIELD_REGEX, '', name)
        population_groupnames.append(groupname)
        target_fieldnames.append(f'{oversupply_fieldname}_{groupname}')
        target_fieldnames.append(f'{undersupply_fieldname}_{groupname}')

    for fieldname in (supply_sum_fieldname,
                      undersupply_fieldname,
                      oversupply_fieldname,
                      *target_fieldnames):
        field = ogr.FieldDefn(fieldname, ogr.OFTReal)
        field.SetWidth(24)
        field.SetPrecision(11)
        target_layer.CreateField(field)

    greenspace_stats = pygeoprocessing.zonal_statistics(
        (greenspace_budget_path, 1), target_aoi_vector_path)
    population_stats = pygeoprocessing.zonal_statistics(
        (population_path, 1), target_aoi_vector_path)
    undersupplied_stats = pygeoprocessing.zonal_statistics(
        (undersupplied_populations_path, 1), target_aoi_vector_path)
    oversupplied_stats = pygeoprocessing.zonal_statistics(
        (oversupplied_populations_path, 1), target_aoi_vector_path)

    target_layer.StartTransaction()
    for feature in target_layer:
        feature_id = feature.GetFID()

        avg_greenspace_supply_demand = (
            greenspace_stats[feature_id]['sum'] /
            population_stats[feature_id]['sum'])
        feature.SetField(supply_sum_fieldname, avg_greenspace_supply_demand)
        undersupply = undersupplied_stats[feature_id]['sum']
        oversupply = oversupplied_stats[feature_id]['sum']
        feature.SetField(undersupply_fieldname, undersupply)
        feature.SetField(oversupply_fieldname, oversupply)
        for pop_groupname in population_groupnames:
            group_proportion = feature.GetField(f'pop_{pop_groupname}')
            feature.SetField(
                f'{oversupply_fieldname}_{pop_groupname}',
                oversupply * group_proportion)
            feature.SetField(
                f'{undersupply_fieldname}_{pop_groupname}',
                undersupply * group_proportion)
        target_layer.SetFeature(feature)
    target_layer.CommitTransaction()

    target_layer = None
    target_vector = None


def _greenspace_budget_op(greenspace_supply, greenspace_demand):
    """Calculate the per-capita greenspace budget.

    This is the amount of greenspace that each pixel has above (positive
    values) or below (negative values) the user-defined ``greenspace_demand``
    value.

    Args:
        greenspace_supply (numpy.array): The supply of greenspace available to
            each person in the population.  This is ``Ai`` in the User's Guide.
            This matrix must have ``FLOAT32_NODATA`` as its nodata value.
        greenspace_demand (float): The per-person greenspace requirement, in

    Returns:
        A ``numpy.array`` of the calculated greenspace budget.
    """
    budget = numpy.full(
        greenspace_supply.shape, FLOAT32_NODATA, dtype=numpy.float32)
    valid_pixels = ~numpy.isclose(greenspace_supply, FLOAT32_NODATA)
    budget[valid_pixels] = greenspace_supply[valid_pixels] - greenspace_demand
    return budget


def _greenspace_supply_demand_op(greenspace_budget, population):
    """Calculate the supply/demand of greenspace per person.

    Args:
        greenspace_budget (numpy.array): The area of greenspace budgeted to
            each person, relative to a minimum required per-person area of
            greenspace.  This matrix must have ``FLOAT32_NODATA`` as its nodata
            value.  This matrix must be the same size and shape as
            ``population``.
        population (numpy.array): Pixel values represent the population count
            of the pixel.  This matrix must be the same size and shape as
            ``greenspace_budget``, and must have ``FLOAT32_NODATA`` as its
            nodata value.

    Returns:
        A ``numpy.array`` of the area (in square meters) of greenspace supplied
        to each individual in each pixel.
    """
    supply_demand = numpy.full(
        greenspace_budget.shape, FLOAT32_NODATA, dtype=numpy.float32)
    valid_pixels = (
        ~numpy.isclose(greenspace_budget, FLOAT32_NODATA) &
        ~numpy.isclose(population, FLOAT32_NODATA))
    supply_demand[valid_pixels] = (
        greenspace_budget[valid_pixels] * population[valid_pixels])
    return supply_demand


def _calculate_greenspace_population_ratio(
        greenspace_area_raster_path, convolved_population_raster_path,
        target_ratio_raster_path):
    """Calculate the greenspace-population ratio R_j.

    Args:
        greenspace_area_raster_path (string): The path to a raster representing
            the area of the pixel that represents greenspace.  Pixel values
            will be ``0`` if there is no greenspace.
        convolved_population_raster_path (string): The path to a raster
            representing population counts that have been convolved over some
            search kernel and perhaps weighted.
        target_ratio_raster_path (string): The path to where the target
            greenspace-population raster should be written.

    Returns:
        ``None``.
    """
    greenspace_nodata = pygeoprocessing.get_raster_info(
        greenspace_area_raster_path)['nodata'][0]
    population_nodata = pygeoprocessing.get_raster_info(
        convolved_population_raster_path)['nodata'][0]

    def _greenspace_population_ratio(greenspace_area, convolved_population):
        """Calculate the greenspace-population ratio R_j.

        Args:
            greenspace_area (numpy.array): A numpy array representing the area
                of greenspace in the pixel.  Pixel values will be ``0`` if
                there is no greenspace.  Pixel values may also match
                ``greenspace_nodata``.
            convolved_population (numpy.array): A numpy array where each pixel
                represents the total number of people within a search radius of
                each pixel, perhaps weighted by a search kernel.

        Returns:
            A numpy array with the ratio ``R_j`` representing the
            greenspace-population ratio with the following constraints:

                * ``convolved_population`` pixels that are numerically close to
                  ``0`` are snapped to ``0`` to avoid unrealistically small
                  denominators in the final ratio.
                * Any non-greenspace pixels will have a value of ``0`` in the
                  output matrix.
        """
        # ASSUMPTION: population nodata value is not close to 0.
        #  Shouldn't be if we're coming from convolution.
        out_array = numpy.full(
            greenspace_area.shape, FLOAT32_NODATA, dtype=numpy.float32)

        # Small negative values should already have been filtered out in
        # another function after the convolution.
        # This avoids divide-by-zero errors when taking the ratio.
        valid_pixels = (convolved_population > 0)

        # R_j is a ratio only calculated for the greenspace pixels.
        greenspace_pixels = ~numpy.isclose(greenspace_area, 0)
        valid_pixels &= greenspace_pixels
        if population_nodata is not None:
            valid_pixels &= ~utils.array_equals_nodata(
                convolved_population, population_nodata)

        if greenspace_nodata is not None:
            valid_pixels &= ~utils.array_equals_nodata(
                greenspace_area, greenspace_nodata)

        # The user's guide specifies that if the population in the search
        # radius is numerically 0, the greenspace/population ratio should be
        # set to the greenspace area.
        # A consequence of this is that as the population approaches 0 from the
        # positive side, the ratio will approach infinity.
        # After checking with the science team, we decided that where the
        # population is less than or equal to 1, the calculated
        # greenspace/population ratio would be set to the available greenspace
        # on that pixel.
        population_close_to_zero = (convolved_population <= 1.0)
        out_array[population_close_to_zero] = (
            greenspace_area[population_close_to_zero])
        out_array[~greenspace_pixels] = 0

        valid_pixels_with_population = (
            valid_pixels & (~population_close_to_zero))
        out_array[valid_pixels_with_population] = (
            greenspace_area[valid_pixels_with_population] /
            convolved_population[valid_pixels_with_population])

        # eliminate pixel values < 0
        out_array[valid_pixels & (out_array < 0)] = 0

        return out_array

    pygeoprocessing.raster_calculator(
        [(greenspace_area_raster_path, 1),
         (convolved_population_raster_path, 1)],
        _greenspace_population_ratio, target_ratio_raster_path,
        gdal.GDT_Float32, FLOAT32_NODATA)


def _convolve_and_set_lower_bounds_for_population(
        signal_path_band, kernel_path_band, target_path, working_dir):
    """Convolve a raster and set all values below 0 to 0.

    Args:
        signal_path_band (tuple): A 2-tuple of (signal_raster_path, band_index)
            to use as the signal raster in the convolution.
        kernel_path_band (tuple): A 2-tuple of (kernel_raster_path, band_index)
            to use as the kernel raster in the convolution.  This kernel should
            be non-normalized.
        target_path (string): Where the target raster should be written.
        working_dir (string): The working directory that
            ``pygeoprocessing.convolve_2d`` may use for its intermediate files.

    Returns:
        ``None``
    """
    pygeoprocessing.convolve_2d(
        signal_path_band=signal_path_band,
        kernel_path_band=kernel_path_band,
        target_path=target_path,
        working_dir=working_dir)

    # Sometimes there are negative values that should have been clamped to 0 in
    # the convolution but weren't, so let's clamp them to avoid support issues
    # later on.
    target_raster = gdal.OpenEx(target_path, gdal.GA_Update)
    target_band = target_raster.GetRasterBand(1)
    target_nodata = target_band.GetNoDataValue()
    for block_data in pygeoprocessing.iterblocks(
            (target_path, 1), offset_only=True):
        block = target_band.ReadAsArray(**block_data)
        valid_pixels = slice(None)
        if target_nodata is not None:
            valid_pixels = ~numpy.isclose(block, target_nodata)
        block[(block < 0) & valid_pixels] = 0
        target_band.WriteArray(
            block, xoff=block_data['xoff'], yoff=block_data['yoff'])

    target_band = None
    target_raster = None


def _square_off_pixels(raster_path):
    """Create square pixels from the provided raster.

    The pixel dimensions produced will respect the sign of the original pixel
    dimensions and will be the mean of the absolute source pixel dimensions.

    Args:
        raster_path (string): The path to a raster on disk.

    Returns:
        A 2-tuple of ``(pixel_width, pixel_height)``, in projected units.
    """
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    pixel_width, pixel_height = raster_info['pixel_size']

    if abs(pixel_width) == abs(pixel_height):
        return (pixel_width, pixel_height)

    pixel_tuple = ()
    average_absolute_size = (abs(pixel_width) + abs(pixel_height)) / 2
    for pixel_dimension_size in (pixel_width, pixel_height):
        # This loop allows either or both pixel dimension(s) to be negative
        sign_factor = 1
        if pixel_dimension_size < 0:
            sign_factor = -1

        pixel_tuple += (average_absolute_size * sign_factor,)

    return pixel_tuple


# TODO: refactor this into raster_calculator and align_and_resize...
def _resample_population_raster(
        source_population_raster_path, target_population_raster_path,
        lulc_pixel_size, lulc_bb, lulc_projection_wkt, working_dir):
    """Resample a population raster without losing or gaining people.

    Population rasters are an interesting special case where the data are
    neither continuous nor categorical, and the total population count
    typically matters.  Common resampling methods for continuous
    (interpolation) and categorical (nearest-neighbor) datasets leave room for
    the total population of a resampled raster to significantly change.  This
    function resamples a population raster with the following steps:

        1. Convert a population count raster to population density per pixel
        2. Warp the population density raster to the target spatial reference
           and pixel size using bilinear interpolation.
        3. Convert the warped density raster back to population counts.

    Args:
        source_population_raster_path (string): The source population raster.
            Pixel values represent the number of people occupying the pixel.
            Must be linearly projected in meters.
        target_population_raster_path (string): The path to where the target,
            warped population raster will live on disk.
        lulc_pixel_size (tuple): A tuple of the pixel size for the target
            raster.  Passed directly to ``pygeoprocessing.warp_raster``.
        lulc_bb (tuple): A tuple of the bounding box for the target raster.
            Passed directly to ``pygeoprocessing.warp_raster``.
        lulc_projection_wkt (string): The Well-Known Text of the target
            spatial reference fro the target raster.  Passed directly to
            ``pygeoprocessing.warp_raster``.  Assumed to be a linear projection
            in meters.
        working_dir (string): The path to a directory on disk.  A new directory
            is created within this directory for the storage of temporary files
            and then deleted upon successful completion of the function.

    Returns:
        ``None``
    """
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    tmp_working_dir = tempfile.mkdtemp(dir=working_dir)
    population_raster_info = pygeoprocessing.get_raster_info(
        source_population_raster_path)
    pixel_area = numpy.multiply(*population_raster_info['pixel_size'])
    population_nodata = population_raster_info['nodata'][0]

    population_srs = osr.SpatialReference()
    population_srs.ImportFromWkt(population_raster_info['projection_wkt'])

    # Convert population pixel area to square km
    population_pixel_area = (
        pixel_area * population_srs.GetLinearUnits()) / 1e6

    def _convert_population_to_density(population):
        """Convert population counts to population per square km.

        Args:
            population (numpy.array): A numpy array where pixel values
                represent the number of people who reside in a pixel.

        Returns:
            """
        out_array = numpy.full(
            population.shape, FLOAT32_NODATA, dtype=numpy.float32)

        valid_mask = slice(None)
        if population_nodata is not None:
            valid_mask = ~numpy.isclose(population, population_nodata)

        out_array[valid_mask] = population[valid_mask] / population_pixel_area
        return out_array

    # Step 1: convert the population raster to population density per sq. km
    density_raster_path = os.path.join(tmp_working_dir, 'pop_density.tif')
    pygeoprocessing.raster_calculator(
        [(source_population_raster_path, 1)],
        _convert_population_to_density,
        density_raster_path, gdal.GDT_Float32, FLOAT32_NODATA)

    # Step 2: align to the LULC
    warped_density_path = os.path.join(tmp_working_dir, 'warped_density.tif')
    pygeoprocessing.warp_raster(
        density_raster_path,
        target_pixel_size=lulc_pixel_size,
        target_raster_path=warped_density_path,
        resample_method='bilinear',
        target_bb=lulc_bb,
        target_projection_wkt=lulc_projection_wkt)

    # Step 3: convert the warped population raster back from density to the
    # population per pixel
    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(lulc_projection_wkt)
    # Calculate target pixel area in km to match above
    target_pixel_area = (
        numpy.multiply(*lulc_pixel_size) * target_srs.GetLinearUnits()) / 1e6

    def _convert_density_to_population(density):
        """Convert a population density raster back to population counts.

        Args:
            density (numpy.array): An array of the population density per
                square kilometer.

        Returns:
            A ``numpy.array`` of the population counts given the target pixel
            size of the output raster."""
        # We're using a float32 array here because doing these unit
        # conversions is likely to end up with partial people spread out
        # between multiple pixels.  So it's preserving an unrealistic degree of
        # precision, but that's probably OK because pixels are imprecise
        # measures anyways.
        out_array = numpy.full(
            density.shape, FLOAT32_NODATA, dtype=numpy.float32)

        # We already know that the nodata value is FLOAT32_NODATA
        valid_mask = ~numpy.isclose(density, FLOAT32_NODATA)
        out_array[valid_mask] = density[valid_mask] * target_pixel_area
        return out_array

    pygeoprocessing.raster_calculator(
        [(warped_density_path, 1)],
        _convert_density_to_population,
        target_population_raster_path, gdal.GDT_Float32, FLOAT32_NODATA)

    shutil.rmtree(tmp_working_dir, ignore_errors=True)


def dichotomous_decay_kernel_raster(expected_distance, kernel_filepath,
                                    normalize=False):
    """Create a raster-based, discontinuous decay kernel based on a dichotomy.

    This kernel has a value of ``1`` for all pixels within
    ``expected_distance`` from the center of the kernel.  All values outside of
    this distance are ``0``.

    Args:
        expected_distance (int or float): The distance (in pixels) after which
            the kernel becomes 0.
        kernel_filepath (string): The string path on disk to where this kernel
            should be stored.
        normalize=False (bool): Whether to divide the kernel values by the sum
            of all values in the kernel.

    Returns:
        ``None``
    """
    pixel_radius = math.ceil(expected_distance)
    kernel_size = pixel_radius * 2 + 1  # allow for a center pixel
    driver = gdal.GetDriverByName('GTiff')
    kernel_dataset = driver.Create(
        kernel_filepath.encode('utf-8'), kernel_size, kernel_size, 1,
        gdal.GDT_Float32, options=[
            'BIGTIFF=IF_SAFER', 'TILED=YES', 'BLOCKXSIZE=256',
            'BLOCKYSIZE=256'])

    # Make some kind of geotransform, it doesn't matter what but
    # will make GIS libraries behave better if it's all defined
    kernel_dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    kernel_dataset.SetProjection(srs.ExportToWkt())

    kernel_band = kernel_dataset.GetRasterBand(1)
    kernel_nodata = FLOAT32_NODATA
    kernel_band.SetNoDataValue(kernel_nodata)

    kernel_band = None
    kernel_dataset = None

    kernel_raster = gdal.OpenEx(kernel_filepath, gdal.GA_Update)
    kernel_band = kernel_raster.GetRasterBand(1)
    band_x_size = kernel_band.XSize
    band_y_size = kernel_band.YSize
    running_sum = 0
    for block_data in pygeoprocessing.iterblocks(
            (kernel_filepath, 1), offset_only=True):
        array_xmin = block_data['xoff'] - pixel_radius
        array_xmax = min(
            array_xmin + block_data['win_xsize'],
            band_x_size - pixel_radius)
        array_ymin = block_data['yoff'] - pixel_radius
        array_ymax = min(
            array_ymin + block_data['win_ysize'],
            band_y_size - pixel_radius)

        pixel_dist_from_center = numpy.hypot(
            *numpy.mgrid[
                array_ymin:array_ymax,
                array_xmin:array_xmax])
        search_kernel = numpy.array(
            pixel_dist_from_center <= expected_distance, dtype=numpy.uint8)
        running_sum += search_kernel.sum()
        kernel_band.WriteArray(
            search_kernel,
            yoff=block_data['yoff'],
            xoff=block_data['xoff'])

    kernel_raster.FlushCache()
    kernel_band = None
    kernel_raster = None

    if normalize:
        kernel_raster = gdal.OpenEx(kernel_filepath, gdal.GA_Update)
        kernel_band = kernel_raster.GetRasterBand(1)
        for block_data, kernel_block in pygeoprocessing.iterblocks(
                (kernel_filepath, 1)):
            # divide by sum to normalize
            kernel_block /= running_sum
            kernel_band.WriteArray(
                kernel_block, xoff=block_data['xoff'], yoff=block_data['yoff'])

        kernel_raster.FlushCache()
        kernel_band = None
        kernel_raster = None


def density_decay_kernel_raster(expected_distance, kernel_filepath,
                                normalize=False):
    """Create a raster-based density decay kernel.

    Args:
        expected_distance (int or float): The distance (in pixels) after which
            the kernel becomes 0.
        kernel_filepath (string): The string path on disk to where this kernel
            should be stored.
        normalize=False (bool): Whether to divide the kernel values by the sum
            of all values in the kernel.

    Returns:
        ``None``
    """
    pixel_radius = math.ceil(expected_distance)
    kernel_size = pixel_radius * 2 + 1  # allow for a center pixel
    driver = gdal.GetDriverByName('GTiff')
    kernel_dataset = driver.Create(
        kernel_filepath.encode('utf-8'), kernel_size, kernel_size, 1,
        gdal.GDT_Float32, options=[
            'BIGTIFF=IF_SAFER', 'TILED=YES', 'BLOCKXSIZE=256',
            'BLOCKYSIZE=256'])

    # Make some kind of geotransform, it doesn't matter what but
    # will make GIS libraries behave better if it's all defined
    kernel_dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    kernel_dataset.SetProjection(srs.ExportToWkt())

    kernel_band = kernel_dataset.GetRasterBand(1)
    kernel_nodata = float(numpy.finfo(numpy.float32).min)
    kernel_band.SetNoDataValue(kernel_nodata)

    kernel_band = None
    kernel_dataset = None

    kernel_raster = gdal.OpenEx(kernel_filepath, gdal.GA_Update)
    kernel_band = kernel_raster.GetRasterBand(1)
    band_x_size = kernel_band.XSize
    band_y_size = kernel_band.YSize
    running_sum = 0
    for block_data in pygeoprocessing.iterblocks(
            (kernel_filepath, 1), offset_only=True):
        array_xmin = block_data['xoff'] - pixel_radius
        array_xmax = min(
            array_xmin + block_data['win_xsize'],
            band_x_size - pixel_radius)
        array_ymin = block_data['yoff'] - pixel_radius
        array_ymax = min(
            array_ymin + block_data['win_ysize'],
            band_y_size - pixel_radius)

        pixel_dist_from_center = numpy.hypot(
            *numpy.mgrid[
                array_ymin:array_ymax,
                array_xmin:array_xmax])

        density = numpy.zeros(
            pixel_dist_from_center.shape, dtype=numpy.float32)
        pixels_in_radius = (pixel_dist_from_center <= expected_distance)
        density[pixels_in_radius] = (
            0.75 * (1 - (pixel_dist_from_center[
                pixels_in_radius] / expected_distance) ** 2))
        running_sum += density.sum()

        kernel_band.WriteArray(
            density,
            yoff=block_data['yoff'],
            xoff=block_data['xoff'])

    kernel_raster.FlushCache()
    kernel_band = None
    kernel_raster = None

    if normalize:
        kernel_raster = gdal.OpenEx(kernel_filepath, gdal.GA_Update)
        kernel_band = kernel_raster.GetRasterBand(1)
        for block_data, kernel_block in pygeoprocessing.iterblocks(
                (kernel_filepath, 1)):
            # divide by sum to normalize
            kernel_block /= running_sum
            kernel_band.WriteArray(
                kernel_block, xoff=block_data['xoff'], yoff=block_data['yoff'])

        kernel_raster.FlushCache()
        kernel_band = None
        kernel_raster = None


def validate(args, limit_to=None):
    # TODO: validate that split population groups match everywhere defined.
    return validation.validate(
        args, ARGS_SPEC['args'], ARGS_SPEC['args_with_spatial_overlap'])
