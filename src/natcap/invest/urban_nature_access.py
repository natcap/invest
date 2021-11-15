import os
import logging

import pygeoprocessing
import taskgraph

from . import validation
from . import spec_utils
from . import utils
from .spec_utils import u
from .. import MODEL_METADATA

LOGGER = logging.getLogger(__name__)
ARGS_SPEC = {
    'model_name': MODEL_METADATA['urban_nature_access'].model_title,
    'pyname': MODEL_METADATA['urban_nature_access'].pyname,
    'userguide_html': MODEL_METADATA['urban_nature_access'].userguide,
    'args_with_spatial_overlap': {
        'spatial_keys': [],
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
            'about': "",  # TODO
        },
        'lulc_attribute_table': {
            'name': 'LULC attribute table',
            'type': 'csv',
            'columns': {
                'lucode': {'type': 'integer'},
                'greenspace': {'type': 'number', 'units': u.none,
                               'about': ''}  # TODO,
            },
            'about': '',  # TODO
        },
        'population_raster_path': {
            'type': 'raster',
            'name': 'population raster',
            'bands': {1: {'type': 'float'}},
            'about': "",  # TODO,
        },
        'admin_unit_vector_path': {
            'type': 'vector',
            'name': 'administrative boundaries',
            'geometries': spec_utils.POLYGONS,
            'about': "",  # TODO
        },
        'greenspace_demand': {
            'type': 'number',
            'name': 'greenspace demand per capita',
            'units': u.m**2,  # defined as m² per capita
            'expression': "value > 0",
            'about': "",  # TODO,
        },
        'search_radius': {
            'type': 'number',
            'name': 'search radius',
            'units': u.m,
            'expression': "value > 0",
            'about': "",  # TODO,
        }
    }
}


_OUTPUT_BASE_FILES = {}
_INTERMEDIATE_BASE_FILES = {}

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
            landcover codes.
        args['lulc_attribute_table'] (string): (required) A string path to a
            CSV with the following columns:

            * ``lucode``: the integer landcover code represented.
            * ``greenspace``: ``0`` or ``1`` indicating whether this landcover
              code is (``1``) or is not (``0``) a greenspace pixel.

        args['population_raster_path'] (string): (required) A string path to a
            GDAL-compatible raster where pixels represent the population of
            that pixel.
        args['admin_unit_vector_path'] (string): (required) A string path to a
            GDAL-compatible vector containing polygon administrative
            boundaries.
        args['greenspace_demand'] (number): (required) A positive, nonzero
            number indicating the required greenspace, in m² per capita.
        args['search_radius'] (number): (required) A positive, nonzero number
            indicating the maximum distance that people travel for recreation.

    Returns:
        ``None``
    """
    LOGGER.info('Starting Urban Nature Access Model')

    # for initial PR, get the basic workflow going with a single test.
    # * Test on basic datasets
    # * align inputs  (what's the best way to reproject a population raster?)
    # * validation

    output_dir = os.path.join(args['workspace_dir'], 'output')
    intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')
    utils.make_directories([output_dir, intermediate_dir])

    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    file_registry = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_dir)],
        file_suffix)

    work_token_dir = os.path.join(intermediate_dir, '_taskgraph_working_dir')
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous execution
    graph = taskgraph.TaskGraph(work_token_dir, n_workers)

    LOGGER.info('Finished Urban Nature Access Model')


def validate(args, limit_to=None):
    return validation.validate(args, ARGS_SPEC['args'])
