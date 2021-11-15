import os
import logging

import pygeoprocessing
import taskgraph
from osgeo import gdal
from osgeo import osr
import numpy

from . import validation
from . import spec_utils
from . import utils
from .spec_utils import u
from .. import MODEL_METADATA

LOGGER = logging.getLogger(__name__)
UINT32_NODATA = numpy.finfo(numpy.uint32).max
FLOAT32_NODATA = numpy.finfo(numpy.float32).max
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
            'bands': {
                1: {'type': 'number', 'units': u.none}
            },
            'projected': True,
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

    graph.close()
    graph.join()
    LOGGER.info('Finished Urban Nature Access Model')


def _resample_population_raster(
        source_population_raster_path, target_population_raster_path,
        target_pixel_size, target_bb, target_projection_wkt, working_dir):
    # Assumes population raster is linearly projected
    # Reasoning: the total population of a pixel is great, but we need to be
    # able to convert that pixel size without losing or gaining people due to
    # resampling.
    #
    # Approach:
    # * Assume population raster is linearly projected.
    # * raster_calculator: convert population raster to density of people/pixel
    # * align_and_resize_raster_stack: resample (bilinear should be fine) to
    #   LULC raster's alignment/resolution
    # * raster_calculator: Reconvert the population raster from density to
    #   population

    # Option 1: I could do each task right in the execute function
    # Option 2: I could do the 2 raster_calculator calls and 1 warp_raster call
    #     right here in this function

    population_raster_info = pygeoprocessing.get_raster_info(
        source_population_raster_path)
    pixel_area = numpy.multiply(population_raster_info['pixel_size'])
    population_nodata = population_raster_info['nodata'][0]

    population_srs = osr.SpatialReference()
    population_srs.ImportFromWKT(population_raster_info['projection_wkt'])

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

        valid_mask = slice()
        if population_nodata is not None:
            valid_mask = ~numpy.isclose(population, population_nodata)

        out_array[valid_mask] = population[valid_mask] / population_pixel_area
        return out_array

    # Step 1: convert the population raster to population density per sq. km
    density_raster_path = os.path.join(working_dir, 'pop_density.tif')
    pygeoprocessing.raster_calculator(
        [(source_population_raster_path, 1)],
        _convert_population_to_density,
        density_raster_path, gdal.GDT_Float32, FLOAT32_NODATA)

    # Step 2: align to the LULC
    warped_density_path = os.path.join(working_dir, 'warped_density.tif')
    pygeoprocessing.warp_raster(
        density_raster_path,
        target_pixel_size=target_pixel_size,
        target_raster_path=warped_density_path,
        resample_method='bilinear',
        target_bb=target_bb,
        target_projection_wkt=target_projection_wkt)

    # Step 3: convert the warped population raster back from density to the
    # population per pixel
    target_srs = osr.SpatialReference()
    target_srs.ImportFromWKT(target_projection_wkt)
    # Calculate target pixel area in km to match above
    target_pixel_area = (
        numpy.multiply(target_pixel_size) * target_srs.GetLinearUnits()) / 1e6

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


def validate(args, limit_to=None):
    return validation.validate(args, ARGS_SPEC['args'])
