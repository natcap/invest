import logging
import math
import os
import shutil
import tempfile

import numpy
import pygeoprocessing
import taskgraph
from osgeo import gdal
from osgeo import osr

from natcap.invest import MODEL_METADATA
from . import spec_utils
from . import utils
from . import validation
from .spec_utils import u

LOGGER = logging.getLogger(__name__)
UINT32_NODATA = int(numpy.iinfo(numpy.uint32).max)
FLOAT32_NODATA = float(numpy.finfo(numpy.float32).min)
BYTE_NODATA = 255
ARGS_SPEC = {
    'model_name': MODEL_METADATA['urban_nature_access'].model_title,
    'pyname': MODEL_METADATA['urban_nature_access'].pyname,
    'userguide_html': MODEL_METADATA['urban_nature_access'].userguide,
    'args_with_spatial_overlap': {
        'spatial_keys': [
            'lulc_raster_path', 'population_raster_path',
            'admin_unit_vector_path'],
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
        'admin_unit_vector_path': {
            'type': 'vector',
            'name': 'administrative boundaries',
            'geometries': spec_utils.POLYGONS,
            'fields': {},  # TODO, complete required fields (if any)
            'about': "",  # TODO, will know more about this when I implement.
        },
        'greenspace_demand': {
            'type': 'number',
            'name': 'greenspace demand per capita',
            'units': u.m**2,  # defined as m² per capita
            'expression': "value > 0",
            'about': (
                "The amount of greenspace that each resident should have. "
                "This is often defined by local urban planning documents."
            )
        },
        'search_radius': {
            'type': 'number',
            'name': 'search radius',
            'units': u.m,
            'expression': "value > 0",
            'about': "",  # TODO, will know more about this when I implement.
        },
        'decay_function': {
            'name': 'decay function',
            'type': 'option_string',
            'required': False,
            'options': [
                'dichotomy',
                'exponential',
                'gaussian',
                'density',
            ],
            'about': '',  # TODO
        }
    }
}


_OUTPUT_BASE_FILES = {
    'greenspace_supply': 'greenspace_supply.tif',
}
_INTERMEDIATE_BASE_FILES = {
    'aligned_population': 'aligned_population.tif',
    'aligned_lulc': 'aligned_lulc.tif',
    'greenspace_area': 'greenspace_area.tif',
    'kernel': 'kernel.tif',
    'greenspace_population_ratio': 'greenspace_population_ratio.tif',
    'convolved_population': 'convolved_population.tif',
    'greenspace_budget': 'greenspace_budget.tif',
    'greenspace_supply_demand_budget': 'greenspace_supply_demand_budget.tif',
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

            * ``lucode``: the integer landcover code represented.
            * ``greenspace``: ``0`` or ``1`` indicating whether this landcover
              code is (``1``) or is not (``0``) a greenspace pixel.

        args['population_raster_path'] (string): (required) A string path to a
            GDAL-compatible raster where pixels represent the population of
            that pixel.  Must be linearly projected in meters.
        args['admin_unit_vector_path'] (string): (required) A string path to a
            GDAL-compatible vector containing polygon administrative
            boundaries.
        args['greenspace_demand'] (number): (required) A positive, nonzero
            number indicating the required greenspace, in m² per capita.
        args['search_radius'] (number): (required) A positive, nonzero number
            indicating the maximum distance that people travel for recreation.
        args['kernel_type'] (string): (optional) The selected kernel type.
            Must be one of the keys in ``KERNEL_TYPES``.  If not provided, the
            ``'dichotomy'`` kernel will be used.

    Returns:
        ``None``
    """
    LOGGER.info('Starting Urban Nature Access Model')

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

    kernel_types = {
        'dichotomy': dichotomous_decay_kernel_raster,
        # "exponential" is more consistent with other InVEST models'
        # terminology.  "Power function" is used in the design doc.
        'exponential': utils.exponential_decay_kernel_raster,
        'gaussian': utils.gaussian_decay_kernel_raster,
        'density': density_decay_kernel_raster,
    }
    # Since we have these keys defined in two places, I want to be super sure
    # that the labels match.
    assert sorted(kernel_types.keys()) == (
        sorted(ARGS_SPEC['args']['decay_function']['options']))

    if 'kernel_type' not in args:
        kernel_type = 'dichotomy'
    elif args['kernel_type'] not in kernel_types:
        raise ValueError(
            f'Kernel type "{args["kernel_type"]}" is not recognized. '
            f"Must be one of {', '.join(kernel_types.keys())}")
    else:
        kernel_type = args['kernel_type']

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

    greenspace_lulc_lookup = utils.build_lookup_from_csv(
        args['lulc_attribute_table'], 'lucode')
    squared_pixel_area = abs(numpy.multiply(*squared_lulc_pixel_size))
    greenspace_area_map = {
        lucode: int(attributes['greenspace']) * squared_pixel_area
        for lucode, attributes in greenspace_lulc_lookup.items()
    }
    greenspace_area_map[lulc_raster_info['nodata'][0]] = FLOAT32_NODATA
    greenspace_reclassification_task = graph.add_task(
        utils.reclassify_raster,
        kwargs={
            'raster_path_band': (file_registry['aligned_lulc'], 1),
            'value_map': greenspace_area_map,
            'target_raster_path': file_registry['greenspace_area'],
            'target_datatype': gdal.GDT_Float32,
            'target_nodata': FLOAT32_NODATA,
            'error_details': {
                'raster_name': ARGS_SPEC['args']['lulc_raster_path']['name'],
                'column_name': 'greenspace',
                'table_name': ARGS_SPEC['args'][
                    'lulc_attribute_table']['name'],
            },
        },
        target_path_list=[file_registry['greenspace_area']],
        task_name='Identify the area of greenspace in pixels',
        dependent_task_list=[lulc_alignment_task]
    )

    search_radius_in_pixels = abs(
        float(args['search_radius']) / squared_lulc_pixel_size[0])
    kernel_creation_task = graph.add_task(
        # All kernel creation types have the same function signature
        kernel_types[kernel_type],
        args=(search_radius_in_pixels, file_registry['kernel']),
        kwargs={'normalize': False},  # Model math calls for un-normalized
        task_name=f'2SFCA - Create {kernel_type} kernel',
        target_path_list=[file_registry['kernel']]
    )

    convolved_population_task = graph.add_task(
        _convolve_and_set_lower_bounds_for_population,
        kwargs={
            'signal_path_band': (file_registry['aligned_population'], 1),
            'kernel_path_band': (file_registry['kernel'], 1),
            'target_path': file_registry['convolved_population'],
            'working_dir': intermediate_dir,
        },
        task_name='2SFCA - Convolve population',
        target_path_list=[file_registry['convolved_population']],
        dependent_task_list=[
            kernel_creation_task,
            population_alignment_task,
        ])

    greenspace_nodata = pygeoprocessing.get_raster_info(
        file_registry['greenspace_area'])['nodata'][0]
    population_nodata = pygeoprocessing.get_raster_info(
        file_registry['convolved_population'])['nodata'][0]
    greenspace_population_ratio_task = graph.add_task(
        pygeoprocessing.raster_calculator,
        kwargs={
            'base_raster_path_band_const_list': [
                (file_registry['greenspace_area'], 1),
                (file_registry['convolved_population'], 1),
                (greenspace_nodata, 'raw'),
                (population_nodata, 'raw')],
            'local_op': _greenspace_population_ratio,
            'target_raster_path': file_registry['greenspace_population_ratio'],
            'datatype_target': gdal.GDT_Float32,
            'nodata_target': FLOAT32_NODATA
        },
        task_name='2SFCA: Calculate R_j greenspace/population ratio',
        target_path_list=[file_registry['greenspace_population_ratio']],
        dependent_task_list=[
            convolved_population_task,
            greenspace_reclassification_task,
        ])

    convolved_greenspace_population_ratio_task = graph.add_task(
        pygeoprocessing.convolve_2d,
        kwargs={
            'signal_path_band': (
                file_registry['greenspace_population_ratio'], 1),
            'kernel_path_band': (file_registry['kernel'], 1),
            'target_path': file_registry['greenspace_supply'],
            'working_dir': intermediate_dir,
            # Insurance against future pygeoprocessing API changes.  The target
            # nodata right now is the minimum possible numpy float32 value,
            # which is also what we use here as FLOAT32_NODATA.
            'target_nodata': FLOAT32_NODATA,
        },
        task_name='2SFCA - greenspace supply',
        target_path_list=[file_registry['greenspace_supply']],
        dependent_task_list=[
            kernel_creation_task,
            greenspace_population_ratio_task,
        ])

    per_capita_greenspace_budgets_task = graph.add_task(
        _calculate_per_capita_greenspace_budgets,
        kwargs={
            'greenspace_supply_raster_path': file_registry[
                'greenspace_supply'],
            'population_raster_path': file_registry[
                'aligned_population'],
            'greenspace_demand': float(args['greenspace_demand']),
            'target_greenspace_budget_path': file_registry[
                'greenspace_budget'],
            'target_supply_demand_budget_path': file_registry[
                'greenspace_supply_demand_budget']
        },
        task_name='Calculate greenspace budgets',
        target_path_list=[
            file_registry['greenspace_budget'],
            file_registry['greenspace_supply_demand_budget']
        ],
        dependent_task_list=[
            convolved_greenspace_population_ratio_task,
            population_alignment_task
        ])

    graph.close()
    graph.join()
    LOGGER.info('Finished Urban Nature Access Model')


def _calculate_per_capita_greenspace_budgets(
        greenspace_supply_raster_path, population_raster_path,
        greenspace_demand, target_greenspace_budget_path,
        target_supply_demand_budget_path):
    """Calculate per-capita greenspace budgets.

    This function will produce two target rasters:

        * The per-capita supply-demand budget at the pixel level
        * The greenspace supply-demand budget at the pixel level

    The latter depends on the former, so running them in a single function
    allows us to avoid reading another raster from disk.

    .. note::
        All rasters provided as parameters to this function must all have the
        same projections, pixel length and width, and have the same numbers of
        pixels in the X and Y dimensions.  They must also have a nodata value
        of ``FLOAT32_NODATA``.

    Args:
        greenspace_supply_raster_path (string): The path to the greenspace
            supply raster on disk.
        population_raster_path (string): The path to the population raster on
            disk.
        greenspace_demand (float): The per-person demand for greenspace.
        target_greenspace_budget_path (string): The path to where the
            greenspace budget raster produced by this function will be written
            on disk.  If a file exists at this location, it will be
            overwritten.
        target_supply_demand_budget_path (string): The path to where the
            greenspace supply/demand budget raster produced by this function
            will be written on disk.  If a file exists at this location, it
            will be overwritten.

    Returns:
        ``None``
    """
    for target_path in (
            target_greenspace_budget_path,
            target_supply_demand_budget_path):
        pygeoprocessing.new_raster_from_base(
            base_path=greenspace_supply_raster_path,
            target_path=target_path,
            datatype=gdal.GDT_Float32,
            band_nodata_list=[FLOAT32_NODATA]
        )

    population_raster = gdal.OpenEx(population_raster_path)
    population_band = population_raster.GetRasterBand(1)

    greenspace_budget_raster = gdal.OpenEx(
        target_greenspace_budget_path, gdal.GA_Update)
    greenspace_budget_band = greenspace_budget_raster.GetRasterBand(1)

    supply_demand_raster = gdal.OpenEx(
        target_supply_demand_budget_path, gdal.GA_Update)
    supply_demand_band = supply_demand_raster.GetRasterBand(1)

    for block_info, greenspace_supply in pygeoprocessing.iterblocks(
            (greenspace_supply_raster_path, 1)):
        valid_pixels = ~numpy.isclose(greenspace_supply, FLOAT32_NODATA)

        # This calculates the per-capita greenspace budget
        greenspace_budget = numpy.full(
            greenspace_supply.shape, FLOAT32_NODATA, dtype=numpy.float32)
        greenspace_budget[valid_pixels] = (
            greenspace_supply[valid_pixels] - greenspace_demand)
        greenspace_budget_band.WriteArray(
            greenspace_budget, xoff=block_info['xoff'],
            yoff=block_info['yoff'])

        # This calculates the greenspace supply-demand budget
        population = population_band.ReadAsArray(**block_info)
        valid_pixels &= ~numpy.isclose(population, FLOAT32_NODATA)
        supply_demand = numpy.full(
            greenspace_supply.shape, FLOAT32_NODATA, dtype=numpy.float32)
        supply_demand[valid_pixels] = (
            greenspace_budget[valid_pixels] * population[valid_pixels])
        supply_demand_band.WriteArray(
            supply_demand, xoff=block_info['xoff'], yoff=block_info['yoff'])

    population_band = None
    population_raster = None
    greenspace_budget_band = None
    greenspace_budget_raster = None
    supply_demand_band = None
    supply_demand_raster = None


def _greenspace_population_ratio(
        greenspace_area, convolved_population, greenspace_nodata,
        population_nodata):
    """Calculate the greenspace-population ratio R_j.

    Args:
        greenspace_area (numpy.array): A numpy array representing the area of
            greenspace in the pixel.  Pixel values will be ``0`` if there is no
            greenspace.  Pixel values may also match ``greenspace_nodata``.
        convolved_population (numpy.array): A numpy array where each pixel
            represents the total number of people within a search radius of
            each pixel, perhaps adjusted by a search kernel.
        greenspace_nodata (int or float): The nodata value of the
            ``greenspace_area`` matrix.
        population_nodata (int or float): The nodata value of the
            ``convolved_population`` matrix.

    Returns:
        A numpy array with the ratio ``R_j`` representing the
        greenspace-population ratio with the following constraints:

            * ``convolved_population`` pixels that are numerically close to
              ``0`` are snapped to ``0`` to avoid unrealistically small
              denominators in the final ratio.
            * Any non-greenspace pixels will have a value of ``0.0`` in the
              output matrix.
    """
    # ASSUMPTION: population nodata value is not close to 0.
    #  Shouldn't be if we're coming from convolution.
    out_array = numpy.full(
        greenspace_area.shape, FLOAT32_NODATA, dtype=numpy.float32)

    # Small negative values should already have been filtered out in another
    # function after the convolution.
    # This avoids divide-by-zero errors when taking the ratio.
    valid_pixels = (convolved_population > 0)

    # R_j is a ratio only calculated for the greenspace pixels.
    greenspace_pixels = ~numpy.isclose(greenspace_area, 0.0)
    valid_pixels &= greenspace_pixels
    if population_nodata is not None:
        valid_pixels &= ~numpy.isclose(
            convolved_population, population_nodata)

    if greenspace_nodata is not None:
        valid_pixels &= ~numpy.isclose(greenspace_area, greenspace_nodata)

    # If the population in the search radius is numerically 0, the model
    # specifies that the ratio should be set to the greenspace area.
    population_close_to_zero = numpy.isclose(convolved_population, 0.0)
    out_array[population_close_to_zero] = (
        greenspace_pixels[population_close_to_zero])
    out_array[~greenspace_pixels] = 0.0  # set non-greenspace non-nodata to 0
    out_array[valid_pixels] = (
        greenspace_area[valid_pixels] / convolved_population[valid_pixels])

    return out_array


def _convolve_and_set_lower_bounds_for_population(
        signal_path_band, kernel_path_band, target_path, working_dir):
    """Convolve a raster and set all values below 0 to 0.

    Args:
        signal_path_band (tuple): A 2-tuple of (signal_raster_path, band_index)
            to use as the signal raster in the convolution.
        kernel_path_band (tuple): A 2-tuple of (kernel_raster_path, band_index)
            to use as the kernel raster in the convolution.
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
        block[(block < 0.0) & valid_pixels] = 0.0
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
    running_sum = 0.0
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
    running_sum = 0.0
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
    return validation.validate(
        args, ARGS_SPEC['args'], ARGS_SPEC['args_with_spatial_overlap'])
