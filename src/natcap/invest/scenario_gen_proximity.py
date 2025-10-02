"""Scenario Generation: Proximity Based."""
import collections
import heapq
import logging
import math
import os
import shutil
import struct
import tempfile
import time

import numpy
import pygeoprocessing
import pygeoprocessing.kernels
import scipy
import taskgraph
from osgeo import gdal

from . import gettext
from . import spec
from . import utils
from . import validation
from .unit_registry import u

LOGGER = logging.getLogger(__name__)

MISSING_CONVERT_OPTION_MSG = gettext(
    'One or more of "convert_nearest_to_edge" or "convert_farthest_from_edge" '
    'must be selected')

MODEL_SPEC = spec.ModelSpec(
    model_id="scenario_generator_proximity",
    model_title=gettext("Scenario Generator: Proximity Based"),
    userguide="scenario_gen_proximity.html",
    validate_spatial_overlap=True,
    different_projections_ok=False,
    aliases=("sgp",),
    module_name=__name__,
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["base_lulc_path", "aoi_path"],
        ["area_to_convert", "focal_landcover_codes",
         "convertible_landcover_codes", "replacement_lucode"],
        ["convert_farthest_from_edge", "convert_nearest_to_edge",
         "n_fragmentation_steps"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.SingleBandRasterInput(
            id="base_lulc_path",
            name=gettext("base LULC map"),
            about=gettext("Base map from which to generate scenarios."),
            data_type=int,
            units=None,
            projected=True
        ),
        spec.IntegerInput(
            id="replacement_lucode",
            name=gettext("replacement landcover code"),
            about=gettext("The LULC code to which habitat will be converted.")
        ),
        spec.NumberInput(
            id="area_to_convert",
            name=gettext("maximum area to convert"),
            about=gettext("Maximum area to be converted to agriculture."),
            units=u.hectare,
            expression="value > 0"
        ),
        spec.StringInput(
            id="focal_landcover_codes",
            name=gettext("focal landcover codes"),
            about=(
                "A space-separated list of LULC codes that are used to determine the"
                " proximity when referring to 'towards' or 'away' from the base landcover"
                " codes"
            ),
            regexp="[0-9 ]+"
        ),
        spec.StringInput(
            id="convertible_landcover_codes",
            name=gettext("convertible landcover codes"),
            about=gettext(
                "A space-separated list of LULC codes that can be converted to"
                " agriculture."
            ),
            regexp="[0-9 ]+"
        ),
        spec.IntegerInput(
            id="n_fragmentation_steps",
            name=gettext("number of conversion steps"),
            about=gettext(
                "The number of steps that the simulation should take to fragment the"
                " habitat of interest in the fragmentation scenario. This parameter is"
                " used to divide the conversion simulation into equal subareas of the"
                " requested max area. During each sub-step the distance transform is"
                " recalculated from the base landcover codes.  This can affect the final"
                " result if the base types are also convertible types."
            ),
            units=u.none,
            expression="value > 0"
        ),
        spec.AOI.model_copy(update=dict(
            id="aoi_path",
            about=gettext(
                "Area over which to run the conversion. Provide this input if change is"
                " only desired in a subregion of the Base LULC map."
            ),
            required=False
        )),
        spec.BooleanInput(
            id="convert_farthest_from_edge",
            name=gettext("convert farthest from edge"),
            about=(
                "Convert the 'convertible' landcover codes starting at the furthest pixel"
                " from the 'focal' land cover areas and working inwards."
            )
        ),
        spec.BooleanInput(
            id="convert_nearest_to_edge",
            name=gettext("convert nearest to edge"),
            about=(
                "Convert the 'convertible' landcover codes starting at the nearest pixels"
                " to the 'focal' land cover areas and working outwards."
            )
        )
    ],
    outputs=[
        spec.SingleBandRasterOutput(
            id="nearest_to_edge",
            path="nearest_to_edge.tif",
            about=gettext("Map of the nearest-to-edge conversion scenario."),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="farthest_from_edge",
            path="farthest_from_edge.tif",
            about=gettext("Map of the farthest-from-edge conversion scenario."),
            data_type=int,
            units=None
        ),
        spec.CSVOutput(
            id="nearest_to_edge_csv",
            path="nearest_to_edge.csv",
            about=gettext(
                "Table of land cover classes and the amount of each that was converted"
                " for the nearest-to-edge conversion scenario."
            ),
            columns=[
                spec.IntegerOutput(
                    id="original lucode",
                    about=gettext("Original LULC code of the land cover class")
                ),
                spec.IntegerOutput(
                    id="replacement lucode",
                    about=gettext("LULC code to which habitat was converted")
                ),
                spec.NumberOutput(
                    id="area converted (Ha)",
                    about=gettext("Total area converted from this land cover class"),
                    units=u.hectare
                ),
                spec.IntegerOutput(
                    id="pixels converted",
                    about=gettext(
                        "Number of pixels converted from this land cover class"
                    )
                )
            ],
            index_col="original lucode"
        ),
        spec.CSVOutput(
            id="farthest_from_edge_csv",
            path="farthest_from_edge.csv",
            about=gettext(
                "Table of land cover classes and the amount of each that was converted"
                " for the farthest-from-edge conversion scenario."
            ),
            columns=[
                spec.IntegerOutput(
                    id="original lucode",
                    about=gettext("Original LULC code of the land cover class")
                ),
                spec.IntegerOutput(
                    id="replacement lucode",
                    about=gettext("LULC code to which habitat was converted")
                ),
                spec.NumberOutput(
                    id="area converted (Ha)",
                    about=gettext("Total area converted from this land cover class"),
                    units=u.hectare
                ),
                spec.IntegerOutput(
                    id="pixels converted",
                    about=gettext(
                        "Number of pixels converted from this land cover class"
                    )
                )
            ],
            index_col="original lucode"
        ),
        spec.SingleBandRasterOutput(
            id="aoi_masked_lulc",
            path="intermediate_outputs/aoi_masked_lulc.tif",
            about=gettext("Copy of the LULC raster masked to the AOI extent."),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="farthest_from_edge_distance",
            path="intermediate_outputs/farthest_from_edge_distance.tif",
            about=gettext(
                "Map of the distance from each pixel to the farthest edge of the"
                " focal landcover."
            ),
            data_type=float,
            units=u.pixel
        ),
        spec.SingleBandRasterOutput(
            id="nearest_to_edge_distance",
            path="intermediate_outputs/nearest_to_edge_distance.tif",
            about=gettext(
                "Map of the distance from each pixel to the nearest edge of the"
                " focal landcover."
            ),
            data_type=float,
            units=u.pixel
        ),
        spec.SingleBandRasterOutput(
            id="tmp_non_base_mask",
            path="tmp/non_base_mask.tif",
            about=gettext("Temporary mask raster for non-base pixels."),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="tmp_base_mask",
            path="tmp/base_mask.tif",
            about=gettext("Temporary mask raster for base pixels."),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="tmp_mask",
            path="tmp/mask.tif",
            about=gettext("Temporary mask raster."),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="tmp_gaussian_kernel",
            path="tmp/gaussian_kernel.tif",
            about=gettext("Temporary Gaussian kernel raster for smoothing."),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="tmp_distance_from_base_mask_edge",
            path="tmp/distance_from_base_mask_edge.tif",
            about=gettext("Temporary raster for distance from base mask edge."),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="tmp_distance_from_non_base_mask_edge",
            path="tmp/distance_from_non_base_mask_edge.tif",
            about=gettext("Temporary raster for distance from non-base mask edge."),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="tmp_convertible_distances",
            path="tmp/convertible_distances.tif",
            about=gettext("Temporary raster for convertible pixel distances."),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="tmp_distance_from_edge",
            path="tmp/distance_from_edge.tif",
            about=gettext("Temporary raster for combined edge distances."),
            data_type=float,
            units=None
        ),
        spec.TASKGRAPH_CACHE
    ]
)


# This sets the largest number of elements that will be packed at once and
# addresses a memory leak issue that happens when many arguments are passed
# to the function via the * operator
_LARGEST_STRUCT_PACK = 1024

# Max number of elements to read/cache at once.  Used throughout the code to
# load arrays to and from disk
_BLOCK_SIZE = 2**20


def execute(args):
    """Scenario Generator: Proximity-Based.

    Main entry point for proximity based scenario generator model.

    Args:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output files
        args['base_lulc_path'] (string): path to the base landcover map
        args['replacement_lucode'] (string or int): code to replace when
            converting pixels
        args['area_to_convert'] (string or float): max area (Ha) to convert
        args['focal_landcover_codes'] (string): a space separated string of
            landcover codes that are used to determine the proximity when
            refering to "towards" or "away" from the base landcover codes
        args['convertible_landcover_codes'] (string): a space separated string
            of landcover codes that can be converted in the generation phase
            found in `args['base_lulc_path']`.
        args['n_fragmentation_steps'] (string): an int as a string indicating
            the number of steps to take for the fragmentation conversion
        args['aoi_path'] (string): (optional) path to a shapefile that
            indicates area of interest.  If present, the expansion scenario
            operates only under that AOI and the output raster is clipped to
            that shape.
        args['convert_farthest_from_edge'] (boolean): if True will run the
            conversion simulation starting from the furthest pixel from the
            edge and work inwards.  Workspace will contain output files named
            'toward_base{suffix}.{tif,csv}.
        args['convert_nearest_to_edge'] (boolean): if True will run the
            conversion simulation starting from the nearest pixel on the
            edge and work inwards.  Workspace will contain output files named
            'toward_base{suffix}.{tif,csv}.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process.

    Returns:
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths

    """
    args, file_registry, task_graph = MODEL_SPEC.setup(args)

    if (not args['convert_farthest_from_edge'] and
            not args['convert_nearest_to_edge']):
        raise ValueError("Neither scenario was selected.")

    # convert all the input strings to lists of ints
    convertible_type_list = numpy.array([
        int(x) for x in args['convertible_landcover_codes'].split()])
    focal_landcover_codes = numpy.array([
        int(x) for x in args['focal_landcover_codes'].split()])

    aoi_mask_task_list = []
    if args['aoi_path']:
        working_lulc_path = file_registry['aoi_masked_lulc']
        aoi_mask_task_list.append(task_graph.add_task(
            func=_mask_raster_by_vector,
            args=((args['base_lulc_path'], 1), args['aoi_path'],
                  args['workspace_dir'], working_lulc_path,
                  file_registry['tmp_mask']),
            target_path_list=[working_lulc_path],
            task_name='aoi_mask'))
    else:
        working_lulc_path = args['base_lulc_path']

    scenarios = [
        (args['convert_farthest_from_edge'], 'farthest_from_edge', -1.0),
        (args['convert_nearest_to_edge'], 'nearest_to_edge', 1.0)]

    for scenario_enabled, basename, score_weight in scenarios:
        if not scenario_enabled:
            continue
        LOGGER.info('executing %s scenario', basename)
        task_graph.add_task(
            func=_convert_landscape,
            args=(working_lulc_path, args['replacement_lucode'],
                  args['area_to_convert'], focal_landcover_codes,
                  convertible_type_list, score_weight,
                  args['n_fragmentation_steps'],
                  file_registry[f"{basename}_distance"],
                  file_registry[basename], file_registry[f"{basename}_csv"],
                  args['workspace_dir'], file_registry),
            target_path_list=[
                file_registry[f"{basename}_distance"],
                file_registry[basename],
                file_registry[f"{basename}_csv"]],
            dependent_task_list=aoi_mask_task_list,
            task_name=f'convert_landscape_{basename}')

    task_graph.close()
    task_graph.join()
    return file_registry.registry


def _mask_raster_by_vector(
        base_raster_path_band, vector_path, working_dir, target_raster_path, mask_raster_path):
    """Mask pixels outside of the vector to nodata.

    Args:
        base_raster_path (string): path/band tuple to raster to process
        vector_path (string): path to single layer raster that is used to
            indicate areas to preserve from the base raster.  Areas outside
            of this vector are set to nodata.
        working_dir (str): path to temporary directory.
        target_raster_path (string): path to a single band raster that will be
            created of the same dimensions and data type as
            `base_raster_path_band` where any pixels that lie outside of
            `vector_path` coverage will be set to nodata.
        mask_raster_path (string): path to a mask raster to be created.

    Returns:
        None.

    """
    # Warp input raster to be same bounding box as AOI if smaller.
    base_raster_info = pygeoprocessing.get_raster_info(
        base_raster_path_band[0])
    nodata = base_raster_info['nodata'][base_raster_path_band[1]-1]
    target_pixel_size = base_raster_info['pixel_size']
    vector_info = pygeoprocessing.get_vector_info(vector_path)
    target_bounding_box = pygeoprocessing.merge_bounding_box_list(
        [base_raster_info['bounding_box'],
         vector_info['bounding_box']], 'intersection')
    pygeoprocessing.warp_raster(
        base_raster_path_band[0], target_pixel_size, target_raster_path,
        'near', target_bb=target_bounding_box)

    # Create mask raster same size as the warped raster.
    tmp_dir = tempfile.mkdtemp(dir=working_dir)
    pygeoprocessing.new_raster_from_base(
        target_raster_path, mask_raster_path, gdal.GDT_Byte, [0],
        fill_value_list=[0])

    # Rasterize the vector onto the mask raster
    pygeoprocessing.rasterize(vector_path, mask_raster_path, [1], None)

    # Parallel iterate over warped raster and mask raster to mask out original.
    target_raster = gdal.OpenEx(
        target_raster_path, gdal.GA_Update | gdal.OF_RASTER)
    target_band = target_raster.GetRasterBand(1)
    mask_raster = gdal.OpenEx(mask_raster_path, gdal.OF_RASTER)
    mask_band = mask_raster.GetRasterBand(1)

    for offset_dict in pygeoprocessing.iterblocks(
            (mask_raster_path, 1), offset_only=True):
        data_array = target_band.ReadAsArray(**offset_dict)
        mask_array = mask_band.ReadAsArray(**offset_dict)
        data_array[mask_array != 1] = nodata
        target_band.WriteArray(
            data_array, xoff=offset_dict['xoff'], yoff=offset_dict['yoff'])
    target_band.FlushCache()
    target_band = None
    target_raster = None
    mask_band = None
    mask_raster = None
    try:
        shutil.rmtree(tmp_dir)
    except OSError:
        LOGGER.warning("Unable to delete temporary file %s", mask_raster_path)


def _convert_landscape(
        base_lulc_path, replacement_lucode, area_to_convert,
        focal_landcover_codes, convertible_type_list, score_weight, n_steps,
        smooth_distance_from_edge_path, output_landscape_raster_path,
        stats_path, workspace_dir, file_registry):
    """Expand replacement lucodes in relation to the focal lucodes.

    If the sign on `score_weight` is positive, expansion occurs marches
    away from the focal types, while if `score_weight` is negative conversion
    marches toward the focal types.

    Args:
        base_lulc_path (string): path to landcover raster that will be used as
            the base landcover map to agriculture pixels
        replacement_lucode (int): agriculture landcover code type found in the
            raster at `base_lulc_path`
        area_to_convert (float): area (Ha) to convert to agriculture
        focal_landcover_codes (list of int): landcover codes that are used to
            calculate proximity
        convertible_type_list (list of int): landcover codes that are allowable
            to be converted to agriculture
        score_weight (float): this value is used to multiply the distance from
            the focal landcover types when prioritizing which pixels in
            `convertable_type_list` are to be converted.  If negative,
            conversion occurs toward the focal types, if positive occurs away
            from the focal types.
        n_steps (int): number of steps to convert the landscape.  On each step
            the distance transform will be applied on the
            current value of the `focal_landcover_codes` pixels in
            `output_landscape_raster_path`.  On the first step the distance
            is calculated from `base_lulc_path`.
        smooth_distance_from_edge_path (string): an intermediate output showing
            the pixel distance from the edge of the base landcover types
        output_landscape_raster_path (string): an output raster that will
            contain the final fragmented forest layer.
        stats_path (string): a path to an output csv that records the number
            type, and area of pixels converted in `output_landscape_raster_path`
        workspace_dir (string): workspace directory that will be used to
            hold temporary files. On a successful run of this function,
            the temporary directory will be removed.
        file_registry: FileRegistry object for output filepaths.

    Returns:
        None.

    """
    temp_dir = tempfile.mkdtemp(prefix='temp_dir', dir=workspace_dir)

    pygeoprocessing.kernels.normal_distribution_kernel(
        file_registry['tmp_gaussian_kernel'], sigma=1)

    # create the output raster first as a copy of the base landcover so it can
    # be looped on for each step
    lulc_raster_info = pygeoprocessing.get_raster_info(base_lulc_path)
    lulc_nodata = lulc_raster_info['nodata'][0]
    mask_nodata = 2
    pygeoprocessing.raster_map(
        op=lambda x: x,
        rasters=[base_lulc_path],
        target_path=output_landscape_raster_path)

    # convert everything furthest from edge for each of n_steps
    pixel_area_ha = (
        abs(lulc_raster_info['pixel_size'][0]) *
        abs(lulc_raster_info['pixel_size'][1])) / 10000.0
    max_pixels_to_convert = int(math.ceil(area_to_convert / pixel_area_ha))
    convertible_type_nodata = -1
    pixels_left_to_convert = max_pixels_to_convert
    pixels_to_convert = max_pixels_to_convert / n_steps
    stats_cache = collections.defaultdict(int)

    # pylint complains when these are defined inside the loop
    invert_mask = None
    distance_nodata = None

    for step_index in range(n_steps):
        LOGGER.info('step %d of %d', step_index+1, n_steps)
        pixels_left_to_convert -= pixels_to_convert

        # Often the last segement of the steps will overstep the  number of
        # pixels to convert, this check converts the exact amount
        if pixels_left_to_convert < 0:
            pixels_to_convert += pixels_left_to_convert

        for invert_mask, mask_path, distance_path in [
                (False, file_registry['tmp_non_base_mask'],
                    file_registry['tmp_distance_from_non_base_mask_edge']),
                (True, file_registry['tmp_base_mask'],
                    file_registry['tmp_distance_from_base_mask_edge'])]:

            def _mask_base_op(lulc_array):
                """Create a mask of valid non-base pixels only."""
                base_mask = numpy.isin(
                    lulc_array.flatten(), focal_landcover_codes).reshape(
                        lulc_array.shape)
                if invert_mask:
                    base_mask = ~base_mask
                return base_mask

            pygeoprocessing.raster_map(
                op=_mask_base_op,
                rasters=[output_landscape_raster_path],
                target_path=mask_path,
                target_dtype=numpy.uint8,
                target_nodata=mask_nodata)

            # create distance transform for the current mask
            pygeoprocessing.distance_transform_edt(
                (mask_path, 1),
                distance_path,
                working_dir=temp_dir)

        # combine inner and outer distance transforms into one
        distance_nodata = pygeoprocessing.get_raster_info(
            file_registry['tmp_distance_from_base_mask_edge'])['nodata'][0]

        def _combine_masks(base_distance_array, non_base_distance_array):
            """Create a mask of valid non-base pixels only."""
            result = non_base_distance_array
            valid_base_mask = base_distance_array > 0.0
            result[valid_base_mask] = base_distance_array[valid_base_mask]
            return result
        pygeoprocessing.raster_calculator(
            [(file_registry['tmp_distance_from_base_mask_edge'], 1),
             (file_registry['tmp_distance_from_non_base_mask_edge'], 1)],
            _combine_masks, file_registry['tmp_distance_from_edge'],
            gdal.GDT_Float32, distance_nodata)

        # smooth the distance transform to avoid scanline artifacts
        pygeoprocessing.convolve_2d(
            (file_registry['tmp_distance_from_edge'], 1),
            (file_registry['tmp_gaussian_kernel'], 1),
            smooth_distance_from_edge_path)

        # turn inside and outside masks into a single mask
        def _mask_to_convertible_codes(distance_from_base_edge, lulc):
            """Mask out the distance transform to a set of lucodes."""
            convertible_mask = numpy.isin(
                lulc.flatten(), convertible_type_list).reshape(lulc.shape)
            return numpy.where(
                convertible_mask, distance_from_base_edge,
                convertible_type_nodata)
        pygeoprocessing.raster_calculator(
            [(smooth_distance_from_edge_path, 1),
             (output_landscape_raster_path, 1)],
            _mask_to_convertible_codes,
            file_registry['tmp_convertible_distances'],
            gdal.GDT_Float32, convertible_type_nodata)

        LOGGER.info(
            'convert %d pixels to lucode %d', pixels_to_convert,
            replacement_lucode)
        _convert_by_score(
            file_registry['tmp_convertible_distances'], pixels_to_convert,
            output_landscape_raster_path, replacement_lucode, stats_cache,
            score_weight)

    _log_stats(stats_cache, replacement_lucode, pixel_area_ha, stats_path)
    try:
        shutil.rmtree(temp_dir)
    except OSError:
        LOGGER.warning(
            "Could not delete temporary working directory '%s'", temp_dir)


def _log_stats(stats_cache, replacement_lucode, pixel_area, stats_path):
    """Write pixel change statistics to a file in tabular format.

    Args:
        stats_cache (dict): a dictionary mapping pixel lucodes to number of
            pixels changed
        replacement_lucode (int): lucode to which habitat was converted
        pixel_area (float): size of pixels in hectares so an area column can
            be generated
        stats_path (string): path to a csv file that the table should be
            generated to

    Returns:
        None

    """
    with open(stats_path, 'w') as csv_output_file:
        csv_output_file.write(
            'original lucode,replacement lucode,'
            'area converted (Ha),pixels converted\n')
        for lucode in sorted(stats_cache):
            csv_output_file.write(
                '%s,%s,%s,%s\n' % (
                    lucode, replacement_lucode,
                    stats_cache[lucode] * pixel_area,
                    stats_cache[lucode]))


def _sort_to_disk(dataset_path, score_weight=1.0):
    """Return an iterable of non-nodata pixels in sorted order.

    Args:
        dataset_path (string): a path to a floating point GDAL dataset
        score_weight (float): a number to multiply all values by, which can be
            used to reverse the order of the iteration if negative.

    Returns:
        an iterable that produces (value * score_weight, flat_index) in
        decreasing sorted order by value * score_weight

    """
    def _read_score_index_from_disk(
            score_file_path, index_file_path):
        """Create generator of float/int value from the given filenames.

        Reads a buffer of `buffer_size` big before to avoid keeping the
        file open between generations.

        score_file_path (string): a path to a file that has 32 bit floats
            packed consecutively
        index_file_path (string): a path to a file that has 32 bit ints
            packed consecutively

        Yields:
            next (score, index) tuple in the given score and index files.

        """
        try:
            score_buffer = ''
            index_buffer = ''
            file_offset = 0
            buffer_offset = 0  # initialize to 0 to trigger the first load

            # ensure buffer size that is not a perfect multiple of 4
            read_buffer_size = int(math.sqrt(_BLOCK_SIZE))
            read_buffer_size = read_buffer_size - read_buffer_size % 4

            while True:
                if buffer_offset == len(score_buffer):
                    score_file = open(score_file_path, 'rb')
                    index_file = open(index_file_path, 'rb')
                    score_file.seek(file_offset)
                    index_file.seek(file_offset)

                    score_buffer = score_file.read(read_buffer_size)
                    index_buffer = index_file.read(read_buffer_size)
                    score_file.close()
                    index_file.close()

                    file_offset += read_buffer_size
                    buffer_offset = 0
                packed_score = score_buffer[buffer_offset:buffer_offset+4]
                packed_index = index_buffer[buffer_offset:buffer_offset+4]
                buffer_offset += 4
                if not packed_score:
                    break
                yield (struct.unpack('f', packed_score)[0],
                       struct.unpack('i', packed_index)[0])
        finally:
            # deletes the files when generator goes out of scope or ends
            os.remove(score_file_path)
            os.remove(index_file_path)

    def _sort_cache_to_iterator(
            index_cache, score_cache):
        """Flushe the current cache to a heap and return it.

        Args:
            index_cache (1d numpy.array): contains flat indexes to the
                score pixels `score_cache`
            score_cache (1d numpy.array): contains score pixels

        Returns:
            Iterable to visit scores/indexes in increasing score order.

        """
        # sort the whole bunch to disk
        score_file = tempfile.NamedTemporaryFile(delete=False)
        index_file = tempfile.NamedTemporaryFile(delete=False)

        sort_index = score_cache.argsort()
        score_cache = score_cache[sort_index]
        index_cache = index_cache[sort_index]
        for index in range(0, score_cache.size, _LARGEST_STRUCT_PACK):
            score_block = score_cache[index:index+_LARGEST_STRUCT_PACK]
            index_block = index_cache[index:index+_LARGEST_STRUCT_PACK]
            score_file.write(
                struct.pack('%sf' % score_block.size, *score_block))
            index_file.write(
                struct.pack('%si' % index_block.size, *index_block))

        score_file_path = score_file.name
        index_file_path = index_file.name
        score_file.close()
        index_file.close()

        return _read_score_index_from_disk(score_file_path, index_file_path)

    dataset_info = pygeoprocessing.get_raster_info(dataset_path)
    nodata = dataset_info['nodata'][0]
    nodata *= score_weight  # scale the nodata so they can be filtered out

    # This will be a list of file iterators we'll pass to heap.merge
    iters = []

    n_cols = dataset_info['raster_size'][0]

    for scores_data, scores_block in pygeoprocessing.iterblocks(
            (dataset_path, 1), largest_block=_BLOCK_SIZE):
        # flatten and scale the results
        scores_block = scores_block.flatten() * score_weight

        col_coords, row_coords = numpy.meshgrid(
            range(scores_data['xoff'], scores_data['xoff'] +
                  scores_data['win_xsize']),
            range(scores_data['yoff'], scores_data['yoff'] +
                  scores_data['win_ysize']))

        flat_indexes = (col_coords + row_coords * n_cols).flatten()

        sort_index = scores_block.argsort()
        sorted_scores = scores_block[sort_index]
        sorted_indexes = flat_indexes[sort_index]

        # search for nodata values are so we can splice them out
        left_index = numpy.searchsorted(sorted_scores, nodata, side='left')
        right_index = numpy.searchsorted(
            sorted_scores, nodata, side='right')

        # remove nodata values
        score_cache = numpy.concatenate(
            (sorted_scores[0:left_index], sorted_scores[right_index::]))
        index_cache = numpy.concatenate(
            (sorted_indexes[0:left_index], sorted_indexes[right_index::]))

        iters.append(_sort_cache_to_iterator(index_cache, score_cache))

    return heapq.merge(*iters)


def _convert_by_score(
        score_path, max_pixels_to_convert, out_raster_path, convert_value,
        stats_cache, score_weight):
    """Convert up to max pixels in ranked order of score.

    Args:
        score_path (string): path to a raster whose non-nodata values score the
            pixels to convert.  The pixels in `out_raster_path` are converted
            from the lowest score to the highest.  This scale can be modified
            by the parameter `score_weight`.
        max_pixels_to_convert (int): number of pixels to convert in
            `out_raster_path` up to the number of non nodata valued pixels in
            `score_path`.
        out_raster_path (string): a path to an existing raster that is of the
            same dimensions and projection as `score_path`.  The pixels in this
            raster are modified depending on the value of `score_path` and set
            to the value in `convert_value`.
        convert_value (int/float): type is dependant on out_raster_path. Any
            pixels converted in `out_raster_path` are set to the value of this
            variable.
        reverse_sort (boolean): If true, pixels are visited in descreasing
            order of `score_path`, otherwise increasing.
        stats_cache (collections.defaultdict(int)): contains the number of
            pixels converted indexed by original pixel id.

    Returns:
        None.

    """
    def _flush_cache_to_band(
            data_array, row_array, col_array, valid_index, dirty_blocks,
            out_band, stats_counter):
        """Flush block cache to the output band.

        Provided as an internal function because the exact operation needs
        to be invoked inside the processing loop and again at the end to
        finalize the scan.

        Args:
            data_array (numpy array): 1D array of valid data in buffer
            row_array (numpy array): 1D array to indicate row indexes for
                `data_array`
            col_array (numpy array): 1D array to indicate col indexes for
                `data_array`
            valid_index (int): value indicates the non-inclusive left valid
                entry in the parallel input arrays
            dirty_blocks (set): contains tuples indicating the block row and
                column indexes that will need to be set in `out_band`.  Allows
                us to skip the examination of the entire sparse matrix.
            out_band (gdal.Band): output band to write to
            stats_counter (collections.defaultdict(int)): is updated so that
                the key corresponds to ids in out_band that get set by the
                sparse matrix, and the number of pixels converted is added
                to the value of that entry.

        Returns:
            None

        """
        # construct sparse matrix so it can be indexed later
        sparse_matrix = scipy.sparse.csc_matrix(
            (data_array[:valid_index],
             (row_array[:valid_index], col_array[:valid_index])),
            shape=(n_rows, n_cols))

        # classic memory block iteration
        for block_row_index, block_col_index in dirty_blocks:
            row_index = block_row_index * out_block_row_size
            col_index = block_col_index * out_block_col_size
            row_index_end = row_index + out_block_row_size
            col_index_end = col_index + out_block_col_size
            row_win = out_block_row_size
            col_win = out_block_col_size
            if row_index_end > n_rows:
                row_index_end = n_rows
                row_win = n_rows - row_index
            if col_index_end > n_cols:
                col_index_end = n_cols
                col_win = n_cols - col_index

            # slice out values, some must be non-zero because of set
            mask_array = sparse_matrix[
                row_index:row_index_end, col_index:col_index_end].toarray()

            # read old array so we can write over the top
            out_array = out_band.ReadAsArray(
                xoff=col_index, yoff=row_index,
                win_xsize=col_win, win_ysize=row_win)

            # keep track of the stats of what ids changed
            for unique_id in numpy.unique(out_array[mask_array]):
                stats_counter[unique_id] += numpy.count_nonzero(
                    out_array[mask_array] == unique_id)

            out_array[mask_array] = convert_value
            out_band.WriteArray(out_array, xoff=col_index, yoff=row_index)

    out_ds = gdal.OpenEx(out_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    out_band = out_ds.GetRasterBand(1)
    out_block_col_size, out_block_row_size = out_band.GetBlockSize()
    n_rows = out_band.YSize
    n_cols = out_band.XSize
    pixels_converted = 0

    row_array = numpy.empty((_BLOCK_SIZE,), dtype=numpy.uint32)
    col_array = numpy.empty((_BLOCK_SIZE,), dtype=numpy.uint32)
    data_array = numpy.empty((_BLOCK_SIZE,), dtype=bool)
    next_index = 0
    dirty_blocks = set()

    last_time = time.time()
    for _, flatindex in _sort_to_disk(score_path, score_weight=score_weight):
        if pixels_converted >= max_pixels_to_convert:
            break
        col_index = flatindex % n_cols
        row_index = flatindex // n_cols
        row_array[next_index] = row_index
        col_array[next_index] = col_index
        # data_array will only ever recieve True elements, necessary for the
        # sparse matrix to function since it requires a data array as long
        # as the row and column arrays
        data_array[next_index] = True
        next_index += 1
        dirty_blocks.add(
            (row_index // out_block_row_size, col_index // out_block_col_size))
        pixels_converted += 1

        if time.time() - last_time > 5.0:
            LOGGER.info(
                "converted %d of %d pixels", pixels_converted,
                max_pixels_to_convert)
            last_time = time.time()

        if next_index == _BLOCK_SIZE:
            # next_index points beyond the end of the cache, flush and reset
            _flush_cache_to_band(
                data_array, row_array, col_array, next_index, dirty_blocks,
                out_band, stats_cache)
            dirty_blocks = set()
            next_index = 0

    # flush any remaining cache
    _flush_cache_to_band(
        data_array, row_array, col_array, next_index, dirty_blocks, out_band,
        stats_cache)


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to `execute`'s contract.

    Args:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in `execute` docstring.
        limit_to (str): (optional) if not None indicates that validation
            should only occur on the args[limit_to] value. The intent that
            individual key validation could be significantly less expensive
            than validating the entire `args` dictionary.

    Returns:
        list of ([invalid key_a, invalid_keyb, ...], 'warning/error message')
            tuples. Where an entry indicates that the invalid keys caused
            the error message in the second part of the tuple. This should
            be an empty list if validation succeeds.

    """
    validation_warnings = validation.validate(args, MODEL_SPEC)
    invalid_keys = validation.get_invalid_keys(validation_warnings)

    if ('convert_nearest_to_edge' not in invalid_keys and
            'convert_farthest_from_edge' not in invalid_keys):
        if (not args['convert_nearest_to_edge'] and
                not args['convert_farthest_from_edge']):
            validation_warnings.append((
                ['convert_nearest_to_edge', 'convert_farthest_from_edge'],
                MISSING_CONVERT_OPTION_MSG))

    return validation_warnings
