# coding=UTF-8
"""A collection of raster and vector algorithms and utilities."""
import collections
import functools
import logging
import math
import os
import pprint
import queue
import shutil
import sys
import tempfile
import threading
import time

from . import geoprocessing_core
from .geoprocessing_core import DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy
import numpy.ma
import rtree
import scipy.interpolate
import scipy.ndimage
import scipy.signal
import scipy.signal.signaltools
import scipy.sparse
import shapely.ops
import shapely.prepared
import shapely.wkb

# This is used to efficiently pass data to the raster stats worker if available
if sys.version_info >= (3, 8):
    import multiprocessing.shared_memory


class ReclassificationMissingValuesError(Exception):
    """Raised when a raster value is not a valid key to a dictionary.

    Attributes:
        msg (str) - error message
        missing_values (list) - a list of the missing values from the raster
            that are not keys in the dictionary
    """

    def __init__(self, msg, missing_values):
        self.msg = msg
        self.missing_values = missing_values
        super().__init__(msg, missing_values)


LOGGER = logging.getLogger(__name__)

# Used in joining finished TaskGraph Tasks.
_MAX_TIMEOUT = 60.0

_VALID_GDAL_TYPES = (
    set([getattr(gdal, x) for x in dir(gdal.gdalconst) if 'GDT_' in x]))

_LOGGING_PERIOD = 5.0  # min 5.0 seconds per update log message for the module
_LARGEST_ITERBLOCK = 2**16  # largest block for iterblocks to read in cells

_GDAL_TYPE_TO_NUMPY_LOOKUP = {
    gdal.GDT_Byte: numpy.uint8,
    gdal.GDT_Int16: numpy.int16,
    gdal.GDT_Int32: numpy.int32,
    gdal.GDT_UInt16: numpy.uint16,
    gdal.GDT_UInt32: numpy.uint32,
    gdal.GDT_Float32: numpy.float32,
    gdal.GDT_Float64: numpy.float64,
    gdal.GDT_CFloat32: numpy.csingle,
    gdal.GDT_CFloat64: numpy.complex64,
}

# In GDAL 3.0 spatial references no longer ignore Geographic CRS Axis Order
# and conform to Lat first, Lon Second. Transforms expect (lat, lon) order
# as opposed to the GIS friendly (lon, lat). See
# https://trac.osgeo.org/gdal/wiki/rfc73_proj6_wkt2_srsbarn Axis order
# issues. SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) swaps the
# axis order, which will use Lon,Lat order for Geographic CRS, but otherwise
# leaves Projected CRS alone
DEFAULT_OSR_AXIS_MAPPING_STRATEGY = osr.OAMS_TRADITIONAL_GIS_ORDER


def raster_calculator(
        base_raster_path_band_const_list, local_op, target_raster_path,
        datatype_target, nodata_target,
        calc_raster_stats=True, largest_block=_LARGEST_ITERBLOCK,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Apply local a raster operation on a stack of rasters.

    This function applies a user defined function across a stack of
    rasters' pixel stack. The rasters in ``base_raster_path_band_list`` must
    be spatially aligned and have the same cell sizes.

    Args:
        base_raster_path_band_const_list (sequence): a sequence containing:

            * ``(str, int)`` tuples, referring to a raster path/band index pair
              to use as an input.
            * ``numpy.ndarray`` s of up to two dimensions.  These inputs must
              all be broadcastable to each other AND the size of the raster
              inputs.
            * ``(object, 'raw')`` tuples, where ``object`` will be passed
              directly into the ``local_op``.

            All rasters must have the same raster size. If only arrays are
            input, numpy arrays must be broadcastable to each other and the
            final raster size will be the final broadcast array shape. A value
            error is raised if only "raw" inputs are passed.
        local_op (function): a function that must take in as many parameters as
            there are elements in ``base_raster_path_band_const_list``. The
            parameters in ``local_op`` will map 1-to-1 in order with the values
            in ``base_raster_path_band_const_list``. ``raster_calculator`` will
            call ``local_op`` to generate the pixel values in ``target_raster``
            along memory block aligned processing windows. Note any
            particular call to ``local_op`` will have the arguments from
            ``raster_path_band_const_list`` sliced to overlap that window.
            If an argument from ``raster_path_band_const_list`` is a
            raster/path band tuple, it will be passed to ``local_op`` as a 2D
            numpy array of pixel values that align with the processing window
            that ``local_op`` is targeting. A 2D or 1D array will be sliced to
            match the processing window and in the case of a 1D array tiled in
            whatever dimension is flat. If an argument is a scalar it is
            passed as as scalar.
            The return value must be a 2D array of the same size as any of the
            input parameter 2D arrays and contain the desired pixel values
            for the target raster.
        target_raster_path (string): the path of the output raster.  The
            projection, size, and cell size will be the same as the rasters
            in ``base_raster_path_const_band_list`` or the final broadcast
            size of the constant/ndarray values in the list.
        datatype_target (gdal datatype; int): the desired GDAL output type of
            the target raster.
        nodata_target (numerical value): the desired nodata value of the
            target raster.
        calc_raster_stats (boolean): If True, calculates and sets raster
            statistics (min, max, mean, and stdev) for target raster.
        largest_block (int): Attempts to internally iterate over raster blocks
            with this many elements.  Useful in cases where the blocksize is
            relatively small, memory is available, and the function call
            overhead dominates the iteration.  Defaults to 2**20.  A value of
            anything less than the original blocksize of the raster will
            result in blocksizes equal to the original size.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to
            geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None

    Raises:
        ValueError: invalid input provided
    """
    if not base_raster_path_band_const_list:
        raise ValueError(
            "`base_raster_path_band_const_list` is empty and "
            "should have at least one value.")

    # It's a common error to not pass in path/band tuples, so check for that
    # and report error if so
    bad_raster_path_list = False
    if not isinstance(base_raster_path_band_const_list, (list, tuple)):
        bad_raster_path_list = True
    else:
        for value in base_raster_path_band_const_list:
            if (not _is_raster_path_band_formatted(value) and
                not isinstance(value, numpy.ndarray) and
                not (isinstance(value, tuple) and len(value) == 2 and
                     value[1] == 'raw')):
                bad_raster_path_list = True
                break
    if bad_raster_path_list:
        raise ValueError(
            "Expected a sequence of path / integer band tuples, "
            "ndarrays, or (value, 'raw') pairs for "
            "`base_raster_path_band_const_list`, instead got: "
            "%s" % pprint.pformat(base_raster_path_band_const_list))

    # check that any rasters exist on disk and have enough bands
    not_found_paths = []
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    base_raster_path_band_list = [
        path_band for path_band in base_raster_path_band_const_list
        if _is_raster_path_band_formatted(path_band)]
    for value in base_raster_path_band_list:
        if gdal.OpenEx(value[0], gdal.OF_RASTER) is None:
            not_found_paths.append(value[0])
    gdal.PopErrorHandler()
    if not_found_paths:
        raise ValueError(
            "The following files were expected but do not exist on the "
            "filesystem: " + str(not_found_paths))

    # check that band index exists in raster
    invalid_band_index_list = []
    for value in base_raster_path_band_list:
        raster = gdal.OpenEx(value[0], gdal.OF_RASTER)
        if not (1 <= value[1] <= raster.RasterCount):
            invalid_band_index_list.append(value)
        raster = None
    if invalid_band_index_list:
        raise ValueError(
            "The following rasters do not contain requested band "
            "indexes: %s" % invalid_band_index_list)

    # check that the target raster is not also an input raster
    if target_raster_path in [x[0] for x in base_raster_path_band_list]:
        raise ValueError(
            "%s is used as a target path, but it is also in the base input "
            "path list %s" % (
                target_raster_path, str(base_raster_path_band_const_list)))

    # check that raster inputs are all the same dimensions
    raster_info_list = [
        get_raster_info(path_band[0])
        for path_band in base_raster_path_band_const_list
        if _is_raster_path_band_formatted(path_band)]
    geospatial_info_set = set()
    for raster_info in raster_info_list:
        geospatial_info_set.add(raster_info['raster_size'])
    if len(geospatial_info_set) > 1:
        raise ValueError(
            "Input Rasters are not the same dimensions. The "
            "following raster are not identical %s" % str(
                geospatial_info_set))

    numpy_broadcast_list = [
        x for x in base_raster_path_band_const_list
        if isinstance(x, numpy.ndarray)]
    stats_worker_thread = None
    try:
        # numpy.broadcast can only take up to 32 arguments, this loop works
        # around that restriction:
        while len(numpy_broadcast_list) > 1:
            numpy_broadcast_list = (
                [numpy.broadcast(*numpy_broadcast_list[:32])] +
                numpy_broadcast_list[32:])
        if numpy_broadcast_list:
            numpy_broadcast_size = numpy_broadcast_list[0].shape
    except ValueError:
        # this gets raised if numpy.broadcast fails
        raise ValueError(
            "Numpy array inputs cannot be broadcast into a single shape %s" %
            numpy_broadcast_list)

    if numpy_broadcast_list and len(numpy_broadcast_list[0].shape) > 2:
        raise ValueError(
            "Numpy array inputs must be 2 dimensions or less %s" %
            numpy_broadcast_list)

    # if there are both rasters and arrays, check the numpy shape will
    # be broadcastable with raster shape
    if raster_info_list and numpy_broadcast_list:
        # geospatial lists x/y order and numpy does y/x so reverse size list
        raster_shape = tuple(reversed(raster_info_list[0]['raster_size']))
        invalid_broadcast_size = False
        if len(numpy_broadcast_size) == 1:
            # if there's only one dimension it should match the last
            # dimension first, in the raster case this is the columns
            # because of the row/column order of numpy. No problem if
            # that value is ``1`` because it will be broadcast, otherwise
            # it should be the same as the raster.
            if (numpy_broadcast_size[0] != raster_shape[1] and
                    numpy_broadcast_size[0] != 1):
                invalid_broadcast_size = True
        else:
            for dim_index in range(2):
                # no problem if 1 because it'll broadcast, otherwise must
                # be the same value
                if (numpy_broadcast_size[dim_index] !=
                        raster_shape[dim_index] and
                        numpy_broadcast_size[dim_index] != 1):
                    invalid_broadcast_size = True
        if invalid_broadcast_size:
            raise ValueError(
                "Raster size %s cannot be broadcast to numpy shape %s" % (
                    raster_shape, numpy_broadcast_size))

    # create a "canonical" argument list that's bands, 2d numpy arrays, or
    # raw values only
    base_canonical_arg_list = []
    base_raster_list = []
    base_band_list = []
    for value in base_raster_path_band_const_list:
        # the input has been tested and value is either a raster/path band
        # tuple, 1d ndarray, 2d ndarray, or (value, 'raw') tuple.
        if _is_raster_path_band_formatted(value):
            # it's a raster/path band, keep track of open raster and band
            # for later so we can __swig_destroy__ them.
            base_raster_list.append(gdal.OpenEx(value[0], gdal.OF_RASTER))
            base_band_list.append(
                base_raster_list[-1].GetRasterBand(value[1]))
            base_canonical_arg_list.append(base_band_list[-1])
        elif isinstance(value, numpy.ndarray):
            if value.ndim == 1:
                # easier to process as a 2d array for writing to band
                base_canonical_arg_list.append(
                    value.reshape((1, value.shape[0])))
            else:  # dimensions are two because we checked earlier.
                base_canonical_arg_list.append(value)
        elif isinstance(value, tuple):
            base_canonical_arg_list.append(value)
        else:
            raise ValueError(
                "An unexpected ``value`` occurred. This should never happen. "
                "Value: %r" % value)

    # create target raster
    if raster_info_list:
        # if rasters are passed, the target is the same size as the raster
        n_cols, n_rows = raster_info_list[0]['raster_size']
    elif numpy_broadcast_list:
        # numpy arrays in args and no raster result is broadcast shape
        # expanded to two dimensions if necessary
        if len(numpy_broadcast_size) == 1:
            n_rows, n_cols = 1, numpy_broadcast_size[0]
        else:
            n_rows, n_cols = numpy_broadcast_size
    else:
        raise ValueError(
            "Only (object, 'raw') values have been passed. Raster "
            "calculator requires at least a raster or numpy array as a "
            "parameter. This is the input list: %s" % pprint.pformat(
                base_raster_path_band_const_list))

    if datatype_target not in _VALID_GDAL_TYPES:
        raise ValueError(
            'Invalid target type, should be a gdal.GDT_* type, received '
            '"%s"' % datatype_target)

    # create target raster
    raster_driver = gdal.GetDriverByName(raster_driver_creation_tuple[0])
    try:
        os.makedirs(os.path.dirname(target_raster_path))
    except OSError:
        pass
    target_raster = raster_driver.Create(
        target_raster_path, n_cols, n_rows, 1, datatype_target,
        options=raster_driver_creation_tuple[1])

    target_band = target_raster.GetRasterBand(1)
    if nodata_target is not None:
        target_band.SetNoDataValue(nodata_target)
    if base_raster_list:
        # use the first raster in the list for the projection and geotransform
        target_raster.SetProjection(base_raster_list[0].GetProjection())
        target_raster.SetGeoTransform(base_raster_list[0].GetGeoTransform())
    target_band.FlushCache()
    target_raster.FlushCache()

    try:
        last_time = time.time()

        block_offset_list = list(iterblocks(
            (target_raster_path, 1), offset_only=True,
            largest_block=largest_block))

        if calc_raster_stats:
            # if this queue is used to send computed valid blocks of
            # the raster to an incremental statistics calculator worker
            stats_worker_queue = queue.Queue()
            exception_queue = queue.Queue()

            if sys.version_info >= (3, 8):
                # The stats worker keeps running variables as a float64, so
                # all input rasters are dtype float64 -- make the shared memory
                # size equivalent.
                block_size_bytes = (
                    numpy.dtype(numpy.float64).itemsize *
                    block_offset_list[0]['win_xsize'] *
                    block_offset_list[0]['win_ysize'])

                shared_memory = multiprocessing.shared_memory.SharedMemory(
                    create=True, size=block_size_bytes)

        else:
            stats_worker_queue = None

        if calc_raster_stats:
            # To avoid doing two passes on the raster to calculate standard
            # deviation, we implement a continuous statistics calculation
            # as the raster is computed. This computational effort is high
            # and benefits from running in parallel. This queue and worker
            # takes a valid block of a raster and incrementally calculates
            # the raster's statistics. When ``None`` is pushed to the queue
            # the worker will finish and return a (min, max, mean, std)
            # tuple.
            LOGGER.info('starting stats_worker')
            stats_worker_thread = threading.Thread(
                target=geoprocessing_core.stats_worker,
                args=(stats_worker_queue, len(block_offset_list)))
            stats_worker_thread.daemon = True
            stats_worker_thread.start()
            LOGGER.info('started stats_worker %s', stats_worker_thread)

        pixels_processed = 0
        n_pixels = n_cols * n_rows

        # iterate over each block and calculate local_op
        for block_offset in block_offset_list:
            # read input blocks
            offset_list = (block_offset['yoff'], block_offset['xoff'])
            blocksize = (block_offset['win_ysize'], block_offset['win_xsize'])
            data_blocks = []
            for value in base_canonical_arg_list:
                if isinstance(value, gdal.Band):
                    data_blocks.append(value.ReadAsArray(**block_offset))
                    # I've encountered the following error when a gdal raster
                    # is corrupt, often from multiple threads writing to the
                    # same file. This helps to catch the error early rather
                    # than lead to confusing values of ``data_blocks`` later.
                    if not isinstance(data_blocks[-1], numpy.ndarray):
                        raise ValueError(
                            f"got a {data_blocks[-1]} when trying to read "
                            f"{value.GetDataset().GetFileList()} at "
                            f"{block_offset}, expected numpy.ndarray.")
                elif isinstance(value, numpy.ndarray):
                    # must be numpy array and all have been conditioned to be
                    # 2d, so start with 0:1 slices and expand if possible
                    slice_list = [slice(0, 1)] * 2
                    tile_dims = list(blocksize)
                    for dim_index in [0, 1]:
                        if value.shape[dim_index] > 1:
                            slice_list[dim_index] = slice(
                                offset_list[dim_index],
                                offset_list[dim_index] +
                                blocksize[dim_index],)
                            tile_dims[dim_index] = 1
                    data_blocks.append(
                        numpy.tile(value[tuple(slice_list)], tile_dims))
                else:
                    # must be a raw tuple
                    data_blocks.append(value[0])

            target_block = local_op(*data_blocks)

            if (not isinstance(target_block, numpy.ndarray) or
                    target_block.shape != blocksize):
                raise ValueError(
                    "Expected `local_op` to return a numpy.ndarray of "
                    "shape %s but got this instead: %s" % (
                        blocksize, target_block))

            target_band.WriteArray(
                target_block, yoff=block_offset['yoff'],
                xoff=block_offset['xoff'])

            # send result to stats calculator
            if stats_worker_queue:
                # guard against an undefined nodata target
                if nodata_target is not None:
                    target_block = target_block[target_block != nodata_target]
                target_block = target_block.astype(numpy.float64).flatten()

                if sys.version_info >= (3, 8):
                    shared_memory_array = numpy.ndarray(
                        target_block.shape, dtype=target_block.dtype,
                        buffer=shared_memory.buf)
                    shared_memory_array[:] = target_block[:]

                    stats_worker_queue.put((
                        shared_memory_array.shape, shared_memory_array.dtype,
                        shared_memory))
                else:
                    stats_worker_queue.put(target_block)

            pixels_processed += blocksize[0] * blocksize[1]
            last_time = _invoke_timed_callback(
                last_time, lambda: LOGGER.info(
                    '%.1f%% complete',
                    float(pixels_processed) / n_pixels * 100.0),
                _LOGGING_PERIOD)

        LOGGER.info('100.0% complete')

        if calc_raster_stats:
            LOGGER.info("Waiting for raster stats worker result.")
            stats_worker_thread.join(_MAX_TIMEOUT)
            if stats_worker_thread.is_alive():
                raise RuntimeError("stats_worker_thread.join() timed out")
            payload = stats_worker_queue.get(True, _MAX_TIMEOUT)
            if payload is not None:
                target_min, target_max, target_mean, target_stddev = payload
                target_band.SetStatistics(
                    float(target_min), float(target_max), float(target_mean),
                    float(target_stddev))
                target_band.FlushCache()
    finally:
        # This block ensures that rasters are destroyed even if there's an
        # exception raised.
        base_band_list[:] = []
        for raster in base_raster_list:
            gdal.Dataset.__swig_destroy__(raster)
        base_raster_list[:] = []
        target_band.FlushCache()
        target_band = None
        target_raster.FlushCache()
        gdal.Dataset.__swig_destroy__(target_raster)
        target_raster = None

        if calc_raster_stats and stats_worker_thread:
            if stats_worker_thread.is_alive():
                stats_worker_queue.put(None, True, _MAX_TIMEOUT)
                LOGGER.info("Waiting for raster stats worker result.")
                stats_worker_thread.join(_MAX_TIMEOUT)
                if stats_worker_thread.is_alive():
                    raise RuntimeError("stats_worker_thread.join() timed out")
                if sys.version_info >= (3, 8):
                    shared_memory.close()
                    shared_memory.unlink()

            # check for an exception in the workers, otherwise get result
            # and pass to writer
            try:
                exception = exception_queue.get_nowait()
                LOGGER.error("Exception encountered at termination.")
                raise exception
            except queue.Empty:
                pass


def align_and_resize_raster_stack(
        base_raster_path_list, target_raster_path_list, resample_method_list,
        target_pixel_size, bounding_box_mode, base_vector_path_list=None,
        raster_align_index=None, base_projection_wkt_list=None,
        target_projection_wkt=None, vector_mask_options=None,
        gdal_warp_options=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
        osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY):
    """Generate rasters from a base such that they align geospatially.

    This function resizes base rasters that are in the same geospatial
    projection such that the result is an aligned stack of rasters that have
    the same cell size, dimensions, and bounding box. This is achieved by
    clipping or resizing the rasters to intersected, unioned, or equivocated
    bounding boxes of all the raster and vector input.

    Args:
        base_raster_path_list (sequence): a sequence of base raster paths that
            will be transformed and will be used to determine the target
            bounding box.
        target_raster_path_list (sequence): a sequence of raster paths that
            will be created to one-to-one map with ``base_raster_path_list``
            as aligned versions of those original rasters. If there are
            duplicate paths in this list, the function will raise a ValueError.
        resample_method_list (sequence): a sequence of resampling methods
            which one to one map each path in ``base_raster_path_list`` during
            resizing.  Each element must be one of
            "near|bilinear|cubic|cubicspline|lanczos|mode".
        target_pixel_size (list/tuple): the target raster's x and y pixel size
            example: (30, -30).
        bounding_box_mode (string): one of "union", "intersection", or
            a sequence of floats of the form [minx, miny, maxx, maxy] in the
            target projection coordinate system.  Depending
            on the value, output extents are defined as the union,
            intersection, or the explicit bounding box.
        base_vector_path_list (sequence): a sequence of base vector paths
            whose bounding boxes will be used to determine the final bounding
            box of the raster stack if mode is 'union' or 'intersection'.  If
            mode is 'bb=[...]' then these vectors are not used in any
            calculation.
        raster_align_index (int): indicates the index of a
            raster in ``base_raster_path_list`` that the target rasters'
            bounding boxes pixels should align with.  This feature allows
            rasters whose raster dimensions are the same, but bounding boxes
            slightly shifted less than a pixel size to align with a desired
            grid layout.  If ``None`` then the bounding box of the target
            rasters is calculated as the precise intersection, union, or
            bounding box.
        base_projection_wkt_list (sequence): if not None, this is a sequence of
            base projections of the rasters in ``base_raster_path_list``. If a
            value is ``None`` the ``base_sr`` is assumed to be whatever is
            defined in that raster. This value is useful if there are rasters
            with no projection defined, but otherwise known.
        target_projection_wkt (string): if not None, this is the desired
            projection of all target rasters in Well Known Text format. If
            None, the base SRS will be passed to the target.
        vector_mask_options (dict): optional, if not None, this is a
            dictionary of options to use an existing vector's geometry to
            mask out pixels in the target raster that do not overlap the
            vector's geometry. Keys to this dictionary are:

            * ``'mask_vector_path'`` (str): path to the mask vector file.
              This vector will be automatically projected to the target
              projection if its base coordinate system does not match the
              target.
            * ``'mask_layer_name'`` (str): the layer name to use for masking.
              If this key is not in the dictionary the default is to use
              the layer at index 0.
            * ``'mask_vector_where_filter'`` (str): an SQL WHERE string.
              This will be used to filter the geometry in the mask. Ex: ``'id
              > 10'`` would use all features whose field value of 'id' is >
              10.

        gdal_warp_options (sequence): if present, the contents of this list
            are passed to the ``warpOptions`` parameter of ``gdal.Warp``. See
            the `GDAL Warp documentation
            <https://gdal.org/api/gdalwarp_cpp.html#_CPPv415GDALWarpOptions>`_
            for valid options.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.
        osr_axis_mapping_strategy (int): OSR axis mapping strategy for
            ``SpatialReference`` objects. Defaults to
            ``geoprocessing.DEFAULT_OSR_AXIS_MAPPING_STRATEGY``. This parameter
            should not be changed unless you know what you are doing.

    Returns:
        None

    Raises:
        ValueError
            If any combination of the raw bounding boxes, raster
            bounding boxes, vector bounding boxes, and/or vector_mask
            bounding box does not overlap to produce a valid target.
        ValueError
            If any of the input or target lists are of different
            lengths.
        ValueError
            If there are duplicate paths on the target list which would
            risk corrupted output.
        ValueError
            If some combination of base, target, and embedded source
            reference systems results in an ambiguous target coordinate
            system.
        ValueError
            If ``vector_mask_options`` is not None but the
            ``mask_vector_path`` is undefined or doesn't point to a valid
            file.
        ValueError
            If ``pixel_size`` is not a 2 element sequence of numbers.

    """
    # make sure that the input lists are of the same length
    list_lengths = [
        len(base_raster_path_list), len(target_raster_path_list),
        len(resample_method_list)]
    if len(set(list_lengths)) != 1:
        raise ValueError(
            "base_raster_path_list, target_raster_path_list, and "
            "resample_method_list must be the same length "
            " current lengths are %s" % (str(list_lengths)))

    unique_targets = set(target_raster_path_list)
    if len(unique_targets) != len(target_raster_path_list):
        seen = set()
        duplicate_list = []
        for path in target_raster_path_list:
            if path not in seen:
                seen.add(path)
            else:
                duplicate_list.append(path)
        raise ValueError(
            "There are duplicated paths on the target list. This is an "
            "invalid state of ``target_path_list``. Duplicates: %s" % (
                duplicate_list))

    # we can accept 'union', 'intersection', or a 4 element list/tuple
    if bounding_box_mode not in ["union", "intersection"] and (
            not isinstance(bounding_box_mode, (list, tuple)) or
            len(bounding_box_mode) != 4):
        raise ValueError("Unknown bounding_box_mode %s" % (
            str(bounding_box_mode)))

    n_rasters = len(base_raster_path_list)
    if ((raster_align_index is not None) and
            ((raster_align_index < 0) or (raster_align_index >= n_rasters))):
        raise ValueError(
            "Alignment index is out of bounds of the datasets index: %s"
            " n_elements %s" % (raster_align_index, n_rasters))

    _assert_is_valid_pixel_size(target_pixel_size)

    # used to get bounding box, projection, and possible alignment info
    raster_info_list = [
        get_raster_info(path) for path in base_raster_path_list]

    # get the literal or intersecting/unioned bounding box
    if isinstance(bounding_box_mode, (list, tuple)):
        # if it's a sequence or tuple, it must be a manual bounding box
        LOGGER.debug(
            "assuming manual bounding box mode of %s", bounding_box_mode)
        target_bounding_box = bounding_box_mode
    else:
        # either intersection or union, get list of bounding boxes, reproject
        # if necessary, and reduce to a single box
        if base_vector_path_list is not None:
            # vectors are only interesting for their bounding boxes, that's
            # this construction is inside an else.
            vector_info_list = [
                get_vector_info(path) for path in base_vector_path_list]
        else:
            vector_info_list = []

        raster_bounding_box_list = []
        for raster_index, raster_info in enumerate(raster_info_list):
            # this block calculates the base projection of ``raster_info`` if
            # ``target_projection_wkt`` is defined, thus implying a
            # reprojection will be necessary.
            if target_projection_wkt:
                if base_projection_wkt_list and \
                        base_projection_wkt_list[raster_index]:
                    # a base is defined, use that
                    base_raster_projection_wkt = \
                        base_projection_wkt_list[raster_index]
                else:
                    # otherwise use the raster's projection and there must
                    # be one since we're reprojecting
                    base_raster_projection_wkt = raster_info['projection_wkt']
                    if not base_raster_projection_wkt:
                        raise ValueError(
                            "no projection for raster %s" %
                            base_raster_path_list[raster_index])
                # since the base spatial reference is potentially different
                # than the target, we need to transform the base bounding
                # box into target coordinates so later we can calculate
                # accurate bounding box overlaps in the target coordinate
                # system
                raster_bounding_box_list.append(
                    transform_bounding_box(
                        raster_info['bounding_box'],
                        base_raster_projection_wkt, target_projection_wkt))
            else:
                raster_bounding_box_list.append(raster_info['bounding_box'])

        # include the vector bounding box information to make a global list
        # of target bounding boxes
        bounding_box_list = [
            vector_info['bounding_box'] if target_projection_wkt is None else
            transform_bounding_box(
                vector_info['bounding_box'],
                vector_info['projection_wkt'], target_projection_wkt)
            for vector_info in vector_info_list] + raster_bounding_box_list

        target_bounding_box = merge_bounding_box_list(
            bounding_box_list, bounding_box_mode)

    if vector_mask_options:
        # ensure the mask exists and intersects with the target bounding box
        if 'mask_vector_path' not in vector_mask_options:
            raise ValueError(
                'vector_mask_options passed, but no value for '
                '"mask_vector_path": %s', vector_mask_options)
        mask_vector_info = get_vector_info(
            vector_mask_options['mask_vector_path'])
        mask_vector_projection_wkt = mask_vector_info['projection_wkt']
        if mask_vector_projection_wkt is not None and \
                target_projection_wkt is not None:
            mask_vector_bb = transform_bounding_box(
                mask_vector_info['bounding_box'],
                mask_vector_info['projection_wkt'], target_projection_wkt)
        else:
            mask_vector_bb = mask_vector_info['bounding_box']
        # Calling `merge_bounding_box_list` will raise an ValueError if the
        # bounding box of the mask and the target do not intersect. The
        # result is otherwise not used.
        _ = merge_bounding_box_list(
            [target_bounding_box, mask_vector_bb], 'intersection')

    if raster_align_index is not None and raster_align_index >= 0:
        # bounding box needs alignment
        align_bounding_box = (
            raster_info_list[raster_align_index]['bounding_box'])
        align_pixel_size = (
            raster_info_list[raster_align_index]['pixel_size'])
        # adjust bounding box so lower left corner aligns with a pixel in
        # raster[raster_align_index]
        for index in [0, 1]:
            n_pixels = int(
                (target_bounding_box[index] - align_bounding_box[index]) /
                float(align_pixel_size[index]))
            target_bounding_box[index] = (
                n_pixels * align_pixel_size[index] +
                align_bounding_box[index])

    for index, (base_path, target_path, resample_method) in enumerate(zip(
            base_raster_path_list, target_raster_path_list,
            resample_method_list)):
        warp_raster(
            base_path, target_pixel_size, target_path, resample_method,
            target_bb=target_bounding_box,
            raster_driver_creation_tuple=(raster_driver_creation_tuple),
            target_projection_wkt=target_projection_wkt,
            base_projection_wkt=(
                    None if not base_projection_wkt_list else
                    base_projection_wkt_list[index]),
            vector_mask_options=vector_mask_options,
            gdal_warp_options=gdal_warp_options)
        LOGGER.info(
            '%d of %d aligned: %s', index+1, n_rasters,
            os.path.basename(target_path))

    LOGGER.info("aligned all %d rasters.", n_rasters)


def new_raster_from_base(
        base_path, target_path, datatype, band_nodata_list,
        fill_value_list=None, n_rows=None, n_cols=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Create new raster by coping spatial reference/geotransform of base.

    A convenience function to simplify the creation of a new raster from the
    basis of an existing one.  Depending on the input mode, one can create
    a new raster of the same dimensions, geotransform, and georeference as
    the base.  Other options are provided to change the raster dimensions,
    number of bands, nodata values, data type, and core raster creation
    options.

    Args:
        base_path (string): path to existing raster.
        target_path (string): path to desired target raster.
        datatype: the pixel datatype of the output raster, for example
            gdal.GDT_Float32.  See the following header file for supported
            pixel types:
            http://www.gdal.org/gdal_8h.html#22e22ce0a55036a96f652765793fb7a4
        band_nodata_list (sequence): list of nodata values, one for each band,
            to set on target raster.  If value is 'None' the nodata value is
            not set for that band.  The number of target bands is inferred
            from the length of this list.
        fill_value_list (sequence): list of values to fill each band with. If
            None, no filling is done.
        n_rows (int): if not None, defines the number of target raster rows.
        n_cols (int): if not None, defines the number of target raster
            columns.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None
    """
    base_raster = gdal.OpenEx(base_path, gdal.OF_RASTER)
    if n_rows is None:
        n_rows = base_raster.RasterYSize
    if n_cols is None:
        n_cols = base_raster.RasterXSize
    driver = gdal.GetDriverByName(raster_driver_creation_tuple[0])

    local_raster_creation_options = list(raster_driver_creation_tuple[1])
    # PIXELTYPE is sometimes used to define signed vs. unsigned bytes and
    # the only place that is stored is in the IMAGE_STRUCTURE metadata
    # copy it over if it exists and it not already defined by the input
    # creation options. It's okay to get this info from the first band since
    # all bands have the same datatype
    base_band = base_raster.GetRasterBand(1)
    metadata = base_band.GetMetadata('IMAGE_STRUCTURE')
    if 'PIXELTYPE' in metadata and not any(
            ['PIXELTYPE' in option for option in
             local_raster_creation_options]):
        local_raster_creation_options.append(
            'PIXELTYPE=' + metadata['PIXELTYPE'])

    block_size = base_band.GetBlockSize()
    # It's not clear how or IF we can determine if the output should be
    # striped or tiled.  Here we leave it up to the default inputs or if its
    # obviously not striped we tile.
    if not any(
            ['TILED' in option for option in local_raster_creation_options]):
        # TILED not set, so lets try to set it to a reasonable value
        if block_size[0] != n_cols:
            # if x block is not the width of the raster it *must* be tiled
            # otherwise okay if it's striped or tiled, I can't construct a
            # test case to cover this, but there is nothing in the spec that
            # restricts this so I have it just in case.
            local_raster_creation_options.append('TILED=YES')

    if not any(
            ['BLOCK' in option for option in local_raster_creation_options]):
        # not defined, so lets copy what we know from the current raster
        local_raster_creation_options.extend([
            'BLOCKXSIZE=%d' % block_size[0],
            'BLOCKYSIZE=%d' % block_size[1]])

    # make target directory if it doesn't exist
    try:
        os.makedirs(os.path.dirname(target_path))
    except OSError:
        pass

    base_band = None
    n_bands = len(band_nodata_list)
    target_raster = driver.Create(
        target_path, n_cols, n_rows, n_bands, datatype,
        options=local_raster_creation_options)
    target_raster.SetProjection(base_raster.GetProjection())
    target_raster.SetGeoTransform(base_raster.GetGeoTransform())
    base_raster = None

    for index, nodata_value in enumerate(band_nodata_list):
        if nodata_value is None:
            continue
        target_band = target_raster.GetRasterBand(index + 1)
        try:
            target_band.SetNoDataValue(nodata_value.item())
        except AttributeError:
            target_band.SetNoDataValue(nodata_value)

    target_raster.FlushCache()
    last_time = time.time()
    pixels_processed = 0
    n_pixels = n_cols * n_rows
    if fill_value_list is not None:
        for index, fill_value in enumerate(fill_value_list):
            if fill_value is None:
                continue
            target_band = target_raster.GetRasterBand(index + 1)
            # some rasters are very large and a fill can appear to cause
            # computation to hang. This block, though possibly slightly less
            # efficient than ``band.Fill`` will give real-time feedback about
            # how the fill is progressing.
            for offsets in iterblocks((target_path, 1), offset_only=True):
                fill_array = numpy.empty(
                    (offsets['win_ysize'], offsets['win_xsize']))
                pixels_processed += (
                    offsets['win_ysize'] * offsets['win_xsize'])
                fill_array[:] = fill_value
                target_band.WriteArray(
                    fill_array, offsets['xoff'], offsets['yoff'])

                last_time = _invoke_timed_callback(
                    last_time, lambda: LOGGER.info(
                        f'filling new raster {target_path} with {fill_value} '
                        f'-- {float(pixels_processed)/n_pixels*100.0:.2f}% '
                        f'complete'),
                    _LOGGING_PERIOD)
            target_band = None
    target_band = None
    target_raster = None


def create_raster_from_vector_extents(
        base_vector_path, target_raster_path, target_pixel_size,
        target_pixel_type, target_nodata, fill_value=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Create a blank raster based on a vector file extent.

    Args:
        base_vector_path (string): path to vector shapefile to base the
            bounding box for the target raster.
        target_raster_path (string): path to location of generated geotiff;
            the upper left hand corner of this raster will be aligned with the
            bounding box of the source vector and the extent will be exactly
            equal or contained the source vector's bounding box depending on
            whether the pixel size divides evenly into the source bounding
            box; if not coordinates will be rounded up to contain the original
            extent.
        target_pixel_size (list/tuple): the x/y pixel size as a sequence
            Example::

                [30.0, -30.0]

        target_pixel_type (int): gdal GDT pixel type of target raster
        target_nodata (numeric): target nodata value. Can be None if no nodata
            value is needed.
        fill_value (int/float): value to fill in the target raster; no fill if
            value is None
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None
    """
    # Determine the width and height of the tiff in pixels based on the
    # maximum size of the combined envelope of all the features
    vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    shp_extent = None
    for layer_index in range(vector.GetLayerCount()):
        layer = vector.GetLayer(layer_index)
        for feature in layer:
            try:
                # envelope is [xmin, xmax, ymin, ymax]
                feature_extent = feature.GetGeometryRef().GetEnvelope()
                if shp_extent is None:
                    shp_extent = list(feature_extent)
                else:
                    # expand bounds of current bounding box to include that
                    # of the newest feature
                    shp_extent = [
                        f(shp_extent[index], feature_extent[index])
                        for index, f in enumerate([min, max, min, max])]
            except AttributeError as error:
                # For some valid OGR objects the geometry can be undefined
                # since it's valid to have a NULL entry in the attribute table
                # this is expressed as a None value in the geometry reference
                # this feature won't contribute
                LOGGER.warning(error)
        layer = None

    if target_pixel_type not in _VALID_GDAL_TYPES:
        raise ValueError(
            'Invalid target type, should be a gdal.GDT_* type, received '
            '"%s"' % target_pixel_type)

    # round up on the rows and cols so that the target raster encloses the
    # base vector
    n_cols = int(numpy.ceil(
        abs((shp_extent[1] - shp_extent[0]) / target_pixel_size[0])))
    n_cols = max(1, n_cols)

    n_rows = int(numpy.ceil(
        abs((shp_extent[3] - shp_extent[2]) / target_pixel_size[1])))
    n_rows = max(1, n_rows)

    driver = gdal.GetDriverByName(raster_driver_creation_tuple[0])
    n_bands = 1
    raster = driver.Create(
        target_raster_path, n_cols, n_rows, n_bands, target_pixel_type,
        options=raster_driver_creation_tuple[1])
    raster.GetRasterBand(1).SetNoDataValue(target_nodata)

    # Set the transform based on the upper left corner and given pixel
    # dimensions
    if target_pixel_size[0] < 0:
        x_source = shp_extent[1]
    else:
        x_source = shp_extent[0]
    if target_pixel_size[1] < 0:
        y_source = shp_extent[3]
    else:
        y_source = shp_extent[2]
    raster_transform = [
        x_source, target_pixel_size[0], 0.0,
        y_source, 0.0, target_pixel_size[1]]
    raster.SetGeoTransform(raster_transform)

    # Use the same projection on the raster as the shapefile
    raster.SetProjection(vector.GetLayer(0).GetSpatialRef().ExportToWkt())

    # Initialize everything to nodata
    if fill_value is not None:
        band = raster.GetRasterBand(1)
        band.Fill(fill_value)
        band.FlushCache()
        band = None
    layer = None
    vector = None
    raster = None
    vector = None


def interpolate_points(
        base_vector_path, vector_attribute_field, target_raster_path_band,
        interpolation_mode):
    """Interpolate point values onto an existing raster.

    Args:
        base_vector_path (string): path to a shapefile that contains point
            vector layers.
        vector_attribute_field (field): a string in the vector referenced at
            ``base_vector_path`` that refers to a numeric value in the
            vector's attribute table.  This is the value that will be
            interpolated across the raster.
        target_raster_path_band (tuple): a path/band number tuple to an
            existing raster which likely intersects or is nearby the source
            vector. The band in this raster will take on the interpolated
            numerical values  provided at each point.
        interpolation_mode (string): the interpolation method to use for
            scipy.interpolate.griddata, one of 'linear', near', or 'cubic'.

    Returns:
        None
    """
    source_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    point_list = []
    value_list = []
    for layer_index in range(source_vector.GetLayerCount()):
        layer = source_vector.GetLayer(layer_index)
        for point_feature in layer:
            value = point_feature.GetField(vector_attribute_field)
            # Add in the numpy notation which is row, col
            # Here the point geometry is in the form x, y (col, row)
            geometry = point_feature.GetGeometryRef()
            point = geometry.GetPoint()
            point_list.append([point[1], point[0]])
            value_list.append(value)

    point_array = numpy.array(point_list)
    value_array = numpy.array(value_list)

    # getting the offsets first before the raster is opened in update mode
    offset_list = list(
        iterblocks(target_raster_path_band, offset_only=True))
    target_raster = gdal.OpenEx(
        target_raster_path_band[0], gdal.OF_RASTER | gdal.GA_Update)
    band = target_raster.GetRasterBand(target_raster_path_band[1])
    nodata = band.GetNoDataValue()
    geotransform = target_raster.GetGeoTransform()
    for offset in offset_list:
        grid_y, grid_x = numpy.mgrid[
            offset['yoff']:offset['yoff']+offset['win_ysize'],
            offset['xoff']:offset['xoff']+offset['win_xsize']]
        grid_y = grid_y * geotransform[5] + geotransform[3]
        grid_x = grid_x * geotransform[1] + geotransform[0]

        # this is to be consistent with GDAL 2.0's change of 'nearest' to
        # 'near' for an interpolation scheme that SciPy did not change.
        if interpolation_mode == 'near':
            interpolation_mode = 'nearest'
        raster_out_array = scipy.interpolate.griddata(
            point_array, value_array, (grid_y, grid_x), interpolation_mode,
            nodata)
        band.WriteArray(raster_out_array, offset['xoff'], offset['yoff'])


def zonal_statistics(
        base_raster_path_band, aggregate_vector_path,
        aggregate_layer_name=None, ignore_nodata=True,
        polygons_might_overlap=True, working_dir=None):
    """Collect stats on pixel values which lie within polygons.

    This function summarizes raster statistics including min, max,
    mean, and pixel count over the regions on the raster that are
    overlapped by the polygons in the vector layer. Statistics are calculated
    in two passes, where first polygons aggregate over pixels in the raster
    whose centers intersect with the polygon. In the second pass, any polygons
    that are not aggregated use their bounding box to intersect with the
    raster for overlap statistics.

    Note:
        There may be some degenerate cases where the bounding box vs. actual
        geometry intersection would be incorrect, but these are so unlikely as
        to be manually constructed. If you encounter one of these please email
        the description and dataset to richsharp@stanford.edu.

    Args:
        base_raster_path_band (tuple): a str/int tuple indicating the path to
            the base raster and the band index of that raster to analyze.
        aggregate_vector_path (string): a path to a polygon vector whose
            geometric features indicate the areas in
            ``base_raster_path_band`` to calculate zonal statistics.
        aggregate_layer_name (string): name of shapefile layer that will be
            used to aggregate results over.  If set to None, the first layer
            in the DataSource will be used as retrieved by ``.GetLayer()``.
            Note: it is normal and expected to set this field at None if the
            aggregating shapefile is a single layer as many shapefiles,
            including the common 'ESRI Shapefile', are.
        ignore_nodata: if true, then nodata pixels are not accounted for when
            calculating min, max, count, or mean.  However, the value of
            ``nodata_count`` will always be the number of nodata pixels
            aggregated under the polygon.
        polygons_might_overlap (boolean): if True the function calculates
            aggregation coverage close to optimally by rasterizing sets of
            polygons that don't overlap.  However, this step can be
            computationally expensive for cases where there are many polygons.
            Setting this flag to False directs the function rasterize in one
            step.
        working_dir (string): If not None, indicates where temporary files
            should be created during this run.

    Returns:
        nested dictionary indexed by aggregating feature id, and then by one
        of 'min' 'max' 'sum' 'count' and 'nodata_count'.  Example::

            {0: {'min': 0,
                 'max': 1,
                 'sum': 1.7,
                 'count': 3,
                 'nodata_count': 1
                 }
            }

    Raises:
        ValueError
            if ``base_raster_path_band`` is incorrectly formatted.
        RuntimeError
            if the aggregate vector or layer cannot open.

    """
    if not _is_raster_path_band_formatted(base_raster_path_band):
        raise ValueError(
            "`base_raster_path_band` not formatted as expected.  Expects "
            "(path, band_index), received %s" % repr(base_raster_path_band))
    aggregate_vector = gdal.OpenEx(aggregate_vector_path, gdal.OF_VECTOR)
    if aggregate_vector is None:
        raise RuntimeError(
            "Could not open aggregate vector at %s" % aggregate_vector_path)
    LOGGER.debug(aggregate_vector)
    if aggregate_layer_name is not None:
        aggregate_layer = aggregate_vector.GetLayerByName(
            aggregate_layer_name)
    else:
        aggregate_layer = aggregate_vector.GetLayer()
    if aggregate_layer is None:
        raise RuntimeError(
            "Could not open layer %s on %s" % (
                aggregate_layer_name, aggregate_vector_path))

    # create a new aggregate ID field to map base vector aggregate fields to
    # local ones that are guaranteed to be integers.
    local_aggregate_field_name = 'original_fid'
    rasterize_layer_args = {
        'options': [
            'ALL_TOUCHED=FALSE',
            'ATTRIBUTE=%s' % local_aggregate_field_name]
        }

    # clip base raster to aggregating vector intersection
    raster_info = get_raster_info(base_raster_path_band[0])
    # -1 here because bands are 1 indexed
    raster_nodata = raster_info['nodata'][base_raster_path_band[1]-1]
    with tempfile.NamedTemporaryFile(
            prefix='clipped_raster', suffix='.tif', delete=False,
            dir=working_dir) as clipped_raster_file:
        clipped_raster_path = clipped_raster_file.name
    try:
        align_and_resize_raster_stack(
            [base_raster_path_band[0]], [clipped_raster_path], ['near'],
            raster_info['pixel_size'], 'intersection',
            base_vector_path_list=[aggregate_vector_path],
            raster_align_index=0)
        clipped_raster = gdal.OpenEx(clipped_raster_path, gdal.OF_RASTER)
        clipped_band = clipped_raster.GetRasterBand(base_raster_path_band[1])
    except ValueError as e:
        if 'Bounding boxes do not intersect' in repr(e):
            LOGGER.error(
                "aggregate vector %s does not intersect with the raster %s",
                aggregate_vector_path, base_raster_path_band)
            aggregate_stats = collections.defaultdict(
                lambda: {
                    'min': None, 'max': None, 'count': 0, 'nodata_count': 0,
                    'sum': 0.0})
            for feature in aggregate_layer:
                _ = aggregate_stats[feature.GetFID()]
            return dict(aggregate_stats)
        else:
            # this would be very unexpected to get here, but if it happened
            # and we didn't raise an exception, execution could get weird.
            raise

    # make a shapefile that non-overlapping layers can be added to
    driver = ogr.GetDriverByName('MEMORY')
    disjoint_vector = driver.CreateDataSource('disjoint_vector')
    spat_ref = aggregate_layer.GetSpatialRef()

    # Initialize these dictionaries to have the shapefile fields in the
    # original datasource even if we don't pick up a value later
    LOGGER.info("build a lookup of aggregate field value to FID")

    aggregate_layer_fid_set = set(
        [agg_feat.GetFID() for agg_feat in aggregate_layer])

    # Loop over each polygon and aggregate
    if polygons_might_overlap:
        LOGGER.info("creating disjoint polygon set")
        disjoint_fid_sets = calculate_disjoint_polygon_set(
            aggregate_vector_path, bounding_box=raster_info['bounding_box'])
    else:
        disjoint_fid_sets = [aggregate_layer_fid_set]

    with tempfile.NamedTemporaryFile(
            prefix='aggregate_fid_raster', suffix='.tif',
            delete=False, dir=working_dir) as agg_fid_raster_file:
        agg_fid_raster_path = agg_fid_raster_file.name

    agg_fid_nodata = -1
    new_raster_from_base(
        clipped_raster_path, agg_fid_raster_path, gdal.GDT_Int32,
        [agg_fid_nodata])
    # fetch the block offsets before the raster is opened for writing
    agg_fid_offset_list = list(
        iterblocks((agg_fid_raster_path, 1), offset_only=True))
    agg_fid_raster = gdal.OpenEx(
        agg_fid_raster_path, gdal.GA_Update | gdal.OF_RASTER)
    aggregate_stats = collections.defaultdict(lambda: {
        'min': None, 'max': None, 'count': 0, 'nodata_count': 0, 'sum': 0.0})
    last_time = time.time()
    LOGGER.info("processing %d disjoint polygon sets", len(disjoint_fid_sets))
    for set_index, disjoint_fid_set in enumerate(disjoint_fid_sets):
        last_time = _invoke_timed_callback(
            last_time, lambda: LOGGER.info(
                "zonal stats approximately %.1f%% complete on %s",
                100.0 * float(set_index+1) / len(disjoint_fid_sets),
                os.path.basename(aggregate_vector_path)),
            _LOGGING_PERIOD)
        disjoint_layer = disjoint_vector.CreateLayer(
            'disjoint_vector', spat_ref, ogr.wkbPolygon)
        disjoint_layer.CreateField(
            ogr.FieldDefn(local_aggregate_field_name, ogr.OFTInteger))
        disjoint_layer_defn = disjoint_layer.GetLayerDefn()
        # add polygons to subset_layer
        disjoint_layer.StartTransaction()
        for index, feature_fid in enumerate(disjoint_fid_set):
            last_time = _invoke_timed_callback(
                last_time, lambda: LOGGER.info(
                    "polygon set %d of %d approximately %.1f%% processed "
                    "on %s", set_index+1, len(disjoint_fid_sets),
                    100.0 * float(index+1) / len(disjoint_fid_set),
                    os.path.basename(aggregate_vector_path)),
                _LOGGING_PERIOD)
            agg_feat = aggregate_layer.GetFeature(feature_fid)
            agg_geom_ref = agg_feat.GetGeometryRef()
            disjoint_feat = ogr.Feature(disjoint_layer_defn)
            disjoint_feat.SetGeometry(agg_geom_ref.Clone())
            disjoint_feat.SetField(
                local_aggregate_field_name, feature_fid)
            disjoint_layer.CreateFeature(disjoint_feat)
        disjoint_layer.CommitTransaction()

        LOGGER.info(
            "disjoint polygon set %d of %d 100.0%% processed on %s",
            set_index+1, len(disjoint_fid_sets), os.path.basename(
                aggregate_vector_path))

        # nodata out the mask
        agg_fid_band = agg_fid_raster.GetRasterBand(1)
        agg_fid_band.Fill(agg_fid_nodata)
        LOGGER.info(
            "rasterizing disjoint polygon set %d of %d %s", set_index+1,
            len(disjoint_fid_sets),
            os.path.basename(aggregate_vector_path))
        rasterize_callback = _make_logger_callback(
            "rasterizing polygon " + str(set_index+1) + " of " +
            str(len(disjoint_fid_set)) + " set %.1f%% complete")
        gdal.RasterizeLayer(
            agg_fid_raster, [1], disjoint_layer,
            callback=rasterize_callback, **rasterize_layer_args)
        agg_fid_raster.FlushCache()

        # Delete the features we just added to the subset_layer
        disjoint_layer = None
        disjoint_vector.DeleteLayer(0)

        # create a key array
        # and parallel min, max, count, and nodata count arrays
        LOGGER.info(
            "summarizing rasterized disjoint polygon set %d of %d %s",
            set_index+1, len(disjoint_fid_sets),
            os.path.basename(aggregate_vector_path))
        for agg_fid_offset in agg_fid_offset_list:
            agg_fid_block = agg_fid_band.ReadAsArray(**agg_fid_offset)
            clipped_block = clipped_band.ReadAsArray(**agg_fid_offset)
            valid_mask = (agg_fid_block != agg_fid_nodata)
            valid_agg_fids = agg_fid_block[valid_mask]
            valid_clipped = clipped_block[valid_mask]
            for agg_fid in numpy.unique(valid_agg_fids):
                masked_clipped_block = valid_clipped[
                    valid_agg_fids == agg_fid]
                if raster_nodata is not None:
                    clipped_nodata_mask = numpy.isclose(
                        masked_clipped_block, raster_nodata)
                else:
                    clipped_nodata_mask = numpy.zeros(
                        masked_clipped_block.shape, dtype=numpy.bool)
                aggregate_stats[agg_fid]['nodata_count'] += (
                    numpy.count_nonzero(clipped_nodata_mask))
                if ignore_nodata:
                    masked_clipped_block = (
                        masked_clipped_block[~clipped_nodata_mask])
                if masked_clipped_block.size == 0:
                    continue

                if aggregate_stats[agg_fid]['min'] is None:
                    aggregate_stats[agg_fid]['min'] = (
                        masked_clipped_block[0])
                    aggregate_stats[agg_fid]['max'] = (
                        masked_clipped_block[0])

                aggregate_stats[agg_fid]['min'] = min(
                    numpy.min(masked_clipped_block),
                    aggregate_stats[agg_fid]['min'])
                aggregate_stats[agg_fid]['max'] = max(
                    numpy.max(masked_clipped_block),
                    aggregate_stats[agg_fid]['max'])
                aggregate_stats[agg_fid]['count'] += (
                    masked_clipped_block.size)
                aggregate_stats[agg_fid]['sum'] += numpy.sum(
                    masked_clipped_block)
    unset_fids = aggregate_layer_fid_set.difference(aggregate_stats)
    LOGGER.debug(
        "unset_fids: %s of %s ", len(unset_fids),
        len(aggregate_layer_fid_set))
    clipped_gt = numpy.array(
        clipped_raster.GetGeoTransform(), dtype=numpy.float32)
    LOGGER.debug("gt %s for %s", clipped_gt, base_raster_path_band)
    for unset_fid in unset_fids:
        unset_feat = aggregate_layer.GetFeature(unset_fid)
        unset_geom_ref = unset_feat.GetGeometryRef()
        if unset_geom_ref is None:
            LOGGER.warn(
                f'no geometry in {aggregate_vector_path} FID: {unset_fid}')
            continue
        unset_geom_envelope = list(unset_geom_ref.GetEnvelope())
        if clipped_gt[1] < 0:
            unset_geom_envelope[0], unset_geom_envelope[1] = (
                unset_geom_envelope[1], unset_geom_envelope[0])
        if clipped_gt[5] < 0:
            unset_geom_envelope[2], unset_geom_envelope[3] = (
                unset_geom_envelope[3], unset_geom_envelope[2])

        xoff = int((unset_geom_envelope[0] - clipped_gt[0]) / clipped_gt[1])
        yoff = int((unset_geom_envelope[2] - clipped_gt[3]) / clipped_gt[5])
        win_xsize = int(numpy.ceil(
            (unset_geom_envelope[1] - clipped_gt[0]) /
            clipped_gt[1])) - xoff
        win_ysize = int(numpy.ceil(
            (unset_geom_envelope[3] - clipped_gt[3]) /
            clipped_gt[5])) - yoff

        # clamp offset to the side of the raster if it's negative
        if xoff < 0:
            win_xsize += xoff
            xoff = 0
        if yoff < 0:
            win_ysize += yoff
            yoff = 0

        # clamp the window to the side of the raster if too big
        if xoff+win_xsize > clipped_band.XSize:
            win_xsize = clipped_band.XSize-xoff
        if yoff+win_ysize > clipped_band.YSize:
            win_ysize = clipped_band.YSize-yoff

        if win_xsize <= 0 or win_ysize <= 0:
            continue

        # here we consider the pixels that intersect with the geometry's
        # bounding box as being the proxy for the intersection with the
        # polygon itself. This is not a bad approximation since the case
        # that caused the polygon to be skipped in the first phase is that it
        # is as small as a pixel. There could be some degenerate cases that
        # make this estimation very wrong, but we do not know of any that
        # would come from natural data. If you do encounter such a dataset
        # please email the description and datset to richsharp@stanford.edu.
        unset_fid_block = clipped_band.ReadAsArray(
            xoff=xoff, yoff=yoff, win_xsize=win_xsize, win_ysize=win_ysize)

        if raster_nodata is not None:
            unset_fid_nodata_mask = numpy.isclose(
                unset_fid_block, raster_nodata)
        else:
            unset_fid_nodata_mask = numpy.zeros(
                unset_fid_block.shape, dtype=numpy.bool)

        valid_unset_fid_block = unset_fid_block[~unset_fid_nodata_mask]
        if valid_unset_fid_block.size == 0:
            aggregate_stats[unset_fid]['min'] = 0.0
            aggregate_stats[unset_fid]['max'] = 0.0
            aggregate_stats[unset_fid]['sum'] = 0.0
        else:
            aggregate_stats[unset_fid]['min'] = numpy.min(
                valid_unset_fid_block)
            aggregate_stats[unset_fid]['max'] = numpy.max(
                valid_unset_fid_block)
            aggregate_stats[unset_fid]['sum'] = numpy.sum(
                valid_unset_fid_block)
        aggregate_stats[unset_fid]['count'] = valid_unset_fid_block.size
        aggregate_stats[unset_fid]['nodata_count'] = numpy.count_nonzero(
            unset_fid_nodata_mask)

    unset_fids = aggregate_layer_fid_set.difference(aggregate_stats)
    LOGGER.debug(
        "remaining unset_fids: %s of %s ", len(unset_fids),
        len(aggregate_layer_fid_set))
    # fill in the missing polygon fids in the aggregate stats by invoking the
    # accessor in the defaultdict
    for fid in unset_fids:
        _ = aggregate_stats[fid]

    LOGGER.info(
        "all done processing polygon sets for %s", os.path.basename(
            aggregate_vector_path))

    # clean up temporary files
    gdal.Dataset.__swig_destroy__(agg_fid_raster)
    gdal.Dataset.__swig_destroy__(aggregate_vector)
    gdal.Dataset.__swig_destroy__(clipped_raster)
    clipped_band = None
    clipped_raster = None
    agg_fid_raster = None
    disjoint_layer = None
    disjoint_vector = None
    aggregate_layer = None
    aggregate_vector = None
    for filename in [agg_fid_raster_path, clipped_raster_path]:
        os.remove(filename)

    return dict(aggregate_stats)


def get_vector_info(vector_path, layer_id=0):
    """Get information about an GDAL vector.

    Args:
        vector_path (str): a path to a GDAL vector.
        layer_id (str/int): name or index of underlying layer to analyze.
            Defaults to 0.

    Raises:
        ValueError if ``vector_path`` does not exist on disk or cannot be
        opened as a gdal.OF_VECTOR.

    Returns:
        raster_properties (dictionary):
            a dictionary with the following key-value pairs:

            * ``'projection_wkt'`` (string): projection of the vector in Well
              Known Text.
            * ``'bounding_box'`` (sequence): sequence of floats representing
              the bounding box in projected coordinates in the order
              [minx, miny, maxx, maxy].
            * ``'file_list'`` (sequence): sequence of string paths to the files
              that make up this vector.

    """
    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    if not vector:
        raise ValueError(
            "Could not open %s as a gdal.OF_VECTOR" % vector_path)
    vector_properties = {}
    vector_properties['file_list'] = vector.GetFileList()
    layer = vector.GetLayer(iLayer=layer_id)
    # projection is same for all layers, so just use the first one
    spatial_ref = layer.GetSpatialRef()
    if spatial_ref:
        vector_projection_wkt = spatial_ref.ExportToWkt()
    else:
        vector_projection_wkt = None
    vector_properties['projection_wkt'] = vector_projection_wkt
    layer_bb = layer.GetExtent()
    layer = None
    vector = None
    # convert form [minx,maxx,miny,maxy] to [minx,miny,maxx,maxy]
    vector_properties['bounding_box'] = [layer_bb[i] for i in [0, 2, 1, 3]]
    return vector_properties


def get_raster_info(raster_path):
    """Get information about a GDAL raster (dataset).

    Args:
       raster_path (String): a path to a GDAL raster.

    Raises:
        ValueError
            if ``raster_path`` is not a file or cannot be opened as a
            ``gdal.OF_RASTER``.

    Returns:
        raster_properties (dictionary):
            a dictionary with the properties stored under relevant keys.

        * ``'pixel_size'`` (tuple): (pixel x-size, pixel y-size)
          from geotransform.
        * ``'raster_size'`` (tuple):  number of raster pixels in (x, y)
          direction.
        * ``'nodata'`` (sequence): a sequence of the nodata values in the bands
          of the raster in the same order as increasing band index.
        * ``'n_bands'`` (int): number of bands in the raster.
        * ``'geotransform'`` (tuple): a 6-tuple representing the geotransform
          of (x orign, x-increase, xy-increase, y origin, yx-increase,
          y-increase).
        * ``'datatype'`` (int): An instance of an enumerated gdal.GDT_* int
          that represents the datatype of the raster.
        * ``'projection_wkt'`` (string): projection of the raster in Well Known
          Text.
        * ``'bounding_box'`` (sequence): sequence of floats representing the
          bounding box in projected coordinates in the order
          [minx, miny, maxx, maxy]
        * ``'block_size'`` (tuple): underlying x/y raster block size for
          efficient reading.
        * ``'numpy_type'`` (numpy type): this is the equivalent numpy datatype
          for the raster bands including signed bytes.

    """
    raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
    if not raster:
        raise ValueError(
            "Could not open %s as a gdal.OF_RASTER" % raster_path)
    raster_properties = {}
    raster_properties['file_list'] = raster.GetFileList()
    projection_wkt = raster.GetProjection()
    if not projection_wkt:
        projection_wkt = None
    raster_properties['projection_wkt'] = projection_wkt
    geo_transform = raster.GetGeoTransform()
    raster_properties['geotransform'] = geo_transform
    raster_properties['pixel_size'] = (geo_transform[1], geo_transform[5])
    raster_properties['raster_size'] = (
        raster.GetRasterBand(1).XSize,
        raster.GetRasterBand(1).YSize)
    raster_properties['n_bands'] = raster.RasterCount
    raster_properties['nodata'] = [
        raster.GetRasterBand(index).GetNoDataValue() for index in range(
            1, raster_properties['n_bands']+1)]
    # blocksize is the same for all bands, so we can just get the first
    raster_properties['block_size'] = raster.GetRasterBand(1).GetBlockSize()

    # we dont' really know how the geotransform is laid out, all we can do is
    # calculate the x and y bounds, then take the appropriate min/max
    x_bounds = [
        geo_transform[0], geo_transform[0] +
        raster_properties['raster_size'][0] * geo_transform[1] +
        raster_properties['raster_size'][1] * geo_transform[2]]
    y_bounds = [
        geo_transform[3], geo_transform[3] +
        raster_properties['raster_size'][0] * geo_transform[4] +
        raster_properties['raster_size'][1] * geo_transform[5]]

    raster_properties['bounding_box'] = [
        numpy.min(x_bounds), numpy.min(y_bounds),
        numpy.max(x_bounds), numpy.max(y_bounds)]

    # datatype is the same for the whole raster, but is associated with band
    band = raster.GetRasterBand(1)
    band_datatype = band.DataType
    raster_properties['datatype'] = band_datatype
    raster_properties['numpy_type'] = (
        _GDAL_TYPE_TO_NUMPY_LOOKUP[band_datatype])
    # this part checks to see if the byte is signed or not
    if band_datatype == gdal.GDT_Byte:
        metadata = band.GetMetadata('IMAGE_STRUCTURE')
        if 'PIXELTYPE' in metadata and metadata['PIXELTYPE'] == 'SIGNEDBYTE':
            raster_properties['numpy_type'] = numpy.int8
    band = None
    raster = None
    return raster_properties


def reproject_vector(
        base_vector_path, target_projection_wkt, target_path, layer_id=0,
        driver_name='ESRI Shapefile', copy_fields=True,
        osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY):
    """Reproject OGR DataSource (vector).

    Transforms the features of the base vector to the desired output
    projection in a new ESRI Shapefile.

    Args:
        base_vector_path (string): Path to the base shapefile to transform.
        target_projection_wkt (string): the desired output projection in Well
            Known Text (by layer.GetSpatialRef().ExportToWkt())
        target_path (string): the filepath to the transformed shapefile
        layer_id (str/int): name or index of layer in ``base_vector_path`` to
            reproject. Defaults to 0.
        driver_name (string): String to pass to ogr.GetDriverByName, defaults
            to 'ESRI Shapefile'.
        copy_fields (bool or iterable): If True, all the fields in
            ``base_vector_path`` will be copied to ``target_path`` during the
            reprojection step. If it is an iterable, it will contain the
            field names to exclusively copy. An unmatched fieldname will be
            ignored. If ``False`` no fields are copied into the new vector.
        osr_axis_mapping_strategy (int): OSR axis mapping strategy for
            ``SpatialReference`` objects. Defaults to
            ``geoprocessing.DEFAULT_OSR_AXIS_MAPPING_STRATEGY``. This parameter
            should not be changed unless you know what you are doing.

    Returns:
        None
    """
    base_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)

    # if this file already exists, then remove it
    if os.path.isfile(target_path):
        LOGGER.warning(
            "%s already exists, removing and overwriting", target_path)
        os.remove(target_path)

    target_sr = osr.SpatialReference(target_projection_wkt)

    # create a new shapefile from the orginal_datasource
    target_driver = ogr.GetDriverByName(driver_name)
    target_vector = target_driver.CreateDataSource(target_path)

    layer = base_vector.GetLayer(layer_id)
    layer_dfn = layer.GetLayerDefn()

    # Create new layer for target_vector using same name and
    # geometry type from base vector but new projection
    target_layer = target_vector.CreateLayer(
        layer_dfn.GetName(), target_sr, layer_dfn.GetGeomType())

    # this will map the target field index to the base index it came from
    # in case we don't need to copy all the fields
    target_to_base_field_id_map = {}
    if copy_fields:
        # Get the number of fields in original_layer
        original_field_count = layer_dfn.GetFieldCount()
        # For every field that's copying, create a duplicate field in the
        # new layer

        for fld_index in range(original_field_count):
            original_field = layer_dfn.GetFieldDefn(fld_index)
            field_name = original_field.GetName()
            if copy_fields is True or field_name in copy_fields:
                target_field = ogr.FieldDefn(
                    field_name, original_field.GetType())
                target_layer.CreateField(target_field)
                target_to_base_field_id_map[fld_index] = len(
                    target_to_base_field_id_map)

    # Get the SR of the original_layer to use in transforming
    base_sr = layer.GetSpatialRef()

    base_sr.SetAxisMappingStrategy(osr_axis_mapping_strategy)
    target_sr.SetAxisMappingStrategy(osr_axis_mapping_strategy)

    # Create a coordinate transformation
    coord_trans = osr.CreateCoordinateTransformation(base_sr, target_sr)

    # Copy all of the features in layer to the new shapefile
    target_layer.StartTransaction()
    error_count = 0
    last_time = time.time()
    LOGGER.info("starting reprojection")
    for feature_index, base_feature in enumerate(layer):
        last_time = _invoke_timed_callback(
            last_time, lambda: LOGGER.info(
                "reprojection approximately %.1f%% complete on %s",
                100.0 * float(feature_index+1) / (layer.GetFeatureCount()),
                os.path.basename(target_path)),
            _LOGGING_PERIOD)

        geom = base_feature.GetGeometryRef()
        if geom is None:
            # we encountered this error occasionally when transforming clipped
            # global polygons.  Not clear what is happening but perhaps a
            # feature was retained that otherwise wouldn't have been included
            # in the clip
            error_count += 1
            continue

        # Transform geometry into format desired for the new projection
        error_code = geom.Transform(coord_trans)
        if error_code != 0:  # error
            # this could be caused by an out of range transformation
            # whatever the case, don't put the transformed poly into the
            # output set
            error_count += 1
            continue

        # Copy original_datasource's feature and set as new shapes feature
        target_feature = ogr.Feature(target_layer.GetLayerDefn())
        target_feature.SetGeometry(geom)

        # For all the fields in the feature set the field values from the
        # source field
        for target_index, base_index in (
                target_to_base_field_id_map.items()):
            target_feature.SetField(
                target_index, base_feature.GetField(base_index))

        target_layer.CreateFeature(target_feature)
        target_feature = None
        base_feature = None
    target_layer.CommitTransaction()
    LOGGER.info(
        "reprojection 100.0%% complete on %s", os.path.basename(target_path))
    if error_count > 0:
        LOGGER.warning(
            '%d features out of %d were unable to be transformed and are'
            ' not in the output vector at %s', error_count,
            layer.GetFeatureCount(), target_path)
    layer = None
    base_vector = None


def reclassify_raster(
        base_raster_path_band, value_map, target_raster_path, target_datatype,
        target_nodata, values_required=True,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Reclassify pixel values in a raster.

    A function to reclassify values in raster to any output type. By default
    the values except for nodata must be in ``value_map``.

    Args:
        base_raster_path_band (tuple): a tuple including file path to a raster
            and the band index to operate over. ex: (path, band_index)
        value_map (dictionary): a dictionary of values of
            {source_value: dest_value, ...} where source_value's type is the
            same as the values in ``base_raster_path`` at band ``band_index``.
            Must contain at least one value.
        target_raster_path (string): target raster output path; overwritten if
            it exists
        target_datatype (gdal type): the numerical type for the target raster
        target_nodata (numerical type): the nodata value for the target raster
            Must be the same type as target_datatype
        values_required (bool): If True, raise a ValueError if there is a
            value in the raster that is not found in ``value_map``.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None

    Raises:
        ReclassificationMissingValuesError
            if ``values_required`` is ``True``
            and a pixel value from ``base_raster_path_band`` is not a key in
            ``value_map``.

    """
    if len(value_map) == 0:
        raise ValueError("value_map must contain at least one value")
    if not _is_raster_path_band_formatted(base_raster_path_band):
        raise ValueError(
            "Expected a (path, band_id) tuple, instead got '%s'" %
            base_raster_path_band)
    raster_info = get_raster_info(base_raster_path_band[0])
    nodata = raster_info['nodata'][base_raster_path_band[1]-1]
    value_map_copy = value_map.copy()
    # possible that nodata value is not defined, so test for None first
    # otherwise if nodata not predefined, remap it into the dictionary
    if nodata is not None and nodata not in value_map_copy:
        value_map_copy[nodata] = target_nodata
    keys = sorted(numpy.array(list(value_map_copy.keys())))
    values = numpy.array([value_map_copy[x] for x in keys])

    def _map_dataset_to_value_op(original_values):
        """Convert a block of original values to the lookup values."""
        if values_required:
            unique = numpy.unique(original_values)
            has_map = numpy.in1d(unique, keys)
            if not all(has_map):
                missing_values = unique[~has_map]
                raise ReclassificationMissingValuesError(
                    f'The following {missing_values.size} raster values'
                    f' {missing_values} from "{base_raster_path_band[0]}"'
                    ' do not have corresponding entries in the ``value_map``:'
                    f' {value_map}.', missing_values)
        index = numpy.digitize(original_values.ravel(), keys, right=True)
        return values[index].reshape(original_values.shape)

    raster_calculator(
        [base_raster_path_band], _map_dataset_to_value_op,
        target_raster_path, target_datatype, target_nodata,
        raster_driver_creation_tuple=raster_driver_creation_tuple)


def warp_raster(
        base_raster_path, target_pixel_size, target_raster_path,
        resample_method, target_bb=None, base_projection_wkt=None,
        target_projection_wkt=None, n_threads=None, vector_mask_options=None,
        gdal_warp_options=None, working_dir=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
        osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY):
    """Resize/resample raster to desired pixel size, bbox and projection.

    Args:
        base_raster_path (string): path to base raster.
        target_pixel_size (list/tuple): a two element sequence indicating
            the x and y pixel size in projected units.
        target_raster_path (string): the location of the resized and
            resampled raster.
        resample_method (string): the resampling technique, one of
            ``near|bilinear|cubic|cubicspline|lanczos|average|mode|max|min|med|q1|q3``
        target_bb (sequence): if None, target bounding box is the same as the
            source bounding box.  Otherwise it's a sequence of float
            describing target bounding box in target coordinate system as
            [minx, miny, maxx, maxy].
        base_projection_wkt (string): if not None, interpret the projection of
            ``base_raster_path`` as this.
        target_projection_wkt (string): if not None, desired target projection
            in Well Known Text format.
        n_threads (int): optional, if not None this sets the ``N_THREADS``
            option for ``gdal.Warp``.
        vector_mask_options (dict): optional, if not None, this is a
            dictionary of options to use an existing vector's geometry to
            mask out pixels in the target raster that do not overlap the
            vector's geometry. Keys to this dictionary are:

            * ``'mask_vector_path'``: (str) path to the mask vector file. This
              vector will be automatically projected to the target
              projection if its base coordinate system does not match
              the target.
            * ``'mask_layer_id'``: (int/str) the layer index or name to use for
              masking, if this key is not in the dictionary the default
              is to use the layer at index 0.
            * ``'mask_vector_where_filter'``: (str) an SQL WHERE string that can
              be used to filter the geometry in the mask. Ex:
              'id > 10' would use all features whose field value of
              'id' is > 10.

        gdal_warp_options (sequence): if present, the contents of this list
            are passed to the ``warpOptions`` parameter of ``gdal.Warp``. See
            the GDAL Warp documentation for valid options.
        working_dir (string): if defined uses this directory to make
            temporary working files for calculation. Otherwise uses system's
            temp directory.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.
        osr_axis_mapping_strategy (int): OSR axis mapping strategy for
            ``SpatialReference`` objects. Defaults to
            ``geoprocessing.DEFAULT_OSR_AXIS_MAPPING_STRATEGY``. This parameter
            should not be changed unless you know what you are doing.

    Returns:
        None

    Raises:
        ValueError
            if ``pixel_size`` is not a 2 element sequence of numbers.
        ValueError
            if ``vector_mask_options`` is not None but the
            ``mask_vector_path`` is undefined or doesn't point to a valid
            file.

    """
    _assert_is_valid_pixel_size(target_pixel_size)

    base_raster_info = get_raster_info(base_raster_path)
    if target_projection_wkt is None:
        target_projection_wkt = base_raster_info['projection_wkt']

    if target_bb is None:
        # ensure it's a sequence so we can modify it
        working_bb = list(get_raster_info(base_raster_path)['bounding_box'])
        # transform the working_bb if target_projection_wkt is not None
        if target_projection_wkt is not None:
            LOGGER.debug(
                "transforming bounding box from %s ", working_bb)
            working_bb = transform_bounding_box(
                base_raster_info['bounding_box'],
                base_raster_info['projection_wkt'], target_projection_wkt)
            LOGGER.debug(
                "transforming bounding to %s ", working_bb)
    else:
        # ensure it's a sequence so we can modify it
        working_bb = list(target_bb)

    # determine the raster size that bounds the input bounding box and then
    # adjust the bounding box to be that size
    target_x_size = int(abs(
        float(working_bb[2] - working_bb[0]) / target_pixel_size[0]))
    target_y_size = int(abs(
        float(working_bb[3] - working_bb[1]) / target_pixel_size[1]))

    # sometimes bounding boxes are numerically perfect, this checks for that
    x_residual = (
        abs(target_x_size * target_pixel_size[0]) -
        (working_bb[2] - working_bb[0]))
    if not numpy.isclose(x_residual, 0.0):
        target_x_size += 1
    y_residual = (
        abs(target_y_size * target_pixel_size[1]) -
        (working_bb[3] - working_bb[1]))
    if not numpy.isclose(y_residual, 0.0):
        target_y_size += 1

    if target_x_size == 0:
        LOGGER.warning(
            "bounding_box is so small that x dimension rounds to 0; "
            "clamping to 1.")
        target_x_size = 1
    if target_y_size == 0:
        LOGGER.warning(
            "bounding_box is so small that y dimension rounds to 0; "
            "clamping to 1.")
        target_y_size = 1

    # this ensures the bounding boxes perfectly fit a multiple of the target
    # pixel size
    working_bb[2] = working_bb[0] + abs(target_pixel_size[0] * target_x_size)
    working_bb[3] = working_bb[1] + abs(target_pixel_size[1] * target_y_size)

    reproject_callback = _make_logger_callback(
        "Warp %.1f%% complete %s")

    warp_options = []
    if n_threads:
        warp_options.append('NUM_THREADS=%d' % n_threads)
    if gdal_warp_options:
        warp_options.extend(gdal_warp_options)

    mask_vector_path = None
    mask_layer_id = 0
    mask_vector_where_filter = None
    if vector_mask_options:
        # translate pygeoprocessing terminology into GDAL warp options.
        if 'mask_vector_path' not in vector_mask_options:
            raise ValueError(
                'vector_mask_options passed, but no value for '
                '"mask_vector_path": %s', vector_mask_options)
        mask_vector_path = vector_mask_options['mask_vector_path']
        if not os.path.exists(mask_vector_path):
            raise ValueError(
                'The mask vector at %s was not found.', mask_vector_path)
        if 'mask_layer_id' in vector_mask_options:
            mask_layer_id = vector_mask_options['mask_layer_id']
        if 'mask_vector_where_filter' in vector_mask_options:
            mask_vector_where_filter = (
                vector_mask_options['mask_vector_where_filter'])

    if vector_mask_options:
        temp_working_dir = tempfile.mkdtemp(dir=working_dir)
        warped_raster_path = os.path.join(
            temp_working_dir, os.path.basename(target_raster_path).replace(
                '.tif', '_nonmasked.tif'))
    else:
        # if there is no vector path the result is the warp
        warped_raster_path = target_raster_path
    base_raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)

    raster_creation_options = list(raster_driver_creation_tuple[1])
    if (base_raster_info['numpy_type'] == numpy.int8 and
            'PIXELTYPE' not in ' '.join(raster_creation_options)):
        raster_creation_options.append('PIXELTYPE=SIGNEDBYTE')

    # WarpOptions.this is None when an invalid option is passed, and it's a
    # truthy SWIG proxy object when it's given a valid resample arg.
    if not gdal.WarpOptions(resampleAlg=resample_method)[0].this:
        raise ValueError(
            f'Invalid resample method: "{resample_method}"')

    gdal.Warp(
        warped_raster_path, base_raster,
        format=raster_driver_creation_tuple[0],
        outputBounds=working_bb,
        xRes=abs(target_pixel_size[0]),
        yRes=abs(target_pixel_size[1]),
        resampleAlg=resample_method,
        outputBoundsSRS=target_projection_wkt,
        srcSRS=base_projection_wkt,
        dstSRS=target_projection_wkt,
        multithread=True if warp_options else False,
        warpOptions=warp_options,
        creationOptions=raster_creation_options,
        callback=reproject_callback,
        callback_data=[target_raster_path])

    if vector_mask_options:
        # there was a cutline vector, so mask it out now, otherwise target
        # is already the result.
        mask_raster(
            (warped_raster_path, 1), vector_mask_options['mask_vector_path'],
            target_raster_path,
            mask_layer_id=mask_layer_id,
            where_clause=mask_vector_where_filter,
            target_mask_value=None, working_dir=temp_working_dir,
            all_touched=False,
            raster_driver_creation_tuple=raster_driver_creation_tuple)
        shutil.rmtree(temp_working_dir)


def rasterize(
        vector_path, target_raster_path, burn_values=None, option_list=None,
        layer_id=0, where_clause=None):
    """Project a vector onto an existing raster.

    Burn the layer at ``layer_id`` in ``vector_path`` to an existing
    raster at ``target_raster_path_band``.

    Args:
        vector_path (string): filepath to vector to rasterize.
        target_raster_path (string): path to an existing raster to burn vector
            into.  Can have multiple bands.
        burn_values (list/tuple): optional sequence of values to burn into
            each band of the raster.  If used, should have the same length as
            number of bands at the ``target_raster_path`` raster.  If ``None``
            then ``option_list`` must have a valid value.
        option_list (list/tuple): optional a sequence of burn options, if None
            then a valid value for ``burn_values`` must exist. Otherwise, each
            element is a string of the form:

            * ``"ATTRIBUTE=?"``: Identifies an attribute field on the features
              to be used for a burn in value. The value will be burned into all
              output bands. If specified, ``burn_values`` will not be used and
              can be None.
            * ``"CHUNKYSIZE=?"``: The height in lines of the chunk to operate
              on. The larger the chunk size the less times we need to make a
              pass through all the shapes. If it is not set or set to zero the
              default chunk size will be used. Default size will be estimated
              based on the GDAL cache buffer size using formula:
              ``cache_size_bytes/scanline_size_bytes``, so the chunk will not
              exceed the cache.
            * ``"ALL_TOUCHED=TRUE/FALSE"``: May be set to ``TRUE`` to set all
              pixels touched by the line or polygons, not just those whose
              center is within the polygon or that are selected by Brezenhams
              line algorithm. Defaults to ``FALSE``.
            * ``"BURN_VALUE_FROM"``: May be set to "Z" to use the Z values of
              the geometries. The value from burn_values or the
              attribute field value is added to this before burning. In
              default case dfBurnValue is burned as it is (richpsharp:
              note, I'm not sure what this means, but copied from formal
              docs). This is implemented properly only for points and
              lines for now. Polygons will be burned using the Z value
              from the first point.
            * ``"MERGE_ALG=REPLACE/ADD"``: REPLACE results in overwriting of
              value, while ADD adds the new value to the existing
              raster, suitable for heatmaps for instance.

            Example::

                ["ATTRIBUTE=npv", "ALL_TOUCHED=TRUE"]

        layer_id (str/int): name or index of the layer to rasterize. Defaults
            to 0.
        where_clause (str): If not None, is an SQL query-like string to filter
            which features are used to rasterize, (e.x. where="value=1").

    Returns:
        None
    """
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    raster = gdal.OpenEx(target_raster_path, gdal.GA_Update | gdal.OF_RASTER)
    gdal.PopErrorHandler()
    if raster is None:
        raise ValueError(
            "%s doesn't exist, but needed to rasterize." % target_raster_path)
    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)

    rasterize_callback = _make_logger_callback(
        "RasterizeLayer %.1f%% complete %s")

    if burn_values is None:
        burn_values = []
    if option_list is None:
        option_list = []

    if not burn_values and not option_list:
        raise ValueError(
            "Neither `burn_values` nor `option_list` is set. At least "
            "one must have a value.")

    if not isinstance(burn_values, (list, tuple)):
        raise ValueError(
            "`burn_values` is not a list/tuple, the value passed is '%s'",
            repr(burn_values))

    if not isinstance(option_list, (list, tuple)):
        raise ValueError(
            "`option_list` is not a list/tuple, the value passed is '%s'",
            repr(option_list))

    layer = vector.GetLayer(layer_id)
    if where_clause:
        layer.SetAttributeFilter(where_clause)
    result = gdal.RasterizeLayer(
        raster, [1], layer, burn_values=burn_values,
        options=option_list, callback=rasterize_callback)
    raster.FlushCache()
    gdal.Dataset.__swig_destroy__(raster)

    if result != 0:
        raise RuntimeError('Rasterize returned a nonzero exit code.')


def calculate_disjoint_polygon_set(
        vector_path, layer_id=0, bounding_box=None):
    """Create a sequence of sets of polygons that don't overlap.

    Determining the minimal number of those sets is an np-complete problem so
    this is an approximation that builds up sets of maximal subsets.

    Args:
        vector_path (string): a path to an OGR vector.
        layer_id (str/int): name or index of underlying layer in
            ``vector_path`` to calculate disjoint set. Defaults to 0.
        bounding_box (sequence): sequence of floats representing a bounding
            box to filter any polygons by. If a feature in ``vector_path``
            does not intersect this bounding box it will not be considered
            in the disjoint calculation. Coordinates are in the order
            [minx, miny, maxx, maxy].

    Returns:
        subset_list (sequence): sequence of sets of FIDs from vector_path

    """
    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    vector_layer = vector.GetLayer(layer_id)
    feature_count = vector_layer.GetFeatureCount()

    if feature_count == 0:
        raise RuntimeError('Vector must have geometries but does not: %s'
                           % vector_path)

    last_time = time.time()
    LOGGER.info("build shapely polygon list")

    if bounding_box is None:
        bounding_box = get_vector_info(vector_path)['bounding_box']
    bounding_box = shapely.prepared.prep(shapely.geometry.box(*bounding_box))

    # As much as I want this to be in a comprehension, a comprehension version
    # of this loop causes python 3.6 to crash on linux in GDAL 2.1.2 (which is
    # what's in the debian:stretch repos.)
    shapely_polygon_lookup = {}
    for poly_feat in vector_layer:
        poly_geom_ref = poly_feat.GetGeometryRef()
        if poly_geom_ref is None:
            LOGGER.warn(
                f'no geometry in {vector_path} FID: {poly_feat.GetFID()}, '
                'skipping...')
            continue
        shapely_polygon_lookup[poly_feat.GetFID()] = (
            shapely.wkb.loads(poly_geom_ref.ExportToWkb()))
        poly_geom_ref = None

    LOGGER.info("build shapely rtree index")
    r_tree_index_stream = [
        (poly_fid, poly.bounds, None)
        for poly_fid, poly in shapely_polygon_lookup.items()
        if bounding_box.intersects(poly)]
    if r_tree_index_stream:
        poly_rtree_index = rtree.index.Index(r_tree_index_stream)
    else:
        LOGGER.warning("no polygons intersected the bounding box")
        return []

    vector_layer = None
    vector = None
    LOGGER.info(
        'poly feature lookup 100.0%% complete on %s',
        os.path.basename(vector_path))

    LOGGER.info('build poly intersection lookup')
    poly_intersect_lookup = collections.defaultdict(set)
    for poly_index, (poly_fid, poly_geom) in enumerate(
            shapely_polygon_lookup.items()):
        last_time = _invoke_timed_callback(
            last_time, lambda: LOGGER.info(
                "poly intersection lookup approximately %.1f%% complete "
                "on %s", 100.0 * float(poly_index+1) / len(
                    shapely_polygon_lookup), os.path.basename(vector_path)),
            _LOGGING_PERIOD)
        possible_intersection_set = list(poly_rtree_index.intersection(
            poly_geom.bounds))
        # no reason to prep the polygon to intersect itself
        if len(possible_intersection_set) > 1:
            polygon = shapely.prepared.prep(poly_geom)
        else:
            polygon = poly_geom
        for intersect_poly_fid in possible_intersection_set:
            if intersect_poly_fid == poly_fid or polygon.intersects(
                    shapely_polygon_lookup[intersect_poly_fid]):
                poly_intersect_lookup[poly_fid].add(intersect_poly_fid)
        polygon = None
    LOGGER.info(
        'poly intersection feature lookup 100.0%% complete on %s',
        os.path.basename(vector_path))

    # Build maximal subsets
    subset_list = []
    while len(poly_intersect_lookup) > 0:
        # sort polygons by increasing number of intersections
        intersections_list = [
            (len(poly_intersect_set), poly_fid, poly_intersect_set)
            for poly_fid, poly_intersect_set in
            poly_intersect_lookup.items()]
        intersections_list.sort()

        # build maximal subset
        maximal_set = set()
        for _, poly_fid, poly_intersect_set in intersections_list:
            last_time = _invoke_timed_callback(
                last_time, lambda: LOGGER.info(
                    "maximal subset build approximately %.1f%% complete "
                    "on %s", 100.0 * float(
                        feature_count - len(poly_intersect_lookup)) /
                    feature_count, os.path.basename(vector_path)),
                _LOGGING_PERIOD)
            if not poly_intersect_set.intersection(maximal_set):
                # no intersection, add poly_fid to the maximal set and remove
                # the polygon from the lookup
                maximal_set.add(poly_fid)
                del poly_intersect_lookup[poly_fid]
        # remove all the polygons from intersections once they're computed
        for poly_fid, poly_intersect_set in poly_intersect_lookup.items():
            poly_intersect_lookup[poly_fid] = (
                poly_intersect_set.difference(maximal_set))
        subset_list.append(maximal_set)
    LOGGER.info(
        'maximal subset build 100.0%% complete on %s',
        os.path.basename(vector_path))
    return subset_list


def distance_transform_edt(
        base_region_raster_path_band, target_distance_raster_path,
        sampling_distance=(1., 1.), working_dir=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Calculate the euclidean distance transform on base raster.

    Calculates the euclidean distance transform on the base raster in units of
    pixels multiplied by an optional scalar constant. The implementation is
    based off the algorithm described in:  Meijster, Arnold, Jos BTM Roerdink,
    and Wim H. Hesselink. "A general algorithm for computing distance
    transforms in linear time." Mathematical Morphology and its applications
    to image and signal processing. Springer, Boston, MA, 2002. 331-340.

    The base mask raster represents the area to distance transform from as
    any pixel that is not 0 or nodata. It is computationally convenient to
    calculate the distance transform on the entire raster irrespective of
    nodata placement and thus produces a raster that will have distance
    transform values even in pixels that are nodata in the base.

    Args:
        base_region_raster_path_band (tuple): a tuple including file path to a
            raster and the band index to define the base region pixels. Any
            pixel  that is not 0 and nodata are considered to be part of the
            region.
        target_distance_raster_path (string): path to the target raster that
            is the exact euclidean distance transform from any pixel in the
            base raster that is not nodata and not 0. The units are in
            ``(pixel distance * sampling_distance)``.
        sampling_distance (tuple/list): an optional parameter used to scale
            the pixel distances when calculating the distance transform.
            Defaults to (1.0, 1.0). First element indicates the distance
            traveled in the x direction when changing a column index, and the
            second element in y when changing a row index. Both values must
            be > 0.
        working_dir (string): If not None, indicates where temporary files
            should be created during this run.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None
    """
    working_raster_paths = {}
    for raster_prefix in ['region_mask_raster', 'g_raster']:
        with tempfile.NamedTemporaryFile(
                prefix=raster_prefix, suffix='.tif', delete=False,
                dir=working_dir) as tmp_file:
            working_raster_paths[raster_prefix] = tmp_file.name
    nodata = (get_raster_info(base_region_raster_path_band[0])['nodata'])[
        base_region_raster_path_band[1]-1]
    nodata_out = 255

    def mask_op(base_array):
        """Convert base_array to 1 if not 0 and nodata, 0 otherwise."""
        if nodata is not None:
            return ~numpy.isclose(base_array, nodata) & (base_array != 0)
        else:
            return base_array != 0

    if not isinstance(sampling_distance, (tuple, list)):
        raise ValueError(
            "`sampling_distance` should be a tuple/list, instead it's %s" % (
                type(sampling_distance)))

    sample_d_x, sample_d_y = sampling_distance
    if sample_d_x <= 0. or sample_d_y <= 0.:
        raise ValueError(
            "Sample distances must be > 0.0, instead got %s",
            sampling_distance)

    raster_calculator(
        [base_region_raster_path_band], mask_op,
        working_raster_paths['region_mask_raster'], gdal.GDT_Byte, nodata_out,
        calc_raster_stats=False,
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    geoprocessing_core._distance_transform_edt(
        working_raster_paths['region_mask_raster'],
        working_raster_paths['g_raster'], sampling_distance[0],
        sampling_distance[1], target_distance_raster_path,
        raster_driver_creation_tuple)

    for path in working_raster_paths.values():
        try:
            os.remove(path)
        except OSError:
            LOGGER.warning("couldn't remove file %s", path)


def _next_regular(base):
    """Find the next regular number greater than or equal to base.

    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    This source was taken directly from scipy.signaltools and saves us from
    having to access a protected member in a library that could change in
    future releases:

    https://github.com/scipy/scipy/blob/v0.17.1/scipy/signal/signaltools.py#L211

    Args:
        base (int): a positive integer to start to find the next Hamming
            number.

    Returns:
        The next regular number greater than or equal to ``base``.

    """
    if base <= 6:
        return base

    # Quickly check if it's already a power of 2
    if not (base & (base-1)):
        return base

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < base:
        p35 = p5
        while p35 < base:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(base / p35))
            quotient = -(-base // p35)

            # Quickly find next power of 2 >= quotient
            p2 = 2**((quotient - 1).bit_length())

            N = p2 * p35
            if N == base:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == base:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == base:
            return p5
    if p5 < match:
        match = p5
    return match


def convolve_2d(
        signal_path_band, kernel_path_band, target_path,
        ignore_nodata_and_edges=False, mask_nodata=True,
        normalize_kernel=False, target_datatype=gdal.GDT_Float64,
        target_nodata=None, working_dir=None, set_tol_to_zero=1e-8,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Convolve 2D kernel over 2D signal.

    Convolves the raster in ``kernel_path_band`` over ``signal_path_band``.
    Nodata values are treated as 0.0 during the convolution and masked to
    nodata for the output result where ``signal_path`` has nodata.

    Args:
        signal_path_band (tuple): a 2 tuple of the form
            (filepath to signal raster, band index).
        kernel_path_band (tuple): a 2 tuple of the form
            (filepath to kernel raster, band index), all pixel values should
            be valid -- output is not well defined if the kernel raster has
            nodata values.
        target_path (string): filepath to target raster that's the convolution
            of signal with kernel.  Output will be a single band raster of
            same size and projection as ``signal_path_band``. Any nodata pixels
            that align with ``signal_path_band`` will be set to nodata.
        ignore_nodata_and_edges (boolean): If true, any pixels that are equal
            to ``signal_path_band``'s nodata value or signal pixels where the
            kernel extends beyond the edge of the raster are not included when
            averaging the convolution filter. This has the effect of
            "spreading" the result as though nodata and edges beyond the bounds
            of the raster are 0s. If set to false this tends to "pull" the
            signal away from nodata holes or raster edges. Set this value
            to ``True`` to avoid distortions signal values near edges for
            large integrating kernels.
        normalize_kernel (boolean): If true, the result is divided by the
            sum of the kernel.
        mask_nodata (boolean): If true, ``target_path`` raster's output is
            nodata where ``signal_path_band``'s pixels were nodata. Note that
            setting ``ignore_nodata_and_edges`` to ``True`` while setting
            ``mask_nodata`` to False would be a nonsensical result and would
            result in exposing the numerical noise where the nodata values were
            ignored. An exception is thrown in this case.
        target_datatype (GDAL type): a GDAL raster type to set the output
            raster type to, as well as the type to calculate the convolution
            in.  Defaults to GDT_Float64.  Note signed byte is not
            supported.
        target_nodata (int/float): nodata value to set on output raster.
            If ``target_datatype`` is not gdal.GDT_Float64, this value must
            be set.  Otherwise defaults to the minimum value of a float32.
        raster_creation_options (sequence): an argument list that will be
            passed to the GTiff driver for creating ``target_path``.  Useful
            for blocksizes, compression, and more.
        working_dir (string): If not None, indicates where temporary files
            should be created during this run.
        set_tol_to_zero (float): any value within +- this from 0.0 will get
            set to 0.0. This is to handle numerical roundoff errors that
            sometimes result in "numerical zero", such as -1.782e-18 that
            cannot be tolerated by users of this function. If `None` no
            adjustment will be done to output values.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        ``None``

    Raises:
        ValueError
            if ``ignore_nodata_and_edges`` is ``True`` and ``mask_nodata``
            is ``False``

    """
    if target_datatype is not gdal.GDT_Float64 and target_nodata is None:
        raise ValueError(
            "`target_datatype` is set, but `target_nodata` is None. "
            "`target_nodata` must be set if `target_datatype` is not "
            "`gdal.GDT_Float64`.  `target_nodata` is set to None.")

    if ignore_nodata_and_edges and not mask_nodata:
        raise ValueError(
            'ignore_nodata_and_edges is True while mask_nodata is False -- '
            'this would yield a nonsensical result.')

    bad_raster_path_list = []
    for raster_id, raster_path_band in [
            ('signal', signal_path_band), ('kernel', kernel_path_band)]:
        if (not _is_raster_path_band_formatted(raster_path_band)):
            bad_raster_path_list.append((raster_id, raster_path_band))
    if bad_raster_path_list:
        raise ValueError(
            "Expected raster path band sequences for the following arguments "
            f"but instead got: {bad_raster_path_list}")

    # The nodata value is reset to a different value at the end of this
    # function. Here 0 is chosen as a default value since data are
    # incrementally added to the raster
    new_raster_from_base(
        signal_path_band[0], target_path, target_datatype, [0],
        raster_driver_creation_tuple=raster_driver_creation_tuple)

    signal_raster_info = get_raster_info(signal_path_band[0])
    kernel_raster_info = get_raster_info(kernel_path_band[0])

    n_cols_signal, n_rows_signal = signal_raster_info['raster_size']
    n_cols_kernel, n_rows_kernel = kernel_raster_info['raster_size']
    s_path_band = signal_path_band
    k_path_band = kernel_path_band
    s_nodata = signal_raster_info['nodata'][0]

    # we need the original signal raster info because we want the output to
    # be clipped and NODATA masked to it
    signal_raster = gdal.OpenEx(signal_path_band[0], gdal.OF_RASTER)
    signal_band = signal_raster.GetRasterBand(signal_path_band[1])
    # getting the offset list before it's opened for updating
    target_offset_list = list(iterblocks((target_path, 1), offset_only=True))
    target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER | gdal.GA_Update)
    target_band = target_raster.GetRasterBand(1)

    # if we're ignoring nodata, we need to make a parallel convolved signal
    # of the nodata mask
    if ignore_nodata_and_edges:
        raster_file, mask_raster_path = tempfile.mkstemp(
            suffix='.tif', prefix='convolved_mask',
            dir=os.path.dirname(target_path))
        os.close(raster_file)
        new_raster_from_base(
            signal_path_band[0], mask_raster_path, gdal.GDT_Float64,
            [0.0], raster_driver_creation_tuple=raster_driver_creation_tuple)
        mask_raster = gdal.OpenEx(
            mask_raster_path, gdal.GA_Update | gdal.OF_RASTER)
        mask_band = mask_raster.GetRasterBand(1)

    LOGGER.info('starting convolve')
    last_time = time.time()

    # calculate the kernel sum for normalization
    kernel_nodata = kernel_raster_info['nodata'][0]
    kernel_sum = 0.0
    for _, kernel_block in iterblocks(kernel_path_band):
        if kernel_nodata is not None and ignore_nodata_and_edges:
            kernel_block[numpy.isclose(kernel_block, kernel_nodata)] = 0.0
        kernel_sum += numpy.sum(kernel_block)

    # limit the size of the work queue since a large kernel / signal with small
    # block size can have a large memory impact when queuing offset lists.
    work_queue = queue.Queue(10)
    signal_offset_list = list(iterblocks(s_path_band, offset_only=True))
    kernel_offset_list = list(iterblocks(k_path_band, offset_only=True))
    n_blocks = len(signal_offset_list) * len(kernel_offset_list)

    LOGGER.debug('start fill work queue thread')

    def _fill_work_queue():
        """Asynchronously fill the work queue."""
        LOGGER.debug('fill work queue')
        for signal_offset in signal_offset_list:
            for kernel_offset in kernel_offset_list:
                work_queue.put((signal_offset, kernel_offset))
        work_queue.put(None)
        LOGGER.debug('work queue full')

    fill_work_queue_worker = threading.Thread(
        target=_fill_work_queue)
    fill_work_queue_worker.daemon = True
    fill_work_queue_worker.start()

    # limit the size of the write queue so we don't accidentally load a whole
    # array into memory
    LOGGER.debug('start worker thread')
    write_queue = queue.Queue(10)
    worker = threading.Thread(
        target=_convolve_2d_worker,
        args=(
            signal_path_band, kernel_path_band,
            ignore_nodata_and_edges, normalize_kernel,
            set_tol_to_zero, work_queue, write_queue))
    worker.daemon = True
    worker.start()

    n_blocks_processed = 0
    LOGGER.info(f'{n_blocks} sent to workers, wait for worker results')
    while True:
        write_payload = write_queue.get()
        if write_payload:
            (index_dict, result, mask_result,
             left_index_raster, right_index_raster,
             top_index_raster, bottom_index_raster,
             left_index_result, right_index_result,
             top_index_result, bottom_index_result) = write_payload
        else:
            worker.join(_MAX_TIMEOUT)
            break

        output_array = numpy.empty(
            (index_dict['win_ysize'], index_dict['win_xsize']),
            dtype=numpy.float32)

        # the inital data value in target_band is 0 because that is the
        # temporary nodata selected so that manual resetting of initial
        # data values weren't necessary. at the end of this function the
        # target nodata value is set to `target_nodata`.
        current_output = target_band.ReadAsArray(**index_dict)

        # read the signal block so we know where the nodata are
        potential_nodata_signal_array = signal_band.ReadAsArray(**index_dict)

        valid_mask = numpy.ones(
            potential_nodata_signal_array.shape, dtype=bool)

        # guard against a None nodata value
        if s_nodata is not None and mask_nodata:
            valid_mask[:] = (
                ~numpy.isclose(potential_nodata_signal_array, s_nodata))
        output_array[:] = target_nodata
        output_array[valid_mask] = (
            (result[top_index_result:bottom_index_result,
                    left_index_result:right_index_result])[valid_mask] +
            current_output[valid_mask])
        target_band.WriteArray(
            output_array, xoff=index_dict['xoff'],
            yoff=index_dict['yoff'])

        if ignore_nodata_and_edges:
            # we'll need to save off the mask convolution so we can divide
            # it in total later
            current_mask = mask_band.ReadAsArray(**index_dict)

            output_array[valid_mask] = (
                (mask_result[
                    top_index_result:bottom_index_result,
                    left_index_result:right_index_result])[valid_mask] +
                current_mask[valid_mask])
            mask_band.WriteArray(
                output_array, xoff=index_dict['xoff'],
                yoff=index_dict['yoff'])

        n_blocks_processed += 1
        last_time = _invoke_timed_callback(
            last_time, lambda: LOGGER.info(
                "convolution worker approximately %.1f%% complete on %s",
                100.0 * float(n_blocks_processed) / (n_blocks),
                os.path.basename(target_path)),
            _LOGGING_PERIOD)

    LOGGER.info(
        f"convolution worker 100.0% complete on "
        f"{os.path.basename(target_path)}")

    target_band.FlushCache()
    if ignore_nodata_and_edges:
        signal_nodata = get_raster_info(signal_path_band[0])['nodata'][
            signal_path_band[1]-1]
        LOGGER.info(
            "need to normalize result so nodata values are not included")
        mask_pixels_processed = 0
        mask_band.FlushCache()
        for target_offset_data in target_offset_list:
            target_block = target_band.ReadAsArray(
                **target_offset_data).astype(numpy.float64)
            signal_block = signal_band.ReadAsArray(**target_offset_data)
            mask_block = mask_band.ReadAsArray(**target_offset_data)
            if mask_nodata and signal_nodata is not None:
                valid_mask = ~numpy.isclose(signal_block, signal_nodata)
            else:
                valid_mask = numpy.ones(target_block.shape, dtype=numpy.bool)
            valid_mask &= (mask_block > 0)
            # divide the target_band by the mask_band
            target_block[valid_mask] /= mask_block[valid_mask].astype(
                numpy.float64)

            # scale by kernel sum if necessary since mask division will
            # automatically normalize kernel
            if not normalize_kernel:
                target_block[valid_mask] *= kernel_sum

            target_band.WriteArray(
                target_block, xoff=target_offset_data['xoff'],
                yoff=target_offset_data['yoff'])

            mask_pixels_processed += target_block.size
            last_time = _invoke_timed_callback(
                last_time, lambda: LOGGER.info(
                    "convolution nodata normalizer approximately %.1f%% "
                    "complete on %s", 100.0 * float(mask_pixels_processed) / (
                        n_cols_signal * n_rows_signal),
                    os.path.basename(target_path)),
                _LOGGING_PERIOD)

        mask_raster = None
        mask_band = None
        os.remove(mask_raster_path)
        LOGGER.info(
            f"convolution nodata normalize 100.0% complete on "
            f"{os.path.basename(target_path)}")

    # set the nodata value from 0 to a reasonable value for the result
    if target_nodata is None:
        target_band.DeleteNoDataValue()
    else:
        target_band.SetNoDataValue(target_nodata)

    target_band = None
    target_raster = None


def iterblocks(
        raster_path_band, largest_block=_LARGEST_ITERBLOCK,
        offset_only=False):
    """Iterate across all the memory blocks in the input raster.

    Result is a generator of block location information and numpy arrays.

    This is especially useful when a single value needs to be derived from the
    pixel values in a raster, such as the sum total of all pixel values, or
    a sequence of unique raster values.  In such cases, ``raster_local_op``
    is overkill, since it writes out a raster.

    As a generator, this can be combined multiple times with itertools.izip()
    to iterate 'simultaneously' over multiple rasters, though the user should
    be careful to do so only with prealigned rasters.

    Args:
        raster_path_band (tuple): a path/band index tuple to indicate
            which raster band iterblocks should iterate over.
        largest_block (int): Attempts to iterate over raster blocks with
            this many elements.  Useful in cases where the blocksize is
            relatively small, memory is available, and the function call
            overhead dominates the iteration.  Defaults to 2**20.  A value of
            anything less than the original blocksize of the raster will
            result in blocksizes equal to the original size.
        offset_only (boolean): defaults to False, if True ``iterblocks`` only
            returns offset dictionary and doesn't read any binary data from
            the raster.  This can be useful when iterating over writing to
            an output.

    Yields:
        If ``offset_only`` is false, on each iteration, a tuple containing a
        dict of block data and a 2-dimensional numpy array are
        yielded. The dict of block data has these attributes:

        * ``data['xoff']`` - The X offset of the upper-left-hand corner of the
          block.
        * ``data['yoff']`` - The Y offset of the upper-left-hand corner of the
          block.
        * ``data['win_xsize']`` - The width of the block.
        * ``data['win_ysize']`` - The height of the block.

        If ``offset_only`` is True, the function returns only the block offset
        data and does not attempt to read binary data from the raster.

    """
    if not _is_raster_path_band_formatted(raster_path_band):
        raise ValueError(
            "`raster_path_band` not formatted as expected.  Expects "
            "(path, band_index), received %s" % repr(raster_path_band))
    raster = gdal.OpenEx(raster_path_band[0], gdal.OF_RASTER)
    if raster is None:
        raise ValueError(
            "Raster at %s could not be opened." % raster_path_band[0])
    band = raster.GetRasterBand(raster_path_band[1])
    block = band.GetBlockSize()
    cols_per_block = block[0]
    rows_per_block = block[1]

    n_cols = raster.RasterXSize
    n_rows = raster.RasterYSize

    block_area = cols_per_block * rows_per_block
    # try to make block wider
    if int(largest_block / block_area) > 0:
        width_factor = int(largest_block / block_area)
        cols_per_block *= width_factor
        if cols_per_block > n_cols:
            cols_per_block = n_cols
        block_area = cols_per_block * rows_per_block
    # try to make block taller
    if int(largest_block / block_area) > 0:
        height_factor = int(largest_block / block_area)
        rows_per_block *= height_factor
        if rows_per_block > n_rows:
            rows_per_block = n_rows

    n_col_blocks = int(math.ceil(n_cols / float(cols_per_block)))
    n_row_blocks = int(math.ceil(n_rows / float(rows_per_block)))

    for row_block_index in range(n_row_blocks):
        row_offset = row_block_index * rows_per_block
        row_block_width = n_rows - row_offset
        if row_block_width > rows_per_block:
            row_block_width = rows_per_block
        for col_block_index in range(n_col_blocks):
            col_offset = col_block_index * cols_per_block
            col_block_width = n_cols - col_offset
            if col_block_width > cols_per_block:
                col_block_width = cols_per_block

            offset_dict = {
                'xoff': col_offset,
                'yoff': row_offset,
                'win_xsize': col_block_width,
                'win_ysize': row_block_width,
            }
            if offset_only:
                yield offset_dict
            else:
                yield (offset_dict, band.ReadAsArray(**offset_dict))

    band = None
    gdal.Dataset.__swig_destroy__(raster)
    raster = None


def transform_bounding_box(
        bounding_box, base_projection_wkt, target_projection_wkt,
        edge_samples=11,
        osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY):
    """Transform input bounding box to output projection.

    This transform accounts for the fact that the reprojected square bounding
    box might be warped in the new coordinate system.  To account for this,
    the function samples points along the original bounding box edges and
    attempts to make the largest bounding box around any transformed point
    on the edge whether corners or warped edges.

    Args:
        bounding_box (sequence): a sequence of 4 coordinates in ``base_epsg``
            coordinate system describing the bound in the order
            [xmin, ymin, xmax, ymax].
        base_projection_wkt (string): the spatial reference of the input
            coordinate system in Well Known Text.
        target_projection_wkt (string): the spatial reference of the desired
            output coordinate system in Well Known Text.
        edge_samples (int): the number of interpolated points along each
            bounding box edge to sample along. A value of 2 will sample just
            the corners while a value of 3 will also sample the corners and
            the midpoint.
        osr_axis_mapping_strategy (int): OSR axis mapping strategy for
            ``SpatialReference`` objects. Defaults to
            ``geoprocessing.DEFAULT_OSR_AXIS_MAPPING_STRATEGY``. This parameter
            should not be changed unless you know what you are doing.

    Returns:
        A list of the form [xmin, ymin, xmax, ymax] that describes the largest
        fitting bounding box around the original warped bounding box in
        ``new_epsg`` coordinate system.

    """
    base_ref = osr.SpatialReference()
    base_ref.ImportFromWkt(base_projection_wkt)

    target_ref = osr.SpatialReference()
    target_ref.ImportFromWkt(target_projection_wkt)

    base_ref.SetAxisMappingStrategy(osr_axis_mapping_strategy)
    target_ref.SetAxisMappingStrategy(osr_axis_mapping_strategy)

    # Create a coordinate transformation
    transformer = osr.CreateCoordinateTransformation(base_ref, target_ref)

    def _transform_point(point):
        """Transform an (x,y) point tuple from base_ref to target_ref."""
        trans_x, trans_y, _ = (transformer.TransformPoint(*point))
        return (trans_x, trans_y)

    # The following list comprehension iterates over each edge of the bounding
    # box, divides each edge into ``edge_samples`` number of points, then
    # reduces that list to an appropriate ``bounding_fn`` given the edge.
    # For example the left edge needs to be the minimum x coordinate so
    # we generate ``edge_samples` number of points between the upper left and
    # lower left point, transform them all to the new coordinate system
    # then get the minimum x coordinate "min(p[0] ...)" of the batch.
    # points are numbered from 0 starting upper right as follows:
    # 0--3
    # |  |
    # 1--2
    p_0 = numpy.array((bounding_box[0], bounding_box[3]))
    p_1 = numpy.array((bounding_box[0], bounding_box[1]))
    p_2 = numpy.array((bounding_box[2], bounding_box[1]))
    p_3 = numpy.array((bounding_box[2], bounding_box[3]))
    raw_bounding_box = [
        bounding_fn(
            [_transform_point(
                p_a * v + p_b * (1 - v)) for v in numpy.linspace(
                    0, 1, edge_samples)])
        for p_a, p_b, bounding_fn in [
            (p_0, p_1, lambda p_list: min([p[0] for p in p_list])),
            (p_1, p_2, lambda p_list: min([p[1] for p in p_list])),
            (p_2, p_3, lambda p_list: max([p[0] for p in p_list])),
            (p_3, p_0, lambda p_list: max([p[1] for p in p_list]))]]

    # sometimes a transform will be so tight that a sampling around it may
    # flip the coordinate system. This flips it back. I found this when
    # transforming the bounding box of Gibraltar in a utm coordinate system
    # to lat/lng.
    minx, maxx = sorted([raw_bounding_box[0], raw_bounding_box[2]])
    miny, maxy = sorted([raw_bounding_box[1], raw_bounding_box[3]])
    transformed_bounding_box = [minx, miny, maxx, maxy]
    return transformed_bounding_box


def merge_rasters(
        raster_path_list, target_path, bounding_box=None, target_nodata=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Merge the given rasters into a single raster.

    This operation creates a mosaic of the rasters in ``raster_path_list``.
    The result is a raster of the size of the union of the bounding box of
    the inputs where the contents of each raster's bands are copied into the
    correct georeferenced target's bands.

    Note the input rasters must be in the same projection, same pixel size,
    same number of bands, and same datatype. If any of these are not true,
    the operation raises a ValueError with an appropriate error message.

    Args:
        raster_path_list (sequence): list of file paths to rasters
        target_path (string): path to the geotiff file that will be created
            by this operation.
        bounding_box (sequence): if not None, clip target path to be within
            these bounds. Format is [minx,miny,maxx,maxy]
        target_nodata (float): if not None, set the target raster's nodata
            value to this. Otherwise use the shared nodata value in the
            ``raster_path_list``. It is an error if different rasters in
            ``raster_path_list`` have different nodata values and
            ``target_nodata`` is None.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None
    """
    raster_info_list = [
        get_raster_info(path) for path in raster_path_list]
    pixel_size_set = set([
        x['pixel_size'] for x in raster_info_list])
    if len(pixel_size_set) != 1:
        raise ValueError(
            "Pixel sizes of all rasters are not the same. "
            "Here's the sizes: %s" % str([
                (path, x['pixel_size']) for path, x in zip(
                    raster_path_list, raster_info_list)]))
    n_bands_set = set([x['n_bands'] for x in raster_info_list])
    if len(n_bands_set) != 1:
        raise ValueError(
            "Number of bands per raster are not the same. "
            "Here's the band counts: %s" % str([
                (path, x['n_bands']) for path, x in zip(
                    raster_path_list, raster_info_list)]))

    datatype_set = set([x['datatype'] for x in raster_info_list])
    if len(datatype_set) != 1:
        raise ValueError(
            "Rasters have different datatypes. "
            "Here's the datatypes: %s" % str([
                (path, x['datatype']) for path, x in zip(
                    raster_path_list, raster_info_list)]))

    if target_nodata is None:
        nodata_set = set([x['nodata'][0] for x in raster_info_list])
        if len(nodata_set) != 1:
            raise ValueError(
                "Nodata per raster are not the same. "
                "Path and nodata values: %s" % str([
                    (path, x['nodata']) for path, x in zip(
                        raster_path_list, raster_info_list)]))

    projection_set = set([x['projection_wkt'] for x in raster_info_list])
    if len(projection_set) != 1:
        raise ValueError(
            "Projections are not identical. Here's the projections: %s" % str(
                [(path, x['projection_wkt']) for path, x in zip(
                    raster_path_list, raster_info_list)]))

    pixeltype_set = set()
    for path in raster_path_list:
        raster = gdal.OpenEx(path, gdal.OF_RASTER)
        band = raster.GetRasterBand(1)
        metadata = band.GetMetadata('IMAGE_STRUCTURE')
        band = None
        if 'PIXELTYPE' in metadata:
            pixeltype_set.add('PIXELTYPE=' + metadata['PIXELTYPE'])
        else:
            pixeltype_set.add(None)
    if len(pixeltype_set) != 1:
        raise ValueError(
            "PIXELTYPE different between rasters."
            "Here is the set of types (should only have 1): %s" % str(
                pixeltype_set))

    bounding_box_list = [x['bounding_box'] for x in raster_info_list]
    target_bounding_box = merge_bounding_box_list(bounding_box_list, 'union')
    if bounding_box is not None:
        LOGGER.debug("target bounding_box %s", target_bounding_box)
        target_bounding_box = merge_bounding_box_list(
            [target_bounding_box, bounding_box], 'intersection')
        LOGGER.debug("bounding_box %s", bounding_box)
        LOGGER.debug("merged target bounding_box %s", target_bounding_box)

    driver = gdal.GetDriverByName(raster_driver_creation_tuple[0])
    target_pixel_size = pixel_size_set.pop()
    n_cols = int(math.ceil(abs(
        (target_bounding_box[2]-target_bounding_box[0]) /
        target_pixel_size[0])))
    n_rows = int(math.ceil(abs(
        (target_bounding_box[3]-target_bounding_box[1]) /
        target_pixel_size[1])))

    target_geotransform = [
        target_bounding_box[0], target_pixel_size[0], 0,
        target_bounding_box[1], 0, target_pixel_size[1]]

    # I haven't been able to get the geotransform to ever have a negative x
    # or positive y, but there's nothing in the spec that would restrict it
    # so we still test here.
    if target_pixel_size[0] < 0:
        target_geotransform[0] = target_bounding_box[2]
    if target_pixel_size[1] < 0:
        target_geotransform[3] = target_bounding_box[3]

    # there's only one element in the sets so okay to pop right in the call,
    # we won't need it after anyway
    n_bands = n_bands_set.pop()
    target_raster = driver.Create(
        target_path, n_cols, n_rows, n_bands,
        datatype_set.pop(), options=raster_driver_creation_tuple[1])
    target_raster.SetProjection(raster.GetProjection())
    target_raster.SetGeoTransform(target_geotransform)
    if target_nodata is None:
        nodata = nodata_set.pop()
    else:
        nodata = target_nodata
    # consider what to do if rasters have nodata defined, but do not fill
    # up the mosaic.
    if nodata is not None:
        # geotiffs only have 1 nodata value set through the band
        target_raster.GetRasterBand(1).SetNoDataValue(nodata)
        for band_index in range(n_bands):
            target_raster.GetRasterBand(band_index+1).Fill(nodata)
    target_band_list = [
        target_raster.GetRasterBand(band_index) for band_index in range(
            1, n_bands+1)]

    # the raster was left over from checking pixel types, remove it after
    raster = None

    for raster_info, raster_path in zip(raster_info_list, raster_path_list):
        # figure out where raster_path starts w/r/t target_raster
        raster_start_x = int((
            raster_info['geotransform'][0] -
            target_geotransform[0]) / target_pixel_size[0])
        raster_start_y = int((
            raster_info['geotransform'][3] -
            target_geotransform[3]) / target_pixel_size[1])
        for band_offset in range(n_bands):
            for offset_info, data_block in iterblocks(
                    (raster_path, band_offset+1)):
                # its possible the block reads in coverage that is outside the
                # target bounds entirely. nothing to do but skip
                if offset_info['yoff'] + raster_start_y > n_rows:
                    continue
                if offset_info['xoff'] + raster_start_x > n_cols:
                    continue
                if (offset_info['xoff'] + raster_start_x +
                        offset_info['win_xsize'] < 0):
                    continue
                if (offset_info['yoff'] + raster_start_y +
                        offset_info['win_ysize'] < 0):
                    continue

                # invariant: the window described in ``offset_info``
                # intersects with the target raster.

                # check to see if window hangs off the left/top part of raster
                # and determine how far to adjust down
                x_clip_min = 0
                if raster_start_x + offset_info['xoff'] < 0:
                    x_clip_min = abs(raster_start_x + offset_info['xoff'])
                y_clip_min = 0
                if raster_start_y + offset_info['yoff'] < 0:
                    y_clip_min = abs(raster_start_y + offset_info['yoff'])
                x_clip_max = 0

                # check if window hangs off right/bottom part of target raster
                if (offset_info['xoff'] + raster_start_x +
                        offset_info['win_xsize'] >= n_cols):
                    x_clip_max = (
                        offset_info['xoff'] + raster_start_x +
                        offset_info['win_xsize'] - n_cols)
                y_clip_max = 0

                if (offset_info['yoff'] + raster_start_y +
                        offset_info['win_ysize'] >= n_rows):
                    y_clip_max = (
                        offset_info['yoff'] + raster_start_y +
                        offset_info['win_ysize'] - n_rows)

                target_band_list[band_offset].WriteArray(
                    data_block[
                        y_clip_min:offset_info['win_ysize']-y_clip_max,
                        x_clip_min:offset_info['win_xsize']-x_clip_max],
                    xoff=offset_info['xoff']+raster_start_x+x_clip_min,
                    yoff=offset_info['yoff']+raster_start_y+y_clip_min)

    del target_band_list[:]
    target_raster = None


def mask_raster(
        base_raster_path_band, mask_vector_path, target_mask_raster_path,
        mask_layer_id=0, target_mask_value=None, working_dir=None,
        all_touched=False, where_clause=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Mask a raster band with a given vector.

    Args:
        base_raster_path_band (tuple): a (path, band number) tuple indicating
            the data to mask.
        mask_vector_path (path): path to a vector that will be used to mask
            anything outside of the polygon that overlaps with
            ``base_raster_path_band`` to ``target_mask_value`` if defined or
            else ``base_raster_path_band``'s nodata value.
        target_mask_raster_path (str): path to desired target raster that
            is a copy of ``base_raster_path_band`` except any pixels that do
            not intersect with ``mask_vector_path`` are set to
            ``target_mask_value`` or ``base_raster_path_band``'s nodata value
            if ``target_mask_value`` is None.
        mask_layer_id (str/int): an index or name to identify the mask
            geometry layer in ``mask_vector_path``, default is 0.
        target_mask_value (numeric): If not None, this value is written to
            any pixel in ``base_raster_path_band`` that does not intersect
            with ``mask_vector_path``. Otherwise the nodata value of
            ``base_raster_path_band`` is used.
        working_dir (str): this is a path to a directory that can be used to
            hold temporary files required to complete this operation.
        all_touched (bool): if False, a pixel is only masked if its centroid
            intersects with the mask. If True a pixel is masked if any point
            of the pixel intersects the polygon mask.
        where_clause (str): (optional) if not None, it is an SQL compatible
            where clause that can be used to filter the features that are used
            to mask the base raster.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None
    """
    with tempfile.NamedTemporaryFile(
            prefix='mask_raster', delete=False, suffix='.tif',
            dir=working_dir) as mask_raster_file:
        mask_raster_path = mask_raster_file.name

    new_raster_from_base(
        base_raster_path_band[0], mask_raster_path, gdal.GDT_Byte, [255],
        fill_value_list=[0],
        raster_driver_creation_tuple=raster_driver_creation_tuple)

    base_raster_info = get_raster_info(base_raster_path_band[0])

    rasterize(
        mask_vector_path, mask_raster_path, burn_values=[1],
        layer_id=mask_layer_id,
        option_list=[('ALL_TOUCHED=%s' % all_touched).upper()],
        where_clause=where_clause)

    base_nodata = base_raster_info['nodata'][base_raster_path_band[1]-1]

    if target_mask_value is None:
        mask_value = base_nodata
        if mask_value is None:
            LOGGER.warning(
                "No mask value was passed and target nodata is undefined, "
                "defaulting to 0 as the target mask value.")
            mask_value = 0
    else:
        mask_value = target_mask_value

    def mask_op(base_array, mask_array):
        result = numpy.copy(base_array)
        result[mask_array == 0] = mask_value
        return result

    raster_calculator(
        [base_raster_path_band, (mask_raster_path, 1)], mask_op,
        target_mask_raster_path, base_raster_info['datatype'], base_nodata,
        raster_driver_creation_tuple=raster_driver_creation_tuple)

    os.remove(mask_raster_path)


def _invoke_timed_callback(
        reference_time, callback_lambda, callback_period):
    """Invoke callback if a certain amount of time has passed.

    This is a convenience function to standardize update callbacks from the
    module.

    Args:
        reference_time (float): time to base ``callback_period`` length from.
        callback_lambda (lambda): function to invoke if difference between
            current time and ``reference_time`` has exceeded
            ``callback_period``.
        callback_period (float): time in seconds to pass until
            ``callback_lambda`` is invoked.

    Returns:
        ``reference_time`` if ``callback_lambda`` not invoked, otherwise the
        time when ``callback_lambda`` was invoked.

    """
    current_time = time.time()
    if current_time - reference_time > callback_period:
        callback_lambda()
        return current_time
    return reference_time


def _gdal_to_numpy_type(band):
    """Calculate the equivalent numpy datatype from a GDAL raster band type.

    This function doesn't handle complex or unknown types.  If they are
    passed in, this function will raise a ValueError.

    Args:
        band (gdal.Band): GDAL Band

    Returns:
        numpy_datatype (numpy.dtype): equivalent of band.DataType

    """
    # doesn't include GDT_Byte because that's a special case
    base_gdal_type_to_numpy = {
        gdal.GDT_Int16: numpy.int16,
        gdal.GDT_Int32: numpy.int32,
        gdal.GDT_UInt16: numpy.uint16,
        gdal.GDT_UInt32: numpy.uint32,
        gdal.GDT_Float32: numpy.float32,
        gdal.GDT_Float64: numpy.float64,
    }

    if band.DataType in base_gdal_type_to_numpy:
        return base_gdal_type_to_numpy[band.DataType]

    if band.DataType != gdal.GDT_Byte:
        raise ValueError("Unsupported DataType: %s" % str(band.DataType))

    # band must be GDT_Byte type, check if it is signed/unsigned
    metadata = band.GetMetadata('IMAGE_STRUCTURE')
    if 'PIXELTYPE' in metadata and metadata['PIXELTYPE'] == 'SIGNEDBYTE':
        return numpy.int8
    return numpy.uint8


def merge_bounding_box_list(bounding_box_list, bounding_box_mode):
    """Create a single bounding box by union or intersection of the list.

    Args:
        bounding_box_list (sequence): a sequence of bounding box coordinates
            in the order [minx, miny, maxx, maxy].
        mode (string): either ``'union'`` or ``'intersection'`` for the
            corresponding reduction mode.

    Returns:
        A four tuple bounding box that is the union or intersection of the
        input bounding boxes.

    Raises:
        ValueError
            if the bounding boxes in ``bounding_box_list`` do not
            intersect if the ``bounding_box_mode`` is 'intersection'.

    """
    def _merge_bounding_boxes(bb1, bb2, mode):
        """Merge two bounding boxes through union or intersection.

        Args:
            bb1, bb2 (sequence): sequence of float representing bounding box
                in the form bb=[minx,miny,maxx,maxy]
            mode (string); one of 'union' or 'intersection'

        Returns:
            Reduced bounding box of bb1/bb2 depending on mode.

        """
        def _less_than_or_equal(x_val, y_val):
            return x_val if x_val <= y_val else y_val

        def _greater_than(x_val, y_val):
            return x_val if x_val > y_val else y_val

        if mode == "union":
            comparison_ops = [
                _less_than_or_equal, _less_than_or_equal,
                _greater_than, _greater_than]
        if mode == "intersection":
            comparison_ops = [
                _greater_than, _greater_than,
                _less_than_or_equal, _less_than_or_equal]

        bb_out = [op(x, y) for op, x, y in zip(comparison_ops, bb1, bb2)]
        return bb_out

    result_bb = functools.reduce(
        functools.partial(_merge_bounding_boxes, mode=bounding_box_mode),
        bounding_box_list)
    if result_bb[0] > result_bb[2] or result_bb[1] > result_bb[3]:
        raise ValueError(
            "Bounding boxes do not intersect. Base list: %s mode: %s "
            " result: %s" % (bounding_box_list, bounding_box_mode, result_bb))
    return result_bb


def get_gis_type(path):
    """Calculate the GIS type of the file located at ``path``.

    Args:
        path (str): path to a file on disk.


    Returns:
        A bitwise OR of all GIS types that PyGeoprocessing models, currently
        this is ``pygeoprocessing.UNKNOWN_TYPE``,
        ``pygeoprocessing.RASTER_TYPE``, or ``pygeoprocessing.VECTOR_TYPE``.

    """
    if not os.path.exists(path):
        raise ValueError("%s does not exist", path)
    from pygeoprocessing import UNKNOWN_TYPE
    gis_type = UNKNOWN_TYPE
    gis_raster = gdal.OpenEx(path, gdal.OF_RASTER)
    if gis_raster is not None:
        from pygeoprocessing import RASTER_TYPE
        gis_type |= RASTER_TYPE
        gis_raster = None
    gis_vector = gdal.OpenEx(path, gdal.OF_VECTOR)
    if gis_vector is not None:
        from pygeoprocessing import VECTOR_TYPE
        gis_type |= VECTOR_TYPE
    return gis_type


def _make_logger_callback(message):
    """Build a timed logger callback that prints ``message`` replaced.

    Args:
        message (string): a string that expects 2 placement %% variables,
            first for % complete from ``df_complete``, second from
            ``p_progress_arg[0]``.

    Returns:
        Function with signature:
            logger_callback(df_complete, psz_message, p_progress_arg)

    """
    def logger_callback(df_complete, _, p_progress_arg):
        """Argument names come from the GDAL API for callbacks."""
        try:
            current_time = time.time()
            if ((current_time - logger_callback.last_time) > 5.0 or
                    (df_complete == 1.0 and
                     logger_callback.total_time >= 5.0)):
                # In some multiprocess applications I was encountering a
                # ``p_progress_arg`` of None. This is unexpected and I suspect
                # was an issue for some kind of GDAL race condition. So I'm
                # guarding against it here and reporting an appropriate log
                # if it occurs.
                if p_progress_arg:
                    LOGGER.info(message, df_complete * 100, p_progress_arg[0])
                else:
                    LOGGER.info(message, df_complete * 100, '')
                logger_callback.last_time = current_time
                logger_callback.total_time += current_time
        except AttributeError:
            logger_callback.last_time = time.time()
            logger_callback.total_time = 0.0
        except Exception:
            LOGGER.exception("Unhandled error occurred while logging "
                             "progress.  df_complete: %s, p_progress_arg: %s",
                             df_complete, p_progress_arg)

    return logger_callback


def _is_raster_path_band_formatted(raster_path_band):
    """Return true if raster path band is a (str, int) tuple/list."""
    if not isinstance(raster_path_band, (list, tuple)):
        return False
    elif len(raster_path_band) != 2:
        return False
    elif not isinstance(raster_path_band[0], str):
        return False
    elif not isinstance(raster_path_band[1], int):
        return False
    else:
        return True


def _convolve_2d_worker(
        signal_path_band, kernel_path_band,
        ignore_nodata, normalize_kernel, set_tol_to_zero,
        work_queue, write_queue):
    """Worker function to be used by ``convolve_2d``.

    Args:
        signal_path_band (tuple): a 2 tuple of the form
            (filepath to signal raster, band index).
        kernel_path_band (tuple): a 2 tuple of the form
            (filepath to kernel raster, band index).
        ignore_nodata (boolean): If true, any pixels that are equal to
            ``signal_path_band``'s nodata value are not included when
            averaging the convolution filter.
        normalize_kernel (boolean): If true, the result is divided by the
            sum of the kernel.
        set_tol_to_zero (float): Value to test close to to determine if values
            are zero, and if so, set to zero.
        work_queue (Queue): will contain (signal_offset, kernel_offset)
            tuples that can be used to read raster blocks directly using
            GDAL ReadAsArray(**offset). Indicates the block to operate on.
        write_queue (Queue): mechanism to pass result back to the writer
            contains a (index_dict, result, mask_result,
                 left_index_raster, right_index_raster,
                 top_index_raster, bottom_index_raster,
                 left_index_result, right_index_result,
                 top_index_result, bottom_index_result) tuple that's used
            for writing and masking.

    Returns:
        None
    """
    signal_raster = gdal.OpenEx(signal_path_band[0], gdal.OF_RASTER)
    kernel_raster = gdal.OpenEx(kernel_path_band[0], gdal.OF_RASTER)
    signal_band = signal_raster.GetRasterBand(signal_path_band[1])
    kernel_band = kernel_raster.GetRasterBand(kernel_path_band[1])

    signal_raster_info = get_raster_info(signal_path_band[0])
    kernel_raster_info = get_raster_info(kernel_path_band[0])

    n_cols_signal, n_rows_signal = signal_raster_info['raster_size']
    n_cols_kernel, n_rows_kernel = kernel_raster_info['raster_size']
    signal_nodata = signal_raster_info['nodata'][0]
    kernel_nodata = kernel_raster_info['nodata'][0]

    mask_result = None  # in case no mask is needed, variable is still defined

    # calculate the kernel sum for normalization
    kernel_sum = 0.0
    for _, kernel_block in iterblocks(kernel_path_band):
        if kernel_nodata is not None and ignore_nodata:
            kernel_block[numpy.isclose(kernel_block, kernel_nodata)] = 0.0
        kernel_sum += numpy.sum(kernel_block)

    while True:
        payload = work_queue.get()
        if payload is None:
            break

        signal_offset, kernel_offset = payload

        signal_block = signal_band.ReadAsArray(**signal_offset)
        kernel_block = kernel_band.ReadAsArray(**kernel_offset)

        # don't ever convolve the nodata value
        if signal_nodata is not None:
            signal_nodata_mask = numpy.isclose(signal_block, signal_nodata)
            signal_block[signal_nodata_mask] = 0.0
            if not ignore_nodata:
                signal_nodata_mask[:] = 0
        else:
            signal_nodata_mask = numpy.zeros(
                signal_block.shape, dtype=numpy.bool)

        left_index_raster = (
            signal_offset['xoff'] - n_cols_kernel // 2 +
            kernel_offset['xoff'])
        right_index_raster = (
            signal_offset['xoff'] - n_cols_kernel // 2 +
            kernel_offset['xoff'] + signal_offset['win_xsize'] +
            kernel_offset['win_xsize'] - 1)
        top_index_raster = (
            signal_offset['yoff'] - n_rows_kernel // 2 +
            kernel_offset['yoff'])
        bottom_index_raster = (
            signal_offset['yoff'] - n_rows_kernel // 2 +
            kernel_offset['yoff'] + signal_offset['win_ysize'] +
            kernel_offset['win_ysize'] - 1)

        # it's possible that the piece of the integrating kernel
        # doesn't affect the final result, if so we should skip
        if (right_index_raster < 0 or
                bottom_index_raster < 0 or
                left_index_raster > n_cols_signal or
                top_index_raster > n_rows_signal):
            continue

        if kernel_nodata is not None:
            kernel_block[numpy.isclose(kernel_block, kernel_nodata)] = 0.0

        if normalize_kernel:
            kernel_block /= kernel_sum

        # determine the output convolve shape
        shape = (
            numpy.array(signal_block.shape) +
            numpy.array(kernel_block.shape) - 1)

        # add zero padding so FFT is fast
        fshape = [_next_regular(int(d)) for d in shape]

        signal_fft = numpy.fft.rfftn(signal_block, fshape)
        kernel_fft = numpy.fft.rfftn(kernel_block, fshape)

        # this variable determines the output slice that doesn't include
        # the padded array region made for fast FFTs.
        fslice = tuple([slice(0, int(sz)) for sz in shape])
        # classic FFT convolution
        result = numpy.fft.irfftn(signal_fft * kernel_fft, fshape)[fslice]
        # nix any roundoff error
        if set_tol_to_zero is not None:
            result[numpy.isclose(result, set_tol_to_zero)] = 0.0

        # if we're ignoring nodata, we need to make a convolution of the
        # nodata mask too
        if ignore_nodata:
            mask_fft = numpy.fft.rfftn(
                numpy.where(signal_nodata_mask, 0.0, 1.0), fshape)
            mask_result = numpy.fft.irfftn(
                mask_fft * kernel_fft, fshape)[fslice]

        left_index_result = 0
        right_index_result = result.shape[1]
        top_index_result = 0
        bottom_index_result = result.shape[0]

        # we might abut the edge of the raster, clip if so
        if left_index_raster < 0:
            left_index_result = -left_index_raster
            left_index_raster = 0
        if top_index_raster < 0:
            top_index_result = -top_index_raster
            top_index_raster = 0
        if right_index_raster > n_cols_signal:
            right_index_result -= right_index_raster - n_cols_signal
            right_index_raster = n_cols_signal
        if bottom_index_raster > n_rows_signal:
            bottom_index_result -= (
                bottom_index_raster - n_rows_signal)
            bottom_index_raster = n_rows_signal

        # Add result to current output to account for overlapping edges
        index_dict = {
            'xoff': left_index_raster,
            'yoff': top_index_raster,
            'win_xsize': right_index_raster-left_index_raster,
            'win_ysize': bottom_index_raster-top_index_raster
        }

        write_queue.put(
            (index_dict, result, mask_result,
             left_index_raster, right_index_raster,
             top_index_raster, bottom_index_raster,
             left_index_result, right_index_result,
             top_index_result, bottom_index_result))

    # Indicates worker has terminated
    write_queue.put(None)


def _assert_is_valid_pixel_size(target_pixel_size):
    """Return true if ``target_pixel_size`` is a valid 2 element sequence.

    Raises ValueError if not a two element list/tuple and/or the values in
        the sequence are not numerical.

    """
    def _is_number(x):
        """Return true if x is a number."""
        try:
            if isinstance(x, str):
                return False
            float(x)
            return True
        except (ValueError, TypeError):
            return False

    if not isinstance(target_pixel_size, (list, tuple)):
        raise ValueError(
            "target_pixel_size is not a tuple, its value was '%s'",
            repr(target_pixel_size))

    if (len(target_pixel_size) != 2 or
            not all([_is_number(x) for x in target_pixel_size])):
        raise ValueError(
            "Invalid value for `target_pixel_size`, expected two numerical "
            "elements, got: %s", repr(target_pixel_size))
    return True


def shapely_geometry_to_vector(
        shapely_geometry_list, target_vector_path, projection_wkt,
        vector_format, fields=None, attribute_list=None,
        ogr_geom_type=ogr.wkbPolygon):
    """Convert list of geometry to vector on disk.

    Args:
        shapely_geometry_list (list): a list of Shapely objects.
        target_vector_path (str): path to target vector.
        projection_wkt (str): WKT for target vector.
        vector_format (str): GDAL driver name for target vector.
        fields (dict): a python dictionary mapping string fieldname
            to OGR Fieldtypes, if None no fields are added
        attribute_list (list of dicts): a list of python dictionary mapping
            fieldname to field value for each geometry in
            `shapely_geometry_list`, if None, no attributes are created.
        ogr_geom_type (ogr geometry enumerated type): sets the target layer
            geometry type. Defaults to wkbPolygon.

    Returns:
        None
    """
    if fields is None:
        fields = {}

    if attribute_list is None:
        attribute_list = [{} for _ in range(len(shapely_geometry_list))]

    num_geoms = len(shapely_geometry_list)
    num_attrs = len(attribute_list)
    if num_geoms != num_attrs:
        raise ValueError(
            f"Geometry count ({num_geoms}) and attribute count "
            f"({num_attrs}) do not match.")

    vector_driver = ogr.GetDriverByName(vector_format)
    target_vector = vector_driver.CreateDataSource(target_vector_path)
    layer_name = os.path.basename(os.path.splitext(target_vector_path)[0])
    projection = osr.SpatialReference()
    projection.ImportFromWkt(projection_wkt)
    target_layer = target_vector.CreateLayer(
        layer_name, srs=projection, geom_type=ogr_geom_type)

    for field_name, field_type in fields.items():
        target_layer.CreateField(ogr.FieldDefn(field_name, field_type))
    layer_defn = target_layer.GetLayerDefn()

    for shapely_feature, fields in zip(shapely_geometry_list, attribute_list):
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
        new_feature.SetGeometry(new_geometry)

        for field_name, field_value in fields.items():
            new_feature.SetField(field_name, field_value)
        target_layer.CreateFeature(new_feature)

    target_layer = None
    target_vector = None


def numpy_array_to_raster(
        base_array, target_nodata, pixel_size, origin, projection_wkt,
        target_path,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Create a single band raster of size ``base_array.shape``.

    Args:
        base_array (numpy.array): a 2d numpy array.
        target_nodata (numeric): nodata value of target array, can be None.
        pixel_size (tuple): square dimensions (in ``(x, y)``) of pixel.
        origin (tuple/list): x/y coordinate of the raster origin.
        projection_wkt (str): target projection in wkt.
        target_path (str): path to raster to create that will be of the
            same type of base_array with contents of base_array.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to
            geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None
    """
    numpy_to_gdal_type = {
        numpy.dtype(numpy.bool): gdal.GDT_Byte,
        numpy.dtype(numpy.int8): gdal.GDT_Byte,
        numpy.dtype(numpy.uint8): gdal.GDT_Byte,
        numpy.dtype(numpy.int16): gdal.GDT_Int16,
        numpy.dtype(numpy.int32): gdal.GDT_Int32,
        numpy.dtype(numpy.uint16): gdal.GDT_UInt16,
        numpy.dtype(numpy.uint32): gdal.GDT_UInt32,
        numpy.dtype(numpy.float32): gdal.GDT_Float32,
        numpy.dtype(numpy.float64): gdal.GDT_Float64,
        numpy.dtype(numpy.csingle): gdal.GDT_CFloat32,
        numpy.dtype(numpy.complex64): gdal.GDT_CFloat64,
    }
    raster_driver = gdal.GetDriverByName(raster_driver_creation_tuple[0])
    ny, nx = base_array.shape
    new_raster = raster_driver.Create(
        target_path, nx, ny, 1, numpy_to_gdal_type[base_array.dtype],
        options=raster_driver_creation_tuple[1])
    if projection_wkt is not None:
        new_raster.SetProjection(projection_wkt)
    new_raster.SetGeoTransform(
        [origin[0], pixel_size[0], 0.0, origin[1], 0.0, pixel_size[1]])
    new_band = new_raster.GetRasterBand(1)
    if target_nodata is not None:
        new_band.SetNoDataValue(target_nodata)
    new_band.WriteArray(base_array)
    new_raster.FlushCache()
    new_band = None
    new_raster = None


def raster_to_numpy_array(raster_path, band_id=1):
    """Read the entire contents of the raster band to a numpy array.

    Args:
        raster_path (str): path to raster.
        band_id (int): band in the raster to read.

    Returns:
        numpy array contents of `band_id` in raster.

    """
    raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
    band = raster.GetRasterBand(band_id)
    array = band.ReadAsArray()
    band = None
    raster = None
    return array
