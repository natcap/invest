"""Scenario Generation: Proximity Based."""

import math
import shutil
import os
import logging
import tempfile
import struct
import heapq
import time
import collections
import csv

import numpy
from osgeo import osr
from osgeo import gdal
import pygeoprocessing
import scipy

from . import utils

LOGGER = logging.getLogger('natcap.invest.scenario_generator_proximity_based')

_OUTPUT_BASE_FILES = {
    }

_INTERMEDIATE_BASE_FILES = {
    }

_TMP_BASE_FILES = {
    'base_lulc_path': 'base_lulc.tif'
    }

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

    Parameters:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output files
        args['base_lulc_path'] (string): path to the base landcover map
        args['replacment_lucode'] (string or int): code to replace when
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

    Returns:
        None.
    """
    if (not args['convert_farthest_from_edge'] and
            not args['convert_nearest_to_edge']):
        raise ValueError("Neither scenario was selected.")

    # append a _ to the suffix if it's not empty and doesn't already have one
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    # create working directories
    output_dir = os.path.join(args['workspace_dir'])
    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    tmp_dir = os.path.join(args['workspace_dir'], 'tmp')

    pygeoprocessing.geoprocessing.create_directories(
        [output_dir, intermediate_output_dir, tmp_dir])

    f_reg = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir),
         (_TMP_BASE_FILES, output_dir)], file_suffix)

    area_to_convert = float(args['area_to_convert'])
    replacement_lucode = int(args['replacment_lucode'])

    # convert all the input strings to lists of ints
    convertible_type_list = numpy.array([
        int(x) for x in args['convertible_landcover_codes'].split()])
    focal_landcover_codes = numpy.array([
        int(x) for x in args['focal_landcover_codes'].split()])

    shutil.copy(args['base_lulc_path'], f_reg['base_lulc_path'])
    if 'aoi_path' in args and args['aoi_path'] != '':
        # clip base lulc to a new raster
        pygeoprocessing.clip_dataset_uri(
            args['base_lulc_path'], args['aoi_path'], f_reg['base_lulc_path'],
            assert_projections=True, all_touched=False)

    scenarios = [
        (args['convert_farthest_from_edge'], 'farthest_from_edge', -1.0),
        (args['convert_nearest_to_edge'], 'nearest_to_edge', 1.0)]

    for scenario_enabled, basename, score_weight in scenarios:
        if not scenario_enabled:
            continue
        LOGGER.info('executing %s scenario', basename)
        output_landscape_raster_uri = os.path.join(
            output_dir, basename+file_suffix+'.tif')
        stats_uri = os.path.join(
            output_dir, basename+file_suffix+'.csv')
        distance_from_edge_uri = os.path.join(
            intermediate_output_dir, basename+'_distance'+file_suffix+'.tif')
        _convert_landscape(
            f_reg['base_lulc_path'], replacement_lucode, area_to_convert,
            focal_landcover_codes, convertible_type_list, score_weight,
            int(args['n_fragmentation_steps']), distance_from_edge_uri,
            output_landscape_raster_uri, stats_uri)


def _convert_landscape(
        base_lulc_uri, replacement_lucode, area_to_convert,
        focal_landcover_codes, convertible_type_list, score_weight, n_steps,
        smooth_distance_from_edge_uri, output_landscape_raster_uri,
        stats_uri):
    """Expand replacement lucodes in relation to the focal lucodes.

    If the sign on `score_weight` is positive, expansion occurs marches
    away from the focal types, while if `score_weight` is negative conversion
    marches toward the focal types.

    Parameters:
        base_lulc_uri (string): path to landcover raster that will be used as
            the base landcover map to agriculture pixels
        replacement_lucode (int): agriculture landcover code type found in the
            raster at `base_lulc_uri`
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
            `output_landscape_raster_uri`.  On the first step the distance
            is calculated from `base_lulc_uri`.
        smooth_distance_from_edge_uri (string): an intermediate output showing
            the pixel distance from the edge of the base landcover types
        output_landscape_raster_uri (string): an output raster that will
            contain the final fragmented forest layer.
        stats_uri (string): a path to an output csv that records the number
            type, and area of pixels converted in `output_landscape_raster_uri`

    Returns:
        None.
    """
    tmp_file_registry = {
        'non_base_mask': pygeoprocessing.temporary_filename(),
        'base_mask': pygeoprocessing.temporary_filename(),
        'gaussian_kernel': pygeoprocessing.temporary_filename(),
        'distance_from_base_mask_edge': pygeoprocessing.temporary_filename(),
        'distance_from_non_base_mask_edge':
            pygeoprocessing.temporary_filename(),
        'convertible_distances': pygeoprocessing.temporary_filename(),
        'smooth_distance_from_edge': pygeoprocessing.temporary_filename(),
        'distance_from_edge': pygeoprocessing.temporary_filename(),
    }
    # a sigma of 1.0 gives nice visual results to smooth pixel level artifacts
    # since a pixel is the 1.0 unit
    _make_gaussian_kernel_uri(1.0, tmp_file_registry['gaussian_kernel'])

    # create the output raster first as a copy of the base landcover so it can
    # be looped on for each step
    lulc_nodata = pygeoprocessing.get_nodata_from_uri(base_lulc_uri)
    pixel_size_out = pygeoprocessing.get_cell_size_from_uri(base_lulc_uri)
    mask_nodata = 2
    pygeoprocessing.vectorize_datasets(
        [base_lulc_uri], lambda x: x, output_landscape_raster_uri,
        gdal.GDT_Int32, lulc_nodata, pixel_size_out, "intersection",
        vectorize_op=False, datasets_are_pre_aligned=True)

    # convert everything furthest from edge for each of n_steps
    pixel_area_ha = (
        pygeoprocessing.get_cell_size_from_uri(base_lulc_uri)**2 / 10000.0)
    max_pixels_to_convert = int(math.ceil(area_to_convert / pixel_area_ha))
    convertible_type_nodata = -1
    pixels_left_to_convert = max_pixels_to_convert
    pixels_to_convert = max_pixels_to_convert / n_steps
    stats_cache = collections.defaultdict(int)

    # pylint complains when these are defined inside the loop
    invert_mask = None
    distance_nodata = None

    for step_index in xrange(n_steps):
        LOGGER.info('step %d of %d', step_index+1, n_steps)
        pixels_left_to_convert -= pixels_to_convert

        # Often the last segement of the steps will overstep the  number of
        # pixels to convert, this check converts the exact amount
        if pixels_left_to_convert < 0:
            pixels_to_convert += pixels_left_to_convert

        # create distance transforms for inside and outside the base lulc codes
        LOGGER.info('create distance transform for current landcover')
        for invert_mask, mask_id, distance_id in [
                (False, 'non_base_mask', 'distance_from_non_base_mask_edge'),
                (True, 'base_mask', 'distance_from_base_mask_edge')]:

            def _mask_base_op(lulc_array):
                """Create a mask of valid non-base pixels only."""
                base_mask = numpy.in1d(
                    lulc_array.flatten(), focal_landcover_codes).reshape(
                        lulc_array.shape)
                if invert_mask:
                    base_mask = ~base_mask
                return numpy.where(
                    lulc_array == lulc_nodata, mask_nodata, base_mask)
            pygeoprocessing.vectorize_datasets(
                [output_landscape_raster_uri], _mask_base_op,
                tmp_file_registry[mask_id], gdal.GDT_Byte,
                mask_nodata, pixel_size_out, "intersection",
                vectorize_op=False, datasets_are_pre_aligned=True)

            # create distance transform for the current mask
            pygeoprocessing.distance_transform_edt(
                tmp_file_registry[mask_id], tmp_file_registry[distance_id])

        # combine inner and outer distance transforms into one
        distance_nodata = pygeoprocessing.get_nodata_from_uri(
            tmp_file_registry['distance_from_base_mask_edge'])

        def _combine_masks(base_distance_array, non_base_distance_array):
            """create a mask of valid non-base pixels only."""
            result = non_base_distance_array
            valid_base_mask = base_distance_array > 0.0
            result[valid_base_mask] = base_distance_array[valid_base_mask]
            return result
        pygeoprocessing.vectorize_datasets(
            [tmp_file_registry['distance_from_base_mask_edge'],
             tmp_file_registry['distance_from_non_base_mask_edge']],
            _combine_masks, tmp_file_registry['distance_from_edge'],
            gdal.GDT_Float32, distance_nodata, pixel_size_out, "intersection",
            vectorize_op=False, datasets_are_pre_aligned=True)

        # smooth the distance transform to avoid scanline artifacts
        pygeoprocessing.convolve_2d_uri(
            tmp_file_registry['distance_from_edge'],
            tmp_file_registry['gaussian_kernel'],
            smooth_distance_from_edge_uri)

        # turn inside and outside masks into a single mask
        def _mask_to_convertible_codes(distance_from_base_edge, lulc):
            """Mask out the distance transform to a set of lucodes."""
            convertible_mask = numpy.in1d(
                lulc.flatten(), convertible_type_list).reshape(lulc.shape)
            return numpy.where(
                convertible_mask, distance_from_base_edge,
                convertible_type_nodata)
        pygeoprocessing.vectorize_datasets(
            [smooth_distance_from_edge_uri, output_landscape_raster_uri],
            _mask_to_convertible_codes,
            tmp_file_registry['convertible_distances'], gdal.GDT_Float32,
            convertible_type_nodata, pixel_size_out, "intersection",
            vectorize_op=False, datasets_are_pre_aligned=True)

        LOGGER.info(
            'convert %d pixels to lucode %d', pixels_to_convert,
            replacement_lucode)
        _convert_by_score(
            tmp_file_registry['convertible_distances'], pixels_to_convert,
            output_landscape_raster_uri, replacement_lucode, stats_cache,
            score_weight)

    _log_stats(stats_cache, pixel_area_ha, stats_uri)
    for filename in tmp_file_registry.values():
        os.remove(filename)


def _log_stats(stats_cache, pixel_area, stats_uri):
    """Write pixel change statistics to a file in tabular format.

    Parameters:
        stats_cache (dict): a dictionary mapping pixel lucodes to number of
            pixels changed
        pixel_area (float): size of pixels in hectares so an area column can
            be generated
        stats_uri (string): path to a csv file that the table should be
            generated to

    Returns:
        None
    """
    with open(stats_uri, 'wb') as csv_output_file:
        stats_writer = csv.writer(
            csv_output_file, delimiter=',', quotechar=',',
            quoting=csv.QUOTE_MINIMAL)
        stats_writer.writerow(
            ['lucode', 'area converted (Ha)', 'pixels converted'])
        for lucode in sorted(stats_cache):
            stats_writer.writerow([
                lucode, stats_cache[lucode] * pixel_area, stats_cache[lucode]])


def _sort_to_disk(dataset_uri, score_weight=1.0):
    """Return an iterable of non-nodata pixels in sorted order.

    Parameters:
        dataset_uri (string): a path to a floating point GDAL dataset
        score_weight (float): a number to multiply all values by, which can be
            used to reverse the order of the iteration if negative.

    Returns:
        an iterable that produces (value * score_weight, flat_index) in
        decreasing sorted order by value * score_weight
    """
    def _read_score_index_from_disk(
            score_file_path, index_file_path):
        """Generator to yield a float/int value from the given filenames.

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

        Parameters:
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
        for index in xrange(0, score_cache.size, _LARGEST_STRUCT_PACK):
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

    nodata = pygeoprocessing.get_nodata_from_uri(dataset_uri)
    nodata *= score_weight  # scale the nodata so they can be filtered out

    # This will be a list of file iterators we'll pass to heap.merge
    iters = []

    _, n_cols = pygeoprocessing.get_row_col_from_uri(dataset_uri)

    for scores_data, scores_block in pygeoprocessing.iterblocks(
            dataset_uri, largest_block=_BLOCK_SIZE):
        # flatten and scale the results
        scores_block = scores_block.flatten() * score_weight

        col_coords, row_coords = numpy.meshgrid(
            xrange(scores_data['xoff'], scores_data['xoff'] +
                   scores_data['win_xsize']),
            xrange(scores_data['yoff'], scores_data['yoff'] +
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
        score_uri, max_pixels_to_convert, out_raster_uri, convert_value,
        stats_cache, score_weight):
    """Convert up to max pixels in ranked order of score.

    Parameters:
        score_uri (string): path to a raster whose non-nodata values score the
            pixels to convert.  The pixels in `out_raster_uri` are converted
            from the lowest score to the highest.  This scale can be modified
            by the parameter `score_weight`.
        max_pixels_to_convert (int): number of pixels to convert in
            `out_raster_uri` up to the number of non nodata valued pixels in
            `score_uri`.
        out_raster_uri (string): a path to an existing raster that is of the
            same dimensions and projection as `score_uri`.  The pixels in this
            raster are modified depending on the value of `score_uri` and set
            to the value in `convert_value`.
        convert_value (int/float): type is dependant on out_raster_uri. Any
            pixels converted in `out_raster_uri` are set to the value of this
            variable.
        reverse_sort (boolean): If true, pixels are visited in descreasing
            order of `score_uri`, otherwise increasing.
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

        Parameters:
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

    out_ds = gdal.Open(out_raster_uri, gdal.GA_Update)
    out_band = out_ds.GetRasterBand(1)
    out_block_col_size, out_block_row_size = out_band.GetBlockSize()
    n_rows = out_band.YSize
    n_cols = out_band.XSize
    pixels_converted = 0

    row_array = numpy.empty((_BLOCK_SIZE,), dtype=numpy.uint32)
    col_array = numpy.empty((_BLOCK_SIZE,), dtype=numpy.uint32)
    data_array = numpy.empty((_BLOCK_SIZE,), dtype=numpy.bool)
    next_index = 0
    dirty_blocks = set()

    last_time = time.time()
    for _, flatindex in _sort_to_disk(score_uri, score_weight=score_weight):
        if pixels_converted >= max_pixels_to_convert:
            break
        col_index = flatindex % n_cols
        row_index = flatindex / n_cols
        row_array[next_index] = row_index
        col_array[next_index] = col_index
        # data_array will only ever recieve True elements, necessary for the
        # sparse matrix to function since it requires a data array as long
        # as the row and column arrays
        data_array[next_index] = True
        next_index += 1
        dirty_blocks.add(
            (row_index / out_block_row_size, col_index / out_block_col_size))
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


def _make_gaussian_kernel_uri(sigma, kernel_uri):
    """Create a 2D Gaussian kernel.

    Parameters:
        sigma (float): the sigma as in the classic Gaussian function
        kernel_uri (string): path to raster on disk to write the gaussian
            kernel.

    Returns:
        None.
    """
    # going 3.0 times out from the sigma gives you over 99% of area under
    # the guassian curve
    max_distance = sigma * 3.0
    kernel_size = int(numpy.round(max_distance * 2 + 1))

    driver = gdal.GetDriverByName('GTiff')
    kernel_dataset = driver.Create(
        kernel_uri.encode('utf-8'), kernel_size, kernel_size, 1,
        gdal.GDT_Float32, options=['BIGTIFF=IF_SAFER'])

    # Make some kind of geotransform, it doesn't matter what but
    # will make GIS libraries behave better if it's all defined
    kernel_dataset.SetGeoTransform([444720, 30, 0, 3751320, 0, -30])
    srs = osr.SpatialReference()
    srs.SetUTM(11, 1)
    srs.SetWellKnownGeogCS('NAD27')
    kernel_dataset.SetProjection(srs.ExportToWkt())

    kernel_band = kernel_dataset.GetRasterBand(1)
    kernel_band.SetNoDataValue(-9999)

    col_index = numpy.array(xrange(kernel_size))
    integration = 0.0
    for row_index in xrange(kernel_size):
        distance_kernel_row = numpy.sqrt(
            (row_index - max_distance) ** 2 +
            (col_index - max_distance) ** 2).reshape(1, kernel_size)
        kernel = numpy.where(
            distance_kernel_row > max_distance, 0.0,
            (1 / (2.0 * numpy.pi * sigma ** 2) *
             numpy.exp(-distance_kernel_row**2 / (2 * sigma ** 2))))
        integration += numpy.sum(kernel)
        kernel_band.WriteArray(kernel, xoff=0, yoff=row_index)

    kernel_dataset.FlushCache()
    for kernel_data, kernel_block in pygeoprocessing.iterblocks(kernel_uri):
        kernel_block /= integration
        kernel_band.WriteArray(
            kernel_block, xoff=kernel_data['xoff'], yoff=kernel_data['yoff'])
