"""Scenario Generation: Proximity Based"""

import os
import logging
import numpy
import tempfile
import struct
import heapq
import time
import atexit
import collections
import csv

from osgeo import osr
from osgeo import gdal
import pygeoprocessing
import scipy

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger(
    'natcap.invest.scenario_generator_proximity_based')


def execute(args):
    """Main entry point for proximity based scenario generator model.

    Parameters:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output files
        args['base_lulc_uri'] (string): path to the base landcover map
        args['replacment_lucode'] (string or int): code to replace when
            converting pixels
        args['area_to_convert'] (string or float): max area (Ha) to convert
        args['base_landcover_types'] (string): a space separated string of
            landcover codes that are used to determine the proximity when
            refering to "towards" or "away" from the base landcover types
        args['convertable_landcover_types'] (string): a space separated string
            of landcover codes that can be converted in the generation phase
            found in `args['base_lulc_uri']`.
        args['n_fragmentation_steps'] (string): an int as a string indicating
            the number of steps to take for the fragmentation conversion
        args['aoi_uri'] (string): (optional) path to a shapefile that indicates
            an area of interest.  If present, the expansion scenario operates
            only under that AOI and the output raster is clipped to that shape.
        args['convert_toward_base_edge'] (boolean): if True will run the
            conversion simulation starting from the furthest pixel from the
            edge and work inwards.  Workspace will contain output files named
            'toward_base{suffix}.{tif,csv}.
        args['convert_from_base_edge'] (boolean): if True will run the
            conversion simulation starting from the nearest pixel on the
            edge and work inwards.  Workspace will contain output files named
            'toward_base{suffix}.{tif,csv}.

    Returns:
        None.
    """

    if (not args['convert_toward_base_edge'] and
            not args['convert_from_base_edge']):
        raise ValueError("Neither scenario was selected.")

    # append a _ to the suffix if it's not empty and doesn't already have one
    try:
        file_suffix = args['results_suffix']
        if file_suffix != "" and not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''

    #create working directories
    output_dir = os.path.join(args['workspace_dir'])
    intermediate_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    tmp_dir = os.path.join(args['workspace_dir'], 'tmp')
    pygeoprocessing.geoprocessing.create_directories(
        [output_dir, intermediate_dir, tmp_dir])

    area_to_convert = float(args['area_to_convert'])
    replacement_lucode = int(args['replacment_lucode'])

    # convert all the input strings to lists of ints
    convertable_type_list = numpy.array([
        int(x) for x in args['convertable_landcover_types'].split()])
    base_landcover_types = numpy.array([
        int(x) for x in args['base_landcover_types'].split()])

    if 'aoi_uri' in args and args['aoi_uri'] != '':
        #clip base lulc to a new raster
        base_lulc_uri = pygeoprocessing.temporary_filename()
        pygeoprocessing.clip_dataset_uri(
            args['base_lulc_uri'], args['aoi_uri'], base_lulc_uri,
            assert_projections=True, process_pool=None, all_touched=False)
    else:
        base_lulc_uri = args['base_lulc_uri']

    scenarios = [(args['convert_toward_base_edge'], 'toward_edge', -1.0),
                 (args['convert_from_base_edge'], 'from_edge', 1.0)]

    for scenario_enabled, basename, score_weight in scenarios:
        if not scenario_enabled:
            continue
        LOGGER.info('running %s scenario', basename)
        output_landscape_raster_uri = os.path.join(
            output_dir, basename+file_suffix+'.tif')
        stats_uri = os.path.join(
            output_dir, basename+file_suffix+'.csv')
        _convert_landscape(
            base_lulc_uri, replacement_lucode, area_to_convert,
            base_landcover_types, convertable_type_list, score_weight,
            int(args['n_fragmentation_steps']), output_landscape_raster_uri,
            stats_uri)


def _convert_landscape(
        base_lulc_uri, replacement_lucode, area_to_convert, base_landcover_types,
        convertable_type_list, score_weight, n_steps,
        output_landscape_raster_uri, stats_uri):
    """Expands agriculture into convertable types starting from the furthest
    distance from the edge of the forest, inward.

    Parameters:
        base_lulc_uri (string): path to landcover raster that will be used as
            the base landcover map to agriculture pixels
        replacement_lucode (int): agriculture landcover code type found in the raster
            at `base_lulc_uri`
        area_to_convert (float): area (Ha) to convert to agriculture
        base_landcover_types (list of int): landcover codes that are used to
            calculate proximity
        convertable_type_list (list of int): landcover codes that are allowable
            to be converted to agriculture
        n_steps (int): number of steps to convert the landscape.  On each step
            the distance transform will be applied on the
            current value of the `base_landcover_types` pixels in
            `output_landscape_raster_uri`.  On the first step the distance
            is calculated from `base_lulc_uri`.
        output_landscape_raster_uri (string): an output raster that will
            contain the final fragmented forest layer.
        stats_uri (string): a path to an output csv that records the number
            type, and area of pixels converted in `output_landscape_raster_uri`

    Returns:
        None."""

    tmp_file_registry = {
        'non_base_mask': pygeoprocessing.temporary_filename(),
        'gaussian_kernel': pygeoprocessing.temporary_filename(),
        'distance_from_base_edge': pygeoprocessing.temporary_filename(),
        'convertable_distances': pygeoprocessing.temporary_filename(),
        'smooth_distance_from_edge': pygeoprocessing.temporary_filename(),

    }
    _make_gaussian_kernel_uri(10.0, tmp_file_registry['gaussian_kernel'])

    # create the output raster first as a copy of the base landcover so it can
    # be looped on for each step
    lulc_nodata = pygeoprocessing.get_nodata_from_uri(base_lulc_uri)
    pixel_size_out = pygeoprocessing.get_cell_size_from_uri(base_lulc_uri)
    non_base_nodata = 2
    pygeoprocessing.vectorize_datasets(
        [base_lulc_uri], lambda x: x, output_landscape_raster_uri,
        gdal.GDT_Int32, lulc_nodata, pixel_size_out, "intersection",
        vectorize_op=False)

    # convert everything furthest from edge for each of n_steps
    pixel_area_ha = (
        pygeoprocessing.get_cell_size_from_uri(base_lulc_uri)**2 / 10000.0)
    max_pixels_to_convert = int(area_to_convert / pixel_area_ha)
    convertable_type_nodata = -1
    pixels_left_to_convert = max_pixels_to_convert
    pixels_to_convert = max_pixels_to_convert / n_steps
    stats_cache = collections.defaultdict(int)
    for step_index in xrange(n_steps):
        LOGGER.info('step %d of %d', step_index+1, n_steps)
        pixels_left_to_convert -= pixels_to_convert
        if pixels_left_to_convert < 0:
            pixels_to_convert += pixels_left_to_convert

        #mask non-base types from map
        def _mask_non_base_op(current_lulc):
            """create a mask of valid non-base pixels only"""
            non_base_mask = ~numpy.in1d(
                current_lulc.flatten(), base_landcover_types).reshape(
                    current_lulc.shape)
            return numpy.where(
                current_lulc == lulc_nodata, non_base_nodata, non_base_mask)
        pygeoprocessing.vectorize_datasets(
            [output_landscape_raster_uri], _mask_non_base_op,
            tmp_file_registry['non_base_mask'], gdal.GDT_Byte,
            non_base_nodata, pixel_size_out, "intersection",
            vectorize_op=False)

        # current step's distance transform mask
        pygeoprocessing.distance_transform_edt(
            tmp_file_registry['non_base_mask'],
            tmp_file_registry['distance_from_base_edge'])
        # smooth the distance transform to avoid scanline artifacts
        pygeoprocessing.convolve_2d_uri(
            tmp_file_registry['distance_from_base_edge'],
            tmp_file_registry['gaussian_kernel'],
            tmp_file_registry['smooth_distance_from_edge'])

        def _mask_to_convertable_types(distance_from_base_edge, lulc):
            """masks out the distance transform to a set of given landcover
            codes"""
            convertable_mask = numpy.in1d(
                lulc.flatten(), convertable_type_list).reshape(lulc.shape)
            return numpy.where(
                convertable_mask, distance_from_base_edge,
                convertable_type_nodata)
        pygeoprocessing.vectorize_datasets(
            [tmp_file_registry['smooth_distance_from_edge'],
             output_landscape_raster_uri], _mask_to_convertable_types,
            tmp_file_registry['convertable_distances'], gdal.GDT_Float32,
            convertable_type_nodata, pixel_size_out, "intersection",
            vectorize_op=False)

        #Convert a wad of pixels
        _convert_by_score(
            tmp_file_registry['convertable_distances'], pixels_to_convert,
            output_landscape_raster_uri, replacement_lucode, stats_cache,
            score_weight)

    _log_stats(stats_cache, pixel_area_ha, stats_uri)
    for filename in tmp_file_registry.values():
        os.remove(filename)


def _log_stats(stats_cache, pixel_area, stats_uri):
    """Writes pixel change statistics from a simulation to disk in tabular
    format.

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
    """Sorts the non-nodata pixels in the dataset on disk and returns
    an iterable in sorted order.

    Parameters:
        dataset_uri (string): a path to a floating point GDAL dataset
        score_weight (float): a number to multiply all values by, which can be
            used to reverse the order of the iteration if negative.

    Returns:
        an iterable that produces (value * score_weight, flat_index) in
        decreasing sorted order by value * score_weight"""

    def _read_score_index_from_disk(file_name, buffer_size=8*10000):
        """Generator to yield a float/int value from `file_ name`, does
        reads a buffer of `buffer_size` big before to avoid keeping the
        file open between generations."""

        file_buffer = ''
        file_offset = 0
        buffer_offset = 1

        while True:
            if buffer_offset > len(file_buffer):
                data_file = open(file_name, 'rb')
                data_file.seek(file_offset)
                file_buffer = data_file.read(buffer_size)
                data_file.close()
                file_offset += buffer_size
                buffer_offset = 0
            packed_score = file_buffer[buffer_offset:buffer_offset+8]
            buffer_offset += 8
            if not packed_score:
                break
            yield struct.unpack('fi', packed_score)

    dataset = gdal.Open(dataset_uri)
    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    nodata *= score_weight  # scale the nodata so they can be filtered out

    n_rows = band.YSize
    n_cols = band.XSize

    #This will be a list of file iterators we'll pass to heap.merge
    iters = []

    #Set the row strides to be something reasonable, like 1MB blocks
    row_strides = max(int(10 * 2**20 / (4 * n_cols)), 1)

    for row_index in xrange(0, n_rows, row_strides):

        #It's possible we're on the last set of rows and the stride is too big
        #update if so
        if row_index + row_strides >= n_rows:
            row_strides = n_rows - row_index

        #Extract scores make them negative, calculate flat indexes, and sort
        scores = band.ReadAsArray(0, row_index, n_cols, row_strides).flatten()
        scores *= score_weight  # scale the results
        col_indexes = numpy.tile(numpy.arange(n_cols), (row_strides, 1))
        row_offsets = numpy.arange(row_index, row_index+row_strides) * n_cols
        row_offsets.resize((row_strides, 1))

        flat_indexes = (col_indexes + row_offsets).flatten()

        sort_index = scores.argsort()
        sorted_scores = scores[sort_index]
        sorted_indexes = flat_indexes[sort_index]

        #Determine where the nodata values are so we can splice them out
        left_index = numpy.searchsorted(sorted_scores, nodata, side='left')
        right_index = numpy.searchsorted(sorted_scores, nodata, side='right')

        #Splice out the nodata values and order the array in descreasing order
        sorted_scores = numpy.concatenate(
            (sorted_scores[0:left_index], sorted_scores[right_index::]))
        sorted_indexes = numpy.concatenate(
            (sorted_indexes[0:left_index], sorted_indexes[right_index::]))

        #Dump all the scores and indexes to disk
        sort_file = tempfile.NamedTemporaryFile(delete=False)
        for score, index in zip(sorted_scores, sorted_indexes):
            sort_file.write(struct.pack('fi', score, index))

        #Get the filename and register a command to delete it after the
        #interpreter exits
        sort_file_name = sort_file.name
        sort_file.close()

        def _remove_file(path):
            """Function to remove a file and handle exceptions to register
                in atexit."""
            try:
                os.remove(path)
            except OSError:
                # This happens if the file didn't exist, which is okay because
                # maybe we deleted it in a method
                pass
        atexit.register(_remove_file, sort_file_name)

        iters.append(_read_score_index_from_disk(sort_file_name))

    return heapq.merge(*iters)


def _convert_by_score(
        score_uri, max_pixels_to_convert, out_raster_uri, convert_value,
        stats_cache, score_weight, cache_size=50000):
    """Takes an input score layer and changes the pixels in `out_raster_uri`
    and converts up to `max_pixels_to_convert` them to `convert_value` type.

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
        cache_size (int): number of elements to keep in cache before flushing
            to `out_raster_uri`
        stats_cache (collections.defaultdict(int)): contains the number of
            pixels converted indexed by original pixel id.

    Returns:
        None.
    """

    def _flush_cache_to_band(
            data_array, row_array, col_array, valid_index, dirty_blocks,
            out_band, stats_counter):
        """Internal function to flush the block cache to the output band.
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

    # initialize the cache to cache_size large
    row_array = numpy.empty((cache_size,), dtype=numpy.uint32)
    col_array = numpy.empty((cache_size,), dtype=numpy.uint32)
    data_array = numpy.empty((cache_size,), dtype=numpy.bool)
    next_index = 0
    dirty_blocks = set()

    last_time = time.time()
    for _, flatindex in _sort_to_disk(
            score_uri, score_weight=score_weight):
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

        if next_index == cache_size:
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
    """Creates a 2D gaussian kernel.

    Parameters:
        sigma (float): the sigma as in the classic Gaussian function
        kernel_uri (string): path to raster on disk to write the gaussian
            kernel.

    Returns:
        None.
    """

    max_distance = sigma * 5
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

    for row_index in xrange(kernel_size):
        kernel_row = kernel_band.ReadAsArray(
            xoff=0, yoff=row_index, win_xsize=kernel_size, win_ysize=1)
        kernel_row /= integration
        kernel_band.WriteArray(kernel_row, 0, row_index)
