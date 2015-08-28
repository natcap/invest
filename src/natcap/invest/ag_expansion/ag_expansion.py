"""Scenario Generation: Agriculture Expansion"""

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
    'natcap.invest.cropland_expansion.cropland_expansion')

def execute(args):
    """Main entry point for cropland expansion tool model.

    Args:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output files
        args['base_lulc_uri'] (string): path to the base landcover map
        args['agriculture_lu_code'] (string or int): agriculture landcover code
        args['area_to_convert'] (string or float): max area (Ha) to convert
        args['forest_landcover_types'] (string): a space separated string of
            forest landcover codes found in `args['base_lulc_uri']`
        args['n_fragmentation_steps'] (string): an int as a string indicating
            the number of steps to take for the fragmentation conversion
        args['aoi_uri'] (string): (optional) path to a shapefile that indicates
            an area of interest.  If present, the expansion scenario operates
            only under that AOI and the output raster is clipped to that shape.

    Returns:
        None.
    """

    #append a _ to the suffix if it's not empty and doens't already have one
    try:
        file_suffix = args['results_suffix']
        if file_suffix != "" and not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''

    area_to_convert = float(args['area_to_convert'])
    ag_lucode = int(args['agriculture_lu_code'])

    #create working directories
    output_dir = os.path.join(args['workspace_dir'], 'output')
    intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')
    tmp_dir = os.path.join(args['workspace_dir'], 'tmp')
    pygeoprocessing.geoprocessing.create_directories(
        [output_dir, intermediate_dir, tmp_dir])

    # convert all the input strings to lists of ints
    convertable_type_list = numpy.array([
        int(x) for x in args['convertable_landcover_types'].split()])
    forest_type_list = numpy.array([
        int(x) for x in args['forest_landcover_types'].split()])

    if 'aoi_uri' in args and args['aoi_uri'] != '':
        #clip base lulc to a new raster
        base_lulc_uri = pygeoprocessing.temporary_filename()
        pygeoprocessing.clip_dataset_uri(
            args['base_lulc_uri'], args['aoi_uri'], base_lulc_uri,
            assert_projections=True, process_pool=None, all_touched=False)
    else:
        base_lulc_uri = args['base_lulc_uri']

    if args['expand_from_ag']:
        LOGGER.info('running expand from ag scenario')
        _expand_from_ag(
            base_lulc_uri, intermediate_dir, output_dir, file_suffix,
            ag_lucode, area_to_convert, convertable_type_list)

    if args['expand_from_forest_edge']:
        LOGGER.info('running expand from forest edge scenario')
        _expand_from_forest_edge(
            base_lulc_uri, intermediate_dir, output_dir, file_suffix,
            ag_lucode, area_to_convert, forest_type_list,
            convertable_type_list)

    if args['fragment_forest']:
        LOGGER.info('running forest fragmentation scenario')
        _fragment_forest(
            base_lulc_uri, intermediate_dir, output_dir,
            file_suffix, ag_lucode, area_to_convert, forest_type_list,
            convertable_type_list, int(args['n_fragmentation_steps']))

def _expand_from_ag(
        base_lulc_uri, intermediate_dir, output_dir, file_suffix, ag_lucode,
        area_to_convert, convertable_type_list):
    """Expands agriculture into convertable types starting in increasing
    distance from nearest agriculture.

    Args:
        base_lulc_uri (string): path to landcover raster that will be used as
            the base landcover map to agriculture pixels
        intermediate_dir (string): path to a directory that is safe to write
            intermediate files
        output_dir (string): path to a directory that is safe to write output
            files
        file_suffix (string): string to append to output files
        ag_lucode (int): agriculture landcover code type found in the raster
            at `base_lulc_uri`
        area_to_convert (float): area (Ha) to convert to agriculture
        convertable_type_list (list of int): landcover codes that are allowable
            to be converted to agriculture

    Returns:
        None.
    """

    # make an ag mask so we can get the distance from it
    ag_mask_uri = os.path.join(intermediate_dir, 'ag_mask%s.tif' % file_suffix)
    lulc_nodata = pygeoprocessing.get_nodata_from_uri(base_lulc_uri)
    pixel_size_out = pygeoprocessing.get_cell_size_from_uri(base_lulc_uri)
    ag_mask_nodata = 2
    def _mask_ag_op(lulc):
        """create a mask of ag pixels only"""
        ag_mask = (lulc == ag_lucode)
        return numpy.where(lulc == lulc_nodata, ag_mask_nodata, ag_mask)
    pygeoprocessing.vectorize_datasets(
        [base_lulc_uri], _mask_ag_op, ag_mask_uri, gdal.GDT_Byte,
        ag_mask_nodata, pixel_size_out, "intersection", vectorize_op=False)

    # distance transform mask
    distance_from_ag_uri = os.path.join(
        intermediate_dir, 'distance_from_ag%s.tif' % file_suffix)
    pygeoprocessing.distance_transform_edt(ag_mask_uri, distance_from_ag_uri)

    # smooth the distance transform so we don't get scanline artifacts
    gaussian_kernel_uri = os.path.join(intermediate_dir, 'gaussian_kernel.tif')
    _make_gaussian_kernel_uri(10.0, gaussian_kernel_uri)
    smooth_distance_from_ag_uri = os.path.join(
        intermediate_dir, 'smooth_distance_from_ag%s.tif' % file_suffix)
    pygeoprocessing.convolve_2d_uri(
        distance_from_ag_uri, gaussian_kernel_uri, smooth_distance_from_ag_uri)

    convertable_type_nodata = -1
    convertable_distances_uri = os.path.join(
        intermediate_dir, 'ag_convertable_distances%s.tif' % file_suffix)
    def _mask_to_convertable_types(distance_from_ag, lulc):
        """masks out the distance transform to a set of given landcover codes"""
        convertable_mask = numpy.in1d(
            lulc.flatten(), convertable_type_list).reshape(lulc.shape)
        return numpy.where(
            convertable_mask, distance_from_ag, convertable_type_nodata)

    pygeoprocessing.vectorize_datasets(
        [smooth_distance_from_ag_uri, base_lulc_uri],
        _mask_to_convertable_types, convertable_distances_uri, gdal.GDT_Float32,
        convertable_type_nodata, pixel_size_out, "intersection",
        vectorize_op=False)

    # make a copy of the base for the expanded ag
    ag_expanded_uri = os.path.join(
        output_dir, 'ag_expanded%s.tif' % file_suffix)
    pygeoprocessing.vectorize_datasets(
        [base_lulc_uri], lambda x: x, ag_expanded_uri, gdal.GDT_Int32,
        lulc_nodata, pixel_size_out, "intersection", vectorize_op=False)

    # convert all the closest to edge pixels to ag
    pixel_area_ha = (
        pygeoprocessing.get_cell_size_from_uri(base_lulc_uri)**2 / 10000.0)
    max_pixels_to_convert = int(area_to_convert / pixel_area_ha)
    stats_cache = collections.defaultdict(int)
    _convert_by_score(
        convertable_distances_uri, max_pixels_to_convert, ag_expanded_uri,
        ag_lucode, stats_cache)
    stats_uri = os.path.join(
        output_dir, 'ag_expanded_stats%s.csv' % file_suffix)
    _log_stats(stats_cache, pixel_area_ha, stats_uri)


def _expand_from_forest_edge(
        base_lulc_uri, intermediate_dir, output_dir, file_suffix, ag_lucode,
        area_to_convert, forest_type_list, convertable_type_list):
    """Expands agriculture into convertable types starting from the edge of
    the forest types, inward.

    Args:
        base_lulc_uri (string): path to landcover raster that will be used as
            the base landcover map to agriculture pixels
        intermediate_dir (string): path to a directory that is safe to write
            intermediate files
        output_dir (string): path to a directory that is safe to write output
            files
        file_suffix (string): string to append to output files
        ag_lucode (int): agriculture landcover code type found in the raster
            at `base_lulc_uri`
        area_to_convert (float): area (Ha) to convert to agriculture
        forest_type_list (list of int): landcover codes that are allowable
            to be converted to agriculture
        convertable_type_list (list of int): landcover codes that are allowable
            to be converted to agriculture

    Returns:
        None."""

    # mask everything not forest so we can get a distance to edge of forest
    non_forest_mask_uri = os.path.join(
        intermediate_dir, 'non_forest_mask%s.tif' % file_suffix)
    lulc_nodata = pygeoprocessing.get_nodata_from_uri(base_lulc_uri)
    pixel_size_out = pygeoprocessing.get_cell_size_from_uri(base_lulc_uri)
    ag_mask_nodata = 2
    def _mask_non_forest_op(lulc):
        """create a mask of valid non-forest pixels only"""
        non_forest_mask = ~numpy.in1d(
            lulc.flatten(), forest_type_list).reshape(lulc.shape)
        return numpy.where(lulc == lulc_nodata, ag_mask_nodata, non_forest_mask)
    pygeoprocessing.vectorize_datasets(
        [base_lulc_uri], _mask_non_forest_op, non_forest_mask_uri,
        gdal.GDT_Byte, ag_mask_nodata, pixel_size_out, "intersection",
        vectorize_op=False)

    #distance transform mask
    distance_from_forest_edge_uri = os.path.join(
        intermediate_dir, 'distance_from_forest_edge%s.tif' % file_suffix)
    pygeoprocessing.distance_transform_edt(
        non_forest_mask_uri, distance_from_forest_edge_uri)
    gaussian_kernel_uri = os.path.join(intermediate_dir, 'gaussian_kernel.tif')
    _make_gaussian_kernel_uri(10.0, gaussian_kernel_uri)
    # smooth the distance transform to avoid scanline artifacts
    smooth_distance_from_edge_uri = os.path.join(
        intermediate_dir,
        'smooth_distance_from_forest_edge%s.tif' % file_suffix)
    pygeoprocessing.convolve_2d_uri(
        distance_from_forest_edge_uri, gaussian_kernel_uri,
        smooth_distance_from_edge_uri)

    # make a mask of the convertable landcover types
    convertable_type_nodata = -1
    convertable_distances_uri = os.path.join(
        intermediate_dir, 'forest_edge_convertable_distances%s.tif' %
        file_suffix)
    def _mask_to_convertable_types(distance_from_forest_edge, lulc):
        """masks out the distance transform to a set of given landcover codes"""
        convertable_mask = numpy.in1d(
            lulc.flatten(), convertable_type_list).reshape(lulc.shape)
        return numpy.where(
            convertable_mask, distance_from_forest_edge,
            convertable_type_nodata)
    pygeoprocessing.vectorize_datasets(
        [smooth_distance_from_edge_uri, base_lulc_uri],
        _mask_to_convertable_types, convertable_distances_uri, gdal.GDT_Float32,
        convertable_type_nodata, pixel_size_out, "intersection",
        vectorize_op=False)
    forest_edge_expanded_uri = os.path.join(
        output_dir, 'forest_edge_expanded%s.tif' % file_suffix)
    pygeoprocessing.vectorize_datasets(
        [base_lulc_uri], lambda x: x, forest_edge_expanded_uri, gdal.GDT_Int32,
        lulc_nodata, pixel_size_out, "intersection", vectorize_op=False)

    #Convert all the closest to forest edge pixels to ag.
    pixel_area_ha = (
        pygeoprocessing.get_cell_size_from_uri(base_lulc_uri)**2 / 10000.0)
    max_pixels_to_convert = int(area_to_convert / pixel_area_ha)
    stats_cache = collections.defaultdict(int)
    _convert_by_score(
        convertable_distances_uri, max_pixels_to_convert,
        forest_edge_expanded_uri, ag_lucode, stats_cache)
    stats_uri = os.path.join(
        output_dir, 'forest_edge_expanded_stats%s.csv' % file_suffix)
    _log_stats(stats_cache, pixel_area_ha, stats_uri)


def _fragment_forest(
        base_lulc_uri, intermediate_dir, output_dir, file_suffix, ag_lucode,
        area_to_convert, forest_type_list, convertable_type_list,
        n_steps):
    """Expands agriculture into convertable types starting from the furthest
    distance from the edge of the forward, inward.

    Args:
        base_lulc_uri (string): path to landcover raster that will be used as
            the base landcover map to agriculture pixels
        intermediate_dir (string): path to a directory that is safe to write
            intermediate files
        output_dir (string): path to a directory that is safe to write output
            files
        file_suffix (string): string to append to output files
        ag_lucode (int): agriculture landcover code type found in the raster
            at `base_lulc_uri`
        area_to_convert (float): area (Ha) to convert to agriculture
        forest_type_list (list of int): landcover codes that are allowable
            to be converted to agriculture
        convertable_type_list (list of int): landcover codes that are allowable
            to be converted to agriculture
        n_steps (int): number of steps to convert the landscape; the higher this
            number the more accurate the fragmentation.  If this value equals
            `max_pixels_to_convert` one pixel will be converted per step. Note
            each step will require an expensive distance transform computation.

    Returns:
        None."""

    # create the output raster first so it can be looped on for each frag step
    forest_fragmented_uri = os.path.join(
        output_dir, 'forest_fragmented%s.tif' % file_suffix)
    lulc_nodata = pygeoprocessing.get_nodata_from_uri(base_lulc_uri)
    pixel_size_out = pygeoprocessing.get_cell_size_from_uri(base_lulc_uri)
    ag_mask_nodata = 2
    pygeoprocessing.vectorize_datasets(
        [base_lulc_uri], lambda x: x, forest_fragmented_uri, gdal.GDT_Int32,
        lulc_nodata, pixel_size_out, "intersection", vectorize_op=False)

    # convert everything furthest from edge for each of n_steps
    pixel_area_ha = (
        pygeoprocessing.get_cell_size_from_uri(base_lulc_uri)**2 / 10000.0)
    max_pixels_to_convert = int(area_to_convert / pixel_area_ha)
    convertable_type_nodata = -1
    pixels_left_to_convert = max_pixels_to_convert
    pixels_to_convert = max_pixels_to_convert / n_steps
    stats_cache = collections.defaultdict(int)
    for step_index in xrange(n_steps):
        LOGGER.info('fragmentation step %d of %d', step_index+1, n_steps)
        pixels_left_to_convert -= pixels_to_convert
        if pixels_left_to_convert < 0:
            pixels_to_convert += pixels_left_to_convert

        #mask agriculture types from LULC
        non_forest_mask_uri = os.path.join(
            intermediate_dir, 'non_forest_mask%s.tif' % file_suffix)
        def _mask_non_forest_op(lulc, converted_forest):
            """create a mask of valid non-forest pixels only"""
            non_forest_mask = ~numpy.in1d(
                lulc.flatten(), forest_type_list).reshape(lulc.shape)
            non_forest_mask = non_forest_mask | (converted_forest == ag_lucode)
            return numpy.where(
                lulc == lulc_nodata, ag_mask_nodata, non_forest_mask)
        pygeoprocessing.vectorize_datasets(
            [base_lulc_uri, forest_fragmented_uri], _mask_non_forest_op,
            non_forest_mask_uri, gdal.GDT_Byte, ag_mask_nodata, pixel_size_out,
            "intersection", vectorize_op=False)

        #distance transform mask
        distance_from_forest_edge_uri = os.path.join(
            intermediate_dir, 'distance_from_forest_edge_%d%s.tif' % (
                step_index, file_suffix))
        pygeoprocessing.distance_transform_edt(
            non_forest_mask_uri, distance_from_forest_edge_uri)

        convertable_distances_uri = os.path.join(
            intermediate_dir,
            'toward_forest_edge_convertable_distances%d%s.tif' % (
                step_index, file_suffix))
        def _mask_to_convertable_types(distance_from_forest_edge, lulc):
            """masks out the distance transform to a set of given landcover
            codes"""
            convertable_mask = numpy.in1d(
                lulc.flatten(), convertable_type_list).reshape(lulc.shape)
            return numpy.where(
                convertable_mask, distance_from_forest_edge,
                convertable_type_nodata)
        pygeoprocessing.vectorize_datasets(
            [distance_from_forest_edge_uri, base_lulc_uri],
            _mask_to_convertable_types, convertable_distances_uri,
            gdal.GDT_Float32, convertable_type_nodata, pixel_size_out,
            "intersection", vectorize_op=False)

        #Convert all the furthest from forest edge pixels to ag.
        _convert_by_score(
            convertable_distances_uri, pixels_to_convert,
            forest_fragmented_uri, ag_lucode, stats_cache, reverse_sort=True)

    stats_uri = os.path.join(
        output_dir, 'forest_fragmented_stats%s.csv' % file_suffix)
    _log_stats(stats_cache, pixel_area_ha, stats_uri)


def _log_stats(stats_cache, pixel_area, stats_uri):
    """Writes pixel change statistics from a simulation to disk in tabular
    format.

    Args:
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


def _sort_to_disk(dataset_uri, scale=1.0):
    """Sorts the non-nodata pixels in the dataset on disk and returns
    an iterable in sorted order.

    Args:
        dataset_uri (string): a path to a floating point GDAL dataset
        scale (float): a number to multiply all values by, which can be
            used to reverse the order of the iteration if negative.

    Returns:
        an iterable that produces (value * scale, flat_index) in decreasing
        sorted order by value * scale"""

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
    nodata *= scale # scale the nodata value so they can be filtered out later

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
        scores *= scale # scale the results
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
        stats_cache, reverse_sort=False, cache_size=50000):
    """Takes an input score layer and changes the pixels in `out_raster_uri`
    and converts up to `max_pixels_to_convert` them to `convert_value` type.

    Args:
        score_uri (string): path to a raster whose non-nodata values score the
            pixels to convert.  If `reverse_sort` is True the pixels in
            `out_raster_uri` are converted from the lowest score to the highest.
            The resverse is true if `reverse_sort` is False.
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
                to the value of that entry."""

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

    # shortcut to set valid scale
    scale = -1.0 if reverse_sort else 1.0

    # initialize the cache to cache_size large
    row_array = numpy.empty((cache_size,), dtype=numpy.uint32)
    col_array = numpy.empty((cache_size,), dtype=numpy.uint32)
    data_array = numpy.empty((cache_size,), dtype=numpy.bool)
    next_index = 0
    dirty_blocks = set()

    last_time = time.time()
    for _, flatindex in _sort_to_disk(score_uri, scale=scale):
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

    Args:
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
        kernel_uri.encode('utf-8'), kernel_size, kernel_size, 1, gdal.GDT_Float32,
        options=['BIGTIFF=IF_SAFER'])

    #Make some kind of geotransform, it doesn't matter what but
    #will make GIS libraries behave better if it's all defined
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
