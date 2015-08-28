"""Cropland Expansion Tool"""

import os
import logging
import numpy
import tempfile
import struct
import heapq
import time
import atexit

import gdal
import pygeoprocessing

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
        args['max_pixels_to_convert'] (string or int): max pixels to convert per
        args['forest_landcover_types'] (string): a space separated string of
            forest landcover codes found in `args['base_lulc_uri']`
        args['n_fragmentation_steps'] (string): an int as a string indicating
            the number of steps to take for the fragmentation conversion

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

    max_pixels_to_convert = int(args['max_pixels_to_convert'])
    ag_lucode = int(args['agriculture_lu_code'])

    #create working directories
    output_dir = os.path.join(args['workspace_dir'], 'output')
    intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')
    tmp_dir = os.path.join(args['workspace_dir'], 'tmp')

    pygeoprocessing.geoprocessing.create_directories(
        [output_dir, intermediate_dir, tmp_dir])

    #mask out distance transform for everything that can be converted
    convertable_type_list = numpy.array([
        int(x) for x in args['convertable_landcover_types'].split()])

    forest_type_list = numpy.array([
        int(x) for x in args['forest_landcover_types'].split()])

    if args['expand_from_ag']:
        _expand_from_ag(
            args['base_lulc_uri'], intermediate_dir, output_dir, file_suffix,
            ag_lucode, max_pixels_to_convert, convertable_type_list)

    if args['expand_from_forest_edge']:
        _expand_from_forest_edge(
            args['base_lulc_uri'], intermediate_dir, output_dir, file_suffix,
            ag_lucode, max_pixels_to_convert, forest_type_list,
            convertable_type_list)

    if args['fragment_forest']:
        _fragment_forest(
            args['base_lulc_uri'], intermediate_dir, output_dir,
            file_suffix, ag_lucode, max_pixels_to_convert, forest_type_list,
            convertable_type_list, int(args['n_fragmentation_steps']))

def _expand_from_ag(
        base_lulc_uri, intermediate_dir, output_dir, file_suffix, ag_lucode,
        max_pixels_to_convert, convertable_type_list):
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
        max_pixels_to_convert (int): number of pixels to convert to agriculture
        convertable_type_list (list of int): landcover codes that are allowable
            to be converted to agriculture

    Returns:
        None.
    """
    #mask agriculture types from LULC
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
        ag_mask_nodata, pixel_size_out, "intersection", vectorize_op=False,
        assert_datasets_projected=False)

    #distance transform mask
    distance_from_ag_uri = os.path.join(
        intermediate_dir, 'distance_from_ag%s.tif' % file_suffix)
    pygeoprocessing.distance_transform_edt(ag_mask_uri, distance_from_ag_uri)

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
        [distance_from_ag_uri, base_lulc_uri],
        _mask_to_convertable_types, convertable_distances_uri, gdal.GDT_Float32,
        convertable_type_nodata, pixel_size_out, "intersection",
        vectorize_op=False, assert_datasets_projected=False)

    ag_expanded_uri = os.path.join(
        output_dir, 'ag_expanded%s.tif' % file_suffix)

    pygeoprocessing.new_raster_from_base_uri(
        base_lulc_uri, ag_expanded_uri, 'GTiff', lulc_nodata,
        gdal.GDT_Int32, fill_value=int(lulc_nodata))

    #Convert all the closest to edge pixels to ag.
    _convert_by_score(
        convertable_distances_uri, max_pixels_to_convert, ag_expanded_uri,
        ag_lucode)


def _expand_from_forest_edge(
        base_lulc_uri, intermediate_dir, output_dir, file_suffix, ag_lucode,
        max_pixels_to_convert, forest_type_list, convertable_type_list):
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
        max_pixels_to_convert (int): number of pixels to convert to agriculture
        forest_type_list (list of int): landcover codes that are allowable
            to be converted to agriculture
        convertable_type_list (list of int): landcover codes that are allowable
            to be converted to agriculture

    Returns:
        None."""
    #mask agriculture types from LULC
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
        vectorize_op=False, assert_datasets_projected=False)

    #distance transform mask
    distance_from_forest_edge_uri = os.path.join(
        intermediate_dir, 'distance_from_forest_edge%s.tif' % file_suffix)
    pygeoprocessing.distance_transform_edt(
        non_forest_mask_uri, distance_from_forest_edge_uri)

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
        [distance_from_forest_edge_uri, base_lulc_uri],
        _mask_to_convertable_types, convertable_distances_uri, gdal.GDT_Float32,
        convertable_type_nodata, pixel_size_out, "intersection",
        vectorize_op=False, assert_datasets_projected=False)

    forest_edge_expanded_uri = os.path.join(
        output_dir, 'forest_edge_expanded%s.tif' % file_suffix)

    pygeoprocessing.new_raster_from_base_uri(
        base_lulc_uri, forest_edge_expanded_uri, 'GTiff', lulc_nodata,
        gdal.GDT_Int32, fill_value=int(lulc_nodata))

    #Convert all the closest to forest edge pixels to ag.
    _convert_by_score(
        convertable_distances_uri, max_pixels_to_convert,
        forest_edge_expanded_uri, ag_lucode)


def _fragment_forest(
        base_lulc_uri, intermediate_dir, output_dir, file_suffix, ag_lucode,
        max_pixels_to_convert, forest_type_list, convertable_type_list,
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
        max_pixels_to_convert (int): number of pixels to convert to agriculture
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

    forest_fragmented_uri = os.path.join(
        output_dir, 'forest_fragmented%s.tif' % file_suffix)

    lulc_nodata = pygeoprocessing.get_nodata_from_uri(base_lulc_uri)
    pixel_size_out = pygeoprocessing.get_cell_size_from_uri(base_lulc_uri)
    ag_mask_nodata = 2

    pygeoprocessing.new_raster_from_base_uri(
        base_lulc_uri, forest_fragmented_uri, 'GTiff', lulc_nodata,
        gdal.GDT_Int32, fill_value=int(lulc_nodata))

    convertable_type_nodata = -1
    pixels_left_to_convert = max_pixels_to_convert
    pixels_to_convert = max_pixels_to_convert / n_steps
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
            "intersection", vectorize_op=False, assert_datasets_projected=False)

        #distance transform mask
        distance_from_forest_edge_uri = os.path.join(
            intermediate_dir, 'distance_from_forest_edge_%d%s.tif' % (
                step_index, file_suffix))
        pygeoprocessing.distance_transform_edt(
            non_forest_mask_uri, distance_from_forest_edge_uri)

        convertable_distances_uri = os.path.join(
            intermediate_dir, 'toward_forest_edge_convertable_distances%d%s.tif'
            % (step_index, file_suffix))
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
            "intersection", vectorize_op=False,
            assert_datasets_projected=False)

        #Convert all the furthest from forest edge pixels to ag.
        _convert_by_score(
            convertable_distances_uri, max_pixels_to_convert,
            forest_fragmented_uri, ag_lucode, reverse_sort=True)


def _sort_to_disk(dataset_uri, scale=1.0):
    """Sorts the non-nodata pixels in the dataset on disk and returns
        an iterable in sorted order.

        dataset_uri - a uri to a GDAL dataset
        scale - a number to multiply all values by, this can be used to reverse
            the sort order for example

        returns an iterable that returns (value, flat_index)
           in decreasing sorted order by value"""


    def _read_score_index_from_disk(file_name, buffer_size=8*10000):
        """Generator to yield a float/int value from a file, does buffering
        and file managment to avoid keeping file open while function is not
        invoked"""

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

        #Reset the file pointer and add an iterator for it to the list
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
        reverse_sort=False):
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
        reverse_sort (boolean): If true, pixels are visited in descreasing order
            of `score_uri`, otherwise increasing.

    Returns:
        None.
    """

    out_ds = gdal.Open(out_raster_uri, gdal.GA_Update)
    out_band = out_ds.GetRasterBand(1)

    _, n_cols = pygeoprocessing.get_row_col_from_uri(score_uri)
    count = 0
    convert_value_array = numpy.array([[convert_value]])

    scale = -1.0 if reverse_sort else 1.0
    last_time = time.time()
    for _, flatindex in _sort_to_disk(score_uri, scale=scale):
        if count >= max_pixels_to_convert:
            break
        col_index = flatindex % n_cols
        row_index = flatindex / n_cols
        out_band.WriteArray(convert_value_array, col_index, row_index)
        count += 1
        if time.time() - last_time > 5.0:
            LOGGER.info(
                "converted %d of %d pixels", count, max_pixels_to_convert)
            last_time = time.time()
