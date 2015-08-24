"""Cropland Expansion Tool"""

import os
import logging
import numpy
import tempfile
import struct
import heapq
import time

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

    if args['expand_from_ag']:
        _expand_from_ag(
            args, intermediate_dir, output_dir, file_suffix, ag_lucode,
            max_pixels_to_convert)

    if args['expand_from_forest_edge']:
        _expand_from_forest_edge(args)

    if args['fragment_forest']:
        _fragment_forest(args)

def _expand_from_ag(
        base_lulc_uri, intermediate_dir, output_dir, file_suffix, ag_lucode,
        max_pixels_to_convert):
    """ """
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
        [args['base_lulc_uri']], _mask_ag_op, ag_mask_uri, gdal.GDT_Byte,
        ag_mask_nodata, pixel_size_out, "intersection", vectorize_op=False,
        assert_datasets_projected=False)

    #distance transform mask
    distance_from_ag_uri = os.path.join(
        intermediate_dir, 'distance_from_ag%s.tif' % file_suffix)
    pygeoprocessing.distance_transform_edt(ag_mask_uri, distance_from_ag_uri)

    #mask out distance transform for everything that can be converted
    convertable_type_list = numpy.array([
        int(x) for x in args['convertable_landcover_types'].split()])

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
        [distance_from_ag_uri, args['base_lulc_uri']],
        _mask_to_convertable_types, convertable_distances_uri, gdal.GDT_Float32,
        convertable_type_nodata, pixel_size_out, "intersection",
        vectorize_op=False, assert_datasets_projected=False)

    ag_expanded_uri = os.path.join(
        output_dir, 'ag_expanded%s.tif' % file_suffix)

    pygeoprocessing.new_raster_from_base_uri(
        args['base_lulc_uri'], ag_expanded_uri, 'GTiff', lulc_nodata,
        gdal.GDT_Int32, fill_value=int(lulc_nodata))

    ag_expanded_ds = gdal.Open(ag_expanded_uri, gdal.GA_Update)
    ag_expanded_band = ag_expanded_ds.GetRasterBand(1)

    n_cols = ag_expanded_band.XSize

    #disk sort to select the top N pixels to convert
    count = 0
    last_time = time.time()
    ag_lucode_array = numpy.array([[ag_lucode]])
    for _, flatindex in _sort_to_disk(convertable_distances_uri):
        if count >= max_pixels_to_convert:
            break
        col_index = flatindex % n_cols
        row_index = flatindex / n_cols
        ag_expanded_band.WriteArray(ag_lucode_array, col_index, row_index)
        count += 1
        if time.time() - last_time > 5.0:
            LOGGER.info(
                "converted %d of %d pixels", count, max_pixels_to_convert)
            last_time = time.time()


def _expand_from_forest_edge(args):
    """ """
    pass


def _fragment_forest(args):
    """ """
    pass


def _sort_to_disk(dataset_uri):
    """Sorts the non-nodata pixels in the dataset on disk and returns
        an iterable in sorted order.

        dataset_uri - a uri to a GDAL dataset

        returns an iterable that returns (value, flat_index)
           in decreasing sorted order by value"""

    def _read_score_index_from_disk(f):
        while True:
            #TODO: better buffering here
            packed_score = f.read(8)
            if not packed_score:
                break
            yield struct.unpack('fi', packed_score)

    dataset = gdal.Open(dataset_uri)
    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()

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
        f = tempfile.TemporaryFile()
        for score, index in zip(sorted_scores, sorted_indexes):
            f.write(struct.pack('fi', score, index))

        #Reset the file pointer and add an iterator for it to the list
        f.seek(0)
        iters.append(_read_score_index_from_disk(f))

    return heapq.merge(*iters)
