import tempfile
import struct
import heapq

from osgeo import gdal
import numpy


def sort_to_disk(dataset_uri, dataset_index):
    """Sorts the non-nodata pixels in the dataset on disk and returns
        an iterable in sorted order.

        dataset_uri - a uri to a GDAL dataset
        dataset_index - an integer to keep track of which dataset
            we're encoding.  This will help us later if we merge
            several of these iterators together

        returns an iterable that returns (-value, flat_index, dataset_index)
           in decreasing sorted order by -value"""

    def _read_score_index_from_disk(f):
        while True:
            #TODO: better buffering here
            packed_score = f.read(8)
            if not packed_score:
                break
            yield struct.unpack('fi', packed_score) + (dataset_index,)

    dataset = gdal.Open(dataset_uri)
    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    n_rows = band.YSize
    n_cols = band.XSize

    #This will be a list of file iterators we'll pass to heap.merge
    iters = []

    #Set the row strides to be something reasonable, like 1MB blocks
    row_strides = max(int(2**20 / (4 * n_cols)), 1)
    
    for row_index in xrange(0, n_rows, row_strides):

        #It's possible we're on the last set of rows and the stride is too big
        #update if so
        if row_index + row_strides >= n_rows:
            row_strides = n_rows - row_index

        #Extract scores make them negative, calculate flat indexes, and sort
        scores = -band.ReadAsArray(0,row_index,n_cols,row_strides).flatten()

        col_indexes = numpy.tile(numpy.arange(n_cols), (row_strides, 1))
        row_offsets = numpy.arange(row_index, row_index+row_strides) * n_cols
        row_offsets.resize((row_strides, 1))

        flat_indexes = (col_indexes + row_offsets).flatten()

        sort_index = scores.argsort()
        sorted_scores = scores[sort_index]
        sorted_indexes = flat_indexes[sort_index]

        #Determine where the nodata values are so we can splice them out
        left_index = numpy.searchsorted(sorted_scores,-nodata,side='left')
        right_index = numpy.searchsorted(sorted_scores,-nodata,side='right')

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
