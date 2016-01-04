# cython: profile=False

import logging
import os
import collections

import numpy
cimport numpy
cimport cython
import osgeo
from osgeo import gdal
from cython.operator cimport dereference as deref

from libcpp.set cimport set as c_set
from libcpp.deque cimport deque
from libcpp.map cimport map
from libc.math cimport atan
from libc.math cimport atan2
from libc.math cimport tan
from libc.math cimport sqrt
from libc.math cimport ceil
from libc.math cimport exp

cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

import pygeoprocessing
import pygeoprocessing.routing.routing_core

logging.basicConfig(format='%(asctime)s %(name)-18s %(levelname)-8s \
    %(message)s', lnevel=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('ndr core')

cdef double PI = 3.141592653589793238462643383279502884
cdef int N_BLOCK_ROWS = 8
cdef int N_BLOCK_COLS = 8

cdef class BlockCache:
    cdef numpy.int32_t[:,:] row_tag_cache
    cdef numpy.int32_t[:,:] col_tag_cache
    cdef numpy.int8_t[:,:] cache_dirty
    cdef int n_block_rows
    cdef int n_block_cols
    cdef int block_col_size
    cdef int block_row_size
    cdef int n_rows
    cdef int n_cols
    band_list = []
    block_list = []
    update_list = []

    def __cinit__(
            self, int n_block_rows, int n_block_cols, int n_rows, int n_cols,
            int block_row_size, int block_col_size, band_list, block_list,
            update_list, numpy.int8_t[:,:] cache_dirty):
        self.n_block_rows = n_block_rows
        self.n_block_cols = n_block_cols
        self.block_col_size = block_col_size
        self.block_row_size = block_row_size
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.row_tag_cache = numpy.zeros((n_block_rows, n_block_cols), dtype=numpy.int32)
        self.col_tag_cache = numpy.zeros((n_block_rows, n_block_cols), dtype=numpy.int32)
        self.cache_dirty = cache_dirty
        self.row_tag_cache[:] = -1
        self.col_tag_cache[:] = -1
        self.band_list[:] = band_list
        self.block_list[:] = block_list
        self.update_list[:] = update_list
        list_lengths = [len(x) for x in [band_list, block_list, update_list]]
        if len(set(list_lengths)) > 1:
            raise ValueError(
                "lengths of band_list, block_list, update_list should be equal."
                " instead they are %s", list_lengths)
        raster_dimensions_list = [(b.YSize, b.XSize) for b in band_list]
        for raster_n_rows, raster_n_cols in raster_dimensions_list:
            if raster_n_rows != n_rows or raster_n_cols != n_cols:
                raise ValueError(
                    "A band was passed in that has a different dimension than "
                    "the memory block was specified as.\n"
                    "raster_dimensions_list=%s\n" % str(raster_dimensions_list)
                    + "bands=%s" % str([b.GetDescription() for b in band_list]))

        for band in band_list:
            block_col_size, block_row_size = band.GetBlockSize()
            if block_col_size == 1 or block_row_size == 1:
                LOGGER.warn(
                    'a band in BlockCache is not memory blocked, this might '
                    'make the runtime slow for other algorithms. %s',
                    band.GetDescription())

    def __dealloc__(self):
        self.band_list[:] = []
        self.block_list[:] = []
        self.update_list[:] = []
        self.row_tag_cache = None
        self.col_tag_cache = None
        self.cache_dirty = None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void update_cache(self, int global_row, int global_col, int *row_index, int *col_index, int *row_block_offset, int *col_block_offset):
        cdef int cache_row_size, cache_col_size
        cdef int global_row_offset, global_col_offset
        cdef int row_tag, col_tag

        row_block_offset[0] = global_row % self.block_row_size
        row_index[0] = (global_row // self.block_row_size) % self.n_block_rows
        row_tag = (global_row // self.block_row_size) // self.n_block_rows

        col_block_offset[0] = global_col % self.block_col_size
        col_index[0] = (global_col // self.block_col_size) % self.n_block_cols
        col_tag = (global_col // self.block_col_size) // self.n_block_cols

        cdef int current_row_tag = self.row_tag_cache[row_index[0], col_index[0]]
        cdef int current_col_tag = self.col_tag_cache[row_index[0], col_index[0]]

        if current_row_tag != row_tag or current_col_tag != col_tag:
            if self.cache_dirty[row_index[0], col_index[0]]:
                global_col_offset = (current_col_tag * self.n_block_cols + col_index[0]) * self.block_col_size
                cache_col_size = self.n_cols - global_col_offset
                if cache_col_size > self.block_col_size:
                    cache_col_size = self.block_col_size

                global_row_offset = (current_row_tag * self.n_block_rows + row_index[0]) * self.block_row_size
                cache_row_size = self.n_rows - global_row_offset
                if cache_row_size > self.block_row_size:
                    cache_row_size = self.block_row_size

                for band, block, update in zip(self.band_list, self.block_list, self.update_list):
                    if update:
                        band.WriteArray(block[row_index[0], col_index[0], 0:cache_row_size, 0:cache_col_size],
                            yoff=global_row_offset, xoff=global_col_offset)
                self.cache_dirty[row_index[0], col_index[0]] = 0
            self.row_tag_cache[row_index[0], col_index[0]] = row_tag
            self.col_tag_cache[row_index[0], col_index[0]] = col_tag

            global_col_offset = (col_tag * self.n_block_cols + col_index[0]) * self.block_col_size
            global_row_offset = (row_tag * self.n_block_rows + row_index[0]) * self.block_row_size

            cache_col_size = self.n_cols - global_col_offset
            if cache_col_size > self.block_col_size:
                cache_col_size = self.block_col_size
            cache_row_size = self.n_rows - global_row_offset
            if cache_row_size > self.block_row_size:
                cache_row_size = self.block_row_size

            for band, block in zip(self.band_list, self.block_list):
                band.ReadAsArray(
                    xoff=global_col_offset, yoff=global_row_offset,
                    win_xsize=cache_col_size, win_ysize=cache_row_size,
                    buf_obj=block[row_index[0], col_index[0], 0:cache_row_size, 0:cache_col_size])

    cdef void flush_cache(self):
        cdef int global_row_offset, global_col_offset
        cdef int cache_row_size, cache_col_size
        cdef int row_index, col_index
        for row_index in xrange(self.n_block_rows):
            for col_index in xrange(self.n_block_cols):
                row_tag = self.row_tag_cache[row_index, col_index]
                col_tag = self.col_tag_cache[row_index, col_index]

                if self.cache_dirty[row_index, col_index]:
                    global_col_offset = (col_tag * self.n_block_cols + col_index) * self.block_col_size
                    cache_col_size = self.n_cols - global_col_offset
                    if cache_col_size > self.block_col_size:
                        cache_col_size = self.block_col_size

                    global_row_offset = (row_tag * self.n_block_rows + row_index) * self.block_row_size
                    cache_row_size = self.n_rows - global_row_offset
                    if cache_row_size > self.block_row_size:
                        cache_row_size = self.block_row_size

                    for band, block, update in zip(self.band_list, self.block_list, self.update_list):
                        if update:
                            band.WriteArray(block[row_index, col_index, 0:cache_row_size, 0:cache_col_size],
                                yoff=global_row_offset, xoff=global_col_offset)
        for band in self.band_list:
            band.FlushCache()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ndr_eff_calculation(
    flow_direction_uri, stream_uri, retention_eff_lulc_uri, crit_len_uri,
    effective_retention_uri):

    """This function calculates the flow downhill effective_retention to the stream layers

        Args:
            flow_direction_uri (string) - (input) a path to a raster with
                d-infinity flow directions.
            stream_uri (string) - (input) a raster where 1 indicates a stream
                all other values ignored must be same dimensions and projection
                as flow_direction_uri.
            retention_eff_lulc_uri (string) - (input) a raster indicating the
                maximum retention efficiency that the landcover on that pixel
                can accumulate.
            crit_len_uri (string) - (input) a raster indicating the critical length
                of the retention efficiency that the landcover on this pixel.

            effective_retention_uri (string) - (output) a raster showing
                the effective retention on that pixel to the stream.

        Returns:
            nothing"""

    cdef float effective_retention_nodata = -9999
    pygeoprocessing.new_raster_from_base_uri(
        flow_direction_uri, effective_retention_uri, 'GTiff', effective_retention_nodata,
        gdal.GDT_Float32, fill_value=effective_retention_nodata)

    cdef float processed_cell_nodata = 127
    processed_cell_uri = (
        os.path.join(os.path.dirname(flow_direction_uri), 'processed_cell.tif'))
    pygeoprocessing.new_raster_from_base_uri(
        flow_direction_uri, processed_cell_uri, 'GTiff', processed_cell_nodata,
        gdal.GDT_Byte, fill_value=0)

    processed_cell_ds = gdal.Open(processed_cell_uri, gdal.GA_Update)
    processed_cell_band = processed_cell_ds.GetRasterBand(1)

    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]
    cdef int *inflow_offsets = [4, 5, 6, 7, 0, 1, 2, 3]

    cdef int n_rows, n_cols
    n_rows, n_cols = pygeoprocessing.get_row_col_from_uri(
        flow_direction_uri)

    cdef deque[int] visit_stack

    stream_ds = gdal.Open(stream_uri)
    stream_band = stream_ds.GetRasterBand(1)
    cdef float stream_nodata = pygeoprocessing.get_nodata_from_uri(
        stream_uri)
    cdef float cell_size = pygeoprocessing.get_cell_size_from_uri(stream_uri)

    effective_retention_ds = gdal.Open(effective_retention_uri, gdal.GA_Update)
    effective_retention_band = effective_retention_ds.GetRasterBand(1)

    retention_eff_lulc_ds = gdal.Open(retention_eff_lulc_uri)
    retention_eff_lulc_band = retention_eff_lulc_ds.GetRasterBand(1)

    crit_len_ds = gdal.Open(crit_len_uri)
    crit_len_band = crit_len_ds.GetRasterBand(1)

    outflow_weights_uri = pygeoprocessing.temporary_filename()
    outflow_direction_uri = pygeoprocessing.temporary_filename()
    pygeoprocessing.routing.routing_core.calculate_flow_weights(
        flow_direction_uri, outflow_weights_uri, outflow_direction_uri)
    outflow_weights_ds = gdal.Open(outflow_weights_uri)
    outflow_weights_band = outflow_weights_ds.GetRasterBand(1)
    cdef float outflow_weights_nodata = pygeoprocessing.get_nodata_from_uri(
        outflow_weights_uri)
    outflow_direction_ds = gdal.Open(outflow_direction_uri)
    outflow_direction_band = outflow_direction_ds.GetRasterBand(1)
    cdef int outflow_direction_nodata = pygeoprocessing.get_nodata_from_uri(
        outflow_direction_uri)
    cdef int block_col_size, block_row_size
    block_col_size, block_row_size = stream_band.GetBlockSize()
    cdef int n_global_block_rows = int(ceil(float(n_rows) / block_row_size))
    cdef int n_global_block_cols = int(ceil(float(n_cols) / block_col_size))

    cdef numpy.ndarray[numpy.npy_float32, ndim=4] stream_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
        dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_int8, ndim=4] outflow_direction_block = (
        numpy.zeros(
            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
            dtype=numpy.int8))
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] outflow_weights_block = (
        numpy.zeros(
            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
            dtype=numpy.float32))
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] effective_retention_block = (
        numpy.zeros(
            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
            dtype=numpy.float32))
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] retention_eff_lulc_block = (
        numpy.zeros(
            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
            dtype=numpy.float32))
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] crit_len_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
        dtype=numpy.float32)


    cdef numpy.ndarray[numpy.npy_int8, ndim=4] processed_cell_block = (
        numpy.zeros(
            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
            dtype=numpy.int8))

    band_list = [stream_band, outflow_direction_band, outflow_weights_band,
                 effective_retention_band, processed_cell_band,
                 retention_eff_lulc_band, crit_len_band]
    block_list = [stream_block, outflow_direction_block, outflow_weights_block,
                  effective_retention_block, processed_cell_block,
                  retention_eff_lulc_block, crit_len_block]
    update_list = [False, False, False, True, True, False, False]

    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = (
        numpy.zeros((N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte))

    cdef BlockCache block_cache = BlockCache(
        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size,
        block_col_size, band_list, block_list, update_list, cache_dirty)

    #center point of global index
    cdef int global_row, global_col
    cdef int row_index, col_index
    cdef int row_block_offset, col_block_offset
    cdef int global_block_row, global_block_col

    #neighbor sections of global index
    cdef int neighbor_row, neighbor_col
    cdef int neighbor_row_index, neighbor_col_index
    cdef int neighbor_row_block_offset, neighbor_col_block_offset
    cdef int flat_index


    cdef c_set[int] cells_in_queue

    #build up the stream pixel indexes as starting seed points for the search
    cdef time_t last_time, current_time
    time(&last_time)
    for global_block_row in xrange(n_global_block_rows):
        time(&current_time)
        if current_time - last_time > 5.0:
            LOGGER.info(
                "find_sinks %.1f%% complete",
                (global_block_row + 1.0) / n_global_block_rows * 100)
            last_time = current_time
        for global_block_col in xrange(n_global_block_cols):
            for global_row in xrange(
                    global_block_row*block_row_size,
                    min((global_block_row+1)*block_row_size, n_rows)):
                for global_col in xrange(
                        global_block_col*block_col_size,
                        min((global_block_col+1)*block_col_size, n_cols)):
                    block_cache.update_cache(
                        global_row, global_col, &row_index, &col_index,
                        &row_block_offset, &col_block_offset)
                    if stream_block[
                            row_index, col_index, row_block_offset,
                            col_block_offset] == 1:
                        flat_index = global_row * n_cols + global_col
                        visit_stack.push_front(global_row * n_cols + global_col)
                        cells_in_queue.insert(flat_index)

                        effective_retention_block[row_index, col_index,
                            row_block_offset, col_block_offset] = 0
                        processed_cell_block[row_index, col_index,
                            row_block_offset, col_block_offset] = 1
                        cache_dirty[row_index, col_index] = 1

    cdef int neighbor_outflow_direction, neighbor_index, outflow_direction
    cdef float neighbor_outflow_weight, current_effective_retention
    cdef float outflow_weight, neighbor_effective_retention, step_size
    cdef float downstream_effective_retention, current_stream
    cdef float original_effective_retention
    cdef float retention_eff_lulc, crit_len, intermediate_retention
    cdef float current_step_factor
    cdef int it_flows_here

    while visit_stack.size() > 0:
        flat_index = visit_stack.front()
        visit_stack.pop_front()
        cells_in_queue.erase(flat_index)
        global_row = flat_index / n_cols
        global_col = flat_index % n_cols

        block_cache.update_cache(
            global_row, global_col, &row_index, &col_index,
            &row_block_offset, &col_block_offset)

        update_downstream = False
        current_effective_retention = 0.0

        time(&current_time)
        if current_time - last_time > 0.0001:
            last_time = current_time
            LOGGER.info(
                'visit_stack on stream effective_retention size: %d ', visit_stack.size())

        current_stream = stream_block[
            row_index, col_index, row_block_offset, col_block_offset]
        outflow_direction = outflow_direction_block[
            row_index, col_index, row_block_offset,
            col_block_offset]
        if current_stream == 1:
            effective_retention_block[row_index, col_index,
                row_block_offset, col_block_offset] = 0
            processed_cell_block[row_index, col_index,
                row_block_offset, col_block_offset] = 1
            cache_dirty[row_index, col_index] = 1
        elif outflow_direction == outflow_direction_nodata:
            current_effective_retention = 1.0
        elif processed_cell_block[row_index, col_index, row_block_offset,
                col_block_offset] == 0:
            #add downstream effective_retention to current effective_retention

            outflow_weight = outflow_weights_block[
                row_index, col_index, row_block_offset,
                col_block_offset]

            retention_eff_lulc = retention_eff_lulc_block[
                row_index, col_index, row_block_offset,
                col_block_offset]

            crit_len = crit_len_block[
                row_index, col_index, row_block_offset,
                col_block_offset]

            for neighbor_index in xrange(2):
                #check if downstream neighbors are calcualted
                if neighbor_index == 1:
                    outflow_direction = (outflow_direction + 1) % 8
                    outflow_weight = (1.0 - outflow_weight)

                if outflow_weight <= 0.0:
                    continue

                neighbor_row = global_row + row_offsets[outflow_direction]
                neighbor_col = global_col + col_offsets[outflow_direction]
                if (neighbor_row < 0 or neighbor_row >= n_rows or
                        neighbor_col < 0 or neighbor_col >= n_cols):
                    #out of bounds
                    continue

                block_cache.update_cache(
                    neighbor_row, neighbor_col, &neighbor_row_index,
                    &neighbor_col_index, &neighbor_row_block_offset,
                    &neighbor_col_block_offset)

                if stream_block[neighbor_row_index,
                        neighbor_col_index, neighbor_row_block_offset,
                        neighbor_col_block_offset] == stream_nodata:
                    #out of the valid raster entirely
                    continue

                neighbor_effective_retention = effective_retention_block[
                    neighbor_row_index, neighbor_col_index,
                    neighbor_row_block_offset, neighbor_col_block_offset]

                neighbor_outflow_direction = outflow_direction_block[
                    neighbor_row_index, neighbor_col_index,
                    neighbor_row_block_offset, neighbor_col_block_offset]

                neighbor_outflow_weight = outflow_weights_block[
                    neighbor_row_index, neighbor_col_index,
                    neighbor_row_block_offset, neighbor_col_block_offset]

                if processed_cell_block[neighbor_row_index, neighbor_col_index,
                        neighbor_row_block_offset,
                        neighbor_col_block_offset] == 0:
                    neighbor_flat_index = neighbor_row * n_cols + neighbor_col
                    #insert into the processing queue if it's not already there
                    if (cells_in_queue.find(flat_index) ==
                            cells_in_queue.end()):
                        visit_stack.push_back(flat_index)
                        cells_in_queue.insert(flat_index)

                    if (cells_in_queue.find(neighbor_flat_index) ==
                            cells_in_queue.end()):
                        visit_stack.push_front(neighbor_flat_index)
                        cells_in_queue.insert(neighbor_flat_index)

                    update_downstream = True
                    neighbor_effective_retention = 0.0

                if outflow_direction % 2 == 1:
                    #increase effective_retention by a square root of 2 for diagonal
                    step_size = cell_size * 1.41421356237
                else:
                    step_size = cell_size

                current_step_factor = exp(-5 * step_size / crit_len)
                if neighbor_effective_retention >= retention_eff_lulc:
                    current_effective_retention += (
                        neighbor_effective_retention) * outflow_weight
                else:
                    intermediate_retention =  (
                        neighbor_effective_retention * current_step_factor +
                        retention_eff_lulc * (1 - current_step_factor))
                    if intermediate_retention > retention_eff_lulc:
                        intermediate_retention = retention_eff_lulc
                    current_effective_retention += (
                        intermediate_retention * outflow_weight)

        if not update_downstream:
            #mark flat_index as processed
            block_cache.update_cache(
                global_row, global_col, &row_index, &col_index,
                &row_block_offset, &col_block_offset)
            processed_cell_block[row_index, col_index,
                row_block_offset, col_block_offset] = 1
            effective_retention_block[row_index, col_index,
                row_block_offset, col_block_offset] = (
                    current_effective_retention)
            cache_dirty[row_index, col_index] = 1

            #update any upstream neighbors with this effective_retention
            for neighbor_index in range(8):
                neighbor_row = global_row + row_offsets[neighbor_index]
                neighbor_col = global_col + col_offsets[neighbor_index]
                if (neighbor_row < 0 or neighbor_row >= n_rows or
                        neighbor_col < 0 or neighbor_col >= n_cols):
                    #out of bounds
                    continue

                block_cache.update_cache(
                    neighbor_row, neighbor_col, &neighbor_row_index,
                    &neighbor_col_index, &neighbor_row_block_offset,
                    &neighbor_col_block_offset)

                #streams were already added, skip if they are in the queue
                if (stream_block[neighbor_row_index, neighbor_col_index,
                        neighbor_row_block_offset,
                        neighbor_col_block_offset] == 1 or
                    stream_block[neighbor_row_index, neighbor_col_index,
                        neighbor_row_block_offset,
                        neighbor_col_block_offset] == stream_nodata):
                    continue

                if processed_cell_block[
                        neighbor_row_index,
                        neighbor_col_index,
                        neighbor_row_block_offset,
                        neighbor_col_block_offset] == 1:
                    #don't reprocess it, it's already been updated by two valid
                    #children
                    continue

                neighbor_outflow_direction = outflow_direction_block[
                    neighbor_row_index, neighbor_col_index,
                    neighbor_row_block_offset, neighbor_col_block_offset]
                if neighbor_outflow_direction == outflow_direction_nodata:
                    #if the neighbor has no flow, we can't flow here
                    continue

                neighbor_outflow_weight = outflow_weights_block[
                    neighbor_row_index, neighbor_col_index,
                    neighbor_row_block_offset, neighbor_col_block_offset]

                it_flows_here = False
                if (neighbor_outflow_direction ==
                        inflow_offsets[neighbor_index]):
                    it_flows_here = True
                elif ((neighbor_outflow_direction + 1) % 8 ==
                        inflow_offsets[neighbor_index]):
                    it_flows_here = True
                    neighbor_outflow_weight = 1.0 - neighbor_outflow_weight

                neighbor_flat_index = neighbor_row * n_cols + neighbor_col
                if (it_flows_here and neighbor_outflow_weight > 0.0 and
                    cells_in_queue.find(neighbor_flat_index) ==
                        cells_in_queue.end()):
                    visit_stack.push_back(neighbor_flat_index)
                    cells_in_queue.insert(neighbor_flat_index)

    block_cache.flush_cache()

    for dataset in [outflow_weights_ds, outflow_direction_ds]:
        gdal.Dataset.__swig_destroy__(dataset)
    for dataset_uri in [outflow_weights_uri, outflow_direction_uri]:
        os.remove(dataset_uri)
