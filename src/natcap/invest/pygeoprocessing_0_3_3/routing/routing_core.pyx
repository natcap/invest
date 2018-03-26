# cython: profile=False

import logging
import os
import collections
import tempfile
import shutil

import numpy
cimport numpy
cimport cython
import osgeo
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from cython.operator cimport dereference as deref

from libcpp.set cimport set as c_set
from libcpp.deque cimport deque
from libcpp.map cimport map
from libcpp.stack cimport stack
from libc.math cimport atan
from libc.math cimport atan2
from libc.math cimport tan
from libc.math cimport sqrt
from libc.math cimport ceil

cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

from .. import geoprocessing

LOGGER = logging.getLogger('natcap.invest.pygeoprocessing_0_3_3.routing.routing_core')

cdef double PI = 3.141592653589793238462643383279502884
cdef double INF = numpy.inf
cdef int N_BLOCK_ROWS = 16
cdef int N_BLOCK_COLS = 16

cdef class BlockCache:
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
                    "a band was passed in that has a different dimension than"
                    "the memory block was specified as")

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
        self.row_tag_cache[:] = -1
        self.col_tag_cache[:] = -1
        self.cache_dirty[:] = 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef calculate_transport(
        outflow_direction_uri, outflow_weights_uri, deque[int] &sink_cell_deque,
        source_uri, absorption_rate_uri, loss_uri, flux_uri, absorption_mode,
        stream_uri=None, include_source=1):
    """This is a generalized flux transport algorithm that operates
        on a 2D grid given a per pixel flow direction, per pixel source,
        and per pixel absorption rate.  It produces a grid of loss per
        pixel, and amount of outgoing flux per pixel.

        outflow_direction_uri - a uri to a byte dataset that indicates the
            first counter clockwise outflow neighbor as an index from the
            following diagram

            3 2 1
            4 x 0
            5 6 7

        outflow_weights_uri - a uri to a float32 dataset whose elements
            correspond to the percent outflow from the current cell to its
            first counter-clockwise neighbor
        sink_cell_deque (deque[int])- a deque of flat integer indexes for the
            cells in flow graph that have no outflow
        source_uri - a GDAL dataset that has source flux per pixel
        absorption_rate_uri - a GDAL floating point dataset that has a percent
            of flux absorbed per pixel
        loss_uri - an output URI to to the dataset that will output the
            amount of flux absorbed by each pixel
        flux_uri - a URI to an output dataset that records the amount of flux
            travelling through each pixel
        absorption_mode - either 'flux_only' or 'source_and_flux'. For
            'flux_only' the outgoing flux is (in_flux * absorption + source).
            If 'source_and_flux' then the output flux
            is (in_flux + source) * absorption.
        include_source - if True, source is added to current pixel, otherwise
            pixel starts at 0.
        stream_uri - (optional) a raster to a stream classification layer that
            if 1 indicates a stream 0 if not.  If flux hits a stream the total
            flux is set to zero so that it can't be further routed out of the
            stream if it diverges later.

        returns nothing"""

    #Calculate flow graph

    #Pass transport
    cdef time_t start
    time(&start)

    #Create output arrays for loss and flux
    outflow_direction_dataset = gdal.OpenEx(outflow_direction_uri)
    cdef int n_cols = outflow_direction_dataset.RasterXSize
    cdef int n_rows = outflow_direction_dataset.RasterYSize
    outflow_direction_band = outflow_direction_dataset.GetRasterBand(1)

    cdef int block_col_size, block_row_size
    block_col_size, block_row_size = outflow_direction_band.GetBlockSize()

    #center point of global index
    cdef int global_row, global_col #index into the overall raster
    cdef int row_index, col_index #the index of the cache block
    cdef int row_block_offset, col_block_offset #index into the cache block
    cdef int global_block_row, global_block_col #used to walk the global blocks

    #neighbor sections of global index
    cdef int neighbor_row, neighbor_col #neighbor equivalent of global_{row,col}
    cdef int neighbor_row_index, neighbor_col_index #neighbor cache index
    cdef int neighbor_row_block_offset, neighbor_col_block_offset #index into the neighbor cache block

    #define all the caches
    cdef numpy.ndarray[numpy.npy_int8, ndim=4] outflow_direction_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int8)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] outflow_weights_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] source_block =numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] absorption_rate_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] loss_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] flux_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_int8, ndim=4] stream_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int8)

    cdef numpy.ndarray[numpy.npy_int8, ndim=2] cache_dirty = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.int8)

    cdef int outflow_direction_nodata = geoprocessing.get_nodata_from_uri(
        outflow_direction_uri)

    outflow_weights_dataset = gdal.OpenEx(outflow_weights_uri)
    outflow_weights_band = outflow_weights_dataset.GetRasterBand(1)
    cdef float outflow_weights_nodata = geoprocessing.get_nodata_from_uri(
        outflow_weights_uri)
    source_dataset = gdal.OpenEx(source_uri)
    source_band = source_dataset.GetRasterBand(1)
    cdef float source_nodata = geoprocessing.get_nodata_from_uri(
        source_uri)
    absorption_rate_dataset = gdal.OpenEx(absorption_rate_uri)
    absorption_rate_band = absorption_rate_dataset.GetRasterBand(1)
    cdef float absorption_rate_nodata = geoprocessing.get_nodata_from_uri(
        absorption_rate_uri)

    #Create output arrays for loss and flux
    transport_nodata = -1.0
    loss_dataset = geoprocessing.new_raster_from_base(
        outflow_direction_dataset, loss_uri, 'GTiff', transport_nodata,
        gdal.GDT_Float32)
    loss_band = loss_dataset.GetRasterBand(1)
    flux_dataset = geoprocessing.new_raster_from_base(
        outflow_direction_dataset, flux_uri, 'GTiff', transport_nodata,
        gdal.GDT_Float32)
    flux_band = flux_dataset.GetRasterBand(1)

    cache_dirty[:] = 0
    band_list = [outflow_direction_band, outflow_weights_band, source_band, absorption_rate_band, loss_band, flux_band]
    block_list = [outflow_direction_block, outflow_weights_block, source_block, absorption_rate_block, loss_block, flux_block]
    update_list = [False, False, False, False, True, True]

    cdef int stream_nodata = 0
    if stream_uri != None:
        stream_dataset = gdal.OpenEx(stream_uri)
        stream_band = stream_dataset.GetRasterBand(1)
        stream_nodata = geoprocessing.get_nodata_from_uri(stream_uri)
        band_list.append(stream_band)
        block_list.append(stream_block)
        update_list.append(False)
    else:
        stream_band = None

    cdef BlockCache block_cache = BlockCache(
        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size, block_col_size, band_list, block_list, update_list, cache_dirty)

    #Process flux through the grid
    cdef deque[int] cells_to_process
    for cell in sink_cell_deque:
        cells_to_process.push_front(cell)
    cdef deque[int] cell_neighbor_to_process
    for _ in range(cells_to_process.size()):
        cell_neighbor_to_process.push_front(0)

    #Diagonal offsets are based off the following index notation for neighbors
    #    3 2 1
    #    4 p 0
    #    5 6 7

    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]

    cdef int *inflow_offsets = [4, 5, 6, 7, 0, 1, 2, 3]

    cdef int neighbor_direction
    cdef double absorption_rate
    cdef double outflow_weight
    cdef double in_flux
    cdef int current_neighbor_index
    cdef int current_index
    cdef int absorb_source = (absorption_mode == 'source_and_flux')
    cdef int include_source_int = include_source

    cdef time_t last_time, current_time
    time(&last_time)
    while cells_to_process.size() > 0:
        time(&current_time)
        if current_time - last_time > 5.0:
            LOGGER.info('calculate transport cells_to_process.size() = %d' % (cells_to_process.size()))
            last_time = current_time

        current_index = cells_to_process.front()
        cells_to_process.pop_front()
        with cython.cdivision(True):
            global_row = current_index / n_cols
            global_col = current_index % n_cols
        #see if we need to update the row cache
        block_cache.update_cache(global_row, global_col, &row_index, &col_index, &row_block_offset, &col_block_offset)

        #Ensure we are working on a valid pixel, if not set everything to 0
        if source_block[row_index, col_index, row_block_offset, col_block_offset] == source_nodata:
            flux_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
            loss_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
            cache_dirty[row_index, col_index] = 1

        #We have real data that make the absorption array nodata sometimes
        #right now the best thing to do is treat it as 0.0 so everything else
        #routes
        if absorption_rate_block[row_index, col_index, row_block_offset, col_block_offset] == absorption_rate_nodata:
            absorption_rate_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0

        if flux_block[row_index, col_index, row_block_offset, col_block_offset] == transport_nodata:
            if stream_block[row_index, col_index, row_block_offset, col_block_offset] == 0:
                if include_source_int == 1:
                    flux_block[row_index, col_index, row_block_offset, col_block_offset] = (
                        source_block[row_index, col_index, row_block_offset, col_block_offset])
                else:
                    flux_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
            else:
                flux_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
            loss_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
            cache_dirty[row_index, col_index] = 1
            if absorb_source:
                absorption_rate = absorption_rate_block[row_index, col_index, row_block_offset, col_block_offset]
                loss_block[row_index, col_index, row_block_offset, col_block_offset] = (
                    absorption_rate * flux_block[row_index, col_index, row_block_offset, col_block_offset])
                flux_block[row_index, col_index, row_block_offset, col_block_offset] *= (1 - absorption_rate)

        current_neighbor_index = cell_neighbor_to_process.front()
        cell_neighbor_to_process.pop_front()
        for direction_index in xrange(current_neighbor_index, 8):
            #get percent flow from neighbor to current cell
            neighbor_row = global_row + row_offsets[direction_index]
            neighbor_col = global_col + col_offsets[direction_index]

            #See if neighbor out of bounds
            if (neighbor_row < 0 or neighbor_row >= n_rows or neighbor_col < 0 or neighbor_col >= n_cols):
                continue

            block_cache.update_cache(neighbor_row, neighbor_col, &neighbor_row_index, &neighbor_col_index, &neighbor_row_block_offset, &neighbor_col_block_offset)
            #if neighbor inflows
            neighbor_direction = outflow_direction_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset]
            if neighbor_direction == outflow_direction_nodata:
                continue

            #check if the cell flows directly, or is one index off
            if (inflow_offsets[direction_index] != neighbor_direction and
                    ((inflow_offsets[direction_index] - 1) % 8) != neighbor_direction):
                #then neighbor doesn't inflow into current cell
                continue

            #Calculate the outflow weight
            outflow_weight = outflow_weights_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset]

            if ((inflow_offsets[direction_index] - 1) % 8) == neighbor_direction:
                outflow_weight = 1.0 - outflow_weight

            if outflow_weight <= 0.0:
                continue
            in_flux = flux_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset]

            if in_flux != transport_nodata:
                absorption_rate = absorption_rate_block[row_index, col_index, row_block_offset, col_block_offset]

                #If it's not a stream, route the flux normally
                if stream_block[row_index, col_index, row_block_offset, col_block_offset] == 0:
                    if include_source_int == 0:
                        flux_block[row_index, col_index, row_block_offset, col_block_offset] += (
                            outflow_weight * (in_flux + source_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset])* (1.0 - absorption_rate))

                        loss_block[row_index, col_index, row_block_offset, col_block_offset] += (
                            outflow_weight * (in_flux + source_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset]) * absorption_rate)
                    else:
                        flux_block[row_index, col_index, row_block_offset, col_block_offset] += (
                            outflow_weight * in_flux * (1.0 - absorption_rate))
                        loss_block[row_index, col_index, row_block_offset, col_block_offset] += (
                            outflow_weight * in_flux * absorption_rate)
                else:
                    #Otherwise if it is a stream, all flux routes to the outlet
                    #we don't want it absorbed later
                    flux_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
                    loss_block[row_index, col_index, row_block_offset, col_block_offset] = 0.0
                cache_dirty[row_index, col_index] = 1
            else:
                #we need to process the neighbor, remember where we were
                #then add the neighbor to the process stack
                cells_to_process.push_front(current_index)
                cell_neighbor_to_process.push_front(direction_index)

                #Calculating the flat index for the neighbor and starting
                #at it's neighbor index of 0
                #a global neighbor row needs to be calculated
                cells_to_process.push_front(neighbor_row * n_cols + neighbor_col)
                cell_neighbor_to_process.push_front(0)
                break

    block_cache.flush_cache()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calculate_flow_weights(
    flow_direction_uri, outflow_weights_uri, outflow_direction_uri):
    """This function calculates the flow weights from a d-infinity based
        flow algorithm to assist in walking up the flow graph.

        flow_direction_uri - uri to a flow direction GDAL dataset that's
            used to calculate the flow graph
        outflow_weights_uri - a uri to a float32 dataset that will be created
            whose elements correspond to the percent outflow from the current
            cell to its first counter-clockwise neighbor
        outflow_direction_uri - a uri to a byte dataset that will indicate the
            first counter clockwise outflow neighbor as an index from the
            following diagram

            3 2 1
            4 x 0
            5 6 7

        returns nothing"""

    cdef time_t start
    time(&start)

    flow_direction_dataset = gdal.OpenEx(flow_direction_uri)
    cdef double flow_direction_nodata
    flow_direction_band = flow_direction_dataset.GetRasterBand(1)
    flow_direction_nodata = flow_direction_band.GetNoDataValue()

    cdef int block_col_size, block_row_size
    block_col_size, block_row_size = flow_direction_band.GetBlockSize()

    cdef numpy.ndarray[numpy.npy_float32, ndim=4] flow_direction_block = numpy.empty(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)

    #This is the array that's used to keep track of the connections of the
    #current cell to those *inflowing* to the cell, thus the 8 directions
    cdef int n_cols, n_rows
    n_cols, n_rows = flow_direction_band.XSize, flow_direction_band.YSize

    cdef int outflow_direction_nodata = 9
    outflow_direction_dataset = geoprocessing.new_raster_from_base(
        flow_direction_dataset, outflow_direction_uri, 'GTiff',
        outflow_direction_nodata, gdal.GDT_Byte, fill_value=outflow_direction_nodata)
    outflow_direction_band = outflow_direction_dataset.GetRasterBand(1)
    cdef numpy.ndarray[numpy.npy_byte, ndim=4] outflow_direction_block = (
        numpy.empty((N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int8))

    cdef double outflow_weights_nodata = -1.0
    outflow_weights_dataset = geoprocessing.new_raster_from_base(
        flow_direction_dataset, outflow_weights_uri, 'GTiff',
        outflow_weights_nodata, gdal.GDT_Float32, fill_value=outflow_weights_nodata)
    outflow_weights_band = outflow_weights_dataset.GetRasterBand(1)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] outflow_weights_block = (
        numpy.empty((N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32))

    #center point of global index
    cdef int global_row, global_col, global_block_row, global_block_col #index into the overall raster
    cdef int row_index, col_index #the index of the cache block
    cdef int row_block_offset, col_block_offset #index into the cache block

    #neighbor sections of global index
    cdef int neighbor_row, neighbor_col #neighbor equivalent of global_{row,col}
    cdef int neighbor_row_index, neighbor_col_index #neighbor cache index
    cdef int neighbor_row_block_offset, neighbor_col_block_offset #index into the neighbor cache block

    #define all the caches
    cdef numpy.ndarray[numpy.npy_int8, ndim=2] cache_dirty = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.int8)

    cache_dirty[:] = 0
    band_list = [flow_direction_band, outflow_direction_band, outflow_weights_band]
    block_list = [flow_direction_block, outflow_direction_block, outflow_weights_block]
    update_list = [False, True, True]

    cdef BlockCache block_cache = BlockCache(
        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size, block_col_size, band_list, block_list, update_list, cache_dirty)


    #The number of diagonal offsets defines the neighbors, angle between them
    #and the actual angle to point to the neighbor
    cdef int n_neighbors = 8
    cdef double angle_to_neighbor[8]
    for index in range(8):
        angle_to_neighbor[index] = 2.0*PI*index/8.0

    #diagonal offsets index is 0, 1, 2, 3, 4, 5, 6, 7 from the figure above
    cdef int *diagonal_offsets = [
        1, -n_cols+1, -n_cols, -n_cols-1, -1, n_cols-1, n_cols, n_cols+1]

    #Iterate over flow directions
    cdef int neighbor_direction_index
    cdef long current_index
    cdef double flow_direction, flow_angle_to_neighbor, outflow_weight

    cdef time_t last_time, current_time
    time(&last_time)
    for global_block_row in xrange(int(ceil(float(n_rows) / block_row_size))):
        time(&current_time)
        if current_time - last_time > 5.0:
            LOGGER.info("calculate_flow_weights %.1f%% complete", (global_row + 1.0) / n_rows * 100)
            last_time = current_time
        for global_block_col in xrange(int(ceil(float(n_cols) / block_col_size))):
            for global_row in xrange(global_block_row*block_row_size, min((global_block_row+1)*block_row_size, n_rows)):
                for global_col in xrange(global_block_col*block_col_size, min((global_block_col+1)*block_col_size, n_cols)):
                    block_cache.update_cache(global_row, global_col, &row_index, &col_index, &row_block_offset, &col_block_offset)
                    flow_direction = flow_direction_block[row_index, col_index, row_block_offset, col_block_offset]
                    #make sure the flow direction is defined, if not, skip this cell
                    if flow_direction == flow_direction_nodata:
                        continue
                    found = False
                    for neighbor_direction_index in range(n_neighbors):
                        flow_angle_to_neighbor = abs(angle_to_neighbor[neighbor_direction_index] - flow_direction)
                        if flow_angle_to_neighbor <= PI/4.0:
                            found = True

                            #Determine if the direction we're on is oriented at 90
                            #degrees or 45 degrees.  Given our orientation even number
                            #neighbor indexes are oriented 90 degrees and odd are 45
                            outflow_weight = 0.0

                            if neighbor_direction_index % 2 == 0:
                                outflow_weight = 1.0 - tan(flow_angle_to_neighbor)
                            else:
                                outflow_weight = tan(PI/4.0 - flow_angle_to_neighbor)

                            # clamping the outflow weight in case it's too large or small
                            if outflow_weight >= 1.0 - 1e-6:
                                outflow_weight = 1.0
                            if outflow_weight <= 1e-6:
                                outflow_weight = 1.0
                                neighbor_direction_index = (neighbor_direction_index + 1) % 8
                            outflow_direction_block[row_index, col_index, row_block_offset, col_block_offset] = neighbor_direction_index
                            outflow_weights_block[row_index, col_index, row_block_offset, col_block_offset] = outflow_weight
                            cache_dirty[row_index, col_index] = 1

                            #we found the outflow direction
                            break
                    if not found:
                        LOGGER.warn('no flow direction found for %s %s' % \
                                         (row_index, col_index))
    block_cache.flush_cache()

cdef struct Row_Col_Weight_Tuple:
    int row_index
    int col_index
    int weight


def fill_pits(dem_uri, dem_out_uri):
    """This function fills regions in a DEM that don't drain to the edge
        of the dataset.  The resulting DEM will likely have plateaus where the
        pits are filled.

        dem_uri - the original dem URI
        dem_out_uri - the original dem with pits raised to the highest drain
            value

        returns nothing"""

    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]

    dem_ds = gdal.OpenEx(dem_uri, gdal.GA_ReadOnly)
    cdef int n_rows = dem_ds.RasterYSize
    cdef int n_cols = dem_ds.RasterXSize

    dem_band = dem_ds.GetRasterBand(1)

    #copy the dem to a different dataset so we know the type
    dem_band = dem_ds.GetRasterBand(1)
    raw_nodata_value = geoprocessing.get_nodata_from_uri(dem_uri)

    cdef double nodata_value
    if raw_nodata_value is not None:
        nodata_value = raw_nodata_value
    else:
        LOGGER.warn("Nodata value not set, defaulting to -9999.9")
        nodata_value = -9999.9
    geoprocessing.new_raster_from_base_uri(
        dem_uri, dem_out_uri, 'GTiff', nodata_value, gdal.GDT_Float32,
        INF)
    dem_out_ds = gdal.OpenEx(dem_out_uri, gdal.GA_Update)
    dem_out_band = dem_out_ds.GetRasterBand(1)
    cdef int row_index, col_index, neighbor_index
    cdef float min_dem_value, cur_dem_value, neighbor_dem_value
    cdef int pit_count = 0

    for row_index in range(n_rows):
        dem_out_array = dem_band.ReadAsArray(
            xoff=0, yoff=row_index, win_xsize=n_cols, win_ysize=1)
        dem_out_band.WriteArray(dem_out_array, xoff=0, yoff=row_index)

    cdef numpy.ndarray[numpy.npy_float32, ndim=2] dem_array

    for row_index in range(1, n_rows - 1):
        #load 3 rows at a time
        dem_array = dem_out_band.ReadAsArray(
            xoff=0, yoff=row_index-1, win_xsize=n_cols, win_ysize=3)

        for col_index in range(1, n_cols - 1):
            min_dem_value = nodata_value
            cur_dem_value = dem_array[1, col_index]
            if cur_dem_value == nodata_value:
                continue
            for neighbor_index in range(8):
                neighbor_dem_value = dem_array[
                    1 + row_offsets[neighbor_index],
                    col_index + col_offsets[neighbor_index]]
                if neighbor_dem_value == nodata_value:
                    continue
                if (neighbor_dem_value < min_dem_value or
                    min_dem_value == nodata_value):
                    min_dem_value = neighbor_dem_value
            if min_dem_value > cur_dem_value:
                #it's a pit, bump it up
                dem_array[1, col_index] = min_dem_value
                pit_count += 1

        dem_out_band.WriteArray(
            dem_array[1, :].reshape((1,n_cols)), xoff=0, yoff=row_index)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def flow_direction_inf(dem_uri, flow_direction_uri):
    """Calculates the D-infinity flow algorithm.  The output is a float
        raster whose values range from 0 to 2pi.

        Algorithm from: Tarboton, "A new method for the determination of flow
        directions and upslope areas in grid digital elevation models," Water
        Resources Research, vol. 33, no. 2, pages 309 - 319, February 1997.

        Also resolves flow directions in flat areas of DEM.

        dem_uri (string) - (input) a uri to a single band GDAL Dataset with elevation values
        flow_direction_uri - (input/output) a uri to an existing GDAL dataset with
            of same as dem_uri.  Flow direction will be defined in regions that have
            nodata values in them.  non-nodata values will be ignored.  This is so
            this function can be used as a two pass filter for resolving flow directions
            on a raw dem, then filling plateaus and doing another pass.

       returns nothing"""

    cdef int col_index, row_index, n_cols, n_rows, max_index, facet_index, flat_index
    cdef double e_0, e_1, e_2, s_1, s_2, d_1, d_2, flow_direction, slope, \
        flow_direction_max_slope, slope_max, nodata_flow

    cdef double dem_nodata = geoprocessing.get_nodata_from_uri(dem_uri)
    #if it is not set, set it to a traditional nodata value
    if dem_nodata == None:
        dem_nodata = -9999

    dem_ds = gdal.OpenEx(dem_uri)
    dem_band = dem_ds.GetRasterBand(1)

    #facet elevation and factors for slope and flow_direction calculations
    #from Table 1 in Tarboton 1997.
    #THIS IS IMPORTANT:  The order is row (j), column (i), transposed to GDAL
    #convention.
    cdef int *e_0_offsets = [+0, +0,
                             +0, +0,
                             +0, +0,
                             +0, +0,
                             +0, +0,
                             +0, +0,
                             +0, +0,
                             +0, +0]
    cdef int *e_1_offsets = [+0, +1,
                             -1, +0,
                             -1, +0,
                             +0, -1,
                             +0, -1,
                             +1, +0,
                             +1, +0,
                             +0, +1]
    cdef int *e_2_offsets = [-1, +1,
                             -1, +1,
                             -1, -1,
                             -1, -1,
                             +1, -1,
                             +1, -1,
                             +1, +1,
                             +1, +1]
    cdef int *a_c = [0, 1, 1, 2, 2, 3, 3, 4]
    cdef int *a_f = [1, -1, 1, -1, 1, -1, 1, -1]

    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]

    n_rows, n_cols = geoprocessing.get_row_col_from_uri(dem_uri)
    d_1 = geoprocessing.get_cell_size_from_uri(dem_uri)
    d_2 = d_1
    cdef double max_r = numpy.pi / 4.0

    #Create a flow carray and respective dataset
    cdef float flow_nodata = -9999
    geoprocessing.new_raster_from_base_uri(
        dem_uri, flow_direction_uri, 'GTiff', flow_nodata,
        gdal.GDT_Float32, fill_value=flow_nodata)

    flow_direction_dataset = gdal.OpenEx(flow_direction_uri, gdal.GA_Update)
    flow_band = flow_direction_dataset.GetRasterBand(1)

    #center point of global index
    cdef int block_row_size, block_col_size
    block_col_size, block_row_size = dem_band.GetBlockSize()
    cdef int global_row, global_col, e_0_row, e_0_col, e_1_row, e_1_col, e_2_row, e_2_col #index into the overall raster
    cdef int e_0_row_index, e_0_col_index #the index of the cache block
    cdef int e_0_row_block_offset, e_0_col_block_offset #index into the cache block
    cdef int e_1_row_index, e_1_col_index #the index of the cache block
    cdef int e_1_row_block_offset, e_1_col_block_offset #index into the cache block
    cdef int e_2_row_index, e_2_col_index #the index of the cache block
    cdef int e_2_row_block_offset, e_2_col_block_offset #index into the cache block

    cdef int global_block_row, global_block_col #used to walk the global blocks

    #neighbor sections of global index
    cdef int neighbor_row, neighbor_col #neighbor equivalent of global_{row,col}
    cdef int neighbor_row_index, neighbor_col_index #neighbor cache index
    cdef int neighbor_row_block_offset, neighbor_col_block_offset #index into the neighbor cache block

    #define all the caches
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] flow_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    #DEM block is a 64 bit float so it can capture the resolution of small DEM offsets
    #from the plateau resolution algorithm.
    cdef numpy.ndarray[numpy.npy_float64, ndim=4] dem_block = numpy.zeros(
      (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float64)

    #the BlockCache object needs parallel lists of bands, blocks, and boolean tags to indicate which ones are updated
    band_list = [dem_band, flow_band]
    block_list = [dem_block, flow_block]
    update_list = [False, True]
    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = numpy.zeros((N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte)

    cdef BlockCache block_cache = BlockCache(
        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size, block_col_size, band_list, block_list, update_list, cache_dirty)

    cdef int row_offset, col_offset

    cdef int n_global_block_rows = int(ceil(float(n_rows) / block_row_size))
    cdef int n_global_block_cols = int(ceil(float(n_cols) / block_col_size))
    cdef time_t last_time, current_time
    cdef float current_flow
    time(&last_time)
    #flow not defined on the edges, so just go 1 row in
    for global_block_row in xrange(n_global_block_rows):
        time(&current_time)
        if current_time - last_time > 5.0:
            LOGGER.info("flow_direction_inf %.1f%% complete", (global_row + 1.0) / n_rows * 100)
            last_time = current_time
        for global_block_col in xrange(n_global_block_cols):
            for global_row in xrange(global_block_row*block_row_size, min((global_block_row+1)*block_row_size, n_rows)):
                for global_col in xrange(global_block_col*block_col_size, min((global_block_col+1)*block_col_size, n_cols)):
                    #is cache block not loaded?

                    e_0_row = e_0_offsets[0] + global_row
                    e_0_col = e_0_offsets[1] + global_col

                    block_cache.update_cache(e_0_row, e_0_col, &e_0_row_index, &e_0_col_index, &e_0_row_block_offset, &e_0_col_block_offset)

                    e_0 = dem_block[e_0_row_index, e_0_col_index, e_0_row_block_offset, e_0_col_block_offset]
                    #skip if we're on a nodata pixel skip
                    if e_0 == dem_nodata:
                        continue

                    #Calculate the flow flow_direction for each facet
                    slope_max = 0 #use this to keep track of the maximum down-slope
                    flow_direction_max_slope = 0 #flow direction on max downward slope
                    max_index = 0 #index to keep track of max slope facet

                    for facet_index in range(8):
                        #This defines the three points the facet

                        e_1_row = e_1_offsets[facet_index * 2 + 0] + global_row
                        e_1_col = e_1_offsets[facet_index * 2 + 1] + global_col
                        e_2_row = e_2_offsets[facet_index * 2 + 0] + global_row
                        e_2_col = e_2_offsets[facet_index * 2 + 1] + global_col
                        #make sure one of the facets doesn't hang off the edge
                        if (e_1_row < 0 or e_1_row >= n_rows or
                            e_2_row < 0 or e_2_row >= n_rows or
                            e_1_col < 0 or e_1_col >= n_cols or
                            e_2_col < 0 or e_2_col >= n_cols):
                            continue

                        block_cache.update_cache(e_1_row, e_1_col, &e_1_row_index, &e_1_col_index, &e_1_row_block_offset, &e_1_col_block_offset)
                        block_cache.update_cache(e_2_row, e_2_col, &e_2_row_index, &e_2_col_index, &e_2_row_block_offset, &e_2_col_block_offset)

                        e_1 = dem_block[e_1_row_index, e_1_col_index, e_1_row_block_offset, e_1_col_block_offset]
                        e_2 = dem_block[e_2_row_index, e_2_col_index, e_2_row_block_offset, e_2_col_block_offset]

                        if e_1 == dem_nodata and e_2 == dem_nodata:
                            continue

                        #s_1 is slope along straight edge
                        s_1 = (e_0 - e_1) / d_1 #Eqn 1
                        #slope along diagonal edge
                        s_2 = (e_1 - e_2) / d_2 #Eqn 2

                        #can't calculate flow direction if one of the facets is nodata
                        if e_1 == dem_nodata or e_2 == dem_nodata:
                            #calc max slope here
                            if e_1 != dem_nodata and facet_index % 2 == 0 and e_1 < e_0:
                                #straight line to next pixel
                                slope = s_1
                                flow_direction = 0
                            elif e_2 != dem_nodata and facet_index % 2 == 1 and e_2 < e_0:
                                #diagonal line to next pixel
                                slope = (e_0 - e_2) / sqrt(d_1 **2 + d_2 ** 2)
                                flow_direction = max_r
                            else:
                                continue
                        else:
                            #both facets are defined, this is the core of
                            #d-infinity algorithm
                            flow_direction = atan2(s_2, s_1) #Eqn 3

                            if flow_direction < 0: #Eqn 4
                                #If the flow direction goes off one side, set flow
                                #direction to that side and the slope to the straight line
                                #distance slope
                                flow_direction = 0
                                slope = s_1
                            elif flow_direction > max_r: #Eqn 5
                                #If the flow direciton goes off the diagonal side, figure
                                #out what its value is and
                                flow_direction = max_r
                                slope = (e_0 - e_2) / sqrt(d_1 ** 2 + d_2 ** 2)
                            else:
                                slope = sqrt(s_1 ** 2 + s_2 ** 2) #Eqn 3

                        #update the maxes depending on the results above
                        if slope > slope_max:
                            flow_direction_max_slope = flow_direction
                            slope_max = slope
                            max_index = facet_index

                    #if there's a downward slope, save the flow direction
                    if slope_max > 0:
                        flow_block[e_0_row_index, e_0_col_index, e_0_row_block_offset, e_0_col_block_offset] = (
                            a_f[max_index] * flow_direction_max_slope +
                            a_c[max_index] * PI / 2.0)
                        cache_dirty[e_0_row_index, e_0_col_index] = 1

    block_cache.flush_cache()
    flow_band = None
    gdal.Dataset.__swig_destroy__(flow_direction_dataset)
    flow_direction_dataset = None
    geoprocessing.calculate_raster_stats_uri(flow_direction_uri)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def distance_to_stream(
        flow_direction_uri, stream_uri, distance_uri, factor_uri=None):
    """This function calculates the flow downhill distance to the stream layers

        Args:
            flow_direction_uri (string) - (input) a path to a raster with
                d-infinity flow directions.
            stream_uri (string) - (input) a raster where 1 indicates a stream
                all other values ignored must be same dimensions and projection
                as flow_direction_uri.
            distance_uri (string) - (output) a path to the output raster that
                will be created as same dimensions as the input rasters where
                each pixel is in linear units the drainage from that point to a
                stream.
            factor_uri (string) - (optional input) a floating point raster that
                is used to multiply the stepsize by for each current pixel,
                useful for some models to calculate a user defined downstream
                factor.

        Returns:
            nothing"""

    cdef float distance_nodata = -9999
    geoprocessing.new_raster_from_base_uri(
        flow_direction_uri, distance_uri, 'GTiff', distance_nodata,
        gdal.GDT_Float32, fill_value=distance_nodata)

    cdef float processed_cell_nodata = 127
    processed_cell_uri = (
        os.path.join(os.path.dirname(flow_direction_uri), 'processed_cell.tif'))
    geoprocessing.new_raster_from_base_uri(
        distance_uri, processed_cell_uri, 'GTiff', processed_cell_nodata,
        gdal.GDT_Byte, fill_value=0)

    processed_cell_ds = gdal.OpenEx(processed_cell_uri, gdal.GA_Update)
    processed_cell_band = processed_cell_ds.GetRasterBand(1)

    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]
    cdef int *inflow_offsets = [4, 5, 6, 7, 0, 1, 2, 3]

    cdef int n_rows, n_cols
    n_rows, n_cols = geoprocessing.get_row_col_from_uri(
        flow_direction_uri)
    cdef int INF = n_rows + n_cols

    cdef deque[int] visit_stack

    stream_ds = gdal.OpenEx(stream_uri)
    stream_band = stream_ds.GetRasterBand(1)
    cdef float stream_nodata = geoprocessing.get_nodata_from_uri(
        stream_uri)
    cdef float cell_size = geoprocessing.get_cell_size_from_uri(stream_uri)

    distance_ds = gdal.OpenEx(distance_uri, gdal.GA_Update)
    distance_band = distance_ds.GetRasterBand(1)

    outflow_weights_uri = geoprocessing.temporary_filename()
    outflow_direction_uri = geoprocessing.temporary_filename()
    calculate_flow_weights(
        flow_direction_uri, outflow_weights_uri, outflow_direction_uri)
    outflow_weights_ds = gdal.OpenEx(outflow_weights_uri)
    outflow_weights_band = outflow_weights_ds.GetRasterBand(1)
    cdef float outflow_weights_nodata = geoprocessing.get_nodata_from_uri(
        outflow_weights_uri)
    outflow_direction_ds = gdal.OpenEx(outflow_direction_uri)
    outflow_direction_band = outflow_direction_ds.GetRasterBand(1)
    cdef int outflow_direction_nodata = geoprocessing.get_nodata_from_uri(
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
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] distance_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
        dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_int8, ndim=4] processed_cell_block = (
        numpy.zeros(
            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
            dtype=numpy.int8))

    band_list = [stream_band, outflow_direction_band, outflow_weights_band,
                 distance_band, processed_cell_band]
    block_list = [stream_block, outflow_direction_block, outflow_weights_block,
                  distance_block, processed_cell_block]
    update_list = [False, False, False, True, True]

    cdef numpy.ndarray[numpy.npy_float32, ndim=4] factor_block
    cdef int factor_exists = (factor_uri != None)
    if factor_exists:
        factor_block = numpy.zeros(
            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
            dtype=numpy.float32)
        factor_ds = gdal.OpenEx(factor_uri)
        factor_band = factor_ds.GetRasterBand(1)
        band_list.append(factor_band)
        block_list.append(factor_block)
        update_list.append(False)

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

    cdef float original_distance

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

                        distance_block[row_index, col_index,
                            row_block_offset, col_block_offset] = 0
                        processed_cell_block[row_index, col_index,
                            row_block_offset, col_block_offset] = 1
                        cache_dirty[row_index, col_index] = 1

    cdef int neighbor_outflow_direction, neighbor_index, outflow_direction
    cdef float neighbor_outflow_weight, current_distance, cell_travel_distance
    cdef float outflow_weight, neighbor_distance, step_size
    cdef float factor
    cdef int it_flows_here
    cdef int downstream_index, downstream_calculated
    cdef float downstream_distance
    cdef float current_stream
    cdef int pushed_current = False

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
        current_distance = 0.0

        time(&current_time)
        if current_time - last_time > 5.0:
            last_time = current_time
            LOGGER.info(
                'visit_stack on stream distance size: %d ', visit_stack.size())

        current_stream = stream_block[
            row_index, col_index, row_block_offset, col_block_offset]
        outflow_direction = outflow_direction_block[
            row_index, col_index, row_block_offset,
            col_block_offset]
        if current_stream == 1:
            distance_block[row_index, col_index,
                row_block_offset, col_block_offset] = 0
            processed_cell_block[row_index, col_index,
                row_block_offset, col_block_offset] = 1
            cache_dirty[row_index, col_index] = 1
        elif outflow_direction == outflow_direction_nodata:
            current_distance = INF
        elif processed_cell_block[row_index, col_index, row_block_offset,
                col_block_offset] == 0:
            #add downstream distance to current distance

            outflow_weight = outflow_weights_block[
                row_index, col_index, row_block_offset,
                col_block_offset]

            if factor_exists:
                factor = factor_block[
                    row_index, col_index, row_block_offset, col_block_offset]
            else:
                factor = 1.0

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

                neighbor_distance = distance_block[
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
                    neighbor_distance = 0.0

                if outflow_direction % 2 == 1:
                    #increase distance by a square root of 2 for diagonal
                    step_size = cell_size * 1.41421356237
                else:
                    step_size = cell_size

                current_distance += (
                    neighbor_distance + step_size * factor) * outflow_weight

        if not update_downstream:
            #mark flat_index as processed
            block_cache.update_cache(
                global_row, global_col, &row_index, &col_index,
                &row_block_offset, &col_block_offset)
            processed_cell_block[row_index, col_index,
                row_block_offset, col_block_offset] = 1
            distance_block[row_index, col_index,
                row_block_offset, col_block_offset] = current_distance
            cache_dirty[row_index, col_index] = 1

            #update any upstream neighbors with this distance
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


@cython.boundscheck(False)
@cython.wraparound(False)
def percent_to_sink(
    sink_pixels_uri, export_rate_uri, outflow_direction_uri,
    outflow_weights_uri, effect_uri):
    """This function calculates the amount of load from a single pixel
        to the source pixels given the percent export rate per pixel.

        sink_pixels_uri - the pixels of interest that will receive flux.
            This may be a set of stream pixels, or a single pixel at a
            watershed outlet.

        export_rate_uri - a GDAL floating point dataset that has a percent
            of flux exported per pixel

        outflow_direction_uri - a uri to a byte dataset that indicates the
            first counter clockwise outflow neighbor as an index from the
            following diagram

            3 2 1
            4 x 0
            5 6 7

        outflow_weights_uri - a uri to a float32 dataset whose elements
            correspond to the percent outflow from the current cell to its
            first counter-clockwise neighbor

        effect_uri - the output GDAL dataset that shows the percent of flux
            emanating per pixel that will reach any sink pixel

        returns nothing"""

    LOGGER.info("calculating percent to sink")
    cdef time_t start_time
    time(&start_time)

    sink_pixels_dataset = gdal.OpenEx(sink_pixels_uri)
    sink_pixels_band = sink_pixels_dataset.GetRasterBand(1)
    cdef int sink_pixels_nodata = geoprocessing.get_nodata_from_uri(
        sink_pixels_uri)
    export_rate_dataset = gdal.OpenEx(export_rate_uri)
    export_rate_band = export_rate_dataset.GetRasterBand(1)
    cdef double export_rate_nodata = geoprocessing.get_nodata_from_uri(
        export_rate_uri)
    outflow_direction_dataset = gdal.OpenEx(outflow_direction_uri)
    outflow_direction_band = outflow_direction_dataset.GetRasterBand(1)
    cdef int outflow_direction_nodata = geoprocessing.get_nodata_from_uri(
        outflow_direction_uri)
    outflow_weights_dataset = gdal.OpenEx(outflow_weights_uri)
    outflow_weights_band = outflow_weights_dataset.GetRasterBand(1)
    cdef float outflow_weights_nodata = geoprocessing.get_nodata_from_uri(
        outflow_weights_uri)

    cdef int block_col_size, block_row_size
    block_col_size, block_row_size = sink_pixels_band.GetBlockSize()
    cdef int n_rows = sink_pixels_dataset.RasterYSize
    cdef int n_cols = sink_pixels_dataset.RasterXSize

    cdef double effect_nodata = -1.0
    geoprocessing.new_raster_from_base_uri(
        sink_pixels_uri, effect_uri, 'GTiff', effect_nodata,
        gdal.GDT_Float32, fill_value=effect_nodata)
    effect_dataset = gdal.OpenEx(effect_uri, gdal.GA_Update)
    effect_band = effect_dataset.GetRasterBand(1)

    #center point of global index
    cdef int global_row, global_col #index into the overall raster
    cdef int row_index, col_index #the index of the cache block
    cdef int row_block_offset, col_block_offset #index into the cache block
    cdef int global_block_row, global_block_col #used to walk the global blocks

    #neighbor sections of global index
    cdef int neighbor_row, neighbor_col #neighbor equivalent of global_{row,col}
    cdef int neighbor_row_index, neighbor_col_index #neighbor cache index
    cdef int neighbor_row_block_offset, neighbor_col_block_offset #index into the neighbor cache block

    #define all the caches

    cdef numpy.ndarray[numpy.npy_int32, ndim=4] sink_pixels_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] export_rate_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_int8, ndim=4] outflow_direction_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int8)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] outflow_weights_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] out_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] effect_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    #the BlockCache object needs parallel lists of bands, blocks, and boolean tags to indicate which ones are updated
    block_list = [sink_pixels_block, export_rate_block, outflow_direction_block, outflow_weights_block, effect_block]
    band_list = [sink_pixels_band, export_rate_band, outflow_direction_band, outflow_weights_band, effect_band]
    update_list = [False, False, False, False, True]
    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = numpy.zeros((N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte)

    cdef BlockCache block_cache = BlockCache(
        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size, block_col_size, band_list, block_list, update_list, cache_dirty)

    cdef float outflow_weight, neighbor_outflow_weight
    cdef int neighbor_outflow_direction

    #Diagonal offsets are based off the following index notation for neighbors
    #    3 2 1
    #    4 p 0
    #    5 6 7

    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]
    cdef int *inflow_offsets = [4, 5, 6, 7, 0, 1, 2, 3]
    cdef int flat_index
    cdef deque[int] process_queue
    #Queue the sinks
    for global_block_row in xrange(int(numpy.ceil(float(n_rows) / block_row_size))):
        for global_block_col in xrange(int(numpy.ceil(float(n_cols) / block_col_size))):
            for global_row in xrange(global_block_row*block_row_size, min((global_block_row+1)*block_row_size, n_rows)):
                for global_col in xrange(global_block_col*block_col_size, min((global_block_col+1)*block_col_size, n_cols)):
                    block_cache.update_cache(global_row, global_col, &row_index, &col_index, &row_block_offset, &col_block_offset)
                    if sink_pixels_block[row_index, col_index, row_block_offset, col_block_offset] == 1:
                        effect_block[row_index, col_index, row_block_offset, col_block_offset] = 1.0
                        cache_dirty[row_index, col_index] = 1
                        process_queue.push_back(global_row * n_cols + global_col)

    while process_queue.size() > 0:
        flat_index = process_queue.front()
        process_queue.pop_front()
        with cython.cdivision(True):
            global_row = flat_index / n_cols
            global_col = flat_index % n_cols

        block_cache.update_cache(global_row, global_col, &row_index, &col_index, &row_block_offset, &col_block_offset)
        if export_rate_block[row_index, col_index, row_block_offset, col_block_offset] == export_rate_nodata:
            continue

        #if the outflow weight is nodata, then not a valid pixel
        outflow_weight = outflow_weights_block[row_index, col_index, row_block_offset, col_block_offset]
        if outflow_weight == outflow_weights_nodata:
            continue

        for neighbor_index in range(8):
            neighbor_row = global_row + row_offsets[neighbor_index]
            neighbor_col = global_col + col_offsets[neighbor_index]
            if neighbor_row < 0 or neighbor_row >= n_rows or neighbor_col < 0 or neighbor_col >= n_cols:
                #out of bounds
                continue

            block_cache.update_cache(neighbor_row, neighbor_col, &neighbor_row_index, &neighbor_col_index, &neighbor_row_block_offset, &neighbor_col_block_offset)

            if sink_pixels_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset] == 1:
                #it's already a sink
                continue

            neighbor_outflow_direction = (
                outflow_direction_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset])
            #if the neighbor is no data, don't try to set that
            if neighbor_outflow_direction == outflow_direction_nodata:
                continue

            neighbor_outflow_weight = (
                outflow_weights_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset])
            #if the neighbor is no data, don't try to set that
            if neighbor_outflow_weight == outflow_direction_nodata:
                continue

            it_flows_here = False
            if neighbor_outflow_direction == inflow_offsets[neighbor_index]:
                #the neighbor flows into this cell
                it_flows_here = True

            if (neighbor_outflow_direction - 1) % 8 == inflow_offsets[neighbor_index]:
                #the offset neighbor flows into this cell
                it_flows_here = True
                neighbor_outflow_weight = 1.0 - neighbor_outflow_weight

            if it_flows_here:
                #If we haven't processed that effect yet, set it to 0 and append to the queue
                if effect_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset] == effect_nodata:
                    process_queue.push_back(neighbor_row * n_cols + neighbor_col)
                    effect_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset] = 0.0
                    cache_dirty[neighbor_row_index, neighbor_col_index] = 1

                #the percent of the pixel upstream equals the current percent
                #times the percent flow to that pixels times the
                effect_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset] += (
                    effect_block[row_index, col_index, row_block_offset, col_block_offset] *
                    neighbor_outflow_weight *
                    export_rate_block[row_index, col_index, row_block_offset, col_block_offset])
                cache_dirty[neighbor_row_index, neighbor_col_index] = 1

    block_cache.flush_cache()
    cdef time_t end_time
    time(&end_time)
    LOGGER.info('Done calculating percent to sink elapsed time %ss' % \
                    (end_time - start_time))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef flat_edges(
        dem_uri, flow_direction_uri, deque[int] &high_edges,
        deque[int] &low_edges, int drain_off_edge=0):
    """This function locates flat cells that border on higher and lower terrain
        and places them into sets for further processing.

        Args:

            dem_uri (string) - (input) a uri to a single band GDAL Dataset with
                elevation values
            flow_direction_uri (string) - (input/output) a uri to a single band
                GDAL Dataset with partially defined d_infinity flow directions
            high_edges (deque) - (output) will contain all the high edge cells as
                flat row major order indexes
            low_edges (deque) - (output) will contain all the low edge cells as flat
                row major order indexes
            drain_off_edge (int) - (input) if True will drain flat regions off
                the nodata edge of a DEM"""

    high_edges.clear()
    low_edges.clear()

    cdef int *neighbor_row_offset = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *neighbor_col_offset = [1,  1,  0, -1, -1, -1, 0, 1]

    dem_ds = gdal.OpenEx(dem_uri)
    dem_band = dem_ds.GetRasterBand(1)
    flow_ds = gdal.OpenEx(flow_direction_uri, gdal.GA_Update)
    flow_band = flow_ds.GetRasterBand(1)

    cdef int block_col_size, block_row_size
    block_col_size, block_row_size = dem_band.GetBlockSize()
    cdef int n_rows = dem_ds.RasterYSize
    cdef int n_cols = dem_ds.RasterXSize

    cdef numpy.ndarray[numpy.npy_float32, ndim=4] flow_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
        dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] dem_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
        dtype=numpy.float32)

    band_list = [dem_band, flow_band]
    block_list = [dem_block, flow_block]
    update_list = [False, False]
    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte)

    block_col_size, block_row_size = dem_band.GetBlockSize()

    cdef BlockCache block_cache = BlockCache(
        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size,
        block_col_size, band_list, block_list, update_list, cache_dirty)

    cdef int n_global_block_rows = int(ceil(float(n_rows) / block_row_size))
    cdef int n_global_block_cols = int(ceil(float(n_cols) / block_col_size))

    cdef int global_row, global_col

    cdef int cell_row_index, cell_col_index
    cdef int cell_row_block_index, cell_col_block_index
    cdef int cell_row_block_offset, cell_col_block_offset

    cdef int neighbor_index
    cdef int neighbor_row, neighbor_col
    cdef int neighbor_row_index, neighbor_col_index
    cdef int neighbor_row_block_offset, neighbor_col_block_offset

    cdef float cell_dem, cell_flow, neighbor_dem, neighbor_flow

    cdef float dem_nodata = geoprocessing.get_nodata_from_uri(
        dem_uri)
    cdef float flow_nodata = geoprocessing.get_nodata_from_uri(
        flow_direction_uri)

    cdef time_t last_time, current_time
    time(&last_time)

    cdef neighbor_nodata

    for global_block_row in xrange(n_global_block_rows):
        time(&current_time)
        if current_time - last_time > 5.0:
            LOGGER.info(
                "flat_edges %.1f%% complete", (global_row + 1.0) / n_rows * 100)
            last_time = current_time
        for global_block_col in xrange(n_global_block_cols):
            for global_row in xrange(
                    global_block_row*block_row_size,
                    min((global_block_row+1)*block_row_size, n_rows)):
                for global_col in xrange(
                        global_block_col*block_col_size,
                        min((global_block_col+1)*block_col_size, n_cols)):

                    block_cache.update_cache(
                        global_row, global_col,
                        &cell_row_index, &cell_col_index,
                        &cell_row_block_offset, &cell_col_block_offset)

                    cell_dem = dem_block[cell_row_index, cell_col_index,
                        cell_row_block_offset, cell_col_block_offset]

                    if cell_dem == dem_nodata:
                        continue

                    cell_flow = flow_block[cell_row_index, cell_col_index,
                        cell_row_block_offset, cell_col_block_offset]

                    neighbor_nodata = 0
                    for neighbor_index in xrange(8):
                        neighbor_row = (
                            neighbor_row_offset[neighbor_index] + global_row)
                        neighbor_col = (
                            neighbor_col_offset[neighbor_index] + global_col)

                        if (neighbor_row >= n_rows or neighbor_row < 0 or
                                neighbor_col >= n_cols or neighbor_col < 0):
                            continue

                        block_cache.update_cache(
                            neighbor_row, neighbor_col,
                            &neighbor_row_index, &neighbor_col_index,
                            &neighbor_row_block_offset,
                            &neighbor_col_block_offset)
                        neighbor_dem = dem_block[
                            neighbor_row_index, neighbor_col_index,
                            neighbor_row_block_offset,
                            neighbor_col_block_offset]

                        if neighbor_dem == dem_nodata:
                            neighbor_nodata = 1
                            continue

                        neighbor_flow = flow_block[
                            neighbor_row_index, neighbor_col_index,
                            neighbor_row_block_offset,
                            neighbor_col_block_offset]

                        if (cell_flow != flow_nodata and
                                neighbor_flow == flow_nodata and
                                cell_dem == neighbor_dem):
                            low_edges.push_back(global_row * n_cols + global_col)
                            break
                        elif (cell_flow == flow_nodata and
                              cell_dem < neighbor_dem):
                            high_edges.push_back(global_row * n_cols + global_col)
                            break
                    if drain_off_edge and neighbor_nodata:
                        low_edges.push_back(global_row * n_cols + global_col)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef label_flats(dem_uri, deque[int] &low_edges, labels_uri):
    """A flood fill function to give all the cells of each flat a unique
        label

        Args:
            dem_uri (string) - (input) a uri to a single band GDAL Dataset with
                elevation values
            low_edges (Set) - (input) Contains all the low edge cells of the dem
                written as flat indexes in row major order
            labels_uri (string) - (output) a uri to a single band integer gdal
                dataset that will be created that will contain labels for the
                flat regions of the DEM.
            """

    cdef int *neighbor_row_offset = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *neighbor_col_offset = [1,  1,  0, -1, -1, -1, 0, 1]

    dem_ds = gdal.OpenEx(dem_uri)
    dem_band = dem_ds.GetRasterBand(1)

    cdef int labels_nodata = -1
    labels_ds = geoprocessing.new_raster_from_base(
        dem_ds, labels_uri, 'GTiff', labels_nodata,
        gdal.GDT_Int32)
    labels_band = labels_ds.GetRasterBand(1)

    cdef int block_col_size, block_row_size
    block_col_size, block_row_size = dem_band.GetBlockSize()
    cdef int n_rows = dem_ds.RasterYSize
    cdef int n_cols = dem_ds.RasterXSize

    cdef numpy.ndarray[numpy.npy_float32, ndim=4] labels_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
        dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] dem_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
        dtype=numpy.float32)

    band_list = [dem_band, labels_band]
    block_list = [dem_block, labels_block]
    update_list = [False, True]
    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte)

    block_col_size, block_row_size = dem_band.GetBlockSize()

    cdef BlockCache block_cache = BlockCache(
        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size,
        block_col_size, band_list, block_list, update_list, cache_dirty)

    cdef int n_global_block_rows = int(ceil(float(n_rows) / block_row_size))
    cdef int n_global_block_cols = int(ceil(float(n_cols) / block_col_size))

    cdef int global_row, global_col

    cdef int cell_row_index, cell_col_index
    cdef int cell_row_block_index, cell_col_block_index
    cdef int cell_row_block_offset, cell_col_block_offset

    cdef int neighbor_index
    cdef int neighbor_row, neighbor_col
    cdef int neighbor_row_index, neighbor_col_index
    cdef int neighbor_row_block_offset, neighbor_col_block_offset

    cdef float cell_dem, neighbor_dem, neighbor_label
    cdef float cell_label, flat_cell_label

    cdef float dem_nodata = geoprocessing.get_nodata_from_uri(
        dem_uri)

    cdef time_t last_time, current_time
    time(&last_time)

    cdef int flat_cell_index
    cdef int flat_fill_cell_index
    cdef int label = 1
    cdef int fill_cell_row, fill_cell_col
    cdef deque[int] to_fill
    cdef float flat_height, current_flat_height
    cdef int visit_number = 0
    for _ in xrange(low_edges.size()):
        flat_cell_index = low_edges.front()
        low_edges.pop_front()
        low_edges.push_back(flat_cell_index)
        visit_number += 1
        time(&current_time)
        if current_time - last_time > 5.0:
            LOGGER.info(
                "label_flats %.1f%% complete",
                float(visit_number) / low_edges.size() * 100)
            last_time = current_time
        global_row = flat_cell_index / n_cols
        global_col = flat_cell_index % n_cols

        block_cache.update_cache(
            global_row, global_col,
            &cell_row_index, &cell_col_index,
            &cell_row_block_offset, &cell_col_block_offset)

        cell_label = labels_block[cell_row_index, cell_col_index,
            cell_row_block_offset, cell_col_block_offset]

        flat_height = dem_block[cell_row_index, cell_col_index,
            cell_row_block_offset, cell_col_block_offset]

        if cell_label == labels_nodata:
            #label flats
            to_fill.push_back(flat_cell_index)
            while not to_fill.empty():
                flat_fill_cell_index = to_fill.front()
                to_fill.pop_front()
                fill_cell_row = flat_fill_cell_index / n_cols
                fill_cell_col = flat_fill_cell_index % n_cols
                if (fill_cell_row < 0 or fill_cell_row >= n_rows or
                        fill_cell_col < 0 or fill_cell_col >= n_cols):
                    continue

                block_cache.update_cache(
                    fill_cell_row, fill_cell_col,
                    &cell_row_index, &cell_col_index,
                    &cell_row_block_offset, &cell_col_block_offset)

                current_flat_height = dem_block[cell_row_index, cell_col_index,
                    cell_row_block_offset, cell_col_block_offset]

                if current_flat_height != flat_height:
                    continue

                flat_cell_label = labels_block[
                    cell_row_index, cell_col_index,
                    cell_row_block_offset, cell_col_block_offset]

                if flat_cell_label != labels_nodata:
                    continue

                #set the label
                labels_block[
                    cell_row_index, cell_col_index,
                    cell_row_block_offset, cell_col_block_offset] = label
                cache_dirty[cell_row_index, cell_col_index] = 1

                #visit the neighbors
                for neighbor_index in xrange(8):
                    neighbor_row = (
                        fill_cell_row + neighbor_row_offset[neighbor_index])
                    neighbor_col = (
                        fill_cell_col + neighbor_col_offset[neighbor_index])
                    to_fill.push_back(neighbor_row * n_cols + neighbor_col)

            label += 1
    block_cache.flush_cache()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef clean_high_edges(labels_uri, deque[int] &high_edges):
    """Removes any high edges that do not have labels and reports them if so.

        Args:
            labels_uri (string) - (input) a uri to a single band integer gdal
                dataset that contain labels for the cells that lie in
                flat regions of the DEM.
            high_edges (set) - (input/output) a set containing row major order
                flat indexes

        Returns:
            nothing"""

    labels_ds = gdal.OpenEx(labels_uri)
    labels_band = labels_ds.GetRasterBand(1)

    cdef int block_col_size, block_row_size
    block_col_size, block_row_size = labels_band.GetBlockSize()
    cdef int n_rows = labels_ds.RasterYSize
    cdef int n_cols = labels_ds.RasterXSize

    cdef numpy.ndarray[numpy.npy_int32, ndim=4] labels_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
        dtype=numpy.int32)

    band_list = [labels_band]
    block_list = [labels_block]
    update_list = [False]
    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte)

    cdef BlockCache block_cache = BlockCache(
        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size,
        block_col_size, band_list, block_list, update_list, cache_dirty)

    cdef int labels_nodata = geoprocessing.get_nodata_from_uri(
        labels_uri)
    cdef int flat_cell_label

    cdef int cell_row_index, cell_col_index
    cdef int cell_row_block_index, cell_col_block_index
    cdef int cell_row_block_offset, cell_col_block_offset

    cdef int flat_index
    cdef int flat_row, flat_col
    cdef c_set[int] unlabeled_set
    for _ in xrange(high_edges.size()):
        flat_index = high_edges.front()
        high_edges.pop_front()
        high_edges.push_back(flat_index)
        flat_row = flat_index / n_cols
        flat_col = flat_index % n_cols

        block_cache.update_cache(
            flat_row, flat_col,
            &cell_row_index, &cell_col_index,
            &cell_row_block_offset, &cell_col_block_offset)

        flat_cell_label = labels_block[
            cell_row_index, cell_col_index,
            cell_row_block_offset, cell_col_block_offset]

        #this is a flat that does not have an outlet
        if flat_cell_label == labels_nodata:
            unlabeled_set.insert(flat_index)

    if unlabeled_set.size() > 0:
        #remove high edges that are unlabeled
        for _ in xrange(high_edges.size()):
            flat_index = high_edges.front()
            high_edges.pop_front()
            if unlabeled_set.find(flat_index) != unlabeled_set.end():
                high_edges.push_back(flat_index)
        LOGGER.warn("Not all flats have outlets")
    block_cache.flush_cache()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef drain_flats(
        deque[int] &high_edges, deque[int] &low_edges, labels_uri,
        flow_direction_uri, flat_mask_uri):
    """A wrapper function for draining flats so it can be called from a
        Python level, but use a C++ map at the Cython level.

        Args:
            high_edges (deque[int]) - (input) A list of row major order indicating the
                high edge lists.
            low_edges (deque[int]) - (input)  A list of row major order indicating the
                high edge lists.
            labels_uri (string) - (input) A uri to a gdal raster that has
                unique integer labels for each flat in the DEM.
            flow_direction_uri (string) - (input/output) A uri to a gdal raster
                that has d-infinity flow directions defined for non-flat pixels
                and will have pixels defined for the flat pixels when the
                function returns
            flat_mask_uri (string) - (out) A uri to a gdal raster that will have
                relative heights defined per flat to drain each flat.

        Returns:
            nothing"""

    cdef map[int, int] flat_height

    LOGGER.info('draining away from higher')
    away_from_higher(
        high_edges, labels_uri, flow_direction_uri, flat_mask_uri, flat_height)

    LOGGER.info('draining towards lower')
    towards_lower(
        low_edges, labels_uri, flow_direction_uri, flat_mask_uri, flat_height)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef away_from_higher(
        deque[int] &high_edges, labels_uri, flow_direction_uri, flat_mask_uri,
        map[int, int] &flat_height):
    """Builds a gradient away from higher terrain.

        Take Care, Take Care, Take Care
        The Earth Is Not a Cold Dead Place
        Those Who Tell The Truth Shall Die,
            Those Who Tell The Truth Shall Live Forever

        Args:
            high_edges (deque) - (input) all the high edge cells of the DEM which
                are part of drainable flats.
            labels_uri (string) - (input) a uri to a single band integer gdal
                dataset that contain labels for the cells that lie in
                flat regions of the DEM.
            flow_direction_uri (string) - (input) a uri to a single band
                GDAL Dataset with partially defined d_infinity flow directions
            flat_mask_uri (string) - (output) gdal dataset that contains the
                number of increments to be applied to each cell to form a
                gradient away from higher terrain.  cells not in a flat have a
                value of 0
            flat_height (collections.defaultdict) - (input/output) Has an entry
                for each label value of of labels_uri indicating the maximal
                number of increments to be applied to the flat idientifed by
                that label.

        Returns:
            nothing"""

    cdef int *neighbor_row_offset = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *neighbor_col_offset = [1,  1,  0, -1, -1, -1, 0, 1]

    cdef int flat_mask_nodata = -9999
    #fill up the flat mask with 0s so it can be used to route a dem later
    geoprocessing.new_raster_from_base_uri(
        labels_uri, flat_mask_uri, 'GTiff', flat_mask_nodata,
        gdal.GDT_Int32, fill_value=0)

    labels_ds = gdal.OpenEx(labels_uri)
    labels_band = labels_ds.GetRasterBand(1)
    flat_mask_ds = gdal.OpenEx(flat_mask_uri, gdal.GA_Update)
    flat_mask_band = flat_mask_ds.GetRasterBand(1)
    flow_direction_ds = gdal.OpenEx(flow_direction_uri)
    flow_direction_band = flow_direction_ds.GetRasterBand(1)

    cdef int block_col_size, block_row_size
    block_col_size, block_row_size = labels_band.GetBlockSize()
    cdef int n_rows = labels_ds.RasterYSize
    cdef int n_cols = labels_ds.RasterXSize

    cdef numpy.ndarray[numpy.npy_int32, ndim=4] labels_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
        dtype=numpy.int32)
    cdef numpy.ndarray[numpy.npy_int32, ndim=4] flat_mask_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
        dtype=numpy.int32)
    cdef numpy.ndarray[numpy.npy_int32, ndim=4] flow_direction_block = (
        numpy.zeros(
            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
            dtype=numpy.int32))

    band_list = [labels_band, flat_mask_band, flow_direction_band]
    block_list = [labels_block, flat_mask_block, flow_direction_block]
    update_list = [False, True, False]
    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte)

    cdef BlockCache block_cache = BlockCache(
        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size,
        block_col_size, band_list, block_list, update_list, cache_dirty)

    cdef int cell_row_index, cell_col_index
    cdef int cell_row_block_index, cell_col_block_index
    cdef int cell_row_block_offset, cell_col_block_offset

    cdef int loops = 1

    cdef int neighbor_row, neighbor_col
    cdef int flat_index
    cdef int flat_row, flat_col
    cdef int flat_mask
    cdef int labels_nodata = geoprocessing.get_nodata_from_uri(labels_uri)
    cdef int cell_label, neighbor_label
    cdef float neighbor_flow
    cdef float flow_nodata = geoprocessing.get_nodata_from_uri(
        flow_direction_uri)

    cdef time_t last_time, current_time
    time(&last_time)

    cdef deque[int] high_edges_queue

    #seed the queue with the high edges
    for _ in xrange(high_edges.size()):
        flat_index = high_edges.front()
        high_edges.pop_front()
        high_edges.push_back(flat_index)
        high_edges_queue.push_back(flat_index)

    marker = -1
    high_edges_queue.push_back(marker)

    while high_edges_queue.size() > 1:
        time(&current_time)
        if current_time - last_time > 5.0:
            LOGGER.info(
                "away_from_higher, work queue size: %d complete",
                high_edges_queue.size())
            last_time = current_time

        flat_index = high_edges_queue.front()
        high_edges_queue.pop_front()
        if flat_index == marker:
            loops += 1
            high_edges_queue.push_back(marker)
            continue

        flat_row = flat_index / n_cols
        flat_col = flat_index % n_cols

        block_cache.update_cache(
            flat_row, flat_col,
            &cell_row_index, &cell_col_index,
            &cell_row_block_offset, &cell_col_block_offset)

        flat_mask = flat_mask_block[
            cell_row_index, cell_col_index,
            cell_row_block_offset, cell_col_block_offset]

        cell_label = labels_block[
            cell_row_index, cell_col_index,
            cell_row_block_offset, cell_col_block_offset]

        if flat_mask != 0:
            continue

        #update the cell mask and the max height of the flat
        #making it negative because it's easier to do here than in towards lower
        flat_mask_block[
            cell_row_index, cell_col_index,
            cell_row_block_offset, cell_col_block_offset] = -loops
        cache_dirty[cell_row_index, cell_col_index] = 1
        flat_height[cell_label] = loops

        #visit the neighbors
        for neighbor_index in xrange(8):
            neighbor_row = (
                flat_row + neighbor_row_offset[neighbor_index])
            neighbor_col = (
                flat_col + neighbor_col_offset[neighbor_index])

            if (neighbor_row < 0 or neighbor_row >= n_rows or
                    neighbor_col < 0 or neighbor_col >= n_cols):
                continue

            block_cache.update_cache(
                neighbor_row, neighbor_col,
                &cell_row_index, &cell_col_index,
                &cell_row_block_offset, &cell_col_block_offset)

            neighbor_label = labels_block[
                cell_row_index, cell_col_index,
                cell_row_block_offset, cell_col_block_offset]

            neighbor_flow = flow_direction_block[
                cell_row_index, cell_col_index,
                cell_row_block_offset, cell_col_block_offset]

            if (neighbor_label != labels_nodata and
                    neighbor_label == cell_label and
                    neighbor_flow == flow_nodata):
                high_edges_queue.push_back(neighbor_row * n_cols + neighbor_col)

    block_cache.flush_cache()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef towards_lower(
        deque[int] &low_edges, labels_uri, flow_direction_uri, flat_mask_uri,
        map[int, int] &flat_height):
    """Builds a gradient towards lower terrain.

        Args:
            low_edges (set) - (input) all the low edge cells of the DEM which
                are part of drainable flats.
            labels_uri (string) - (input) a uri to a single band integer gdal
                dataset that contain labels for the cells that lie in
                flat regions of the DEM.
            flow_direction_uri (string) - (input) a uri to a single band
                GDAL Dataset with partially defined d_infinity flow directions
            flat_mask_uri (string) - (input/output) gdal dataset that contains
                the negative step increments from toward_higher and will contain
                the number of steps to be applied to each cell to form a
                gradient away from higher terrain.  cells not in a flat have a
                value of 0
            flat_height (collections.defaultdict) - (input/output) Has an entry
                for each label value of of labels_uri indicating the maximal
                number of increments to be applied to the flat idientifed by
                that label.

        Returns:
            nothing"""

    cdef int *neighbor_row_offset = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *neighbor_col_offset = [1,  1,  0, -1, -1, -1, 0, 1]

    flat_mask_nodata = geoprocessing.get_nodata_from_uri(flat_mask_uri)

    labels_ds = gdal.OpenEx(labels_uri)
    labels_band = labels_ds.GetRasterBand(1)
    flat_mask_ds = gdal.OpenEx(flat_mask_uri, gdal.GA_Update)
    flat_mask_band = flat_mask_ds.GetRasterBand(1)
    flow_direction_ds = gdal.OpenEx(flow_direction_uri)
    flow_direction_band = flow_direction_ds.GetRasterBand(1)

    cdef int block_col_size, block_row_size
    block_col_size, block_row_size = labels_band.GetBlockSize()
    cdef int n_rows = labels_ds.RasterYSize
    cdef int n_cols = labels_ds.RasterXSize

    cdef numpy.ndarray[numpy.npy_int32, ndim=4] labels_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
        dtype=numpy.int32)
    cdef numpy.ndarray[numpy.npy_int32, ndim=4] flat_mask_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
        dtype=numpy.int32)
    cdef numpy.ndarray[numpy.npy_int32, ndim=4] flow_direction_block = (
        numpy.zeros(
            (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
            dtype=numpy.int32))

    band_list = [labels_band, flat_mask_band, flow_direction_band]
    block_list = [labels_block, flat_mask_block, flow_direction_block]
    update_list = [False, True, False]
    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte)

    cdef BlockCache block_cache = BlockCache(
        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size,
        block_col_size, band_list, block_list, update_list, cache_dirty)

    cdef int cell_row_index, cell_col_index
    cdef int cell_row_block_index, cell_col_block_index
    cdef int cell_row_block_offset, cell_col_block_offset

    cdef int loops = 1

    cdef deque[int] low_edges_queue
    cdef int neighbor_row, neighbor_col
    cdef int flat_index
    cdef int flat_row, flat_col
    cdef int flat_mask
    cdef int labels_nodata = geoprocessing.get_nodata_from_uri(labels_uri)
    cdef int cell_label, neighbor_label
    cdef float neighbor_flow
    cdef float flow_nodata = geoprocessing.get_nodata_from_uri(
        flow_direction_uri)

    #seed the queue with the low edges
    for _ in xrange(low_edges.size()):
        flat_index = low_edges.front()
        low_edges.pop_front()
        low_edges.push_back(flat_index)
        low_edges_queue.push_back(flat_index)

    cdef time_t last_time, current_time
    time(&last_time)

    marker = -1
    low_edges_queue.push_back(marker)
    while low_edges_queue.size() > 1:

        time(&current_time)
        if current_time - last_time > 5.0:
            LOGGER.info(
                "toward_lower work queue size: %d", low_edges_queue.size())
            last_time = current_time

        flat_index = low_edges_queue.front()
        low_edges_queue.pop_front()
        if flat_index == marker:
            loops += 1
            low_edges_queue.push_back(marker)
            continue

        flat_row = flat_index / n_cols
        flat_col = flat_index % n_cols

        block_cache.update_cache(
            flat_row, flat_col,
            &cell_row_index, &cell_col_index,
            &cell_row_block_offset, &cell_col_block_offset)

        flat_mask = flat_mask_block[
            cell_row_index, cell_col_index,
            cell_row_block_offset, cell_col_block_offset]

        if flat_mask > 0:
            continue

        cell_label = labels_block[
            cell_row_index, cell_col_index,
            cell_row_block_offset, cell_col_block_offset]

        if flat_mask < 0:
            flat_mask_block[
                cell_row_index, cell_col_index,
                cell_row_block_offset, cell_col_block_offset] = (
                    flat_height[cell_label] + flat_mask + 2 * loops)
        else:
            flat_mask_block[
                cell_row_index, cell_col_index,
                cell_row_block_offset, cell_col_block_offset] = 2 * loops
        cache_dirty[cell_row_index, cell_col_index] = 1

        #visit the neighbors
        for neighbor_index in xrange(8):
            neighbor_row = (
                flat_row + neighbor_row_offset[neighbor_index])
            neighbor_col = (
                flat_col + neighbor_col_offset[neighbor_index])

            if (neighbor_row < 0 or neighbor_row >= n_rows or
                    neighbor_col < 0 or neighbor_col >= n_cols):
                continue

            block_cache.update_cache(
                neighbor_row, neighbor_col,
                &cell_row_index, &cell_col_index,
                &cell_row_block_offset, &cell_col_block_offset)

            neighbor_label = labels_block[
                cell_row_index, cell_col_index,
                cell_row_block_offset, cell_col_block_offset]

            neighbor_flow = flow_direction_block[
                cell_row_index, cell_col_index,
                cell_row_block_offset, cell_col_block_offset]

            if (neighbor_label != labels_nodata and
                    neighbor_label == cell_label and
                    neighbor_flow == flow_nodata):
                low_edges_queue.push_back(neighbor_row * n_cols + neighbor_col)

    block_cache.flush_cache()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def flow_direction_inf_masked_flow_dirs(
        flat_mask_uri, labels_uri, flow_direction_uri):
    """Calculates the D-infinity flow algorithm for regions defined from flat
        drainage resolution.

        Flow algorithm from: Tarboton, "A new method for the determination of
        flow directions and upslope areas in grid digital elevation models,"
        Water Resources Research, vol. 33, no. 2, pages 309 - 319, February
        1997.

        Also resolves flow directions in flat areas of DEM.

        flat_mask_uri (string) - (input) a uri to a single band GDAL Dataset
            that has offset values from the flat region resolution algorithm.
            The offsets in flat_mask are the relative heights only within the
            flat regions defined in labels_uri.
        labels_uri (string) - (input) a uri to a single band integer gdal
                dataset that contain labels for the cells that lie in
                flat regions of the DEM.
        flow_direction_uri - (input/output) a uri to an existing GDAL dataset
            of same size as dem_uri.  Flow direction will be defined in regions
            that have nodata values in them that overlap regions of labels_uri.
            This is so this function can be used as a two pass filter for
            resolving flow directions on a raw dem, then filling plateaus and
            doing another pass.

       returns nothing"""

    cdef int col_index, row_index, n_cols, n_rows, max_index, facet_index, flat_index
    cdef double e_0, e_1, e_2, s_1, s_2, d_1, d_2, flow_direction, slope, \
        flow_direction_max_slope, slope_max, nodata_flow

    flat_mask_ds = gdal.OpenEx(flat_mask_uri)
    flat_mask_band = flat_mask_ds.GetRasterBand(1)

    #facet elevation and factors for slope and flow_direction calculations
    #from Table 1 in Tarboton 1997.
    #THIS IS IMPORTANT:  The order is row (j), column (i), transposed to GDAL
    #convention.
    cdef int *e_0_offsets = [+0, +0,
                             +0, +0,
                             +0, +0,
                             +0, +0,
                             +0, +0,
                             +0, +0,
                             +0, +0,
                             +0, +0]
    cdef int *e_1_offsets = [+0, +1,
                             -1, +0,
                             -1, +0,
                             +0, -1,
                             +0, -1,
                             +1, +0,
                             +1, +0,
                             +0, +1]
    cdef int *e_2_offsets = [-1, +1,
                             -1, +1,
                             -1, -1,
                             -1, -1,
                             +1, -1,
                             +1, -1,
                             +1, +1,
                             +1, +1]
    cdef int *a_c = [0, 1, 1, 2, 2, 3, 3, 4]
    cdef int *a_f = [1, -1, 1, -1, 1, -1, 1, -1]

    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]

    n_rows, n_cols = geoprocessing.get_row_col_from_uri(flat_mask_uri)
    d_1 = geoprocessing.get_cell_size_from_uri(flat_mask_uri)
    d_2 = d_1
    cdef double max_r = numpy.pi / 4.0


    cdef float flow_nodata = geoprocessing.get_nodata_from_uri(
        flow_direction_uri)
    flow_direction_dataset = gdal.OpenEx(flow_direction_uri, gdal.GA_Update)
    flow_band = flow_direction_dataset.GetRasterBand(1)

    cdef float label_nodata = geoprocessing.get_nodata_from_uri(labels_uri)
    label_dataset = gdal.OpenEx(labels_uri)
    label_band = label_dataset.GetRasterBand(1)

    #center point of global index
    cdef int block_row_size, block_col_size
    block_col_size, block_row_size = flat_mask_band.GetBlockSize()
    cdef int global_row, global_col, e_0_row, e_0_col, e_1_row, e_1_col, e_2_row, e_2_col #index into the overall raster
    cdef int e_0_row_index, e_0_col_index #the index of the cache block
    cdef int e_0_row_block_offset, e_0_col_block_offset #index into the cache block
    cdef int e_1_row_index, e_1_col_index #the index of the cache block
    cdef int e_1_row_block_offset, e_1_col_block_offset #index into the cache block
    cdef int e_2_row_index, e_2_col_index #the index of the cache block
    cdef int e_2_row_block_offset, e_2_col_block_offset #index into the cache block

    cdef int global_block_row, global_block_col #used to walk the global blocks

    #neighbor sections of global index
    cdef int neighbor_row, neighbor_col #neighbor equivalent of global_{row,col}
    cdef int neighbor_row_index, neighbor_col_index #neighbor cache index
    cdef int neighbor_row_block_offset, neighbor_col_block_offset #index into the neighbor cache block

    #define all the caches
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] flow_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    #flat_mask block is a 64 bit float so it can capture the resolution of small flat_mask offsets
    #from the plateau resolution algorithm.
    cdef numpy.ndarray[numpy.npy_int32, ndim=4] flat_mask_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int32)
    cdef numpy.ndarray[numpy.npy_int32, ndim=4] label_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int32)

    #the BlockCache object needs parallel lists of bands, blocks, and boolean tags to indicate which ones are updated
    band_list = [flat_mask_band, flow_band, label_band]
    block_list = [flat_mask_block, flow_block, label_block]
    update_list = [False, True, False]
    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = numpy.zeros((N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte)

    cdef BlockCache block_cache = BlockCache(
        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size, block_col_size, band_list, block_list, update_list, cache_dirty)

    cdef int row_offset, col_offset

    cdef int n_global_block_rows = int(ceil(float(n_rows) / block_row_size))
    cdef int n_global_block_cols = int(ceil(float(n_cols) / block_col_size))
    cdef time_t last_time, current_time
    cdef float current_flow
    cdef int current_label, e_1_label, e_2_label
    time(&last_time)
    #flow not defined on the edges, so just go 1 row in
    for global_block_row in xrange(n_global_block_rows):
        time(&current_time)
        if current_time - last_time > 5.0:
            LOGGER.info("flow_direction_inf %.1f%% complete", (global_row + 1.0) / n_rows * 100)
            last_time = current_time
        for global_block_col in xrange(n_global_block_cols):
            for global_row in xrange(global_block_row*block_row_size, min((global_block_row+1)*block_row_size, n_rows)):
                for global_col in xrange(global_block_col*block_col_size, min((global_block_col+1)*block_col_size, n_cols)):
                    #is cache block not loaded?

                    e_0_row = e_0_offsets[0] + global_row
                    e_0_col = e_0_offsets[1] + global_col

                    block_cache.update_cache(e_0_row, e_0_col, &e_0_row_index, &e_0_col_index, &e_0_row_block_offset, &e_0_col_block_offset)

                    current_label = label_block[
                        e_0_row_index, e_0_col_index,
                        e_0_row_block_offset, e_0_col_block_offset]

                    #if a label isn't defiend we're not in a flat region
                    if current_label == label_nodata:
                        continue

                    current_flow = flow_block[
                        e_0_row_index, e_0_col_index,
                        e_0_row_block_offset, e_0_col_block_offset]

                    #this can happen if we have been passed an existing flow
                    #direction raster, perhaps from an earlier iteration in a
                    #multiphase flow resolution algorithm
                    if current_flow != flow_nodata:
                        continue

                    e_0 = flat_mask_block[e_0_row_index, e_0_col_index, e_0_row_block_offset, e_0_col_block_offset]
                    #skip if we're on a nodata pixel skip

                    #Calculate the flow flow_direction for each facet
                    slope_max = 0 #use this to keep track of the maximum down-slope
                    flow_direction_max_slope = 0 #flow direction on max downward slope
                    max_index = 0 #index to keep track of max slope facet

                    for facet_index in range(8):
                        #This defines the three points the facet

                        e_1_row = e_1_offsets[facet_index * 2 + 0] + global_row
                        e_1_col = e_1_offsets[facet_index * 2 + 1] + global_col
                        e_2_row = e_2_offsets[facet_index * 2 + 0] + global_row
                        e_2_col = e_2_offsets[facet_index * 2 + 1] + global_col
                        #make sure one of the facets doesn't hang off the edge
                        if (e_1_row < 0 or e_1_row >= n_rows or
                            e_2_row < 0 or e_2_row >= n_rows or
                            e_1_col < 0 or e_1_col >= n_cols or
                            e_2_col < 0 or e_2_col >= n_cols):
                            continue

                        block_cache.update_cache(e_1_row, e_1_col, &e_1_row_index, &e_1_col_index, &e_1_row_block_offset, &e_1_col_block_offset)
                        block_cache.update_cache(e_2_row, e_2_col, &e_2_row_index, &e_2_col_index, &e_2_row_block_offset, &e_2_col_block_offset)

                        e_1 = flat_mask_block[e_1_row_index, e_1_col_index, e_1_row_block_offset, e_1_col_block_offset]
                        e_2 = flat_mask_block[e_2_row_index, e_2_col_index, e_2_row_block_offset, e_2_col_block_offset]

                        e_1_label = label_block[e_1_row_index, e_1_col_index, e_1_row_block_offset, e_1_col_block_offset]
                        e_2_label = label_block[e_2_row_index, e_2_col_index, e_2_row_block_offset, e_2_col_block_offset]

                        #if labels aren't t the same as the current, we can't flow to them
                        if e_1_label != current_label and e_2_label != current_label:
                            continue

                        #s_1 is slope along straight edge
                        s_1 = (e_0 - e_1) / d_1 #Eqn 1
                        #slope along diagonal edge
                        s_2 = (e_1 - e_2) / d_2 #Eqn 2

                        #can't calculate flow direction if one of the facets is nodata
                        if e_1_label != current_label or e_2_label != current_label:
                            #make sure the flow direction perfectly aligns with
                            #the facet direction so we don't get a case where
                            #we point toward a pixel but the next pixel down
                            #is the correct flow direction
                            if e_1_label == current_label and facet_index % 2 == 0 and e_1 < e_0:
                                #straight line to next pixel
                                slope = s_1
                                flow_direction = 0
                            elif e_2_label == current_label and facet_index % 2 == 1 and e_2 < e_0:
                                #diagonal line to next pixel
                                slope = (e_0 - e_2) / sqrt(d_1 **2 + d_2 ** 2)
                                flow_direction = max_r
                            else:
                                continue
                        else:
                            #both facets are defined, this is the core of
                            #d-infinity algorithm
                            flow_direction = atan2(s_2, s_1) #Eqn 3

                            if flow_direction < 0: #Eqn 4
                                #If the flow direction goes off one side, set flow
                                #direction to that side and the slope to the straight line
                                #distance slope
                                flow_direction = 0
                                slope = s_1
                            elif flow_direction > max_r: #Eqn 5
                                #If the flow direciton goes off the diagonal side, figure
                                #out what its value is and
                                flow_direction = max_r
                                slope = (e_0 - e_2) / sqrt(d_1 ** 2 + d_2 ** 2)
                            else:
                                slope = sqrt(s_1 ** 2 + s_2 ** 2) #Eqn 3

                        #update the maxes depending on the results above
                        if slope > slope_max:
                            flow_direction_max_slope = flow_direction
                            slope_max = slope
                            max_index = facet_index

                    #if there's a downward slope, save the flow direction
                    if slope_max > 0:
                        flow_block[e_0_row_index, e_0_col_index, e_0_row_block_offset, e_0_col_block_offset] = (
                            a_f[max_index] * flow_direction_max_slope +
                            a_c[max_index] * PI / 2.0)
                        cache_dirty[e_0_row_index, e_0_col_index] = 1

    block_cache.flush_cache()
    flow_band = None
    gdal.Dataset.__swig_destroy__(flow_direction_dataset)
    flow_direction_dataset = None
    geoprocessing.calculate_raster_stats_uri(flow_direction_uri)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef find_outlets(dem_uri, flow_direction_uri, deque[int] &outlet_deque):
    """Discover and return the outlets in the dem array

        Args:
            dem_uri (string) - (input) a uri to a gdal dataset representing
                height values
            flow_direction_uri (string) - (input) a uri to gdal dataset
                representing flow direction values
            outlet_deque (deque[int]) - (output) a reference to a c++ set that
                contains the set of flat integer index indicating the outlets
                in dem

        Returns:
            nothing"""

    dem_ds = gdal.OpenEx(dem_uri)
    dem_band = dem_ds.GetRasterBand(1)

    flow_direction_ds = gdal.OpenEx(flow_direction_uri)
    flow_direction_band = flow_direction_ds.GetRasterBand(1)
    cdef float flow_nodata = geoprocessing.get_nodata_from_uri(
        flow_direction_uri)

    cdef int block_col_size, block_row_size
    block_col_size, block_row_size = dem_band.GetBlockSize()
    cdef int n_rows = dem_ds.RasterYSize
    cdef int n_cols = dem_ds.RasterXSize

    cdef numpy.ndarray[numpy.npy_float32, ndim=4] dem_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
        dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] flow_direction_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size),
        dtype=numpy.float32)

    band_list = [dem_band, flow_direction_band]
    block_list = [dem_block, flow_direction_block]
    update_list = [False, False]
    cdef numpy.ndarray[numpy.npy_byte, ndim=2] cache_dirty = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.byte)

    cdef BlockCache block_cache = BlockCache(
        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size,
        block_col_size, band_list, block_list, update_list, cache_dirty)

    cdef float dem_nodata = geoprocessing.get_nodata_from_uri(dem_uri)

    cdef int cell_row_index, cell_col_index
    cdef int cell_row_block_index, cell_col_block_index
    cdef int cell_row_block_offset, cell_col_block_offset
    cdef int flat_index
    cdef float dem_value, flow_direction

    outlet_deque.clear()

    cdef time_t last_time, current_time
    time(&last_time)

    for cell_row_index in xrange(n_rows):
        time(&current_time)
        if current_time - last_time > 5.0:
            LOGGER.info(
                'find outlet percent complete = %.2f, outlet_deque size = %d',
                float(cell_row_index)/n_rows * 100, outlet_deque.size())
            last_time = current_time
        for cell_col_index in xrange(n_cols):

            block_cache.update_cache(
                cell_row_index, cell_col_index,
                &cell_row_block_index, &cell_col_block_index,
                &cell_row_block_offset, &cell_col_block_offset)

            dem_value = dem_block[
                cell_row_block_index, cell_col_block_index,
                cell_row_block_offset, cell_col_block_offset]
            flow_direction = flow_direction_block[
                cell_row_block_index, cell_col_block_index,
                cell_row_block_offset, cell_col_block_offset]

            #it's a valid dem but no flow direction could be defined, it's
            #either a sink or an outlet

            if dem_value != dem_nodata and flow_direction == flow_nodata:
                flat_index = cell_row_index * n_cols + cell_col_index
                outlet_deque.push_front(flat_index)


def resolve_flats(
    dem_uri, flow_direction_uri, flat_mask_uri, labels_uri,
    drain_off_edge=False):
    """Function to resolve the flat regions in the dem given a first attempt
        run at calculating flow direction.  Will provide regions of flat areas
        and their labels.

        Based on: Barnes, Richard, Clarence Lehman, and David Mulla. "An
            efficient assignment of drainage direction over flat surfaces in
            raster digital elevation models." Computers & Geosciences 62
            (2014): 128-135.

        Args:
            dem_uri (string) - (input) a uri to a single band GDAL Dataset with
                elevation values
            flow_direction_uri (string) - (input/output) a uri to a single band
                GDAL Dataset with partially defined d_infinity flow directions
            drain_off_edge (boolean) - input if true will drain flat areas off
                the edge of the raster

        Returns:
            True if there were flats to resolve, False otherwise"""

    cdef deque[int] high_edges
    cdef deque[int] low_edges
    flat_edges(
        dem_uri, flow_direction_uri, high_edges, low_edges,
        drain_off_edge=drain_off_edge)

    if low_edges.size() == 0:
        if high_edges.size() != 0:
            LOGGER.warn('There were undrainable flats')
        else:
            LOGGER.info('There were no flats')
        return False

    LOGGER.info('labeling flats')
    label_flats(dem_uri, low_edges, labels_uri)

    #LOGGER.info('cleaning high edges')
    #clean_high_edges(labels_uri, high_edges)

    drain_flats(
        high_edges, low_edges, labels_uri, flow_direction_uri, flat_mask_uri)

    return True


def route_flux(
        in_flow_direction, in_dem, in_source_uri, in_absorption_rate_uri,
        loss_uri, flux_uri, absorption_mode, aoi_uri=None, stream_uri=None,
        include_source=True):

    """This function will route flux across a landscape given a dem to
        guide flow from a d-infinty flow algorithm, and a custom function
        that will operate on input flux and other user defined arguments
        to determine nodal output flux.

        in_flow_direction - a URI to a d-infinity flow direction raster
        in_dem - a uri to the dem that generated in_flow_direction, they
            should be aligned rasters
        in_source_uri - a GDAL dataset that has source flux per pixel
        in_absorption_rate_uri - a GDAL floating point dataset that has a
            percent of flux absorbed per pixel
        loss_uri - an output URI to to the dataset that will output the
            amount of flux absorbed by each pixel
        flux_uri - a URI to an output dataset that records the amount of flux
            travelling through each pixel
        absorption_mode - either 'flux_only' or 'source_and_flux'. For
            'flux_only' the outgoing flux is (in_flux * absorption + source).
            If 'source_and_flux' then the output flux
            is (in_flux + source) * absorption.
        aoi_uri - an OGR datasource for an area of interest polygon.
            the routing flux calculation will only occur on those pixels
            and neighboring pixels will either be raw outlets or
            non-contibuting inputs depending on the orientation of the DEM.
        stream_uri - (optional) a GDAL dataset that classifies pixels as stream
            (1) or not (0).  If during routing we hit a stream pixel, all
            upstream flux is considered to wash to zero because it will
            reach the outlet.  The advantage here is that it can't then
            route out of the stream
        include_source - if True, source is added to current pixel, otherwise
            pixel starts at 0.

        returns nothing"""

    dem_uri = geoprocessing.temporary_filename(suffix='.tif')
    flow_direction_uri = geoprocessing.temporary_filename(suffix='.tif')
    source_uri = geoprocessing.temporary_filename(suffix='.tif')
    absorption_rate_uri = geoprocessing.temporary_filename(suffix='.tif')
    out_pixel_size = geoprocessing.get_cell_size_from_uri(in_flow_direction)

    geoprocessing.align_dataset_list(
        [in_flow_direction, in_dem, in_source_uri, in_absorption_rate_uri],
        [flow_direction_uri, dem_uri, source_uri, absorption_rate_uri],
        ["nearest", "nearest", "nearest", "nearest"], out_pixel_size,
        "intersection", 0, aoi_uri=aoi_uri, assert_datasets_projected=False)

    outflow_weights_uri = geoprocessing.temporary_filename(suffix='.tif')
    outflow_direction_uri = geoprocessing.temporary_filename(suffix='.tif')

    cdef deque[int] outlet_cell_deque

    find_outlets(dem_uri, flow_direction_uri, outlet_cell_deque)
    calculate_flow_weights(
        flow_direction_uri, outflow_weights_uri, outflow_direction_uri)

    calculate_transport(
        outflow_direction_uri, outflow_weights_uri, outlet_cell_deque,
        source_uri, absorption_rate_uri, loss_uri, flux_uri, absorption_mode,
        stream_uri, include_source)

    cleanup_uri_list = [
        dem_uri, flow_direction_uri, source_uri, absorption_rate_uri,
        outflow_weights_uri, outflow_direction_uri]

    for ds_uri in cleanup_uri_list:
        try:
            os.remove(ds_uri)
        except OSError as exception:
            LOGGER.warn("couldn't remove %s because it's still open", ds_uri)
            LOGGER.warn(exception)

def delineate_watershed(
        outflow_direction_uri, outflow_weights_uri, outlet_shapefile_uri,
        snap_distance, stream_uri, watershed_out_uri,
        snapped_outlet_points_uri):

    cdef time_t last_time, current_time
    time(&last_time)

    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]
    cdef int *inflow_offsets = [4, 5, 6, 7, 0, 1, 2, 3]

    #Pass transport
    cdef time_t start
    time(&start)

    #Create output arrays for loss and flux
    outflow_direction_dataset = gdal.OpenEx(outflow_direction_uri)
    cdef int n_cols = outflow_direction_dataset.RasterXSize
    cdef int n_rows = outflow_direction_dataset.RasterYSize
    outflow_direction_band = outflow_direction_dataset.GetRasterBand(1)

    cdef int block_col_size, block_row_size
    block_col_size, block_row_size = outflow_direction_band.GetBlockSize()

    if os.path.isfile(watershed_out_uri):
        os.remove(watershed_out_uri)

    geoprocessing.create_directories(
        [os.path.dirname(watershed_out_uri)])

    projection_wkt = outflow_direction_dataset.GetProjection()
    output_sr = osr.SpatialReference()
    output_sr.ImportFromWkt(projection_wkt)

    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    temp_target_dir = tempfile.mkdtemp()
    working_watershed_vector = output_driver.CreateDataSource(
        os.path.join(temp_target_dir, 'workingwatersheds.shp'))
    working_watershed_layer = working_watershed_vector.CreateLayer(
        'serviceshed', output_sr, ogr.wkbPolygon)
    field = ogr.FieldDefn('pixel_valu', ogr.OFTReal)
    field.SetWidth(24)
    working_watershed_layer.CreateField(field)

    if os.path.isfile(snapped_outlet_points_uri):
        os.remove(snapped_outlet_points_uri)

    snapped_outlet_points_datasource = output_driver.CreateDataSource(
        snapped_outlet_points_uri)
    snapped_outlet_points_layer = snapped_outlet_points_datasource.CreateLayer(
        'snapped_outlet_points', output_sr, ogr.wkbPoint)

    outlet_ds = gdal.OpenEx(outlet_shapefile_uri)
    outlet_layer = outlet_ds.GetLayer()
    outlet_defn = outlet_layer.GetLayerDefn()
    for index in xrange(outlet_defn.GetFieldCount()):
        field_defn = outlet_defn.GetFieldDefn(index)
        field_type = field_defn.GetType()
        # for some reason the default numerical width can be set to less than
        # the precision of the fields being copied over.  This addresses that
        # issue
        if field_type in [ogr.OFTInteger, ogr.OFTReal]:
            field_defn.SetWidth(24)
        snapped_outlet_points_layer.CreateField(field_defn)
        working_watershed_layer.CreateField(field_defn)

    #center point of global index
    cdef int global_row, global_col #index into the overall raster
    cdef int row_index, col_index #the index of the cache block
    cdef int row_block_offset, col_block_offset #index into the cache block
    cdef int global_block_row, global_block_col #used to walk the global blocks

    #neighbor sections of global index
    cdef int neighbor_row, neighbor_col #neighbor equivalent of global_{row,col}
    cdef int neighbor_row_index, neighbor_col_index #neighbor cache index
    cdef int neighbor_row_block_offset, neighbor_col_block_offset #index into the neighbor cache block

    #define all the caches
    cdef numpy.ndarray[numpy.npy_int8, ndim=4] outflow_direction_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int8)
    cdef numpy.ndarray[numpy.npy_float32, ndim=4] outflow_weights_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_int8, ndim=4] watershed_block = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS, block_row_size, block_col_size), dtype=numpy.int8)

    cdef numpy.ndarray[numpy.npy_int8, ndim=2] cache_dirty = numpy.zeros(
        (N_BLOCK_ROWS, N_BLOCK_COLS), dtype=numpy.int8)

    cdef int outflow_direction_nodata = geoprocessing.get_nodata_from_uri(
        outflow_direction_uri)

    outflow_weights_dataset = gdal.OpenEx(outflow_weights_uri)
    outflow_weights_band = outflow_weights_dataset.GetRasterBand(1)
    cdef int outflow_weights_nodata = geoprocessing.get_nodata_from_uri(
        outflow_weights_uri)

    #Create output arrays for loss and flux
    watershed_nodata = 255
    watershed_mask_uri = geoprocessing.temporary_filename()
    geoprocessing.new_raster_from_base_uri(
        outflow_direction_uri, watershed_mask_uri, 'GTiff', watershed_nodata,
        gdal.GDT_Byte, fill_value=watershed_nodata)
    watershed_dataset = gdal.OpenEx(watershed_mask_uri, gdal.GA_Update)
    watershed_band = watershed_dataset.GetRasterBand(1)

    cache_dirty[:] = 0
    band_list = [outflow_direction_band, outflow_weights_band, watershed_band]
    block_list = [outflow_direction_block, outflow_weights_block, watershed_block]
    update_list = [False, False, True]

    cdef BlockCache block_cache = BlockCache(
        N_BLOCK_ROWS, N_BLOCK_COLS, n_rows, n_cols, block_row_size,
        block_col_size, band_list, block_list, update_list, cache_dirty)

    cdef stack[int] work_stack

    #parse out each point in the input shapefile and determine the x/y coordinate in the raster
    geotransform = outflow_direction_dataset.GetGeoTransform()
    stream_ds = gdal.OpenEx(stream_uri)
    stream_band = stream_ds.GetRasterBand(1)
    count = 0
    layer = outlet_ds.GetLayer()
    n_points_left = layer.GetFeatureCount()
    for point_feature in layer:
        point_geometry = point_feature.GetGeometryRef()
        point = point_geometry.GetPoint()
        x_index = (point[0] - geotransform[0]) // geotransform[1]
        y_index = (point[1] - geotransform[3]) // geotransform[5]
        if x_index < 0 or x_index >= n_cols or y_index < 0 or y_index > n_rows:
            LOGGER.warn('Encountered a point that was outside the bounds of the DEM %s', point_geometry)
            continue
        n_points_left -= 1

        if snap_distance > 0:
            x_center = x_index
            y_center = y_index
            x_left = x_index - snap_distance
            if x_left < 0:
                x_left = 0
            y_top = y_index - snap_distance
            if y_top < 0:
                y_top = 0
            x_right = x_index + snap_distance
            if x_right >= n_cols:
                x_right = n_cols - 1
            y_bottom = y_index + snap_distance
            if y_bottom >= n_rows:
                y_bottom = n_rows - 1

            #snap to the nearest stream pixel
            stream_window = stream_band.ReadAsArray(
                int(x_left), int(y_top), int(x_right - x_left),
                int(y_bottom - y_top))
            row_indexes, col_indexes = numpy.nonzero(
                stream_window == 1)
            if row_indexes.size > 0:
                #calc euclidan distance
                distance_array = (
                    (row_indexes - stream_window.shape[0] / 2) ** 2 +
                    (col_indexes - stream_window.shape[1] / 2) **2) ** 0.5

                #closest element
                min_index = numpy.argmin(distance_array)
                min_row = row_indexes[min_index]
                min_col = col_indexes[min_index]
                offset_row = min_row - (y_center - y_top)
                offset_col = min_col - (x_center - x_left)

                y_index += offset_row
                x_index += offset_col

            point_geometry = ogr.Geometry(ogr.wkbPoint)
            point_geometry.AddPoint(
                geotransform[0] + (x_index + 0.5) * geotransform[1],
                geotransform[3] + (y_index + 0.5) * geotransform[5])

            # Get the output Layer's Feature Definition
            feature_def = snapped_outlet_points_layer.GetLayerDefn()
            snapped_point_feature = ogr.Feature(feature_def)
            snapped_point_feature.SetGeometry(point_geometry)

            for index in xrange(point_feature.GetFieldCount()):
                snapped_point_feature.SetField(
                    index, point_feature.GetField(index))
            snapped_outlet_points_layer.CreateFeature(snapped_point_feature)

        work_stack.push(y_index * n_cols + x_index)
        count += 1
        while work_stack.size() > 0:
            time(&current_time)
            if current_time - last_time > 5.0:
                LOGGER.info(
                    'work_stack_size=%d, n_outlet_points_left=%d',
                    work_stack.size(), n_points_left)
                last_time = current_time

            current_index = work_stack.top()
            work_stack.pop()
            with cython.cdivision(True):
                global_row = current_index / n_cols
                global_col = current_index % n_cols

            block_cache.update_cache(
                global_row, global_col, &row_index, &col_index,
                &row_block_offset, &col_block_offset)

            if watershed_block[
                    row_index, col_index, row_block_offset,
                    col_block_offset] == 1:
                continue

            watershed_block[
                row_index, col_index, row_block_offset,
                col_block_offset] = 1
            cache_dirty[row_index, col_index] = 1

            for direction_index in xrange(8):
                #get percent flow from neighbor to current cell
                neighbor_row = global_row + row_offsets[direction_index]
                neighbor_col = global_col + col_offsets[direction_index]

                #See if neighbor out of bounds
                if (neighbor_row < 0 or neighbor_row >= n_rows or neighbor_col < 0 or neighbor_col >= n_cols):
                    continue

                block_cache.update_cache(neighbor_row, neighbor_col, &neighbor_row_index, &neighbor_col_index, &neighbor_row_block_offset, &neighbor_col_block_offset)
                #if neighbor inflows
                neighbor_direction = outflow_direction_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset]
                if neighbor_direction == outflow_direction_nodata:
                    continue

                #check if the cell flows directly, or is one index off
                if (inflow_offsets[direction_index] != neighbor_direction and
                        ((inflow_offsets[direction_index] - 1) % 8) != neighbor_direction):
                    #then neighbor doesn't inflow into current cell
                    continue

                #Calculate the outflow weight
                outflow_weight = outflow_weights_block[neighbor_row_index, neighbor_col_index, neighbor_row_block_offset, neighbor_col_block_offset]

                if ((inflow_offsets[direction_index] - 1) % 8) == neighbor_direction:
                    outflow_weight = 1.0 - outflow_weight

                if outflow_weight <= 0.0:
                    continue

                work_stack.push(neighbor_row * n_cols + neighbor_col)

        block_cache.flush_cache()
        original_feature_count = working_watershed_layer.GetFeatureCount()
        gdal.Polygonize(
            watershed_band, watershed_band, working_watershed_layer, 0,
            ["8CONNECTED=8"])
        #get the last n features and add the point field values
        #to the polygon feature
        n_added = (
            working_watershed_layer.GetFeatureCount() -
            original_feature_count)
        for added_feature_index in xrange(n_added):
            watershed_feature = working_watershed_layer.GetFeature(
                working_watershed_layer.GetFeatureCount() - 1 -
                added_feature_index)
            for index in xrange(point_feature.GetFieldCount()):
                watershed_feature.SetField(
                    index+1, point_feature.GetField(index))
            working_watershed_layer.SetFeature(watershed_feature)
            watershed_feature = None
        watershed_band.Fill(watershed_nodata)

    watershed_datasource = output_driver.CreateDataSource(
        watershed_out_uri)
    watershed_layer = watershed_datasource.CreateLayer(
        'serviceshed', output_sr, ogr.wkbPolygon)
    field = ogr.FieldDefn('pixel_valu', ogr.OFTReal)
    watershed_layer.CreateField(field)
    for index in xrange(outlet_defn.GetFieldCount()):
        field_defn = outlet_defn.GetFieldDefn(index)
        field_type = field_defn.GetType()
        # for some reason the default numerical width can be set to less than
        # the precision of the fields being copied over.  This addresses that
        # issue
        if field_type in [ogr.OFTInteger, ogr.OFTReal]:
            field_defn.SetWidth(24)
        watershed_layer.CreateField(field_defn)

    for feature_id in xrange(working_watershed_layer.GetFeatureCount()):
        watershed_feature = working_watershed_layer.GetFeature(feature_id)
        pixel_value = watershed_feature.GetField('pixel_valu')
        if pixel_value != watershed_nodata:
            # feature was created off a defined value, not nodata; save it.
            watershed_layer.CreateFeature(watershed_feature)
    # remove the pixel_valu field
    watershed_layer.DeleteField(0)
    watershed_layer = None
    watershed_datasource = None
    working_watershed_layer = None
    working_watershed_vector = None
    # delete the directory the working shapefile is in
    shutil.rmtree(temp_target_dir)
