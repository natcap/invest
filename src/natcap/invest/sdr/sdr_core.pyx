import logging
import os

import pygeoprocessing
cimport cython
from osgeo import gdal

from libc.time cimport time as ctime
from libcpp.stack cimport stack
from ..managed_raster.managed_raster cimport ManagedRaster
from ..managed_raster.managed_raster cimport NeighborTuple
from ..managed_raster.managed_raster cimport ManagedFlowDirRaster
from ..managed_raster.managed_raster cimport is_close
from ..managed_raster.managed_raster cimport INFLOW_OFFSETS
from ..managed_raster.managed_raster cimport DownslopeNeighborIterator
from ..managed_raster.managed_raster cimport UpslopeNeighborIterator


cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

LOGGER = logging.getLogger(__name__)

def calculate_sediment_deposition(
        mfd_flow_direction_path, e_prime_path, f_path, sdr_path,
        target_sediment_deposition_path):
    """Calculate sediment deposition layer.

    This algorithm outputs both sediment deposition (t_i) and flux (f_i)::

        t_i  =      dr_i  * (sum over j ∈ J of f_j * p(i,j)) + E'_i

        f_i  = (1 - dr_i) * (sum over j ∈ J of f_j * p(i,j)) + E'_i


                (sum over k ∈ K of SDR_k * p(i,k)) - SDR_i
        dr_i = --------------------------------------------
                              (1 - SDR_i)

    where:

    - ``p(i,j)`` is the proportion of flow from pixel ``i`` into pixel ``j``
    - ``J`` is the set of pixels that are immediate upslope neighbors of
      pixel ``i``
    - ``K`` is the set of pixels that are immediate downslope neighbors of
      pixel ``i``
    - ``E'`` is ``USLE * (1 - SDR)``, the amount of sediment loss from pixel
      ``i`` that doesn't reach a stream (``e_prime_path``)
    - ``SDR`` is the sediment delivery ratio (``sdr_path``)

    ``f_i`` is recursively defined in terms of ``i``'s upslope neighbors.
    The algorithm begins from seed pixels that are local high points and so
    have no upslope neighbors. It works downslope from each seed pixel,
    only adding a pixel to the stack when all its upslope neighbors are
    already calculated.

    Note that this function is designed to be used in the context of the SDR
    model. Because the algorithm is recursive upslope and downslope of each
    pixel, nodata values in the SDR input would propagate along the flow path.
    This case is not handled because we assume the SDR and flow dir inputs
    will come from the SDR model and have nodata in the same places.

    Args:
        mfd_flow_direction_path (string): a path to a raster with
            pygeoprocessing.routing MFD flow direction values.
        e_prime_path (string): path to a raster that shows sources of
            sediment that wash off a pixel but do not reach the stream.
        f_path (string): path to a raster that shows the sediment flux
            on a pixel for sediment that does not reach the stream.
        sdr_path (string): path to Sediment Delivery Ratio raster.
        target_sediment_deposition_path (string): path to created that
            shows where the E' sources end up across the landscape.

    Returns:
        None.

    """
    LOGGER.info('Calculate sediment deposition')
    cdef float target_nodata = -1
    pygeoprocessing.new_raster_from_base(
        mfd_flow_direction_path, target_sediment_deposition_path,
        gdal.GDT_Float32, [target_nodata])
    pygeoprocessing.new_raster_from_base(
        mfd_flow_direction_path, f_path,
        gdal.GDT_Float32, [target_nodata])

    mfd_flow_dir_path_bytes = mfd_flow_direction_path.encode('UTF-8')
    cdef char* mfd_flow_dir_path_char_ptr = mfd_flow_dir_path_bytes
    cdef ManagedFlowDirRaster mfd_flow_direction_raster = ManagedFlowDirRaster(
        mfd_flow_dir_path_char_ptr, 1, False)

    e_prime_path_bytes = e_prime_path.encode('UTF-8')
    cdef char* e_prime_path_char_ptr = e_prime_path_bytes
    cdef ManagedRaster e_prime_raster = ManagedRaster(
        e_prime_path_char_ptr, 1, False)

    sdr_path_bytes = sdr_path.encode('UTF-8')
    cdef char* sdr_path_char_ptr = sdr_path_bytes
    cdef ManagedRaster sdr_raster = ManagedRaster(sdr_path_char_ptr, 1, False)

    f_path_bytes = f_path.encode('UTF-8')
    cdef char* f_path_char_ptr = f_path_bytes
    cdef ManagedRaster f_raster = ManagedRaster(f_path_char_ptr, 1, True)

    sed_dep_path_bytes = target_sediment_deposition_path.encode('UTF-8')
    cdef char* sed_dep_path_char_ptr = sed_dep_path_bytes
    cdef ManagedRaster sediment_deposition_raster = ManagedRaster(
        sed_dep_path_char_ptr, 1, True)

    cdef long n_cols, n_rows
    flow_dir_info = pygeoprocessing.get_raster_info(mfd_flow_direction_path)
    n_cols, n_rows = flow_dir_info['raster_size']
    cdef int mfd_nodata = 0
    cdef stack[long] processing_stack
    cdef float sdr_nodata = pygeoprocessing.get_raster_info(
        sdr_path)['nodata'][0]
    cdef float e_prime_nodata = pygeoprocessing.get_raster_info(
        e_prime_path)['nodata'][0]
    cdef long col_index, row_index, win_xsize, win_ysize, xoff, yoff
    cdef long global_col, global_row, j, k
    cdef int xs, ys
    cdef long flat_index
    cdef long seed_col = 0
    cdef long seed_row = 0
    cdef long neighbor_row, neighbor_col, ds_neighbor_row, ds_neighbor_col
    cdef int flow_val, neighbor_flow_val, ds_neighbor_flow_val
    cdef int flow_weight, neighbor_flow_weight
    cdef float flow_sum, neighbor_flow_sum
    cdef float downslope_sdr_weighted_sum, sdr_i, sdr_j
    cdef float p_j, p_val
    cdef unsigned long n_pixels_processed = 0
    cdef time_t last_log_time = ctime(NULL)
    cdef float f_j_weighted_sum
    cdef NeighborTuple neighbor2

    for offset_dict in pygeoprocessing.iterblocks(
            (mfd_flow_direction_path, 1), offset_only=True, largest_block=0):
        # use cython variables to avoid python overhead of dict values
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']
        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            LOGGER.info('Sediment deposition %.2f%% complete', 100 * (
                n_pixels_processed / float(n_cols * n_rows)))

        for row_index in range(win_ysize):
            ys = yoff + row_index
            for col_index in range(win_xsize):
                xs = xoff + col_index

                if mfd_flow_direction_raster.get(xs, ys) == mfd_nodata:
                    continue

                # if this can be a seed pixel and hasn't already been
                # calculated, put it on the stack
                if (mfd_flow_direction_raster.is_local_high_point(xs, ys) and
                        sediment_deposition_raster.get(xs, ys) == target_nodata):
                    processing_stack.push(ys * n_cols + xs)

                while processing_stack.size() > 0:
                    # loop invariant: cell has all upslope neighbors
                    # processed. this is true for seed pixels because they
                    # have no upslope neighbors.
                    flat_index = processing_stack.top()
                    processing_stack.pop()
                    global_row = flat_index // n_cols
                    global_col = flat_index % n_cols

                    # (sum over j ∈ J of f_j * p(i,j) in the equation for t_i)
                    # calculate the upslope f_j contribution to this pixel,
                    # the weighted sum of flux flowing onto this pixel from
                    # all neighbors
                    f_j_weighted_sum = 0
                    up_iterator = UpslopeNeighborIterator(
                        mfd_flow_direction_raster, global_col, global_row)
                    neighbor = up_iterator.next()
                    while neighbor.direction < 8:

                        f_j = f_raster.get(neighbor.x, neighbor.y)
                        if is_close(f_j, target_nodata):
                            neighbor = up_iterator.next()
                            continue

                        # add the neighbor's flux value, weighted by the
                        # flow proportion
                        f_j_weighted_sum += neighbor.flow_proportion * f_j
                        neighbor = up_iterator.next()

                    # calculate sum of SDR values of immediate downslope
                    # neighbors, weighted by proportion of flow into each
                    # neighbor
                    # (sum over k ∈ K of SDR_k * p(i,k) in the equation above)
                    downslope_sdr_weighted_sum = 0
                    dn_iterator = DownslopeNeighborIterator(
                        mfd_flow_direction_raster, global_col, global_row)

                    neighbor2 = <NeighborTuple>dn_iterator.next()
                    while neighbor2.direction < 8:

                        sdr_j = sdr_raster.get(neighbor2.x, neighbor2.y)
                        if is_close(sdr_j, sdr_nodata):
                            neighbor2 = dn_iterator.next()
                            continue
                        if sdr_j == 0:
                            # this means it's a stream, for SDR deposition
                            # purposes, we set sdr to 1 to indicate this
                            # is the last step on which to retain sediment
                            sdr_j = 1

                        downslope_sdr_weighted_sum += (
                            sdr_j * neighbor2.flow_proportion)
                        # check if we can add neighbor j to the stack yet
                        #
                        # if there is a downslope neighbor it
                        # couldn't have been pushed on the processing
                        # stack yet, because the upslope was just
                        # completed
                        upslope_neighbors_processed = 1
                        # iterate over each neighbor-of-neighbor
                        #
                        up_iterator = UpslopeNeighborIterator(
                            mfd_flow_direction_raster, neighbor2.x, neighbor2.y)
                        neighbor_of_neighbor = up_iterator.next_skip(neighbor2.direction)
                        while neighbor_of_neighbor.direction < 8:
                            a = sediment_deposition_raster.get(neighbor_of_neighbor.x, neighbor_of_neighbor.y)
                            if (a == target_nodata):
                                upslope_neighbors_processed = 0
                                break
                            neighbor_of_neighbor = up_iterator.next_skip(neighbor2.direction)
                        # if all upslope neighbors of neighbor j are
                        # processed, we can push j onto the stack.
                        if upslope_neighbors_processed:
                            processing_stack.push(
                                neighbor2.y * n_cols + neighbor2.x)

                        neighbor2 = dn_iterator.next()

                    # nodata pixels should propagate to the results
                    sdr_i = sdr_raster.get(global_col, global_row)
                    if is_close(sdr_i, sdr_nodata):
                        continue
                    e_prime_i = e_prime_raster.get(global_col, global_row)
                    if is_close(e_prime_i, e_prime_nodata):
                        continue

                    if dn_iterator.flow_dir_sum:
                        downslope_sdr_weighted_sum /= dn_iterator.flow_dir_sum

                    # This condition reflects property A in the user's guide.
                    if downslope_sdr_weighted_sum < sdr_i:
                        # i think this happens because of our low resolution
                        # flow direction, it's okay to zero out.
                        downslope_sdr_weighted_sum = sdr_i

                    # these correspond to the full equations for
                    # dr_i, t_i, and f_i given in the docstring
                    if sdr_i == 1:
                        # This reflects property B in the user's guide and is
                        # an edge case to avoid division-by-zero.
                        dr_i = 1
                    else:
                        dr_i = (downslope_sdr_weighted_sum - sdr_i) / (1 - sdr_i)

                    # Lisa's modified equations
                    t_i = dr_i * f_j_weighted_sum  # deposition, a.k.a trapped sediment
                    f_i = (1 - dr_i) * f_j_weighted_sum + e_prime_i  # flux

                    # On large flow paths, it's possible for dr_i, f_i and t_i
                    # to have very small negative values that are numerically
                    # equivalent to 0. These negative values were raising
                    # questions on the forums and it's easier to clamp the
                    # values here than to explain IEEE 754.
                    if dr_i < 0:
                        dr_i = 0
                    if t_i < 0:
                        t_i = 0
                    if f_i < 0:
                        f_i = 0

                    sediment_deposition_raster.set(global_col, global_row, t_i)
                    f_raster.set(global_col, global_row, f_i)
        n_pixels_processed += win_xsize * win_ysize

    LOGGER.info('Sediment deposition 100% complete')
    sediment_deposition_raster.close()
