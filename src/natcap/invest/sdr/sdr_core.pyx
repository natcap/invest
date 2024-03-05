# cython: profile=True
# cython: language_level=3
# distutils: language = c++
import logging
import os

import numpy
import pygeoprocessing
cimport numpy
cimport cython
from osgeo import gdal

from libc.time cimport time as ctime
from libcpp.stack cimport stack
from libcpp.vector cimport vector
from ..managed_raster.managed_raster cimport ManagedRaster
from ..managed_raster.managed_raster cimport ManagedFlowDirRaster
from ..managed_raster.managed_raster cimport is_close
from ..managed_raster.managed_raster cimport INFLOW_OFFSETS
# from ..managed_raster.managed_raster cimport route

cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

LOGGER = logging.getLogger(__name__)


cdef cppclass SeedFn:

    SeedFn() except +
    SeedFn(ManagedFlowDirRaster, ManagedRaster) except +

    ManagedFlowDirRaster flow_dir_raster
    ManagedRaster sed_dep_raster

    bint get "operator()"(int x, int y):
        cdef bint is_local_high_point = flow_dir_raster.is_local_high_point(x, y)
        cdef bint is_undefined = is_close(sed_dep_raster.get(x, y),
                                          sed_dep_raster.nodata)
        # if this can be a seed pixel and hasn't already been
        # calculated, put it on the stack
        return is_local_high_point and is_undefined


cdef cppclass RouteFn:

    RouteFn() except +
    RouteFn(
        ManagedFlowDirRaster,
        ManagedRaster, ManagedRaster,
        ManagedRaster, ManagedRaster) except +

    ManagedFlowDirRaster mfd_flow_direction_raster
    ManagedRaster e_prime_raster
    ManagedRaster f_raster
    ManagedRaster sdr_raster
    ManagedRaster sediment_deposition_raster


    vector[long] get "operator()"(int x, int y):
        """Perform routed operations for SDR.

        Args:
            x (int): column index of the pixel in raster space
            y (int): row index of the pixel in raster space

        Returns:
            list of integer indexes of pixels to push onto the stack.
            Flat indexes are used.
        """
        cdef float f_j_weighted_sum = 0
        cdef float downslope_sdr_weighted_sum = 0
        cdef float sdr_i, sdr_j, f_j
        cdef bint upslope_neighbors_processed
        cdef vector[long] next_pixels
        # (sum over j ∈ J of f_j * p(i,j) in the equation for t_i)
        # calculate the upslope f_j contribution to this pixel,
        # the weighted sum of flux flowing onto this pixel from
        # all neighbors
        for neighbor in (
                mfd_flow_direction_raster.get_upslope_neighbors(
                    x, y)):

            f_j = f_raster.get(neighbor.x, neighbor.y)
            if is_close(f_j, f_raster.nodata):
                continue

            # add the neighbor's flux value, weighted by the
            # flow proportion
            f_j_weighted_sum += neighbor.flow_proportion * f_j

        # calculate sum of SDR values of immediate downslope
        # neighbors, weighted by proportion of flow into each
        # neighbor
        # (sum over k ∈ K of SDR_k * p(i,k) in the equation above)
        for neighbor in (
                mfd_flow_direction_raster.get_downslope_neighbors(
                    x, y)):
            sdr_j = sdr_raster.get(neighbor.x, neighbor.y)
            if is_close(sdr_j, sdr_raster.nodata):
                continue
            if sdr_j == 0:
                # this means it's a stream, for SDR deposition
                # purposes, we set sdr to 1 to indicate this
                # is the last step on which to retain sediment
                sdr_j = 1

            downslope_sdr_weighted_sum += (
                sdr_j * neighbor.flow_proportion)

            # check if we can add neighbor j to the stack yet
            #
            # if there is a downslope neighbor it
            # couldn't have been pushed on the processing
            # stack yet, because the upslope was just
            # completed
            upslope_neighbors_processed = True
            # iterate over each neighbor-of-neighbor
            for neighbor_of_neighbor in (
                    mfd_flow_direction_raster.get_upslope_neighbors(
                        neighbor.x, neighbor.y)):
                # no need to push the one we're currently
                # calculating back onto the stack
                if (INFLOW_OFFSETS[neighbor_of_neighbor.direction] ==
                        neighbor.direction):
                    continue
                if is_close(
                        sediment_deposition_raster.get(
                            neighbor_of_neighbor.x, neighbor_of_neighbor.y
                        ), sediment_deposition_raster.nodata):
                    upslope_neighbors_processed = False
                    break
            # if all upslope neighbors of neighbor j are
            # processed, we can push j onto the stack.
            if upslope_neighbors_processed:
                next_pixels.push_back(
                    neighbor.y * mfd_flow_direction_raster.raster_x_size + neighbor.x)

        # nodata pixels should propagate to the results
        sdr_i = sdr_raster.get(x, y)
        e_prime_i = e_prime_raster.get(x, y)
        if not (is_close(sdr_i, sdr_raster.nodata) or
                is_close(e_prime_i, e_prime_raster.nodata)):
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
            sediment_deposition_raster.set(x, y, t_i)
            f_raster.set(x, y, f_i)

        return next_pixels

cdef route(flow_dir_path, SeedFn seed_fn, RouteFn route_fn):
    """
    Args:
        seed_fn (callable): function that accepts an (x, y) coordinate
            and returns a bool indicating if the pixel is a seed
        route_fn (callable): function that accepts an (x, y) coordinate
            and performs whatever routing operation is needed on that pixel.

    Returns:
        None
    """

    cdef long win_xsize, win_ysize, xoff, yoff, flat_index
    cdef int col_index, row_index, global_col, global_row
    cdef stack[long] processing_stack
    cdef long n_cols, n_rows
    cdef vector[long] next_pixels

    flow_dir_info = pygeoprocessing.get_raster_info(flow_dir_path)
    n_cols, n_rows = flow_dir_info['raster_size']

    for offset_dict in pygeoprocessing.iterblocks(
            (flow_dir_path, 1), offset_only=True, largest_block=0):
        # use cython variables to avoid python overhead of dict values
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']
        for row_index in range(win_ysize):
            global_row = yoff + row_index
            for col_index in range(win_xsize):

                global_col = xoff + col_index

                if seed_fn.get(global_col, global_row):
                    processing_stack.push(global_row * n_cols + global_col)

        while processing_stack.size() > 0:
            # loop invariant, we don't push a cell on the stack that
            # hasn't already been set for processing.
            flat_index = processing_stack.top()
            processing_stack.pop()
            global_row = flat_index // n_cols
            global_col = flat_index % n_cols

            next_pixels = route_fn.get(global_col, global_row)
            for index in next_pixels:
                processing_stack.push(index)


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

    cdef ManagedFlowDirRaster mfd_flow_direction_raster = ManagedFlowDirRaster(
        mfd_flow_direction_path, 1, False)
    cdef ManagedRaster e_prime_raster = ManagedRaster(
        e_prime_path, 1, False)
    cdef ManagedRaster sdr_raster = ManagedRaster(sdr_path, 1, False)
    cdef ManagedRaster f_raster = ManagedRaster(f_path, 1, True)
    cdef ManagedRaster sediment_deposition_raster = ManagedRaster(
        target_sediment_deposition_path, 1, True)

    cdef SeedFn seed_fn
    seed_fn.flow_dir_raster = mfd_flow_direction_raster
    seed_fn.sed_dep_raster = sediment_deposition_raster

    cdef RouteFn route_fn
    route_fn.mfd_flow_direction_raster = mfd_flow_direction_raster
    route_fn.e_prime_raster = e_prime_raster
    route_fn.f_raster = f_raster
    route_fn.sdr_raster = sdr_raster
    route_fn.sediment_deposition_raster = sediment_deposition_raster

    route(
        flow_dir_path=mfd_flow_direction_path,
        seed_fn=seed_fn,
        route_fn=route_fn)
    sediment_deposition_raster.close()



