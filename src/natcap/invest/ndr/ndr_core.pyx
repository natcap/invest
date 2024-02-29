# cython: profile=False
# cython: language_level=2
import tempfile
import logging
import os
import collections

import numpy
import pygeoprocessing
cimport numpy
cimport cython
from osgeo import gdal

from libcpp.stack cimport stack
from libc.math cimport exp
from ..managed_raster.managed_raster cimport _ManagedRaster
from ..managed_raster.managed_raster cimport ManagedFlowDirRaster
from ..managed_raster.managed_raster cimport is_close
from ..managed_raster.managed_raster cimport INFLOW_OFFSETS
from ..managed_raster.managed_raster cimport route

cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

LOGGER = logging.getLogger(__name__)

# Within a stream, the effective retention is 0
cdef int STREAM_EFFECTIVE_RETENTION = 0

def ndr_eff_calculation(
        mfd_flow_direction_path, stream_path, retention_eff_lulc_path,
        crit_len_path, effective_retention_path):
    """Calculate flow downhill effective_retention to the channel.

        Args:
            mfd_flow_direction_path (string): a path to a raster with
                pygeoprocessing.routing MFD flow direction values.
            stream_path (string): a path to a raster where 1 indicates a
                stream all other values ignored must be same dimensions and
                projection as mfd_flow_direction_path.
            retention_eff_lulc_path (string): a path to a raster indicating
                the maximum retention efficiency that the landcover on that
                pixel can accumulate.
            crit_len_path (string): a path to a raster indicating the critical
                length of the retention efficiency that the landcover on this
                pixel.
            effective_retention_path (string): path to a raster that is
                created by this call that contains a per-pixel effective
                sediment retention to the stream.

        Returns:
            None.

    """
    cdef float effective_retention_nodata = -1.0
    pygeoprocessing.new_raster_from_base(
        mfd_flow_direction_path, effective_retention_path, gdal.GDT_Float32,
        [effective_retention_nodata])
    fp, to_process_flow_directions_path = tempfile.mkstemp(
        suffix='.tif', prefix='flow_to_process',
        dir=os.path.dirname(effective_retention_path))
    os.close(fp)

    cdef _ManagedRaster stream_raster = _ManagedRaster(stream_path, 1, False)
    cdef _ManagedRaster crit_len_raster = _ManagedRaster(
        crit_len_path, 1, False)
    cdef _ManagedRaster retention_eff_lulc_raster = _ManagedRaster(
        retention_eff_lulc_path, 1, False)
    cdef _ManagedRaster effective_retention_raster = _ManagedRaster(
        effective_retention_path, 1, True)
    cdef ManagedFlowDirRaster mfd_flow_direction_raster = ManagedFlowDirRaster(
        mfd_flow_direction_path, 1, False)

    # cell sizes must be square, so no reason to test at this point.
    cdef float cell_size = abs(pygeoprocessing.get_raster_info(
        stream_path)['pixel_size'][0])

    # create direction raster in bytes
    def _mfd_to_flow_dir_op(mfd_array):
        result = numpy.zeros(mfd_array.shape, dtype=numpy.int8)
        for i in range(8):
            result[:] |= (((mfd_array >> (i*4)) & 0xF) > 0) << i
        return result

    # convert mfd raster to binary mfd
    # each value is an 8-digit binary number
    # where 1 indicates that the pixel drains in that direction
    # and 0 indicates that it does not drain in that direction
    pygeoprocessing.raster_calculator(
        [(mfd_flow_direction_path, 1)], _mfd_to_flow_dir_op,
        to_process_flow_directions_path, gdal.GDT_Byte, None)

    cdef _ManagedRaster to_process_flow_directions_raster = _ManagedRaster(
        to_process_flow_directions_path, 1, True)

    def ndr_seed_fn(col, row):
        """Determine if a given pixel can be a seed pixel.

        To be a seed pixel, it must drain to nodata or off the edge of the raster.

        Args:
            col (int): column index of the pixel in raster space
            row (int): row index of the pixel in raster space

        Returns:
            True if the pixel qualifies as a seed pixel, False otherwise
        """
        cdef bint should_seed = False
        cdef int outflow_dirs = <int>to_process_flow_directions_raster.get(col, row)
        # see if this pixel drains to nodata or the edge, if so it's
        # a drain
        for neighbor in (
                mfd_flow_direction_raster.get_downslope_neighbors(
                    col, row, skip_oob=False)):
            if (neighbor.x < 0 or neighbor.x >= mfd_flow_direction_raster.raster_x_size or
                neighbor.y < 0 or neighbor.y >= mfd_flow_direction_raster.raster_y_size or
                to_process_flow_directions_raster.get(
                    neighbor.x, neighbor.y) == 0):
                should_seed = True
                outflow_dirs &= ~(1 << neighbor.direction)
        if should_seed:
            # mark all outflow directions processed
            to_process_flow_directions_raster.set(
                col, row, outflow_dirs)
        return should_seed

    def ndr_route_fn(col, row):
        """Perform routed operations for NDR.

        Args:
            col (int): column index of the pixel in raster space
            row (int): row index of the pixel in raster space

        Returns:
            list of integer indexes of pixels to push onto the stack.
            Flat indexes are used.
        """
        to_push = []
        cdef int neighbor_outflow_dir, neighbor_outflow_dir_mask, neighbor_process_flow_dir
        cdef float current_step_factor, step_size
        cdef float working_retention_eff
        cdef float crit_len = <float>crit_len_raster.get(col, row)
        cdef float retention_eff_lulc = retention_eff_lulc_raster.get(col, row)
        cdef int flow_dir = <int>mfd_flow_direction_raster.get(col, row)
        if stream_raster.get(col, row) == 1:
            # if it's a stream effective retention is 0.
            effective_retention_raster.set(col, row, STREAM_EFFECTIVE_RETENTION)
        elif (is_close(crit_len, crit_len_raster.nodata) or
              is_close(retention_eff_lulc, retention_eff_lulc_raster.nodata) or
              flow_dir == 0):
            # if it's nodata, effective retention is nodata.
            effective_retention_raster.set(
                col, row, effective_retention_nodata)
        else:
            working_retention_eff = 0.0
            has_outflow = False
            for neighbor in mfd_flow_direction_raster.get_downslope_neighbors(
                    col, row, skip_oob=False):
                has_outflow = True
                if (neighbor.x < 0 or neighbor.x >= mfd_flow_direction_raster.raster_x_size or
                    neighbor.y < 0 or neighbor.y >= mfd_flow_direction_raster.raster_y_size):
                    continue

                if neighbor.direction % 2 == 1:
                    step_size = <float>(cell_size * 1.41421356237)
                else:
                    step_size = cell_size
                # guard against a critical length factor that's 0
                if crit_len > 0:
                    current_step_factor = <float>(
                        exp(-5 * step_size / crit_len))
                else:
                    current_step_factor = 0.0

                neighbor_effective_retention = (
                    effective_retention_raster.get(
                        neighbor.x, neighbor.y))

                # Case 1: downslope neighbor is a stream pixel
                if neighbor_effective_retention == STREAM_EFFECTIVE_RETENTION:
                    intermediate_retention = (
                        retention_eff_lulc * (1 - current_step_factor))

                # Case 2: the current LULC's retention exceeds the neighbor's retention.
                elif retention_eff_lulc > neighbor_effective_retention:
                    intermediate_retention = (
                        (neighbor_effective_retention * current_step_factor) +
                        (retention_eff_lulc * (1 - current_step_factor)))

                # Case 3: the other 2 cases have not been hit.
                else:
                    intermediate_retention = neighbor_effective_retention

                working_retention_eff += (
                    intermediate_retention * neighbor.flow_proportion)

            if has_outflow:
                effective_retention_raster.set(
                    col, row, working_retention_eff)
            else:
                raise Exception("got to a cell that has no outflow!")
        # search upslope to see if we need to push a cell on the stack
        # for i in range(8):
        for neighbor in mfd_flow_direction_raster.get_upslope_neighbors(col, row):
            neighbor_outflow_dir = INFLOW_OFFSETS[neighbor.direction]
            neighbor_outflow_dir_mask = 1 << neighbor_outflow_dir
            neighbor_process_flow_dir = <int>(
                to_process_flow_directions_raster.get(
                    neighbor.x, neighbor.y))
            if neighbor_process_flow_dir == 0:
                # skip, due to loop invariant this must be a nodata pixel
                continue
            if neighbor_process_flow_dir & neighbor_outflow_dir_mask == 0:
                # no outflow
                continue
            # mask out the outflow dir that this iteration processed
            neighbor_process_flow_dir &= ~neighbor_outflow_dir_mask
            to_process_flow_directions_raster.set(
                neighbor.x, neighbor.y, neighbor_process_flow_dir)
            if neighbor_process_flow_dir == 0:
                # if 0 then all downslope have been processed,
                # push on stack, otherwise another downslope pixel will
                # pick it up
                to_push.append(neighbor.y * mfd_flow_direction_raster.raster_x_size + neighbor.x)
        return to_push


    route(mfd_flow_direction_path, ndr_seed_fn, ndr_route_fn)

    to_process_flow_directions_raster.close()
    os.remove(to_process_flow_directions_path)
