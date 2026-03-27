import tempfile
import logging
import os

import numpy
import pygeoprocessing
cimport numpy
cimport cython
from osgeo import gdal

from ..managed_raster.managed_raster cimport D8
from ..managed_raster.managed_raster cimport MFD
from .retention cimport calculate_retention

cdef extern from "time.h" nogil:
    ctypedef int time_t
    time_t time(time_t*)

LOGGER = logging.getLogger(__name__)

# Within a stream, the effective retention is 0
cdef int STREAM_EFFECTIVE_RETENTION = 0

def ndr_eff_calculation(
        flow_direction_path, stream_path, retention_eff_lulc_path,
        crit_len_path, effective_retention_path, algorithm):
    """Calculate flow downhill effective_retention to the channel.

        Args:
            flow_direction_path (string): a path to a raster with
                pygeoprocessing.routing flow direction values (MFD or D8).
            stream_path (string): a path to a raster where 1 indicates a
                stream all other values ignored must be same dimensions and
                projection as flow_direction_path.
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
        flow_direction_path, effective_retention_path, gdal.GDT_Float32,
        [effective_retention_nodata])
    fp, to_process_flow_directions_path = tempfile.mkstemp(
        suffix='.tif', prefix='flow_to_process',
        dir=os.path.dirname(effective_retention_path))
    os.close(fp)
    algorithm = algorithm.lower()

    # create direction raster in bytes
    def _mfd_to_flow_dir_op(mfd_array):
        result = numpy.zeros(mfd_array.shape, dtype=numpy.uint8)
        for i in range(8):
            result[:] |= ((((mfd_array >> (i*4)) & 0xF) > 0) << i).astype(numpy.uint8)
        return result

    # create direction raster in bytes
    def _d8_to_flow_dir_op(d8_array):
        result = numpy.zeros(d8_array.shape, dtype=numpy.uint8)
        for i in range(8):
            result[d8_array == i] = 1 << i
        return result

    flow_dir_op = _mfd_to_flow_dir_op if algorithm == 'mfd' else _d8_to_flow_dir_op

    # convert mfd raster to binary mfd
    # each value is an 8-digit binary number
    # where 1 indicates that the pixel drains in that direction
    # and 0 indicates that it does not drain in that direction
    pygeoprocessing.raster_calculator(
        [(flow_direction_path, 1)], flow_dir_op,
        to_process_flow_directions_path, gdal.GDT_Byte, None)

    if algorithm == 'mfd':
        calculate_retention[MFD](
            flow_direction_path.encode('utf-8'),
            stream_path.encode('utf-8'),
            retention_eff_lulc_path.encode('utf-8'),
            crit_len_path.encode('utf-8'),
            to_process_flow_directions_path.encode('utf-8'),
            effective_retention_path.encode('utf-8'))
    else: # D8
        calculate_retention[D8](
            flow_direction_path.encode('utf-8'),
            stream_path.encode('utf-8'),
            retention_eff_lulc_path.encode('utf-8'),
            crit_len_path.encode('utf-8'),
            to_process_flow_directions_path.encode('utf-8'),
            effective_retention_path.encode('utf-8'))
