import logging

import pygeoprocessing
cimport cython
from osgeo import gdal

from ..managed_raster.managed_raster cimport D8
from ..managed_raster.managed_raster cimport MFD
from .sediment_deposition cimport run_sediment_deposition


LOGGER = logging.getLogger(__name__)

def calculate_sediment_deposition(
        flow_direction_path, e_prime_path, f_path, sdr_path,
        target_sediment_deposition_path, algorithm):
    """Calculate sediment deposition layer.

    This algorithm outputs both sediment deposition (t_i) and flux (f_i)::

        t_i  =      dt_i  * (sum over j ∈ J of f_j * p(j,i))

        f_i  = (1 - dt_i) * (sum over j ∈ J of f_j * p(j,i)) + E'_i


                (sum over k ∈ K of SDR_k * p(i,k)) - SDR_i
        dt_i = --------------------------------------------
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
        flow_direction_path (string): a path to a flow direction raster,
            in either MFD or D8 format. Specify with the ``algorithm`` arg.
        e_prime_path (string): path to a raster that shows sources of
            sediment that wash off a pixel but do not reach the stream.
        f_path (string): path to a raster that shows the sediment flux
            on a pixel for sediment that does not reach the stream.
        sdr_path (string): path to Sediment Delivery Ratio raster.
        target_sediment_deposition_path (string): path to created that
            shows where the E' sources end up across the landscape.
        algorithm (string): MFD or D8

    Returns:
        None.

    """
    LOGGER.info('Calculate sediment deposition')
    cdef float target_nodata = -1
    pygeoprocessing.new_raster_from_base(
        flow_direction_path, target_sediment_deposition_path,
        gdal.GDT_Float32, [target_nodata])
    pygeoprocessing.new_raster_from_base(
        flow_direction_path, f_path,
        gdal.GDT_Float32, [target_nodata])

    if algorithm.lower() == 'd8':
        run_sediment_deposition[D8](
            flow_direction_path.encode('utf-8'), e_prime_path.encode('utf-8'),
            f_path.encode('utf-8'), sdr_path.encode('utf-8'),
            target_sediment_deposition_path.encode('utf-8'))
    else:
        run_sediment_deposition[MFD](
            flow_direction_path.encode('utf-8'), e_prime_path.encode('utf-8'),
            f_path.encode('utf-8'), sdr_path.encode('utf-8'),
            target_sediment_deposition_path.encode('utf-8'))
