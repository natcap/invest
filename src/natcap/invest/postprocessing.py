"""This is a collection of postprocessing functions that are useful for some
    of the InVEST models."""

import logging

import numpy as np
import pylab
from osgeo import gdal

LOGGER = logging.getLogger('natcap.invest.postprocessing')

def plot_flow_direction(flow_dataset_uri, output_uri):
    """Generates a quiver plot (arrows on a grid) of a flow matrix

    flow_dataset_uri - a uri to a GDAL compatable raster whose values are
        radians indicating the direction of outward flow.
    output_uri - the location to disk to save the resulting plot png file

    returns nothing"""

    LOGGER.info('Loading %s' % flow_dataset_uri)
    flow_dataset = gdal.Open(flow_dataset_uri, gdal.GA_ReadOnly)
    flow_matrix = (flow_dataset.GetRasterBand(1).ReadAsArray(0, 0,
        flow_dataset.RasterXSize, flow_dataset.RasterYSize))
    LOGGER.info('Done loading %s' % flow_dataset_uri)

    LOGGER.info('Starting plot of flow direction')
    pylab.figure()
    steps = flow_matrix.size
    xmax = flow_matrix.shape[0]
    ymax = flow_matrix.shape[1]
    steps = 10
    pylab.quiver(np.sin(flow_matrix[0:xmax:steps, 0:ymax:steps]),
                 np.cos(flow_matrix[0:xmax:steps, 0:ymax:steps]))
    pylab.axes().set_aspect('equal')
    LOGGER.info('Saving plot as %s' % output_uri)
    pylab.savefig(output_uri, dpi=1600,
                  units='x')
    LOGGER.info('Done with plot of flow direction')
