"""Example model for demonstrating basic API and testing functionality."""
import os
import logging

from osgeo import gdal
import pygeoprocessing
import numpy

LOGGER = logging.getLogger('natcap.invest._example_model')


def execute(args):
    """
    An example model.

    Demonstrates the interface and testing of the InVEST suite of models.

    Parameters:
        args (dict): A dict of key-value pairs.
            args['workspace_dir'] (string): A URI to the output workspace.
            args['example_lulc'] (string): A URI to the LULC to process.
            args['suffix'] (string, optional): A suffix to be appended to
                the output filenames.

    Returns:
        None
    """
    dir_registry = {
        'intermediate': os.path.join(args['workspace_dir'],
                                     'intermediate_outputs'),
        'output': args['workspace_dir']
    }

    def _add_suffix(path):
        """Add a suffix to the input path."""
        try:
            suffix = args['suffix']
        except KeyError:
            return path

        if not suffix.startswith('_'):
            suffix = '_' + suffix
        filepath_prefix, ext = os.path.splitext(path)
        return filepath_prefix + suffix + ext

    filepath_registry = {
        'sum': _add_suffix(os.path.join(dir_registry['output'], 'sum.tif')),
    }

    # Create directories if needed.
    pygeoprocessing.create_directories(dir_registry.values())

    _simple_op(args['example_lulc'], filepath_registry['sum'])


def _simple_op(lulc_filepath, out_filepath):
    """
    Add 5 to any pixel values that are between 1 and 9 (inclusive).

    Parameters:
        lulc_filepath (string): A filepath to a landcover raster
        out_filepath (string): The filepath to which the output raster will be
            written.

    Returns:
        None
    """
    # An example, simple vectorize_datasets operation.
    lulc_nodata = pygeoprocessing.get_nodata_from_uri(lulc_filepath)
    lulc_cell_size = pygeoprocessing.get_cell_size_from_uri(lulc_filepath)

    # Range, between 1 and 9 (inclusive)
    valid_values = numpy.arange(1, 10)

    def simple_local_op(lulc_values):
        """Add 5 to any landcover codes that have a value between 1 and 9."""
        nodata_mask = (lulc_values != lulc_nodata)
        valid_lulc_codes = numpy.in1d(lulc_values,
                                      valid_values).reshape(lulc_values.shape)
        return numpy.where(nodata_mask & valid_lulc_codes, lulc_values + 5,
                           numpy.where(nodata_mask, lulc_values, lulc_nodata))

    pygeoprocessing.vectorize_datasets(
        [lulc_filepath], simple_local_op, out_filepath, gdal.GDT_Int32,
        lulc_nodata, lulc_cell_size, 'intersection', vectorize_op=False)
