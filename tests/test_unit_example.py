"""Tests for natcap.invest._example_model.

This module is intended only to demonstrate how a model might be tested given
common needs of InVEST models.  Key things to note:

    * Sample data creation that's repeated outside a test method has its own
      function.
    * The module-under-test (natcap.invest._example_model) is imported in
      each test method.
    * Each test method is responsible for the appropriate cleanup of created
      sample data and workspaces.
"""
import unittest
import tempfile
import shutil
import os

from osgeo import gdal
import numpy
import numpy.testing
import pygeoprocessing.testing
from pygeoprocessing.testing import sampledata


def _create_lulc(matrix=None):
    """
    Create an LULC for the _example_model.

    This LULC is created with the following characteristics:
        * SRS is in the SRS_WILLAMETTE.
        * Nodata is -1.
        * Raster type is `gdal.GDT_Int32`
        * Pixel size is 30m

    Parameters:
        matrix=None (numpy.array): A numpy array to use as a landcover matrix.
            The output raster created will be saved with these pixel values.
            If None, a default matrix will be used.

    Returns:
        A string filepath to a new LULC raster on disk.
    """
    if matrix is None:
        lulc_matrix = numpy.array([
            [0, 1, 2, 3, 4],
            [1, 2, 4, 5, 11],
            [1, -1, -1, -1, -1],
            [0, 1, 2, 3, 4]], numpy.int32)
    else:
        lulc_matrix = matrix
    lulc_nodata = -1
    srs = sampledata.SRS_WILLAMETTE
    return pygeoprocessing.testing.create_raster_on_disk(
        [lulc_matrix], srs.origin, srs.projection, lulc_nodata,
        srs.pixel_size(30), datatype=gdal.GDT_Int32)


class ExampleUnitTests(unittest.TestCase):

    """Tests for natcap.invest._example_model."""

    def test_execute(self):
        """Example execution to ensure correctness when called via execute."""
        from natcap.invest import _example_model

        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'example_lulc': _create_lulc(),
        }
        _example_model.execute(args)

        expected_matrix = numpy.array([
            [0, 6, 7, 8, 9],
            [6, 7, 9, 10, 11],
            [6, -1, -1, -1, -1],
            [0, 6, 7, 8, 9]], numpy.int32)
        expected_raster = _create_lulc(expected_matrix)
        sum_raster = os.path.join(args['workspace_dir'], 'sum.tif')
        pygeoprocessing.testing.assert_rasters_equal(sum_raster,
                                                     expected_raster)

        shutil.rmtree(args['workspace_dir'])
        for filename in [args['example_lulc'], expected_raster]:
            os.remove(filename)

    def test_execute_with_suffix(self):
        """When a suffix is added, verify it's added correctly."""
        from natcap.invest import _example_model
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'example_lulc': _create_lulc(),
            'suffix': 'foo',
        }
        _example_model.execute(args)

        self.assertTrue(os.path.exists(os.path.join(args['workspace_dir'],
                                                    'sum_foo.tif')))

    def test_execute_with_suffix_and_underscore(self):
        """When the user's suffix has an underscore, don't add another one."""
        from natcap.invest import _example_model
        args = {
            'workspace_dir': tempfile.mkdtemp(),
            'example_lulc': _create_lulc(),
            'suffix': 'foo',
        }
        _example_model.execute(args)

        self.assertTrue(os.path.exists(os.path.join(args['workspace_dir'],
                                                    'sum_foo.tif')))
