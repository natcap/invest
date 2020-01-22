"""InVEST Model Logging tests."""

import time
import threading
import unittest
import tempfile
import shutil
import socket
import urllib
import os
import logging

try:
    from io import StringIO
    from urllib.parse import urlencode
except ImportError:
    str = unicode
    from StringIO import StringIO
    from urllib import urlencode

import numpy.testing


SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-sample-data')


class ModelLoggingTests(unittest.TestCase):
    """Tests for the InVEST model logging framework."""

    def setUp(self):
        """Initalize a workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_bounding_boxes(self):
        """Usage logger test that we can extract bounding boxes."""
        from natcap.invest import utils
        from natcap.invest.ui import usage

        freshwater_dir = os.path.join(SAMPLE_DATA, 'Base_Data', 'Freshwater')
        model_args = {
            'raster': os.path.join(freshwater_dir, 'dem'),
            'vector': os.path.join(freshwater_dir, 'subwatersheds.shp'),
            'not_a_gis_input': 'foobar'
        }

        output_logfile = os.path.join(self.workspace_dir, 'logfile.txt')
        with utils.log_to_file(output_logfile):
            bb_inter, bb_union = usage._calculate_args_bounding_box(model_args)

        numpy.testing.assert_allclose(
            bb_inter, [-123.584877, 44.273852, -123.400091, 44.726233])
        numpy.testing.assert_allclose(
            bb_union, [-123.658275, 44.415778, -123.253863, 44.725814])

        # Verify that no errors were raised in calculating the bounding boxes.
        self.assertTrue('ERROR' not in open(output_logfile).read(),
                        'Exception logged when there should not have been.')
