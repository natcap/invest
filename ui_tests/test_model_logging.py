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

from pygeoprocessing.testing import scm
import numpy.testing

import Pyro4
import mock

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

    def test_pyro_server(self):
        """Usage logger test server as an RPC."""
        from natcap.invest.ui import usage_logger
        # attempt to get an open port; could result in race condition but
        # will be okay for a test. if this test ever fails because of port
        # in use, that's probably why
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
        sock = None

        server_args = {
            'hostname': 'localhost',
            'port': port,
        }

        server_thread = threading.Thread(
            target=usage_logger.execute, args=(server_args,))
        server_thread.daemon = True
        server_thread.start()
        time.sleep(1)

        logging_server = Pyro4.Proxy(
            "PYRO:natcap.invest.remote_logging@localhost:%d" % port)
        # this makes for an easy expected result
        sample_data = dict(
            (key_field, key_field) for key_field in
            usage_logger.LoggingServer._LOG_FIELD_NAMES)
        logging_server.log_invest_run(sample_data, 'log')

    def test_add_exit_status(self):
        """Usage logger record run and then exit and verify they are added."""
        from natcap.invest.ui import usage_logger

        logging_server = usage_logger.LoggingServer()

        # set up a sample dict whose values are identical to its keys
        # this makes for an easy expected result
        sample_data = dict(
            (key_field, key_field) for key_field in
            usage_logger.LoggingServer._LOG_FIELD_NAMES)

        # This mock needs only to return a valid json string with the expected
        # key-value pairs.
        json_string = str('{"START": "http://foo.bar", "FINISH": "http://foo.bar"}')
        with mock.patch(
                'natcap.invest.ui.usage_logger.urlopen',
                return_value=StringIO(json_string)) as mock_obj:
            logging_server.log_invest_run(sample_data, 'log')
        self.assertEqual(mock_obj.call_count, 2)
        sample_data['ip_address'] = 'local'
        self.assertEqual(
            sorted(mock_obj.call_args[0][0].data.decode('utf-8').split('&')),
            sorted(urlencode(sample_data).split('&')))

        exit_sample_data = dict(
            (key_field, key_field) for key_field in
            usage_logger.LoggingServer._EXIT_LOG_FIELD_NAMES)
        with mock.patch(
                'natcap.invest.ui.usage_logger.urlopen',
                return_value=StringIO(json_string)) as mock_obj:
            logging_server.log_invest_run(exit_sample_data, 'exit')
        self.assertEqual(mock_obj.call_count, 2)
        exit_sample_data['ip_address'] = 'local'
        self.assertEqual(
            sorted(mock_obj.call_args[0][0].data.decode('utf-8').split('&')),
            sorted(urlencode(exit_sample_data).split('&')))

    def test_unknown_mode(self):
        """Usage logger test that an unknown mode raises an exception."""
        from natcap.invest.ui import usage_logger

        logging_server = usage_logger.LoggingServer()

        sample_data = dict(
            (key_field, key_field) for key_field in
            usage_logger.LoggingServer._LOG_FIELD_NAMES)

        with self.assertRaises(ValueError):
            logging_server.log_invest_run(sample_data, 'bad_mode')

    @scm.skip_if_data_missing(SAMPLE_DATA)
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
