"""InVEST Model Logging tests."""

import time
import threading
import unittest
import tempfile
import shutil
import socket
import urllib

import Pyro4
import mock


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

        with mock.patch(
                'natcap.invest.ui.usage_logger.urllib2.urlopen') as mock_obj:
            logging_server.log_invest_run(sample_data, 'log')
        mock_obj.assert_called_once()
        sample_data['ip_address'] = 'local'
        self.assertEqual(sorted(mock_obj.call_args[0][0].get_data().split('&')),
                         sorted(urllib.urlencode(sample_data).split('&')))

        exit_sample_data = dict(
            (key_field, key_field) for key_field in
            usage_logger.LoggingServer._EXIT_LOG_FIELD_NAMES)
        with mock.patch(
                'natcap.invest.ui.usage_logger.urllib2.urlopen') as mock_obj:
            logging_server.log_invest_run(exit_sample_data, 'exit')
        mock_obj.assert_called_once()
        exit_sample_data['ip_address'] = 'local'
        self.assertEqual(sorted(mock_obj.call_args[0][0].get_data().split('&')),
                         sorted(urllib.urlencode(exit_sample_data).split('&')))

    def test_unknown_mode(self):
        """Usage logger test that an unknown mode raises an exception."""
        from natcap.invest.ui import usage_logger

        logging_server = usage_logger.LoggingServer()

        sample_data = dict(
            (key_field, key_field) for key_field in
            usage_logger.LoggingServer._LOG_FIELD_NAMES)

        with self.assertRaises(ValueError):
            logging_server.log_invest_run(sample_data, 'bad_mode')
