"""InVEST Model Logging tests."""

import time
import threading
import unittest
import tempfile
import shutil
import os
import sqlite3
import socket

import Pyro4
import numpy
from osgeo import ogr
from pygeoprocessing.testing import scm


class ModelLoggingTests(unittest.TestCase):
    """Tests for the InVEST model logging framework."""

    def setUp(self):
        """Initalize a workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_create_server(self):
        """Usage logger test server will launch and create a database."""
        from natcap.invest.iui import usage_logger

        database_path = os.path.join(
            self.workspace_dir, 'subdir', 'test_log.db')
        logging_server = usage_logger.LoggingServer(database_path)

        db_connection = sqlite3.connect(database_path)
        db_cursor = db_connection.cursor()
        db_cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' ORDER BY name;")
        # fetchall returns 1 element tuples, this line removes the tuple
        tables = [x[0] for x in db_cursor.fetchall()]
        db_cursor = None
        db_connection.close()
        self.assertTrue(
            usage_logger.LoggingServer._LOG_TABLE_NAME in tables,
            msg="%s not in %s" % (
                usage_logger.LoggingServer._LOG_TABLE_NAME, tables))

    def test_pyro_server(self):
        """Usage logger test server as an RPC."""
        from natcap.invest.iui import usage_logger
        # attempt to get an open port; could result in race condition but
        # will be okay for a test. if this test ever fails because of port
        # in use, that's probably why
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
        sock = None

        database_path = os.path.join(
            self.workspace_dir, 'subdir', 'test_log.db')
        server_args = {
            'hostname': 'localhost',
            'port': port,
            'database_filepath': database_path,
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

        remote_database_as_string = logging_server.get_run_summary_db()
        local_database_as_string = open(database_path, 'rb').read()
        self.assertEqual(local_database_as_string, remote_database_as_string)

    def test_add_extra_records(self):
        """Usage logger record runs with an extra field."""
        from natcap.invest.iui import usage_logger

        database_path = os.path.join(self.workspace_dir, 'test_log.db')
        logging_server = usage_logger.LoggingServer(database_path)

        # set up a sample dict whose values are identical to its keys
        # this makes for an easy expected result
        sample_data = dict(
            (key_field, key_field) for key_field in
            usage_logger.LoggingServer._LOG_FIELD_NAMES)
        # make an extra field which should be ignored on server side
        sample_data['extra_field'] = -238328

        logging_server.log_invest_run(sample_data, 'log')

        db_connection = sqlite3.connect(database_path)
        db_cursor = db_connection.cursor()
        db_cursor.execute(
            "SELECT model_name, invest_release FROM model_log_table;")
        result = db_cursor.next()
        db_cursor = None
        db_connection.close()
        expected_result = ('model_name', 'invest_release')
        self.assertTrue(
            result == expected_result, msg="%s != %s" % (
                result, expected_result))

    def test_add_records(self):
        """Usage logger record runs and verify they are added."""
        from natcap.invest.iui import usage_logger

        database_path = os.path.join(self.workspace_dir, 'test_log.db')
        logging_server = usage_logger.LoggingServer(database_path)

        # set up a sample dict whose values are identical to its keys
        # this makes for an easy expected result
        sample_data = dict(
            (key_field, key_field) for key_field in
            usage_logger.LoggingServer._LOG_FIELD_NAMES)

        logging_server.log_invest_run(sample_data, 'log')

        db_connection = sqlite3.connect(database_path)
        db_cursor = db_connection.cursor()
        db_cursor.execute(
            "SELECT model_name, invest_release FROM model_log_table;")
        result = db_cursor.next()
        db_cursor = None
        db_connection.close()
        expected_result = ('model_name', 'invest_release')
        self.assertEqual(expected_result, result)

    def test_add_exit_status(self):
        """Usage logger record run and then exit and verify they are added."""
        from natcap.invest.iui import usage_logger

        database_path = os.path.join(self.workspace_dir, 'test_log.db')
        logging_server = usage_logger.LoggingServer(database_path)

        # set up a sample dict whose values are identical to its keys
        # this makes for an easy expected result
        sample_data = dict(
            (key_field, key_field) for key_field in
            usage_logger.LoggingServer._LOG_FIELD_NAMES)
        logging_server.log_invest_run(sample_data, 'log')

        exit_sample_data = dict(
            (key_field, key_field) for key_field in
            usage_logger.LoggingServer._EXIT_LOG_FIELD_NAMES)
        logging_server.log_invest_run(exit_sample_data, 'exit')

        db_connection = sqlite3.connect(database_path)
        db_cursor = db_connection.cursor()
        db_cursor.execute(
            "SELECT status FROM %s;" %
            usage_logger.LoggingServer._EXIT_LOG_TABLE_NAME)
        result = db_cursor.next()
        db_cursor = None
        db_connection.close()
        expected_result = ('status',)
        self.assertEqual(expected_result, result)

    def test_download_database(self):
        """Usage logger run summary db is the same as the base db."""
        from natcap.invest.iui import usage_logger

        database_path = os.path.join(self.workspace_dir, 'test_log.db')
        logging_server = usage_logger.LoggingServer(database_path)

        # set up a sample dict whose values are identical to its keys
        # this makes for an easy expected result
        sample_data = dict(
            (key_field, key_field) for key_field in
            usage_logger.LoggingServer._LOG_FIELD_NAMES)

        logging_server.log_invest_run(sample_data, 'log')
        remote_database_as_string = logging_server.get_run_summary_db()

        local_database_as_string = open(database_path, 'rb').read()
        self.assertEqual(local_database_as_string, remote_database_as_string)
