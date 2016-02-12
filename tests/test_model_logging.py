"""InVEST Model Logging tests."""

import unittest
import tempfile
import shutil
import os
import sqlite3

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
        tables = db_cursor.next()
        db_cursor = None
        db_connection.close()
        self.assertTrue(usage_logger.LoggingServer._TABLE_NAME in tables)

    def test_add_records(self):
        """Usage logger record runs and verify they are added."""
        from natcap.invest.iui import usage_logger

        database_path = os.path.join(self.workspace_dir, 'test_log.db')
        logging_server = usage_logger.LoggingServer(database_path)

        # set up a sample dict whose values are identical to its keys
        # this makes for an easy expected result
        sample_data = dict(
            (key_field, key_field) for key_field in
            usage_logger.LoggingServer._FIELD_NAMES)

        logging_server.log_invest_run(sample_data)

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
            usage_logger.LoggingServer._FIELD_NAMES)

        logging_server.log_invest_run(sample_data)

        db_connection = sqlite3.connect(database_path)
        db_cursor = db_connection.cursor()
        db_cursor.execute(
            "SELECT model_name, invest_release FROM model_log_table;")
        result = db_cursor.next()
        db_cursor = None
        db_connection.close()
        expected_result = ('model_name', 'invest_release')
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
            usage_logger.LoggingServer._FIELD_NAMES)

        logging_server.log_invest_run(sample_data)
        remote_database_as_string = logging_server.get_run_summary_db()

        local_database_as_string = open(database_path, 'rb').read()
        self.assertEqual(local_database_as_string, remote_database_as_string)
