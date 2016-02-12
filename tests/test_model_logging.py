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

    def test_x(self):
        """Skeleton test that a server will launch and create a database."""
        from natcap.invest.iui import usage_logger

        database_path = os.path.join(self.workspace_dir, 'test_log.db')
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
