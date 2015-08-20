"""
Functions to assist with remote logging of InVEST usage.
"""

import os
import datetime
import platform
import sys
import locale
import time
from types import DictType
from types import ListType
import traceback
import logging
import sqlite3

import Pyro4
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import pygeoprocessing
from pygeoprocessing import geoprocessing

import natcap.invest

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.remote_logging')


class LoggingServer(object):
    _FIELD_NAMES = [
        'model_name',
        'invest_release',
        'time',
        'ip_address',
        'bounding_box_union',
        'bounding_box_intersection',
        'node_hash',
        'system_full_platform_string',
        'system_preferred_encoding',
        'system_default_language',
        ]
    _TABLE_NAME = 'natcap_model_log_table'
    def __init__(self, database_filepath):
        """Launches a new logger and initalizes an sqlite database at
        `database_filepath` if not previously defined.

        Args:
            database_filepath (string): path to a database filepath, will create
                the file and directory path if it doesn't exists

        Returns:
            None."""

        self.database_filepath = database_filepath
        # make the directory if it doesn't exist and isn't the current directory
        filepath_directory = os.path.dirname(self.database_filepath)
        if filepath_directory != '' and not os.path.exists(filepath_directory):
            os.mkdir(os.path.dirname(self.database_filepath))
        db_connection = sqlite3.connect(self.database_filepath)
        db_cursor = db_connection.cursor()
        db_cursor.execute(
            'CREATE TABLE IF NOT EXISTS %s (%s)' % (
                self._TABLE_NAME,
                ','.join([
                    '%s text' % field_id for field_id in self._FIELD_NAMES])))
        db_connection.commit()
        db_connection.close()


    def log_invest_run(self, data):
        """Logs an invest run to the sqlite database found at database_filepath
        Any self._FIELD_NAMES that match data keys will be inserted into the
        database.

        Args:
            data (dict): a flat dictionary with data about the InVEST run where
                the keys of the dictionary are at least self._FIELD_NAMES

                TODO: list keys here

        Returns:
            None."""

        try:
            # Add info about the client's IP
            data_copy = data.copy()
            data_copy['ip_address'] = (
                Pyro4.current_context.client.sock.getpeername()[0])
            data_copy['time'] = datetime.datetime.now().isoformat(' ')

            # Get data into the same order as the field names
            ordered_data = [
                data_copy[field_id] for field_id in self._FIELD_NAMES]
            # get as many '?'s as there are fields for the insert command
            position_format = ','.join(['?'] * len(self._FIELD_NAMES))

            insert_command = (
                'INSERT OR REPLACE INTO natcap_model_log_table'
                '(%s) VALUES (%s)' % (
                    ','.join(self._FIELD_NAMES), position_format))

            db_connection = sqlite3.connect(self.database_filepath)
            db_cursor = db_connection.cursor()
            # pass in ordered_data to the command
            db_cursor.execute(insert_command, ordered_data)
            db_connection.commit()
            db_connection.close()
        except:
            # print something locally for our log and raise back to client
            traceback.print_exc()
            raise
        extra_fields = set(data_copy).difference(self._FIELD_NAMES)
        if len(extra_fields) > 0:
            LOGGER.warn(
                "Warning there were extra fields %s passed to logger. "
                " Expected: %s Received: %s", sorted(extra_fields),
                sorted(self._FIELD_NAMES), sorted(data_copy))


def launch_logging_server(database_filepath, hostname, port):
    """Function to start a remote procedure call server

    Args:
        database_filepath (string): local filepath to the sqlite database
        hostname (string): network interface to bind to
        port (int): TCP port to bind to

    Returns:
        never"""

    daemon = Pyro4.Daemon(hostname, port)
    uri = daemon.register(
        LoggingServer(database_filepath), 'natcap.invest.remote_logging')
    LOGGER.info("natcap.invest.recreation ready. Object uri = %s", uri)
    daemon.requestLoop()


if __name__ == '__main__':
    DATABASE_FILEPATH = sys.argv[1]
    HOSTNAME = sys.argv[2]
    PORT = int(sys.argv[3])
    launch_logging_server(DATABASE_FILEPATH, HOSTNAME, PORT)
