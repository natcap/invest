"""Functions to assist with remote logging of InVEST usage."""

import os
import datetime
import sys
import tempfile
import traceback
import logging
import sqlite3
import zipfile
import glob

import Pyro4
from osgeo import ogr
from osgeo import osr

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.remote_logging')

Pyro4.config.SERIALIZER = 'marshal'  # lets us pass null bytes in strings


class LoggingServer(object):
    """RPC server for logging invest runs and getting database summaries."""

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
    _TABLE_NAME = 'model_log_table'

    def __init__(self, database_filepath):
        """Launch a logger and initialize an sqlite database.

        Parameters:
            database_filepath (string): path to a database filepath, will
                create the file and directory path if it doesn't exist.

        Returns:
            None.
        """
        self.database_filepath = database_filepath
        # make directory if it doesn't exist and isn't the current directory
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
        """Log some parameters of an InVEST run.

        Metadata is saved to a new record in the sqlite database found at
        `self.database_filepath`.  Any self._FIELD_NAMES that match data keys
        will be inserted into the record.

        Parameters:
            data (dict): a flat dictionary with data about the InVEST run
                where the keys of the dictionary are at least
                self._FIELD_NAMES

        Returns:
            None.
        """
        try:
            # Add info about the client's IP
            data_copy = data.copy()
            if Pyro4.current_context.client is not None:
                data_copy['ip_address'] = (
                    Pyro4.current_context.client.sock.getpeername()[0])
            else:
                data_copy['ip_address'] = 'local'
            data_copy['time'] = datetime.datetime.now().isoformat(' ')

            # Get data into the same order as the field names
            ordered_data = [
                data_copy[field_id] for field_id in self._FIELD_NAMES]
            # get as many '?'s as there are fields for the insert command
            position_format = ','.join(['?'] * len(self._FIELD_NAMES))

            insert_command = (
                'INSERT OR REPLACE INTO %s'
                '(%s) VALUES (%s)' % (
                    (self._TABLE_NAME,) + (
                        ','.join(self._FIELD_NAMES), position_format)))

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

    def get_run_summary_db(self):
        """Retrieve the raw sqlite database of runs as a binary stream."""
        try:
            return open(self.database_filepath, 'rb').read()
        except:
            # print something locally for our log and raise back to client
            traceback.print_exc()
            raise


def execute(args):
    """Function to start a remote procedure call server.

    Parameters:
        args['database_filepath'] (string): local filepath to the sqlite
            database
        args['hostname'] (string): network interface to bind to
        args['port'] (int): TCP port to bind to

    Returns:
        never
    """
    daemon = Pyro4.Daemon(args['hostname'], args['port'])
    uri = daemon.register(
        LoggingServer(args['database_filepath']),
        'natcap.invest.remote_logging')
    LOGGER.info("natcap.invest.usage_logger ready. Object uri = %s", uri)
    daemon.requestLoop()
