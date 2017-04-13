"""Functions to assist with remote logging of InVEST usage."""

import os
import datetime
import logging
import sqlite3

import Pyro4

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.remote_logging')

Pyro4.config.SERIALIZER = 'marshal'  # lets us pass null bytes in strings


class LoggingServer(object):
    """RPC server for logging invest runs and getting database summaries."""

    _LOG_FIELD_NAMES = [
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
        'session_id',
        ]
    _EXIT_LOG_FIELD_NAMES = [
        'session_id',
        'time',
        'ip_address',
        'status',
        ]
    _LOG_TABLE_NAME = 'model_log_table'
    _EXIT_LOG_TABLE_NAME = 'model_exit_status_table'

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
                self._LOG_TABLE_NAME,
                ','.join([
                    '%s text' % field_id for field_id in
                    self._LOG_FIELD_NAMES])))
        db_cursor.execute(
            'CREATE TABLE IF NOT EXISTS %s (%s)' % (
                self._EXIT_LOG_TABLE_NAME,
                ','.join([
                    '%s text' % field_id for field_id in
                    self._EXIT_LOG_FIELD_NAMES])))
        db_connection.commit()
        db_connection.close()

    def log_invest_run(self, data, mode):
        """Log some parameters of an InVEST run.

        Metadata is saved to a new record in the sqlite database found at
        `self.database_filepath`.  The mode specifies if it is a log or an
        exit status notification.  The appropriate table name and fields will
        be used in that case.

        Parameters:
            data (dict): a flat dictionary with data about the InVEST run
                where the keys of the dictionary are at least
                self._LOG_FIELD_NAMES
            mode (string): one of 'log' or 'exit'.  If 'log' uses
                self._LOG_TABLE_NAME and parameters, while 'exit' logs to
                self._LOG_EXIT_TABLE_NAME

        Returns:
            None
        """
        try:
            if mode == 'log':
                table_name = self._LOG_TABLE_NAME
                field_names = self._LOG_FIELD_NAMES
            elif mode == 'exit':
                table_name = self._EXIT_LOG_TABLE_NAME
                field_names = self._EXIT_LOG_FIELD_NAMES
            else:
                raise ValueError(
                    "Unknown mode '%s', expected 'log' or 'exit'" % mode)
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
                data_copy[field_id] for field_id in field_names]
            # get as many '?'s as there are fields for the insert command
            position_format = ','.join(['?'] * len(field_names))

            insert_command = (
                'INSERT OR REPLACE INTO %s'
                '(%s) VALUES (%s)' % (
                    (table_name,) + (
                        ','.join(field_names), position_format)))

            db_connection = sqlite3.connect(self.database_filepath)
            db_cursor = db_connection.cursor()
            # pass in ordered_data to the command
            db_cursor.execute(insert_command, ordered_data)
            db_connection.commit()
            db_connection.close()
        except:
            # print something locally for our log and raise back to client
            LOGGER.exception("log_invest_run failed")
            raise
        extra_fields = set(data_copy).difference(self._LOG_FIELD_NAMES)
        if len(extra_fields) > 0:
            LOGGER.warn(
                "Warning there were extra fields %s passed to logger. "
                " Expected: %s Received: %s", sorted(extra_fields),
                sorted(self._LOG_FIELD_NAMES), sorted(data_copy))

    def get_run_summary_db(self):
        """Retrieve the raw sqlite database of runs as a binary stream."""
        try:
            return open(self.database_filepath, 'rb').read()
        except:
            # print something locally for our log and raise back to client
            LOGGER.exception("get_run_summary_db failed")
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
