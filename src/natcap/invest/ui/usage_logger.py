"""Functions to assist with remote logging of InVEST usage."""

import logging
import json

try:
    from urllib.request import urlopen, Request
    from urllib.parse import urlencode
except ImportError:
    from urllib2 import urlopen, Request
    from urllib import urlencode

import Pyro4

LOGGER = logging.getLogger('natcap.invest.remote_logging')

Pyro4.config.SERIALIZER = 'marshal'  # lets us pass null bytes in strings
_ENDPOINTS_INDEX_URL = (
    'http://data.naturalcapitalproject.org/server_registry/'
    'invest_usage_logger_v2')


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

    @Pyro4.expose
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
        endpoints = json.loads(urlopen(_ENDPOINTS_INDEX_URL).read().strip())

        try:
            if mode == 'log':
                url = endpoints['START']
            elif mode == 'exit':
                url = endpoints['FINISH']
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

            urlopen(Request(url, urlencode(data_copy)))
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


def execute(args):
    """Function to start a remote procedure call server.

    Parameters:
        args['hostname'] (string): network interface to bind to
        args['port'] (int): TCP port to bind to

    Returns:
        never
    """
    daemon = Pyro4.Daemon(args['hostname'], args['port'])
    uri = daemon.register(
        LoggingServer(),
        'natcap.invest.remote_logging')
    LOGGER.info("natcap.invest.usage_logger ready. Object uri = %s", uri)
    daemon.requestLoop()
