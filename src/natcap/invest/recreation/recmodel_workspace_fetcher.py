"""InVEST recreation workspace fetcher."""

import os
import logging
import urllib

import Pyro4
import pygeoprocessing

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.recmodel_client')
# This URL is a NatCap global constant
RECREATION_SERVER_URL = 'http://data.naturalcapitalproject.org/server_registry/invest_recreation_model/'

# this serializer lets us pass null bytes in strings unlike the default
Pyro4.config.SERIALIZER = 'marshal'


def execute(args):
    """Fetch workspace from remote server.

    After the call a .zip file exists at `args['workspace_dir']` named
    `args['workspace_id'] + '.zip'` and contains the zipped workspace of that
    model run.

    Parameters:
        args['workspace_dir'] (string): path to workspace directory
        args['hostname'] (string): FQDN to recreation server
        args['port'] (string or int): port on hostname for recreation server
        args['workspace_id'] (string): workspace identifier

    Returns:
        None
    """
    output_dir = args['workspace_dir']
    pygeoprocessing.create_directories([output_dir])

    # in case the user defines a hostname
    if 'hostname' in args:
        path = "PYRO:natcap.invest.recreation@%s:%s" % (
            args['hostname'], args['port'])
    else:
        # else use a well known path to get active server
        path = urllib.urlopen(RECREATION_SERVER_URL).read().rstrip()
    LOGGER.info("contacting server")
    recmodel_server = Pyro4.Proxy(path)

    LOGGER.info("sending id request %s", args['workspace_id'])
    workspace_aoi_binary = recmodel_server.fetch_workspace_aoi(
        args['workspace_id'])

    # unpack result
    open(os.path.join(
        output_dir, '%s.zip' % args['workspace_id']), 'wb').write(
            workspace_aoi_binary)
    LOGGER.info("fetched aoi")
