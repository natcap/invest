"""InVEST recreation workspace fetcher."""

import os
import logging
import urllib

import Pyro5
import Pyro5.api

from natcap.invest.recreation import recmodel_client
from .. import utils

LOGGER = logging.getLogger('natcap.invest.recmodel_client')

# this serializer lets us pass null bytes in strings unlike the default
Pyro5.config.SERIALIZER = 'marshal'


def execute(args):
    """Fetch workspace from remote server.

    After the call a .zip file exists at `args['workspace_dir']` named
    `args['workspace_id'] + '.zip'` and contains the zipped workspace of that
    model run.

    Args:
        args['workspace_dir'] (string): path to workspace directory
        args['hostname'] (string): FQDN to recreation server
        args['port'] (string or int): port on hostname for recreation server
        args['workspace_id'] (string): workspace identifier
        args['server_id'] (string): one of ('flickr', 'twitter')

    Returns:
        None
    """
    output_dir = args['workspace_dir']
    os.makedirs(output_dir, exist_ok=True)

    # in case the user defines a hostname
    if 'hostname' in args:
        server_url = "PYRO:natcap.invest.recreation@%s:%s" % (
            args['hostname'], args['port'])
    else:
        # else use a well known path to get active server
        server_url = urllib.urlopen(recmodel_client.SERVER_URL).read().rstrip()
    LOGGER.info("contacting server")
    recmodel_manager = Pyro5.api.Proxy(server_url)

    LOGGER.info("sending id request %s", args['workspace_id'])
    zip_binary = recmodel_manager.fetch_aoi_workspaces(
        args['workspace_id'], args['server_id'])

    with open(os.path.join(
            output_dir,
            f"{args['server_id']}_{args['workspace_id']}.zip"), 'wb') as file:
        file.write(zip_binary)
    LOGGER.info(f"fetched {args['server_id']}_{args['workspace_id']}.zip")
