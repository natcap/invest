import logging
import multiprocessing

import natcap.invest.recreation.recmodel_server

logging.basicConfig(
    format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s',
    level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

args = {
    'hostname': '',  # the local IP for the server
    'port': 54322,
    'raw_csv_point_data_path': 'photos_2005-2017_odlla.csv',
    'max_year': 2017,
    'min_year': 2005,
    'cache_workspace': './recmodel_server_cache'  # a local directory
}

if __name__ == '__main__':
    # It's crucial to specify `spawn` here as some OS use 'fork' as the default
    # for new child processes. And a 'fork' will immediately duplicate all
    # the resources (memory) of the parent. We noticed that causing
    # Cannot Allocate Memory errors, as the recmodel_server can start child
    # processes at a point when it's already using a lot of memory.
    multiprocessing.set_start_method('spawn')
    natcap.invest.recreation.recmodel_server.execute(args)
