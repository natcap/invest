import logging
import multiprocessing

import natcap.invest.recreation.recmodel_server

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

args = {
    'hostname': '10.240.0.4',
    'port': 54322,
    'raw_csv_point_data_path': 'photos_2005-2017_odlla.csv',
    'max_year': 2017,
    'min_year': 2005,
    'cache_workspace': './recserver_cache_py36'
}

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    natcap.invest.recreation.recmodel_server.execute(args)
