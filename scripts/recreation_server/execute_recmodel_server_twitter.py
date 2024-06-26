import argparse
import logging
import multiprocessing
import os

import natcap.invest.recreation.recmodel_server

logging.basicConfig(
    format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s',
    level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

args_dict = {
    'hostname': 'localhost',
    'port': 54322,
    'datasets': {
        'flickr': {
            'raw_csv_point_data_path': '~/server/volume/flickr/photos_2005-2017_odlla.csv',
            'min_year': 2005,
            'max_year': 2017
        },
        'twitter': {
            'quadtree_pickle_filename': '~/server/volume/twitter_quadtree/global_twitter_qt.pickle',
            'min_year': 2012,
            'max_year': 2022
        }
    }
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w', dest='cache_workspace',
        help='Path to a local, writeable, directory. Avoid mounted volumes.')
    args = parser.parse_args()
    # It's crucial to specify `spawn` here as some OS use 'fork' as the default
    # for new child processes. And a 'fork' will immediately duplicate all
    # the resources (memory) of the parent. We noticed that causing
    # Cannot Allocate Memory errors, as the recmodel_server can start child
    # processes at a point when it's already using a lot of memory.
    multiprocessing.set_start_method('spawn')
    args_dict['cache_workspace'] = args.cache_workspace
    mounted_volume = os.path.join(args.cache_workspace, 'cache')
    args_dict['quadtree_pickle_filename'] = os.path.join(
        mounted_volume, 'global_twitter_qt.pickle')
    natcap.invest.recreation.recmodel_server.execute(args_dict)
