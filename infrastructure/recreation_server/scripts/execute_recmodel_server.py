import argparse
import logging
import multiprocessing

import natcap.invest.recreation.recmodel_server

logging.basicConfig(
    format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s',
    level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

args_dict = {
    'hostname': '',
    'port': 54323,  # http://data.naturalcapitalproject.org/server_registry/invest_recreation_model_3_19_0/
    'max_allowable_query': 40_000_000,
    'datasets': {
        'flickr': {
            'raw_csv_point_data_path': '',
            'min_year': 2005,
            'max_year': 2017
        },
        'twitter': {
            'quadtree_pickle_filename': '',
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
    parser.add_argument(
        '--flickr_csv', dest='flickr_csv',
        help='Path to the flickr database in the form of a CSV')
    parser.add_argument(
        '--twitter_qt_pickle', dest='twitter_qt_pickle',
        help='Path to the pre-built twitter quadtree pickle file')
    args = parser.parse_args()

    # It's crucial to specify `spawn` here as some OS use 'fork' as the default
    # for new child processes. And a 'fork' will immediately duplicate all
    # the resources (memory) of the parent. We noticed that causing
    # Cannot Allocate Memory errors, as the recmodel_server can start child
    # processes at a point when it's already using a lot of memory.
    multiprocessing.set_start_method('spawn')

    args_dict['cache_workspace'] = args.cache_workspace
    args_dict['datasets']['flickr']['raw_csv_point_data_path'] = args.flickr_csv
    args_dict['datasets']['twitter']['quadtree_pickle_filename'] = args.twitter_qt_pickle
    natcap.invest.recreation.recmodel_server.execute(args_dict)
