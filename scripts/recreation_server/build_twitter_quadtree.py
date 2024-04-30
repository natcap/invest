import argparse
import logging
import os
import sys

import natcap.invest.utils
from natcap.invest.recreation import recmodel_server


LOGGER = logging.getLogger(__name__)
root_logger = logging.getLogger()

handler = logging.StreamHandler(sys.stdout)
filehandler = logging.FileHandler('logfile.txt', 'w', encoding='UTF-8')
formatter = logging.Formatter(
    fmt=natcap.invest.utils.LOG_FMT,
    datefmt='%m/%d/%Y %H:%M:%S ')
handler.setFormatter(formatter)
filehandler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO, handlers=[handler, filehandler])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv_file_list', type=str,
        help='path to text file with list of csv filepaths containing point data')
    parser.add_argument(
        '-w', '--workspace', type=str,
        help='path to directory for writing quadtree files.')
    parser.add_argument(
        '-o', '--output_filename', type=str,
        help='name for the pickle file quadtree index created in the workspace.')
    parser.add_argument(
        '-c', '--n_cores', type=int,
        help='number of available cores for multiprocessing')
    args = parser.parse_args()

    with open(args.csv_file_list, 'r') as file:
        csv_list = [line.rstrip() for line in file]

    ooc_qt_pickle_filename = os.path.join(
        args.workspace, args.output_filename)
    recmodel_server.construct_userday_quadtree(
        recmodel_server.INITIAL_BOUNDING_BOX,
        csv_list,
        'twitter',
        args.workspace,
        ooc_qt_pickle_filename,
        recmodel_server.GLOBAL_MAX_POINTS_PER_NODE,
        recmodel_server.GLOBAL_DEPTH,
        n_workers=args.n_cores,
        build_shapefile=False,
        fast_point_count=True)
