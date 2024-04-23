import argparse
import logging
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
logging.basicConfig(level=logging.DEBUG, handlers=[handler, filehandler])

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
    args = parser.parse_args()

    recmodel_server.construct_userday_quadtree(
        recmodel_server.INITIAL_BOUNDING_BOX,
        args.csv_files,
        'twitter',
        args.workspace,
        args.output_filename,
        recmodel_server.GLOBAL_MAX_POINTS_PER_NODE,
        recmodel_server.GLOBAL_DEPTH)
