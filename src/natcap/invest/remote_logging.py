"""
Functions to assist with remote logging of InVEST usage.
"""

import os
import platform
import sys
import locale
import hashlib
import json
import time
from types import DictType
from types import ListType
import traceback
import logging

import Pyro4
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import pygeoprocessing
from pygeoprocessing import geoprocessing

import natcap.invest

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.remote_logging')

def _hash_data(data):
    """
    Create an md5sum hash of the input data.
    """
    try:
        md5 = hashlib.md5()
        md5.update(json.dumps(data))
        return md5.hexdigest()
    except:
        return None

def _user_hash():
    """Returns a hash for the user, based on the machine."""
    data = {
        'os': platform.platform(),
        'hostname': platform.node(),
        'userdir': os.path.expanduser('~')
    }
    return _hash_data(data)

def log_model(model_name, args_dict):
    data = {
        'model_name': model_name,
        'invest_release': natcap.invest.__version__,
        'user': _user_hash(),
        'system': {
            'os': platform.system(),
            'release': platform.release(),
            'full_platform_string': platform.platform(),
            'fs_encoding': sys.getfilesystemencoding(),
            'preferred_encoding': locale.getdefaultlocale()[1],
            'default_language': locale.getdefaultlocale()[0],
            'python': {
                'version': platform.python_version(),
                'bits': platform.architecture()[0],
            },
        },
        "args": log_args(args_dict),
    }
    return data

def _format_gdal(gdal_path):
    raster = gdal.Open(gdal_path)
    rasterband = raster.GetRasterBand(1)

    geotransform = raster.GetGeoTransform()
    driver = raster.GetDriver()

    raster_data = {
        "cols": raster.RasterXSize,
        "rows": raster.RasterYSize,
        "pixelwidth": float(geotransform[1]),
        "pixelheight": float(geotransform[5]),
        "nodata": rasterband.GetNoDataValue(),
        "bbox": pygeoprocessing.get_bounding_box(gdal_path),
        "projection": pygeoprocessing.get_dataset_projection_wkt_uri(gdal_path),
        "driver": driver.ShortName,
        "type": {
            "gdal": rasterband.DataType,
            "numpy": geoprocessing._gdal_to_numpy_type(rasterband).__name__,
        }
    }
    return raster_data

def _format_ogr(ogr_path):
    vector = ogr.Open(ogr_path)
    layer = vector.GetLayer()
    driver = vector.GetDriver()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(layer.GetSpatialRef().__str__())

    vector_data = {
        "projection": srs.ExportToWkt(),
        "features": layer.GetFeatureCount(),
        "driver": driver.name,
        "bbox": pygeoprocessing.get_datasource_bounding_box(ogr_path),
    }
    return vector_data

def _is_file(filepath):
    if os.path.exists(filepath):
        return True
    return False

def _is_gdal(filepath):
    if _is_file(filepath):
        raster = gdal.Open(filepath)
        if raster is not None:
            return True
    return False

def _is_ogr(filepath):
    if _is_file(filepath):
        vector = ogr.Open(filepath)
        if vector is not None:
            return True
    return False

    # CSV files can be parsed by OGR, so guard against it.
    driver = vector.GetDriver()
    if driver.name == 'CSV':
        return False
    return True

def log_args(args_dict):
    """
    Parse through an args dict and extract interesting spatial information.
    """

    test_funcs = [
        (_is_gdal, _format_gdal, lambda x: 'gdal'),
        (_is_ogr, _format_ogr, lambda x: 'ogr'),
        (lambda x: True, lambda x: x, lambda x: type(x).__name__)
    ]

    def recurse_args(arg):
        out_args_dict = {}
        if type(arg) is DictType:
            for key, value in arg.iteritems():
                out_args_dict[key] = recurse_args(value)
        elif type(arg) is ListType:
            out_args_list = []
            for list_value in arg:
                out_args_list.append(recurse_args(list_value))
            return out_args_list
        else:
            for test_func, format_func, typestring in test_funcs:
                if test_func(arg) is True:
                    return {
                        'type': typestring(arg),
                        'value': arg,
                        'metadata': format_func(arg),
                    }
        return out_args_dict

    return recurse_args(args_dict)

def get_session_id():
    data = {
        'user_hash': _user_hash(),
        'time': str(time.time()),
    }
    return '-'.join(data.values())

def sanitize(dirty_args):
    """
    Remove data we might not want people to see or know we collect.
    """
    new_args = {}
    for key, value in dirty_args.iteritems():
        if key in ['args']:
           continue
        new_args[key] = value
    return new_args


def _default_payload(event, session_id, data):
    default_data = {
        "type": event,
        "user_time": time.time(),
        "session_id": session_id,
        "api": "2.0",
        "data": data,
    }
    return default_data

def format_start(model_name, args, session_id):
    model_data = log_model(model_name, args)
    return _default_payload('start', session_id, model_data)


def format_success(session_id):
    return _default_payload('success', session_id, None)


def format_exception(exception, session_id):
    data = {
        "type": exception.__class__.__name__,
        # Limit the number of stack trace entries
        "traceback": traceback.format_exc(limit=20),
        "text": exception.message,
        "args": list(exception.args),
    }
    return _default_payload('success', session_id, data)


class LoggingServer(object):
    def __init__(self, database_filepath):
        """Launches a new logger and initalizes an sqlite database at
        `database_filepath` if not previously defined.

        Args:
            database_filepath (string): path to a database filepath, will create
                the file and directory path if it doesn't exists

        Returns:
            None."""

        field_names = [
            'model_name',
            'invest_release',
            'node_hash',
            'system_full_platform_string',
            'system_preferred_encoding',
            'system_default_language',
            'time',
            'bounding_box_intersection'
            'bounding_box_union'
            ]

        self.database_filepath = database_filepath
        if not os.path.exists(os.path.dirname(manager_filename)):
            os.mkdir(os.path.dirname(manager_filename))
        db_connection = sqlite3.connect(manager_filename)
        db_cursor = db_connection.cursor()
        db_cursor.execute('''CREATE TABLE IF NOT EXISTS blob_table
            (blob_id text PRIMARY KEY, blob_data blob)''')
        db_connection.commit()
        db_connection.close()


        self.database_filepath = database_filepath

    def log_invest_run(self, data):
        """Logs an invest run to the sqlite database found at database_filepath

        Args:
            data (dict): a possibly nested dictionary of data about the InVEST
                run.

                TODO: list keys here

        Returns:
            None."""

        LOGGER.debug(data)


def launch_logging_server(database_filepath, hostname, port):
    """Function to start a remote procedure call server

    Args:
        database_filepath (string): local filepath to the sqlite database
        hostname (string): network interface to bind to
        port (int): TCP port to bind to

    Returns:
        None"""

    daemon = Pyro4.Daemon(hostname, port)
    uri = daemon.register(
        LoggingServer(database_filepath), 'natcap.invest.remote_logging')
    LOGGER.info("natcap.invest.recreation ready. Object uri = %s", uri)
    daemon.requestLoop()


if __name__ == '__main__':
    DATABASE_FILEPATH = sys.argv[1]
    HOSTNAME = sys.argv[2]
    PORT = int(sys.argv[3])
    launch_logging_server(DATABASE_FILEPATH, HOSTNAME, PORT)
