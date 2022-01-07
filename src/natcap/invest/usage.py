"""Module to that provides functions for usage logging."""
import contextlib
import hashlib
import json
import locale
import logging
import os
import platform
import sys
import threading
import traceback
import uuid
import importlib

from urllib.request import urlopen, Request
from urllib.parse import urlencode

from osgeo import osr
import natcap.invest
import pygeoprocessing

from . import utils

ENCODING = sys.getfilesystemencoding()
LOGGER = logging.getLogger(__name__)

_ENDPOINTS_INDEX_URL = (
    'http://data.naturalcapitalproject.org/server_registry/'
    'invest_usage_logger_v2/index.html')

# This is defined here because it's very useful to know the thread name ahead
# of time so we can exclude any log messages it generates from the logging.
# Python doesn't care about having multiple threads have the same name.
_USAGE_LOGGING_THREAD_NAME = 'usage-logging-thread'


@contextlib.contextmanager
def log_run(model_pyname, args):
    """Context manager to log an InVEST model run and exit status.

    Args:
        model_pyname (string): The string module name that identifies the model.
        args (dict): The full args dictionary.

    Returns:
        ``None``
    """
    invest_interface = 'Qt'  # this cm is only used by the Qt interface
    session_id = str(uuid.uuid4())
    log_thread = threading.Thread(
        target=_log_model,
        args=(model_pyname, args, invest_interface, session_id),
        name=_USAGE_LOGGING_THREAD_NAME)
    log_thread.start()

    try:
        yield
    except Exception:
        exit_status_message = traceback.format_exc()
        raise
    else:
        exit_status_message = ':)'
    finally:
        log_exit_thread = threading.Thread(
            target=_log_exit_status, args=(session_id, exit_status_message),
            name=_USAGE_LOGGING_THREAD_NAME)
        log_exit_thread.start()


def _calculate_args_bounding_box(args, args_spec):
    """Calculate the bounding boxes of any GIS types found in `args_dict`.

    Args:
        args (dict): a string key and any value pair dictionary.
        args_spec (dict): the model ARGS_SPEC describing args

    Returns:
        bb_intersection, bb_union tuple that's either the lat/lng bounding
            intersection and union bounding boxes of the gis types referred to
            in args_dict.  If no GIS types are present, this is a (None, None)
            tuple.
    """

    def _merge_bounding_boxes(bb1, bb2, mode):
        """Merge two bounding boxes through union or intersection.

        Args:
            bb1 (list of float): bounding box of the form
                [minx, maxy, maxx, miny] or None
            bb2 (list of float): bounding box of the form
                [minx, maxy, maxx, miny] or None
            mode (string): either "union" or "intersection" indicating the
                how to combine the two bounding boxes.

        Returns:
            either the intersection or union of bb1 and bb2 depending
            on mode.  If either bb1 or bb2 is None, the other is returned.
            If both are None, None is returned.
        """
        if bb1 is None:
            return bb2
        if bb2 is None:
            return bb1

        if mode == "union":
            comparison_ops = [min, max, max, min]
        if mode == "intersection":
            comparison_ops = [max, min, min, max]

        bb_out = [op(x, y) for op, x, y in zip(comparison_ops, bb1, bb2)]
        return bb_out

    bb_intersection = None
    bb_union = None
    for key, value in args.items():
        # Using gdal.OpenEx to check if an input is spatial caused the
        # model to hang sometimes (possible race condition), so only
        # get the bounding box of inputs that are known to be spatial.
        # Also eliminate any string paths that are empty to prevent an
        # exception. By the time we've made it to this function, all paths
        # should already have been validated so the path is either valid or
        # blank.
        spatial_info = None
        if args_spec['args'][key]['type'] == 'raster' and value.strip() != '':
            spatial_info = pygeoprocessing.get_raster_info(value)
        elif (args_spec['args'][key]['type'] == 'vector'
                and value.strip() != ''):
            spatial_info = pygeoprocessing.get_vector_info(value)

        if spatial_info:
            local_bb = spatial_info['bounding_box']
            projection_wkt = spatial_info['projection_wkt']
            spatial_ref = osr.SpatialReference()
            spatial_ref.ImportFromWkt(projection_wkt)

            try:
                # means there's a GIS type with a well defined bounding box
                # create transform, and reproject local bounding box to
                # lat/lng
                lat_lng_ref = osr.SpatialReference()
                lat_lng_ref.ImportFromEPSG(4326)  # EPSG 4326 is lat/lng
                to_lat_trans = utils.create_coordinate_transformer(
                    spatial_ref, lat_lng_ref)
                for point_index in [0, 2]:
                    local_bb[point_index], local_bb[point_index + 1], _ = (
                        to_lat_trans.TransformPoint(
                            local_bb[point_index],
                            local_bb[point_index+1]))

                bb_intersection = _merge_bounding_boxes(
                    local_bb, bb_intersection, 'intersection')
                bb_union = _merge_bounding_boxes(
                    local_bb, bb_union, 'union')
            except Exception as transform_error:
                # All kinds of exceptions from bad transforms or CSV files
                # or dbf files could get us to this point, just don't
                # bother with the local_bb at all
                LOGGER.exception('Error when transforming coordinates: %s',
                                 transform_error)
        else:
            LOGGER.debug(f'Arg {key} of type {args_spec["args"][key]["type"]} '
                          'excluded from bounding box calculation')

    return bb_intersection, bb_union


def _log_exit_status(session_id, status):
    """Log the completion of a model with the given status.

    Args:
        session_id (string): a unique string that can be used to identify
            the current session between the model initial start and exit.
        status (string): a string describing the exit status of the model,
            'success' would indicate the successful completion while an
            exception string could indicate a failure.

    Returns:
        None
    """
    logger = logging.getLogger('natcap.invest.ui.usage._log_exit_status')

    try:
        payload = {
            'session_id': session_id,
            'status': status,
        }
        log_finish_url = json.loads(urlopen(
            _ENDPOINTS_INDEX_URL).read().strip())['FINISH']

        # The data must be a python string of bytes. This will be ``bytes``
        # in python3.
        urlopen(Request(log_finish_url, urlencode(payload).encode('utf-8')))
    except Exception as exception:
        # An exception was thrown, we don't care.
        logger.warning(
            'an exception encountered when _log_exit_status %s',
            str(exception))


def _log_model(pyname, model_args, invest_interface, session_id=None):
    """Log information about a model run to a remote server.

    Args:
        pyname (string): a python string of the package version.
        model_args (dict): the traditional InVEST argument dictionary.
        invest_interface (string): a string identifying the calling UI,
            e.g. `Qt` or 'Workbench'.

    Returns:
        None
    """
    logger = logging.getLogger('natcap.invest.ui.usage._log_model')

    def _node_hash():
        """Return a hash for the current computational node."""
        data = {
            'os': platform.platform(),
            'hostname': platform.node(),
            'userdir': os.path.expanduser('~')
        }
        md5 = hashlib.md5()
        # a json dump will handle non-ascii encodings
        # but then data must be encoded before hashing in Python 3.
        md5.update(json.dumps(data).encode('utf-8'))
        return md5.hexdigest()

    args_spec = importlib.import_module(pyname).ARGS_SPEC

    try:
        bounding_box_intersection, bounding_box_union = (
            _calculate_args_bounding_box(model_args, args_spec))

        payload = {
            'model_name': pyname,
            'invest_release': natcap.invest.__version__,
            'invest_interface': invest_interface,
            'node_hash': _node_hash(),
            'system_full_platform_string': platform.platform(),
            'system_preferred_encoding': locale.getdefaultlocale()[1],
            'system_default_language': locale.getdefaultlocale()[0],
            'bounding_box_intersection': str(bounding_box_intersection),
            'bounding_box_union': str(bounding_box_union),
            'session_id': session_id,
        }
        log_start_url = json.loads(urlopen(
            _ENDPOINTS_INDEX_URL).read().strip())['START']

        # The data must be a python string of bytes. This will be ``bytes``
        # in python3.
        urlopen(Request(log_start_url, urlencode(payload).encode('utf-8')))
    except Exception as exception:
        # An exception was thrown, we don't care.
        logger.warning(
            'an exception encountered when logging %s', repr(exception))
