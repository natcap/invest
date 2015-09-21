"""init module for natcap.invest"""

import urllib
import locale
import os
import platform
import sys
import hashlib
import json
import pkg_resources
import logging
import Pyro4
import gdal
import ogr
import osr
import pygeoprocessing
import natcap.versioner

pkg_resources.require('pygeoprocessing>=0.3.0a7')

__version__ = natcap.versioner.get_version('natcap.invest')

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

INVEST_USAGE_LOGGER_URL = ('http://data.naturalcapitalproject.org/'
                           'server_registry/invest_usage_logger/')


def is_release():
    """Returns a boolean indicating whether this invest release is actually a
    release or if it's a development release."""
    if 'post' in __version__:
        return False
    return True


def local_dir(source_file):
    """Return the path to where the target_file would be on disk.  If this is
    frozen (as with PyInstaller), this will be the folder with the executable
    in it.  If not, it'll just be the foldername of the source_file being
    passed in."""
    source_dirname = os.path.dirname(source_file)
    if getattr(sys, 'frozen', False):
        # sys.frozen is True when we're in either a py2exe or pyinstaller
        # build.
        # sys._MEIPASS exists, we're in a Pyinstaller build.
        if not getattr(sys, '_MEIPASS', False):
            # only one os.path.dirname() results in the path being relative to
            # the natcap.invest package, when I actually want natcap/invest to
            # be in the filepath.

            # relpath would be something like <modelname>/<data_file>
            relpath = os.path.relpath(source_file, os.path.dirname(__file__))
            pkg_path = os.path.join('natcap', 'invest', relpath)
            return os.path.join(os.path.dirname(sys.executable), os.path.dirname(pkg_path))
        else:
            # assume that if we're in a frozen build, we're in py2exe.  When in
            # py2exe, the directory structure is maintained, so we just return
            # the source_dirname.
            pass
    return source_dirname


def _user_hash():
    """Returns a hash for the user, based on the machine."""
    data = {
        'os': platform.platform(),
        'hostname': platform.node(),
        'userdir': os.path.expanduser('~')
    }

    md5 = hashlib.md5()
    md5.update(json.dumps(data))
    return md5.hexdigest()


def _calculate_args_bounding_box(args_dict):
    """Parse through an args dict and calculate the bounding boxes of any GIS
    types found there.

    Args:
        args_dict (dict): a string key and any value pair dictionary.

    Returns:
        bb_intersection, bb_union tuple that's either the lat/lng bounding
            intersection and union bounding boxes of the gis types referred to
            in args_dict.  If no GIS types are present, this is a (None, None)
            tuple."""

    def _merge_bounding_boxes(bb1, bb2, mode):
        """Helper function to merge two bounding boxes through union or
            intersection

            Args:
                bb1 (list of float): bounding box of the form
                    [minx, maxy, maxx, miny] or None
                bb2 (list of float): bounding box of the form
                    [minx, maxy, maxx, miny] or None

            Returns:
                either the intersection or union of bb1 and bb2 depending
                on mode.  If either bb1 or bb2 is None, the other is returned.
                If both are None, None is returned.
            """
        if bb1 is None:
            return bb2
        if bb2 is None:
            return bb1

        less_than_or_equal = lambda x, y: x if x <= y else y
        greater_than = lambda x, y: x if x > y else y

        if mode == "union":
            comparison_ops = [
                less_than_or_equal, greater_than, greater_than,
                less_than_or_equal]
        if mode == "intersection":
            comparison_ops = [
                greater_than, less_than_or_equal, less_than_or_equal,
                greater_than]

        bb_out = [op(x, y) for op, x, y in zip(comparison_ops, bb1, bb2)]
        return bb_out

    def _merge_local_bounding_boxes(arg, bb_intersection=None, bb_union=None):
        """Allows us to recursively walk a potentially nested dictionary
        and merge the bounding boxes that might be found in the GIS
        types

        Args:
            arg (dict): contains string keys and pairs that might be files to
                gis types.  They can be any other type, including dictionaries.
            bb_intersection (list or None): if list, has the form
                [xmin, ymin, xmax, ymax], where coordinates are in lng, lat
            bb_union (list): same as bb_intersection

        Returns:
            (intersection, union) bounding box tuples of all filepaths to GIS
            data types found in the dictionary and bb_intersection and bb_union
            inputs.  None, None if no arguments were GIS data types and input
            bounding boxes are None."""

        def _is_gdal(arg):
            """tests if input argument is a path to a gdal raster"""
            if (isinstance(arg, str) or
                    isinstance(arg, unicode)) and os.path.exists(arg):
                raster = gdal.Open(arg)
                if raster is not None:
                    return True
            return False

        def _is_ogr(arg):
            """tests if input argument is a path to an ogr vector"""
            if (isinstance(arg, str) or
                    isinstance(arg, unicode)) and os.path.exists(arg):
                vector = ogr.Open(arg)
                if vector is not None:
                    return True
            return False

        if isinstance(arg, dict):
            # if dict, grab the bb's for all the members in it
            for value in arg.itervalues():
                bb_intersection, bb_union = _merge_local_bounding_boxes(
                    value, bb_intersection, bb_union)
        elif isinstance(arg, list):
            # if list, grab the bb's for all the members in it
            for value in arg:
                bb_intersection, bb_union = _merge_local_bounding_boxes(
                    value, bb_intersection, bb_union)
        else:
            # singular value, test if GIS type, if not, don't update bb's
            # this is an undefined bounding box that gets returned when ogr
            # opens a table only
            local_bb = [0., 0., 0., 0.]
            if _is_gdal(arg):
                local_bb = pygeoprocessing.get_bounding_box(arg)
                projection_wkt = pygeoprocessing.get_dataset_projection_wkt_uri(
                    arg)
                spatial_ref = osr.SpatialReference()
                spatial_ref.ImportFromWkt(projection_wkt)
            elif _is_ogr(arg):
                local_bb = pygeoprocessing.get_datasource_bounding_box(arg)
                spatial_ref = pygeoprocessing.get_spatial_ref_uri(arg)

            try:
                # means there's a GIS type with a well defined bounding box
                # create transform, and reproject local bounding box to lat/lng
                lat_lng_ref = osr.SpatialReference()
                lat_lng_ref.ImportFromEPSG(4326)  # EPSG 4326 is lat/lng
                to_lat_trans = osr.CoordinateTransformation(
                    spatial_ref, lat_lng_ref)
                for point_index in [0, 2]:
                    local_bb[point_index], local_bb[point_index + 1], _ = (
                        to_lat_trans.TransformPoint(
                            local_bb[point_index], local_bb[point_index + 1]))

                bb_intersection = _merge_bounding_boxes(
                    local_bb, bb_intersection, 'intersection')
                bb_union = _merge_bounding_boxes(
                    local_bb, bb_union, 'union')
            except Exception:
                # All kinds of exceptions from bad transforms or CSV files
                # or dbf files could get us to this point, just don't bother
                # with the local_bb at all
                pass

        return bb_intersection, bb_union

    return _merge_local_bounding_boxes(args_dict)


def log_model(model_name, model_args, session_id=None):
    """Submit a POST request to the defined URL with the modelname passed in as
    input.  The InVEST version number is also submitted, retrieved from the
    package's resources.

    Args:
        model_name (string): a python string of the package version.
        model_args (dict): the traditional InVEST argument dictionary.

    Returns:
        None."""

    logger = logging.getLogger('natcap.invest.log_model')

    def _node_hash():
        """Returns a hash for the current computational node."""
        data = {
            'os': platform.platform(),
            'hostname': platform.node(),
            'userdir': os.path.expanduser('~')
        }
        md5 = hashlib.md5()
        # a json dump will handle non-ascii encodings
        md5.update(json.dumps(data))
        return md5.hexdigest()

    try:
        bounding_box_intersection, bounding_box_union = (
            _calculate_args_bounding_box(model_args))

        payload = {
            'model_name': model_name,
            'invest_release': __version__,
            'node_hash': _node_hash(),
            'system_full_platform_string': platform.platform(),
            'system_preferred_encoding': locale.getdefaultlocale()[1],
            'system_default_language': locale.getdefaultlocale()[0],
            'bounding_box_intersection': str(bounding_box_intersection),
            'bounding_box_union': str(bounding_box_union),
            'session_id': session_id,
        }

        # http://data.naturalcapitalproject.org/server_registry/invest_usage_logger/
        #path = urllib.urlopen(INVEST_USAGE_LOGGER_URL).read().rstrip()
        # TODO: Debugging for local server
        path = "PYRO:natcap.invest.remote_logging@localhost:54321"
        logging_server = Pyro4.Proxy(path)
        logging_server.log_invest_run(payload)
    except Exception as exception:
        # An exception was thrown, we don't care.
        logger.warn(
            'an exception encountered when logging %s', str(exception))
