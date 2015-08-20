"""init module for natcap.invest"""

import locale
import os
import platform
import sys
import hashlib
import json
import distutils.version
import datetime
import logging

import Pyro4
import natcap.versioner

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

_LOGGER = logging.getLogger('natcap.invest.remote_logging')


try:
    import pygeoprocessing
    REQUIRED_PYGEOPROCESSING_VERSION = '0.3.0a7'
    if (distutils.version.StrictVersion(pygeoprocessing.__version__) <
            distutils.version.StrictVersion(REQUIRED_PYGEOPROCESSING_VERSION)):
        raise Exception(
            "Requires PyGeoprocessing version at least %s.  "
            "Current version %s ",
            REQUIRED_PYGEOPROCESSING_VERSION, pygeoprocessing.__version__)
except ImportError:
    pass

__version__ = natcap.versioner.get_version('natcap.invest')

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
        if getattr(sys, '_MEIPASS', False) != False:
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

def _node_hash():
    """Returns a hash for the current computational node."""
    data = {
        'os': platform.platform(),
        'hostname': platform.node(),
        'userdir': os.path.expanduser('~')
    }
    try:
        md5 = hashlib.md5()
        md5.update(json.dumps(data))
        return md5.hexdigest()
    except:
        return None

def log_model(model_name, model_args):
    """Submit a POST request to the defined URL with the modelname passed in as
    input.  The InVEST version number is also submitted, retrieved from the
    package's resources.

    Args:

        model_name (string): a python string of the package version.
        model_args (dict): the traditional InVEST argument dictionary.

    Returns:
        None."""

    try:
        payload = {
            'model_name': model_name,
            'invest_release': __version__,
            'node_hash': _node_hash(),
            'system_full_platform_string': platform.platform(),
            'system_preferred_encoding': locale.getdefaultlocale()[1],
            'system_default_language': locale.getdefaultlocale()[0],
            # too hard to get reliable timezone
            'time': datetime.datetime.now().isoformat(' '),
            'bounding_box_intersection': "[]",
            'bounding_box_union': "[]"
        }

        path = "PYRO:natcap.invest.remote_logging@localhost:54321"
        logging_server = Pyro4.Proxy(path)
        logging_server.log_invest_run(payload)
    except:
        # An exception was thrown, we don't care.
        _LOGGER.warn('an exception encountered when logging')
