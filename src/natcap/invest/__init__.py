"""init module for natcap.invest"""

from urllib import urlencode
from urllib2 import Request
from urllib2 import urlopen
import locale
import os
import platform
import sys
import hashlib
import json
import pkg_resources

import pygeoprocessing
import natcap.versioner

from natcap.invest.tests import test

__all__ = ['test']

# Verify that the installed pygeoprocessing meets the minimum requirements.
# Pyinstaller binaries do not allow us to use pkg_resources.require(), as
# no EGG_INFO is included in the binary distribution.
# pkg_resources is preferred over distutils.StrictVersion and
# distutils.LooseVersion, since pkg_resources.parse_version is
# PEP440-compliant and it's very likely that a dev version of pygeoprocessing
# will be found.
PYGEOPROCESSING_REQUIRED = '0.3.0a8'
if (pkg_resources.parse_version(pygeoprocessing.__version__) <
        pkg_resources.parse_version(PYGEOPROCESSING_REQUIRED)):
    raise ValueError(('Pygeoprocessing >= {req_version} required, '
                      'but version {found_ver} was found').format(
                          req_version=PYGEOPROCESSING_REQUIRED,
                          found_ver=pygeoprocessing.__version__))

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


def _user_hash():
    """Returns a hash for the user, based on the machine."""
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


def log_model(model_name, model_version=None):
    """Submit a POST request to the defined URL with the modelname passed in as
    input.  The InVEST version number is also submitted, retrieved from the
    package's resources.

        model_name - a python string of the package version.
        model_version=None - a python string of the model's version.  Defaults
            to None if a model version is not provided.

    returns nothing."""

    path = 'http://ncp-dev.stanford.edu/~invest-logger/log-modelname.php'
    data = {
        'model_name': model_name,
        'invest_release': __version__,
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
    }

    if model_version is None:
        model_version = __version__
    data['model_version'] = model_version

    try:
        urlopen(Request(path, urlencode(data)))
    except Exception:
        # An exception was thrown, we don't care.
        print 'an exception encountered when logging'
