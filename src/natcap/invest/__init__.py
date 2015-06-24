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
import distutils.version
import build_utils

try:
    __version__ = build_utils.invest_version()
except:
    __version__ = 'dev'

def is_release():
    """Returns a boolean indicating whether this invest release is actually a
    release or if it's a development release."""
    if __version__[0:3] == 'dev':
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
            # the natcap.invest package, when I actually want natcap.invest to
            # be in the filepath.
            # with 1 dirname, path is 'reporting/reporting_data'
            # with 2 dirnames, path is 'natcap.invest/reporting/reporting_data'
            package_dirname = os.path.dirname(os.path.dirname(__file__))
            relpath = os.path.relpath(source_dirname, package_dirname)
            return os.path.join(os.path.dirname(sys.executable), relpath)
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

    if model_version == None:
        model_version = __version__
    data['model_version'] = model_version

    try:
        urlopen(Request(path, urlencode(data)))
    except:
        # An exception was thrown, we don't care.
        print 'an exception encountered when logging'
        pass

