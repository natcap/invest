"""init module for natcap.invest"""

import os
import sys
import pkg_resources
import logging

import pygeoprocessing
import natcap.versioner

from natcap.invest.tests import test


__all__ = ['is_release', 'local_dir', 'test', 'PYGEOPROCESSING_REQUIRED']


# Verify that the installed pygeoprocessing meets the minimum requirements.
# Pyinstaller binaries do not allow us to use pkg_resources.require(), as
# no EGG_INFO is included in the binary distribution.
# pkg_resources is preferred over distutils.StrictVersion and
# distutils.LooseVersion, since pkg_resources.parse_version is
# PEP440-compliant and it's very likely that a dev version of pygeoprocessing
# will be found.
PYGEOPROCESSING_REQUIRED = '0.3.0a11'
if (pkg_resources.parse_version(pygeoprocessing.__version__) <
        pkg_resources.parse_version(PYGEOPROCESSING_REQUIRED)):
    raise ValueError(('Pygeoprocessing >= {req_version} required, '
                      'but version {found_ver} was found').format(
                          req_version=PYGEOPROCESSING_REQUIRED,
                          found_ver=pygeoprocessing.__version__))

__version__ = natcap.versioner.get_version('natcap.invest')

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')


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
