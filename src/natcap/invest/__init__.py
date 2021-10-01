"""init module for natcap.invest."""
import builtins
import gettext
import logging
import os
import sys

import pkg_resources

# location of our translation message catalog directory
LOCALE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'translations/locales')

LOGGER = logging.getLogger('natcap.invest')
LOGGER.addHandler(logging.NullHandler())
__all__ = ['local_dir', ]

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    # package is not installed.  Log the exception for debugging.
    LOGGER.exception('Could not load natcap.invest version information')

# Check if the function _() is available
# If not, define it as the identity function
# _() is installed into builtins by gettext when we set up to translate
# It wraps every string in every model that we want to translate
# Make sure it's defined so that natcap.invest modules are importable whether
# or not gettext has been installed in the importing namespace
if not callable(getattr(builtins, '_', None)):
    print('_() not already defined; setting to identity')
    def identity(x): return x
    builtins.__dict__['_'] = identity
else:
    print('_() already defined')


def install_language(language_code):
    # globally install the _() function for the requested language
    # fall back to a NullTranslation, which returns the English messages
    print(LOCALE_DIR)
    language = gettext.translation(
        'messages',
        languages=[language_code],
        localedir=LOCALE_DIR,
        fallback=True)
    language.install()
    LOGGER.debug(f'Installed language "{language_code}"')
    print('installed language', language_code)


def local_dir(source_file):
    """Return the path to where `source_file` would be on disk.

    If this is frozen (as with PyInstaller), this will be the folder with the
    executable in it.  If not, it'll just be the foldername of the source_file
    being passed in.
    """
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
            return os.path.join(
                os.path.dirname(sys.executable), os.path.dirname(pkg_path))
        else:
            # assume that if we're in a frozen build, we're in py2exe.  When in
            # py2exe, the directory structure is maintained, so we just return
            # the source_dirname.
            pass
    return source_dirname
