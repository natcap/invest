"""init module for natcap.invest."""
import builtins
import dataclasses
from gettext import translation
import logging
import os
import sys

import babel
import pkg_resources

LOGGER = logging.getLogger('natcap.invest')
LOGGER.addHandler(logging.NullHandler())
__all__ = ['local_dir', ]

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    # package is not installed.  Log the exception for debugging.
    LOGGER.exception('Could not load natcap.invest version information')

# location of our translation message catalog directory
LOCALE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'internationalization/locales')
# all supported language codes, including the default English
LOCALES = sorted(set(os.listdir(LOCALE_DIR) + ['en']))
# map locale codes to the corresponding localized language name
# e.g. 'es': 'español'
LOCALE_NAME_MAP = {
    locale: babel.Locale(locale).display_name for locale in LOCALES
}

def set_locale(locale_code):
    """Set the `gettext` attribute of natcap.invest.

    This is the locale that will be used for translation. The `gettext`
    function returned by `install_locale` will translate to this locale.
    To change the language of a natcap.invest module, call this function
    with the desired language code, then reload the module.

    Args:
        locale_code (str): ISO 639-1 locale code for a language supported
            by invest

    Returns:
        None

    Raises:
        ValueError if the given locale code is not supported by invest
    """
    if locale_code not in LOCALES:
        raise ValueError(
            f"Locale '{locale_code}' is not supported by InVEST. "
            f"Supported locale codes are: {LOCALES}")
    this_module = sys.modules[__name__]
    gettext = translation(
        'messages',
        languages=[locale_code],
        localedir=LOCALE_DIR,
        # fall back to a NullTranslation, which returns the English messages
        fallback=True).gettext
    setattr(this_module, 'gettext', gettext)

# create natcap.invest.gettext, the default translation function
set_locale('en')

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
