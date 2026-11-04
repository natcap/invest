"""init module for natcap.invest."""
import importlib.metadata
import logging
import os
import sys
from gettext import translation

import babel
import certifi

LOGGER = logging.getLogger('natcap.invest')
LOGGER.addHandler(logging.NullHandler())

try:
    __version__ = importlib.metadata.version('natcap.invest')
except importlib.metadata.PackageNotFoundError:
    try:
        __version__ = importlib.metadata.version('natcap_invest')
    except importlib.metadata.PackageNotFoundError:
        # package is not installed.  Log the exception for debugging.
        LOGGER.exception(
            'Could not load natcap.invest (or natcap_invest) version information')

# Needed to reliably open datasets over HTTPS across different
# systems when running in a pyinstaller frozen environment
# https://github.com/natcap/invest/issues/2200
os.environ['SSL_CERT_FILE'] = certifi.where()
# certifi.where() sometimes returns the wrong location when
# running the frozen environment.
# https://github.com/natcap/invest/issues/2816
print('========= looking for cert file ============')
if getattr(sys, '_MEIPASS', False):
    for root, dirs, files in os.walk(sys._MEIPASS):
        if 'cacert.pem' in files:
            print(root)
    print('========================================')

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
# track the current locale setting,
# this can be changed during runtime by the set_locale function
# the gettext function below uses this to set the translation language
LOCALE_CODE = 'en'


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
    setattr(this_module, 'LOCALE_CODE', locale_code)


def gettext(msg):
    """natcap.invest.gettext, the translation function used throughout.

    By referencing translation.gettext within this function, it's updated
    according to the currently active locale.

    Args:
        msg (string): message string to translate
    """
    return translation(
        'messages',
        languages=[LOCALE_CODE],
        localedir=LOCALE_DIR,
        # fall back to a NullTranslation, which returns the English messages
        fallback=True).gettext(msg)
