# Special hook necessary for PyInstaller v2.x (our linux builds)
from PyInstaller.compat import is_darwin

if not is_darwin:
    hiddenimports = ['_proxy', 'utils', 'defs', 'h5ac']
