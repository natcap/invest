# Special hook necessary for PyInstaller v2.x (our linux builds)
from PyInstaller.compat import is_linux

if is_linux:
    hiddenimports = ['_proxy', 'utils', 'defs', 'h5ac']
