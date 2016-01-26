# Special hook necessary for PyInstaller v2.x (our linux builds)
import sys
if sys.platform.startswith('linux'):
    hiddenimports = ['_proxy', 'utils', 'defs', 'h5ac']
