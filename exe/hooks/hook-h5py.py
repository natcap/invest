from sys import platform as _platform
if _platform == "linux" or _platform == "linux2":
    hiddenimports = ['_proxy', 'utils', 'defs', 'h5ac']
