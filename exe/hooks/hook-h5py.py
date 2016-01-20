from sys import platform as _platform
if _platform == "linux" or _platform == "linux2":
    # linux
    hiddenimports = ['_proxy', 'utils', 'defs', 'h5ac']
