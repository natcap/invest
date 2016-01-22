from sys import platform as _platform
if _platform == "linux" or _platform == "linux2":
    from hookutils import collect_submodules
else:
    from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules('ctypes')
