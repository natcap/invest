from sys import platform as _platform
if _platform == "linux" or _platform == "linux2":
    # linux
    from hookutils import collect_submodules
    hiddenimports = collect_submodules('numpy')
else:
    from PyInstaller.utils.hooks import collect_submodules
    hiddenimports = collect_submodules('numpy')
