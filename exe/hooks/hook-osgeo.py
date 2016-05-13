# This file was removed for the purposes of the mac build (since a more robust
# hook for osgeo was added after PyInstaller 3), but we still need this for
# PyInstaller 2.1 which we use on Windows.
try:
    # This is the pyinstaller 2.1-compatible stuff
    from PyInstaller.hooks.hookutils import collect_data_files
    datas = collect_data_files('osgeo')
except ImportError:
    # If we're running pyinstaller 3, defer to the existing osgeo hook that
    # they distribute.
    pass

