from PyInstaller.compat import is_win
from PyInstaller.utils.hooks import collect_dynamic_libs

if is_win:
    # GDAL appears to need `_gdal.cp38-win_amd64.pyd` located specifically in
    # `osgeo/_gdal....pyd` in order to work.  This is because the GDAL python
    # __init__ script specifically looks in the `osgeo` directory in order to
    # find it.  This is apparently only an issue on Windows.
    #
    # This will take the dynamic libraries in osgeo and put them into osgeo,
    # relative to the binaries directory.
    binaries = collect_dynamic_libs('osgeo', 'osgeo')
