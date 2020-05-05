from PyInstaller.compat import is_darwin
from PyInstaller.utils.hooks import (collect_system_data_files,
                                     collect_data_files,
                                     collect_submodules)
import os

if is_darwin:
    # Assume we're using a local conda env to install gdal.
    # glob for gcs.csv instead of passing the env name.
    # import glob
    # datas = collect_system_data_files(
    # 	path=os.path.dirname(glob.glob('**/gcs.csv', recursive=True)[0]),
    #     destdir='gdal-data')
    pass
else:
    datas = collect_data_files('osgeo')
