import sys
import os
import multiprocessing
import platform

multiprocessing.freeze_support()

os.environ['PROJ_LIB'] = os.path.join(sys._MEIPASS, 'proj')

if platform.system() == 'Darwin':
    # This allows Qt 5.13+ to start on Big Sur.
    # See https://bugreports.qt.io/browse/QTBUG-87014
    # and https://github.com/natcap/invest/issues/384
    os.environ['QT_MAC_WANTS_LAYER'] = '1'

elif platform.system() == 'Windows':
    # Remove any paths from the PATH that don't exist
    # This is necessary to prevent a FileNotFoundError when importing gdal
    # with USE_PATH_FOR_GDAL_PYTHON=YES, which is needed to import gdal >=3.3.0
    # https://github.com/OSGeo/gdal/issues/3898
    filtered_paths = []
    for path in os.environ['PATH'].split(';'):
        if os.path.exists(path):
            filtered_paths.append(path)
    os.environ['PATH'] = ';'.join(filtered_paths)
