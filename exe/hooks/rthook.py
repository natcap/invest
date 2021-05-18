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

if platform.system() == 'Windows':
    # Encountered with the GDAL 3.3.0 release.  Specific exception below.
    #
    # On Windows, with Python >= 3.8, DLLs are no longer imported from the
    # PATH.  If gdalXXX.dll is in the PATH, then set the
    # USE_PATH_FOR_GDAL_PYTHON=YES environment variable to feed the PATH into
    # os.add_dll_directory().
    os.environ['USE_PATH_FOR_GDAL_PYTHON'] = 'YES'
