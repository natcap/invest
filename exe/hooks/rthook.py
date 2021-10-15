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

    # Rtree will look in this directory first for libspatialindex_c.dylib.
    # In response to issues with github mac binary builds:
    # https://github.com/natcap/invest/issues/594
    # sys._MEIPASS is the path to where the pyinstaller entrypoint bundle
    # lives.  See the pyinstaller docs for more details.
    os.environ['SPATIALINDEX_C_LIBRARY'] = sys._MEIPASS

if platform.system() == 'Windows':
    # On Windows, with Python >= 3.8, DLLs are no longer imported from the PATH.
    # This is good for security, but it also means we need to be sure that
    # pyinstaller can find our .pyd files.
    # The commit message at
    # https://github.com/OSGeo/gdal/commit/6c8c66e41928b54f341336fa66982029d5bb9745
    # has some helpful information about the intent of the change in how GDAL
    # imports its DLLs.
    os.add_dll_directory(os.path.dirname(sys.executable))
    os.add_dll_directory(sys._MEIPASS)
