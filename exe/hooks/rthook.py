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
