import sys
import os
import multiprocessing
import platform

import sip
sip.setapi('QString', 2)  # qtpy assumes api version 2

multiprocessing.freeze_support()

os.environ['MATPLOTLIBDATA'] = os.path.join(sys._MEIPASS, 'mpl-data')

if platform.system() == 'Darwin':
    os.environ['GDAL_DATA'] = os.path.join(sys._MEIPASS, 'gdal-data', 'gdal')
