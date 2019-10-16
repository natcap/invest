import sys
import os
import multiprocessing
import platform

multiprocessing.freeze_support()

os.environ['MATPLOTLIBDATA'] = os.path.join(sys._MEIPASS, 'mpl-data')

if platform.system() == 'Darwin':
    os.environ['GDAL_DATA'] = os.path.join(sys._MEIPASS, 'gdal-data', 'gdal')
