import sys
import os
import multiprocessing
import platform

multiprocessing.freeze_support()

# if platform.system() == 'Darwin':
#     os.environ['GDAL_DATA'] = os.path.join(sys._MEIPASS, 'gdal-data', 'gdal')
