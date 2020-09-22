import sys
import os
import multiprocessing
import platform

multiprocessing.freeze_support()

os.environ['PROJ_LIB'] = os.path.join(sys._MEIPASS, 'proj')
