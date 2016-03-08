import sys
import os
import multiprocessing

multiprocessing.freeze_support()

os.environ['MATPLOTLIBDATA'] = os.path.join(sys._MEIPASS, 'mpl-data')
