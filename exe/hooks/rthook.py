import sys
import os
import multiprocessing

multiprocessing.freeze_support()

os.environ['MATPLOTLIBDATA'] = os.path.join(sys._MEIPASS, 'mpl-data')

sys.platform.startswith('darwin'):
    os.environ['PATH'] += ':' + os.path.dirname(sys.executable)
    os.environ['PATH'] += ':' + os.path.dirname(sys._MEIPASS)
    os.environ['DYLD_LIBRARY_PATH'] += ':' + os.path.dirname(sys.executable)
    os.environ['DYLD_LIBRARY_PATH'] += ':' + os.path.dirname(sys._MEIPASS)
