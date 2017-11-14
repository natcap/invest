import sys
import os
import multiprocessing
import platform

# sip assumes qt API v2 is used.
# Taken from https://github.com/pyinstaller/pyinstaller/wiki/Recipe-PyQt4-API-Version
import sip
sip.setapi(u'QDate', 2)
sip.setapi(u'QDateTime', 2)
sip.setapi(u'QString', 2)
sip.setapi(u'QTextStream', 2)
sip.setapi(u'QTime', 2)
sip.setapi(u'QUrl', 2)
sip.setapi(u'QVariant', 2)

multiprocessing.freeze_support()

os.environ['MATPLOTLIBDATA'] = os.path.join(sys._MEIPASS, 'mpl-data')

if platform.system() == 'Darwin':
    os.environ['GDAL_DATA'] = os.path.join(sys._MEIPASS, 'gdal-data', 'gdal')
