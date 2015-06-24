import sys
import os

sys.path.append("/usr/share/qgis/python/plugins")
sys.path.append(os.getenv("HOME")+"/.qgis/python/plugins")

import PyQt4
import sextante
import qgis
import qgis.utils

def main():
    app = PyQt4.QtGui.QApplication(sys.argv)
    qgis.core.QgsApplication.setPrefixPath("/usr/lib/qgis", True)
    qgis.core.QgsApplication.initQgis()
    sextante.core.Sextante.Sextante.initialize()
    run_script(qgis.utils.iface)

def run_script(iface):
    """ this shall be called from Script Runner"""
    sextante.alglist()
    sextante.alghelp("grass:r.los")

    dem = sys.argv[1]
    out_uri = sys.argv[2]
    x = sys.argv[3]
    y = sys.argv[4]

    sextante.runalg("grass:r.los",
                    input=dem,
                    coordinate=",".join(x,y),
                    output=out_uri)

if __name__=="__main__":
    main()

