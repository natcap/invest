# Special hook necessary for PyInstaller v2.x (our linux builds)
import sys
if sys.platform.startswith('linux'):
    from PyInstaller.hooks.hookutils import collect_data_files
    datas = collect_data_files('osgeo')
