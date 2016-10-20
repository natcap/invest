from PyInstaller.compat import is_darwin
import os

try:
    # This is the pyinstaller 2.1-compatible stuff
    from PyInstaller.hooks.hookutils import collect_data_files
    datas = collect_data_files('osgeo')
except ImportError:
    if is_darwin:
        # Assume we're using homebrew to install GDAL and collect data files
        # accordingly.
        from PyInstaller.utils.hooks import get_homebrew_path
        from PyInstaller.utils.hooks import collect_system_data_files

        datas = collect_system_data_files(
            path=os.path.join(get_homebrew_path('gdal'), 'share', 'gdal'),
            destdir='gdal-data')
