from PyInstaller.compat import is_win, is_darwin, is_linux

# Special hook necessary for PyInstaller v2.x (our linux builds)
if is_darwin:
    from PyInstaller.utils.hooks import collect_data_files
else:
    from PyInstaller.hooks.hookutils import collect_data_files

datas = collect_data_files('osgeo')
