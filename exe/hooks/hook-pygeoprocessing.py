from PyInstaller.compat import is_darwin, is_win

if is_darwin or is_win:
    from PyInstaller.utils.hooks import collect_data_files
else:
    from PyInstaller.hooks.hookutils import collect_data_files

datas = collect_data_files('pygeoprocessing')
hiddenimports = ['pygeoprocessing.version']
