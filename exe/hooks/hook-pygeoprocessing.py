from PyInstaller.compat import is_darwin

if is_darwin:
    from PyInstaller.utils.hooks import collect_data_files
else:
    from PyInstaller.hooks.hookutils import collect_data_files

datas = collect_data_files('pygeoprocessing')
hiddenimports = ['pygeoprocessing.version']
