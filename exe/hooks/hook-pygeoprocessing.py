from sys import platform as _platform
if _platform == "linux" or _platform == "linux2":
    from PyInstaller.hooks.hookutils import collect_data_files
else:
    from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files('pygeoprocessing')
hiddenimports = ['pygeoprocessing.version']
