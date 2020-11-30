from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules('shapely')
datas = collect_data_files('shapely')
