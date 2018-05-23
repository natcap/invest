from PyInstaller.utils.hooks import collect_data_files, copy_metadata

datas = collect_data_files('pygeoprocessing') + copy_metadata('pygeoprocessing')
