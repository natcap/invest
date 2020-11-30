from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files('qtawesome')
hiddenimports = collect_submodules('qtawesome')
