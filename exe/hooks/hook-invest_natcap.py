from PyInstaller.hooks.hookutils import collect_data_files, collect_submodules
datas = collect_data_files('invest_natcap')
hiddenimports = collect_submodules('invest_natcap')
