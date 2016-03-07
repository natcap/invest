from PyInstaller.compat import is_darwin

if is_darwin:
    from PyInstaller.utils.hooks import collect_data_files, collect_submodules
else:
    from PyInstaller.hooks.hookutils import collect_data_files, collect_submodules
datas = collect_data_files('natcap.versioner')
hiddenimports = collect_submodules('natcap.versioner')
