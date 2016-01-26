import sys
if sys.platform.startswith('linux'):
    from PyInstaller.hooks.hookutils import collect_data_files, collect_submodules
else:
    from PyInstaller.utils.hooks import collect_data_files, collect_submodules
datas = collect_data_files('natcap.versioner')
hiddenimports = collect_submodules('natcap.versioner')
