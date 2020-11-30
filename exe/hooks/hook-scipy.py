# encoding=UTF-8
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

datas = collect_data_files('scipy')
hiddenimports = collect_submodules('scipy')
