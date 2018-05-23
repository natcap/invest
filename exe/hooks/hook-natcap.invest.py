from PyInstaller.utils.hooks import (collect_data_files,
                                     collect_submodules,
                                     copy_metadata)

hiddenimports = collect_submodules('natcap.invest')
datas = copy_metadata("natcap.invest") + collect_data_files('natcap.invest')
