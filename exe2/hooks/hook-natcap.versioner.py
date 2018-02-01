from PyInstaller.utils.hooks import (copy_metadata,
                                     collect_data_files,
                                     collect_submodules)

datas = collect_data_files('natcap.versioner') + copy_metadata('natcap.versioner')
hiddenimports = collect_submodules('natcap.versioner')
