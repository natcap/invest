from PyInstaller.utils.hooks import (collect_data_files,
                                     copy_metadata)

datas = copy_metadata("charset-normalizer") + collect_data_files('charset-normalizer')
