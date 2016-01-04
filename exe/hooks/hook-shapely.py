import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, get_package_paths
hiddenimports = collect_submodules('shapely')

pkg_base, pkg_dir = get_package_paths('shapely')
datas = collect_data_files('shapely')
datas += [(os.path.join(pkg_dir, 'DLLs/geos_c.dll'), '')]
