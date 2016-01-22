from sys import platform as _platform
import os
from PyInstaller.compat import is_darwin, is_win
if _platform == "linux" or _platform == "linux2":
    from hookutils import \
        (collect_submodules, collect_data_files, get_package_paths)
else:
    from PyInstaller.utils.hooks import \
        (collect_submodules, collect_data_files, get_package_paths)

hiddenimports = collect_submodules('shapely')
pkg_base, pkg_dir = get_package_paths('shapely')
datas = collect_data_files('shapely')

if is_win:
    datas += [(os.path.join(pkg_dir, 'DLLs/geos_c.dll'), '')]
elif is_darwin:
    datas += [(os.path.join(pkg_dir, '.dylibs/*.dylib'), '')]
