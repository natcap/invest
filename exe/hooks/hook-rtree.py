from PyInstaller.compat import is_win

from PyInstaller.hooks.hookutils import get_package_paths
import os

if is_win:
    files = [
        'spatialindex_c.dll',
        'spatialindex.dll',
    ]
    pkg_base, pkg_dir = get_package_paths('rtree')
    datas = [(os.path.join(pkg_dir, filename), '') for filename in files]
