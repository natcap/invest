from PyInstaller.compat import is_win, is_darwin
import os

if is_win:
    # Windows and linux are still on pyinstaller 2.x, so the imports differ
    # from 3.x
    from PyInstaller.hooks.hookutils import get_package_paths
    files = [
        'spatialindex_c.dll',
        'spatialindex.dll',
    ]
    pkg_base, pkg_dir = get_package_paths('rtree')
    datas = [(os.path.join(pkg_dir, filename), '') for filename in files]
