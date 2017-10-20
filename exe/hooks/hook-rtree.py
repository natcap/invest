from PyInstaller.compat import is_win
import os

if is_win:
    # Windows and linux are still on pyinstaller 2.x, so the imports differ
    # from 3.x
    files = [
        'spatialindex_c.dll',
        'spatialindex.dll',
    ]
    import rtree
    pkg_dir = rtree.__path__[0]
    datas = [(os.path.join(pkg_dir, filename), '') for filename in files]
