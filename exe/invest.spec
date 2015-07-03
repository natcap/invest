import sys
import os
from PyInstaller.compat import is_win

# Global Variables
current_dir = os.path.join(os.getcwd(), os.path.dirname(sys.argv[1]))

# Analyze Scripts for Dependencies
kwargs = {
    'hookspath': [os.path.join(current_dir, 'hooks')],
    'excludes': None,
    'pathex': [os.getcwd()],
    'hiddenimports': ['natcap', 'natcap.invest'],
}

cli_file = os.path.join(current_dir, '..', 'src', 'natcap', 'invest', 'iui', 'cli.py')
a = Analysis([cli_file], **kwargs)

# Compress pyc and pyo Files into ZlibArchive Objects
pyz = PYZ(a.pure)

# Create the executable file
exename = 'invest'
if is_win:
    exename += '.exe'

exe = EXE(
    pyz,
    a.scripts,
    name=exename,
    exclude_binaries=1,
    debug=False,
    strip=None,
    upx=False,
    console=True)

# Collect Files into Distributable Folder/File
args = [exe, a.binaries, a.zipfiles, a.datas]

dist = COLLECT(
        *args,
        name="invest_dist",
        strip=None,
        upx=True)
