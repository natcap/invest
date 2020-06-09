# coding=UTF-8
# -*- mode: python -*-
import sys
import os
import itertools
import glob
from PyInstaller.compat import is_win, is_darwin

# Global Variables
current_dir = os.getcwd()  # assume we're building from the project root
block_cipher = None
invest_exename = 'invest'
server_exename = 'server'
conda_env = '/usr/local/miniconda/envs/mac-env'


kwargs = {
    'hookspath': [os.path.join(current_dir, 'exe', 'hooks')],
    'excludes': None,
    'pathex': sys.path,
    'runtime_hooks': [os.path.join(current_dir, 'exe', 'hooks', 'rthook.py')],
    'hiddenimports': [
        'natcap',
        'natcap.invest',
        'natcap.invest.ui.launcher',
        'yaml',
        'distutils',
        'distutils.dist',
        'rtree',  # mac builds aren't picking up rtree by default.
        'pkg_resources.py2_warn'
    ],
    'datas': [('qt.conf', '.')],
    'cipher': block_cipher,
}

cli_file = os.path.join(current_dir, 'src', 'natcap', 'invest', 'cli.py')
invest_a = Analysis([cli_file], **kwargs)

# This path matches the directory setup in Makefile (make fetch)
flask_run_file = os.path.join(current_dir, 'gui', 'src', 'server.py')
# All the same kwargs apply because this app also imports natcap.invest
server_a = Analysis([flask_run_file], **kwargs)

MERGE((invest_a, invest_exename, invest_exename),
      (server_a, server_exename, server_exename))

# Compress pyc and pyo Files into ZlibArchive Objects
invest_pyz = PYZ(invest_a.pure, invest_a.zipped_data, cipher=block_cipher)
server_pyz = PYZ(server_a.pure, server_a.zipped_data, cipher=block_cipher)

# Create the executable file.
if is_darwin:
    # add rtree dependency dynamic libraries from conda environment
    invest_a.binaries += [
        (os.path.basename(name), name, 'BINARY') for name in
        glob.glob(os.path.join(conda_env, 'lib/libspatialindex*.dylib'))]
elif is_win:
    # Adapted from
    # https://shanetully.com/2013/08/cross-platform-deployment-of-python-applications-with-pyinstaller/
    # Supposed to gather the mscvr/p DLLs from the local system before
    # packaging.  Skirts the issue of us needing to keep them under version
    # control.
    invest_a.binaries += [
        ('msvcp90.dll', 'C:\\Windows\\System32\\msvcp90.dll', 'BINARY'),
        ('msvcr90.dll', 'C:\\Windows\\System32\\msvcr90.dll', 'BINARY')
    ]

    # .exe extension is required if we're on windows.
    invest_exename += '.exe'
    server_exename += '.exe'

invest_exe = EXE(
    invest_pyz,
    invest_a.scripts,
    name=invest_exename,
    exclude_binaries=True,
    debug=False,
    strip=False,
    upx=False,
    console=True)

server_exe = EXE(
    server_pyz,
    server_a.scripts,
    name=server_exename,
    exclude_binaries=True,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True)

# Collect Files into Distributable Folder/File
invest_dist = COLLECT(
        invest_exe,
        invest_a.binaries,
        invest_a.zipfiles,
        invest_a.datas,
        server_exe,
        server_a.binaries,
        server_a.zipfiles,
        server_a.datas,
        name="invest",  # name of the output folder
        strip=False,
        upx=False)
