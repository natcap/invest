import sys
import os
from PyInstaller.compat import is_win, is_darwin

# Global Variables
current_dir = os.path.join(os.getcwd(), os.path.dirname(sys.argv[1]))

# Analyze Scripts for Dependencies

# Add the release virtual environment to the extended PATH.
# This helps IMMENSELY with trying to get the binaries to work from within
# a virtual environment, even if the virtual environment is hardcoded.
path_extension = []
release_env_dir = os.path.abspath(os.path.join('..', 'release_env'))
if is_win:
    import distutils
    env_path_base = os.path.join(release_env_dir, 'lib')
else:
    env_path_base = os.path.join(release_env_dir, 'lib', 'python2.7')

# We're in a virtualenv if the expected env lib dir exists AND the python
# executable is within the release env dir.
# NOTE: Pyinstaller seems to pick up packages within the global site-packages
# just fine, so we don't need to modify the pathext when we're not in a
# virtualenv.
if os.path.exists(env_path_base) and sys.executable.startswith(release_env_dir):
    env_path_base = os.path.abspath(env_path_base)
    path_extension.insert(0, env_path_base)
    path_extension.insert(0, os.path.join(env_path_base, 'site-packages'))

print 'PATH EXT: %s' % path_extension

kwargs = {
    'hookspath': [os.path.join(current_dir, 'hooks')],
    'excludes': None,
    'pathex': path_extension,
    'hiddenimports': [
        'natcap',
        'natcap.invest',
        'natcap.versioner',
        'natcap.versioner.version',
        'natcap.invest.version',
        'yaml',
        'distutils',
        'distutils.dist',
        'rtree',  # mac builds aren't picking up rtree by default.
    ],
}

cli_file = os.path.join(current_dir, '..', 'src', 'natcap', 'invest', 'iui', 'cli.py')
a = Analysis([cli_file], **kwargs)

# Compress pyc and pyo Files into ZlibArchive Objects
pyz = PYZ(a.pure)

# Create the executable file.
# .exe extension is required if we're on Windows.
exename = 'invest'
if is_win:
    exename += '.exe'

if is_darwin:
    # remove shapely dynamic library collision
    a.binaries = filter(lambda x: x[0] != 'libgeos_c.1.dylib', a.binaries)
    # remove matplotlib dynamic library collision
    a.binaries = filter(lambda x: x[0] != 'libpng16.16.dylib', a.binaries)
    # add gdal dynamic libraries from homebrew
    a.binaries += [('geos_c.dll', '/usr/local/lib/libgeos_c.dylib', 'BINARY')]
    a.binaries += [('libgeos_c.dylib', '/usr/local/lib/libgeos_c.dylib', 'BINARY')]
    a.binaries += [('libgeos_c.1.dylib', '/usr/local/lib/libgeos_c.1.dylib', 'BINARY')]
    a.binaries += [('libgeos-3.5.0.dylib', '/usr/local/lib/libgeos-3.5.0.dylib', 'BINARY')]
    a.binaries += [('libgeotiff.dylib', '/usr/local/lib/libgeotiff.dylib', 'BINARY')]
    a.binaries += [('libgeotiff.2.dylib', '/usr/local/lib/libgeotiff.2.dylib', 'BINARY')]
    a.binaries += [('libpng.dylib', '/usr/local/lib/libpng.dylib', 'BINARY')]
    a.binaries += [('libpng16.16.dylib', '/usr/local/lib/libpng16.16.dylib', 'BINARY')]

exe = EXE(
    pyz,

    # Taken from:
    # https://shanetully.com/2013/08/cross-platform-deployment-of-python-applications-with-pyinstaller/
    # Supposed to gather the mscvr/p DLLs from the local system before
    # packaging.  Skirts the issue of us needing to keep them under version
    # control.
    a.binaries + [
        ('msvcp90.dll', 'C:\\Windows\\System32\\msvcp90.dll', 'BINARY'),
        ('msvcr90.dll', 'C:\\Windows\\System32\\msvcr90.dll', 'BINARY')
    ] if is_win else a.binaries,
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
        upx=False)
