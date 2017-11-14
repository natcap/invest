from PyInstaller.compat import is_linux

# Special hook necessary for PyInstaller v2.x (our linux builds)
if is_linux:
    from PyInstaller.hooks.hookutils import collect_submodules
    hiddenimports = ['scipy.special._ufuncs_cxx', 'scipy.io.matlab.streams', 'scipy.sparse.cgraph._validation'] + collect_submodules('scipy.linalg')
