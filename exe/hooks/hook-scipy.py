from PyInstaller.utils.hooks import collect_submodules
hiddenimports = ['scipy.special._ufuncs_cxx', 'scipy.io.matlab.streams', 'scipy.sparse.cgraph._validation'] + collect_submodules('scipy.linalg')
