from PyInstaller.compat import is_darwin

if is_darwin:
    from PyInstaller.utils.hooks import collect_submodules
else:
    from hookutils import collect_submodules

hiddenimports = collect_submodules('ctypes')
