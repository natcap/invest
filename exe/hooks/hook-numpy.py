from PyInstaller.compat import is_darwin, is_win

if is_darwin or is_win:
    from PyInstaller.utils.hooks import collect_submodules
else:
    from hookutils import collect_submodules

hiddenimports = collect_submodules('numpy')
