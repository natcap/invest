import sys
if sys.platform.startswith('linux'):
    from hookutils import collect_submodules
else:
    from PyInstaller.utils.hooks import collect_submodules
hiddenimports = collect_submodules('numpy')
