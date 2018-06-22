# encoding=UTF-8
"""hook-pandas.py"""

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules('pandas')

