from PyInstaller.utils.hooks import copy_metadata

datas = copy_metadata("natcap.invest")
hiddenimports = ['faulthandler', 'natcap.invest.ui.launcher']
