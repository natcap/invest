from PyInstaller.compat import is_darwin
from PyInstaller.utils.hooks import collect_data_files

if is_darwin:
	import glob
	binaries = [(glob.glob('**/lib/libspatialindex*'), 'dylib')]

else:
	datas = collect_data_files('rtree')
