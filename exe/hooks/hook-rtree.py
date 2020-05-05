from PyInstaller.compat import is_darwin
from PyInstaller.utils.hooks import collect_data_files

if is_darwin:
	import glob
	binaries = [
		(binary, 'dylib') for binary in glob.glob(
			'/usr/local/miniconda/envs/mac-env/lib/libspatialindex*')]

else:
	datas = collect_data_files('rtree')
