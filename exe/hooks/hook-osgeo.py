from PyInstaller.compat import is_win
from PyInstaller.hooks.hookutils import get_package_paths, collect_data_files
# from osgeo import gdal
# import platform
# import os
# import glob

# files = []
# if is_win:
#     files += [
#         'geos_c.dll',
#     ]
#     # If shapely is present, use that version of GEOS_C.dll
#     try:
#         import shapely
#         pkg_dir = os.path.dirname(shapely.__file__)
#     except (ImportError, AssertionError) as error:
#         # ImportError is raised when we can't import shapely
#         # AssertionError is raised when the package path can't be found.
#         print 'Defaulting to osgeo pkg:', error
#         pkg_base, pkg_dir = get_package_paths('osgeo')

#     osgeo_base, osgeo_dir = get_package_paths('osgeo')
#     data_dir = os.path.join(osgeo_dir, 'data', 'gdal')

#     datas = [(os.path.join(pkg_dir, filename), '') for filename in files]

#     datas += [(os.path.join(data_dir, filename), '') for filename in
#             glob.glob(data_dir + '/*')]

# else:
#     if platform.system() == 'Darwin':
#         # Accommodate multiple gdal version installations
#         _base_data_dir = '/usr/local/Cellar/gdal/'
#         _version_glob = "%s*" % gdal.__version__
#         _gdal_version_dirname = glob.glob(_base_data_dir + _version_glob)[-1]
#         data_dir = os.path.join(_gdal_version_dirname, 'share', 'gdal')

#         datas = [(os.path.join(pkg_dir, filename), '') for filename in files]

#         datas += [(os.path.join(data_dir, filename), '') for filename in
#                 glob.glob(data_dir + '/*')]

#     else:
#         datas = collect_data_files('osgeo')

datas = collect_data_files('osgeo')
