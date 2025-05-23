import os
import platform
import subprocess

import numpy
from Cython.Build import cythonize
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.extension import Extension

# Read in requirements.txt and populate the python readme with the
# non-comment, non-environment-specifier contents.
_REQUIREMENTS = [req.split(';')[0].split('#')[0].strip() for req in
                 open('requirements.txt').readlines()
                 if (not req.startswith(('#', 'hg+', 'git+'))
                     and len(req.strip()) > 0)]

include_dirs = [numpy.get_include(), 'src/natcap/invest/managed_raster']
if platform.system() == 'Windows':
    compiler_args = ['/std:c++20']
    compiler_and_linker_args = []
    if 'NATCAP_INVEST_GDAL_LIB_PATH' not in os.environ:
        raise RuntimeError(
            'env variable NATCAP_INVEST_GDAL_LIB_PATH is not defined. '
            'This env variable is required when building on Windows. If '
            'using conda to manage your gdal installation, you may set '
            'NATCAP_INVEST_GDAL_LIB_PATH=%CONDA_PREFIX%/Library".')
    library_dirs = [os.path.join(
        os.environ["NATCAP_INVEST_GDAL_LIB_PATH"].rstrip(), "lib")]
    include_dirs.append(os.path.join(
        os.environ["NATCAP_INVEST_GDAL_LIB_PATH"].rstrip(), "include"))
else:
    compiler_args = [subprocess.run(
        ['gdal-config', '--cflags'], capture_output=True, text=True
    ).stdout.strip()]
    compiler_and_linker_args = ['-std=c++20']
    library_dirs = [subprocess.run(
        ['gdal-config', '--libs'], capture_output=True, text=True
    ).stdout.split()[0][2:]] # get the first argument which is the library path


class build_py(_build_py):
    """Command to compile translation message catalogs before building."""

    def run(self):
        # internationalization: compile human-readable PO message catalogs
        # into machine-readable MO message catalogs used by gettext
        # the MO files are included as package data
        locale_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            'src/natcap/invest/internationalization/locales'))
        for locale in os.listdir(locale_dir):
            subprocess.run([
                'pybabel',
                'compile',
                '--input-file', f'{locale_dir}/{locale}/LC_MESSAGES/messages.po',
                '--output-file', f'{locale_dir}/{locale}/LC_MESSAGES/messages.mo'])
        # then execute the original run method
        _build_py.run(self)


setup(
    install_requires=_REQUIREMENTS,
    ext_modules=cythonize([
        Extension(
            name=f'natcap.invest.{package}.{module}',
            sources=[f'src/natcap/invest/{package}/{module}.pyx'],
            include_dirs=include_dirs,
            extra_compile_args=compiler_args + package_compiler_args + compiler_and_linker_args,
            extra_link_args=compiler_and_linker_args,
            language='c++',
            libraries=['gdal'],
            library_dirs=library_dirs,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
        ) for package, module, package_compiler_args in [
            ('delineateit', 'delineateit_core', []),
            ('recreation', 'out_of_core_quadtree', []),
            # clang-14 defaults to -ffp-contract=on, which causes the
            # arithmetic of A*B+C to be implemented using a contraction, which
            # causes an unexpected change in the precision in some viewshed
            # tests on ARM64 (mac M1).  See these issues for more details:
            #  * https://github.com/llvm/llvm-project/issues/91824
            #  * https://github.com/natcap/invest/issues/1562
            #  * https://github.com/natcap/invest/pull/1564/files
            # Using this flag on gcc and on all versions of clang should work
            # as expected, with consistent results.
            ('scenic_quality', 'viewshed', ['-ffp-contract=off']),
            ('ndr', 'ndr_core', []),
            ('sdr', 'sdr_core', []),
            ('seasonal_water_yield', 'seasonal_water_yield_core', [])
        ]
    ], compiler_directives={'language_level': '3'}),
    include_dirs=[numpy.get_include()],
    cmdclass={
        'build_py': build_py
    }
)
