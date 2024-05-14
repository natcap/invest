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

# Since OSX Mavericks, the stdlib has been renamed.  So if we're on OSX, we
# need to be sure to define which standard c++ library to use.  I don't have
# access to a pre-Mavericks mac, so hopefully this won't break on someone's
# older system.  Tested and it works on Mac OSX Catalina.
compiler_and_linker_args = []
if platform.system() == 'Darwin':
    compiler_and_linker_args = ['-stdlib=libc++']


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
            extra_compile_args=compiler_args + compiler_and_linker_args,
            extra_link_args=compiler_and_linker_args,
            language='c++',
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
        ) for package, module, compiler_args in [
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
