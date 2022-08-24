import platform

import Cython.Build
import numpy
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.extension import Extension

# Read in requirements.txt and populate the python readme with the
# non-comment, non-environment-specifier contents.
_REQUIREMENTS = [req.split(';')[0].split('#')[0].strip() for req in
                 open('requirements.txt').readlines()
                 if (not req.startswith(('#', 'hg+', 'git+'))
                     and len(req.strip()) > 0)]
_GUI_REQUIREMENTS = [req.split(';')[0].split('#')[0].strip() for req in
                     open('requirements-gui.txt').readlines()
                     if not (req.startswith(('#', 'hg+'))
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
        # NOTE: un-comment this when we get message catalogs.
        #
        # internationalization: compile human-readable PO message catalogs
        # into machine-readable MO message catalogs used by gettext
        # the MO files are included as package data
        # locale_dir = os.path.abspath(os.path.join(
        #     os.path.dirname(__file__),
        #     'src/natcap/invest/internationalization/locales'))
        # for locale in os.listdir(locale_dir):
        #     subprocess.run([
        #         'pybabel',
        #         'compile',
        #         '--input-file', f'{locale_dir}/{locale}/LC_MESSAGES/messages.po',
        #         '--output-file', f'{locale_dir}/{locale}/LC_MESSAGES/messages.mo'])
        # then execute the original run method
        _build_py.run(self)


setup(
    install_requires=_REQUIREMENTS,
    extras_require={
        'ui': _GUI_REQUIREMENTS,
    },
    ext_modules=[
        Extension(
            name=f'natcap.invest.{package}.{module}',
            sources=[f'src/natcap/invest/{package}/{module}.pyx'],
            include_dirs=[numpy.get_include()],
            extra_compile_args=compiler_and_linker_args,
            extra_link_args=compiler_and_linker_args,
            language='c++'
        ) for package, module in [
            ('delineateit', 'delineateit_core'),
            ('recreation', 'out_of_core_quadtree'),
            ('scenic_quality', 'viewshed'),
            ('ndr', 'ndr_core'),
            ('sdr', 'sdr_core'),
            ('seasonal_water_yield', 'seasonal_water_yield_core')
        ]
    ],
    cmdclass={
        'build_ext': Cython.Build.build_ext,
        'build_py': build_py
    }
)
