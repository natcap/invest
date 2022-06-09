# encoding=UTF-8
"""setup.py module for natcap.invest.

InVEST - Integrated Valuation of Ecosystem Services and Tradeoffs

Common functionality provided by setup.py:
    build_sphinx

For other commands, try `python setup.py --help-commands`
"""
import os
import platform
import subprocess

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
README = open('README_PYTHON.rst').read().format(
    requirements='\n'.join(['    ' + r for r in _REQUIREMENTS]))

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
    name='natcap.invest',
    description="InVEST Ecosystem Service models",
    long_description=README,
    maintainer='James Douglass',
    maintainer_email='jdouglass@stanford.edu',
    url='http://github.com/natcap/invest',
    namespace_packages=['natcap'],
    packages=[
        'natcap',
        'natcap.invest',
        'natcap.invest.coastal_blue_carbon',
        'natcap.invest.delineateit',
        'natcap.invest.ui',
        'natcap.invest.ndr',
        'natcap.invest.sdr',
        'natcap.invest.recreation',
        'natcap.invest.scenic_quality',
        'natcap.invest.seasonal_water_yield',
    ],
    package_dir={
        'natcap': 'src/natcap'
    },
    include_package_data=True,
    install_requires=_REQUIREMENTS,
    python_requires='>=3.8,<3.11',
    license='BSD',
    long_description_content_type='text/x-rst',
    zip_safe=False,
    keywords='gis invest',
    classifiers=[
        'Intended Audience :: Developers',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Cython',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering :: GIS'
    ],
    ext_modules=[
        Extension(
            name="natcap.invest.delineateit.delineateit_core",
            sources=['src/natcap/invest/delineateit/delineateit_core.pyx'],
            include_dirs=[numpy.get_include()],
            extra_compile_args=compiler_and_linker_args,
            extra_link_args=compiler_and_linker_args,
            language="c++"),
        Extension(
            name="natcap.invest.recreation.out_of_core_quadtree",
            sources=[
                'src/natcap/invest/recreation/out_of_core_quadtree.pyx'],
            include_dirs=[numpy.get_include()],
            extra_compile_args=compiler_and_linker_args,
            extra_link_args=compiler_and_linker_args,
            language="c++"),
        Extension(
            name="natcap.invest.scenic_quality.viewshed",
            sources=[
                'src/natcap/invest/scenic_quality/viewshed.pyx'],
            include_dirs=[numpy.get_include(),
                          'src/natcap/invest/scenic_quality'],
            extra_compile_args=compiler_and_linker_args,
            extra_link_args=compiler_and_linker_args,
            language="c++"),
        Extension(
            name="natcap.invest.ndr.ndr_core",
            sources=['src/natcap/invest/ndr/ndr_core.pyx'],
            include_dirs=[numpy.get_include()],
            extra_compile_args=compiler_and_linker_args,
            extra_link_args=compiler_and_linker_args,
            language="c++"),
        Extension(
            name="natcap.invest.sdr.sdr_core",
            sources=['src/natcap/invest/sdr/sdr_core.pyx'],
            include_dirs=[numpy.get_include()],
            extra_compile_args=compiler_and_linker_args,
            extra_link_args=compiler_and_linker_args,
            language="c++"),
        Extension(
            name=("natcap.invest.seasonal_water_yield."
                  "seasonal_water_yield_core"),
            sources=[
                ("src/natcap/invest/seasonal_water_yield/"
                 "seasonal_water_yield_core.pyx")],
            include_dirs=[numpy.get_include()],
            extra_compile_args=compiler_and_linker_args,
            extra_link_args=compiler_and_linker_args,
            language="c++"),
    ],
    cmdclass={
        'build_ext': Cython.Build.build_ext,
        'build_py': build_py},
    entry_points={
        'console_scripts': [
            'invest = natcap.invest.cli:main'
        ],
    },
    extras_require={
        'ui': _GUI_REQUIREMENTS,
    },
    package_data={
        'natcap.invest': [
            'internationalization/locales/en/.gitignore'
        ]
    }
)
