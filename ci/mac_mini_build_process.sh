#!/bin/bash

# A series of Make commands to build InVEST binaries and
# installers for MacOSX.

# System Pre-requisites (all should be on the PATH):
# make
# python3
# conda
# pandoc

# PyInstaller's exe/invest.spec expects the python environment
# to be named 'env' so it can find and binary files that need moving
PYTHON_ENV=env
make ENV=$PYTHON_ENV env

# calling these targets directly in order to override the PYTHON var in Makefile
make PYTHON=$PYTHON_ENV/bin/python python_packages
make PYTHON=$PYTHON_ENV/bin/python binaries

# activate environment here so that userguide recipe finds sphinx-build
source activate ./$PYTHON_ENV

# overriding this make variable just so it includes the trailing slash,
# which is key for cp and rsync commands in this bash shell
make INVEST_BINARIES_DIR=dist/invest/ mac_zipfile
make deploy