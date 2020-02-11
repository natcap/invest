#!/bin/bash -xe

# A series of Make commands to build InVEST binaries and
# installers for MacOSX.

# System Pre-requisites (all should be on the PATH):
# make
# python3 (with Cython)
# conda
# pandoc

# It might be a good idea to `rm -rf` any existing python environment
# because Make doesn't always re-make an env if there were updates to 
# requirements files. But I'm afraid to add that `rm` to this script.

# PyInstaller's exe/invest.spec expects the python environment
# to be named 'env' so it can find binary files that need moving
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

# This script never makes the sample data, which is fine since we don't 
# distribute sample data with the mac installer. 
# But it means we will see this message on make deploy (and its okay to ignore):
# CommandException: arg (dist/data) does not name a directory, bucket, or bucket subdir.
make deploy