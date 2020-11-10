#!/bin/bash -ve
#
# This script is assumed to be executed from the project root.
#
# Arguments:
#  $1 = the version string to use
#  $2 = the path to the binary dir to package
#  $3 = the path to the directory of html documentation to include
#  $4 = the path to a python file setting the dmgbuild configuration
#  $5 = the path to the final DMG to create

VERSION=$1
BINARY_PATH=$2
DOCS_PATH=$3
DMG_CONFIG_PATH=$4
DMG_PATH=$5

# create a temp dir to pass to dmgbuild
# this will hold the app bundle, docs, and history
TEMPDIR="tmp_dmg_dir"
mkdir TEMPDIR

# copy in the app bundle
cp -r $BINARY_PATH $TEMPDIR

# copy in the html docs
if [ -d $DOCS_PATH ]
then
    cp -r $DOCS_PATH $TEMPDIR/documentation
fi

# Copy in the release notes (HISTORY.rst) as an HTML doc.
pandoc HISTORY.rst -o $TEMPDIR/HISTORY.html

dmgbuild -Dinvestdir=$TEMPDIR -s $DMG_CONFIG_PATH "InVEST ${VERSION}" $DMG_PATH
