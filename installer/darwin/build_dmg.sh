#!/bin/bash -ve
#
# This script is assumed to be executed from the project root.
#
# Arguments:
#  $1 = the version string to use
#  $2 = the path to the binary dir to package.
#  $3 = the path to the directory of html documentation to include.
#  $4 = the path to the final DMG to create

VERSION=$1
BINARY_PATH=$2
DOCS_PATH=$3
DMG_PATH=$4
DMG_CONFIG_PATH=$5

# copy the docs into the dmg
if [ -d $DOCS_PATH ]
then
    cp -r $DOCS_PATH $BINARY_PATH/documentation
fi

# Copy the release notes (HISTORY.rst) into the dmg as an HTML doc.
pandoc HISTORY.rst -o BINARY_PATH/HISTORY.html

dmgbuild -Dinvestdir=$BINARY_PATH -s $DMG_CONFIG_PATH "InVEST ${VERSION}" $DMG_PATH
