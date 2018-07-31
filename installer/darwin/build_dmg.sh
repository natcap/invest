#!/bin/bash -ve
#
# This script is assumed to be executed from the project root.
#
# Arguments:
#  $1 = the version string to use
#  $2 = the path to the binary dir to package.
#  $3 = the path to the directory of html documentation to include.
#
# Script adapted from http://stackoverflow.com/a/1513578/299084

CONFIG_DIR="installer/darwin"
appdirname="InVEST_${1}_unstable"  # the name of the folder the user will drag from the DMG to their applications folder.
title="InVEST ${1}"  # the name of the volume the DMG provides.
finalDMGName="dist/InVEST ${1}.dmg"  # the name of the final DMG file.

# copy the docs into the dmg
docsdir="$3"
if [ -d $docsdir ]
then
    cp -r $docsdir $tempdir/documentation
fi

# Copy the release notes (HISTORY.rst) into the dmg as an HTML doc.
pandoc HISTORY.rst -o $tempdir/HISTORY.html

dmgbuild -Dinvestdir="$tempdir" -s $CONFIG_DIR/dmgconf.py "$title" "$finalDMGName"
    
