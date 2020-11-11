#!/bin/bash -ve
#
# This script is assumed to be executed from the project root.
#
# Arguments:
#  $1 = the version string to use
#  $2 = the path to the binary dir to package
#  $3 = the path to a python file setting the dmgbuild configuration
#  $4 = the path to the final DMG to create

VERSION=$1
BINARY_PATH=$2
DMG_CONFIG_PATH=$3
DMG_PATH=$4

dmgbuild -Dinvestdir=$BINARY_PATH -s $DMG_CONFIG_PATH "InVEST ${VERSION}" $DMG_PATH
