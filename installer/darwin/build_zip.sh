#!/bin/bash -ve
#
# This script is assumed to be executed from the project root.
#
# Arguments:
#  $1 = the version string to use
#  $2 = the path to the application bundle to include.
#  $3 = the path to the directory of html documentation to include.

# This script writes the bundle to "build/app_bundle/InVEST-${1}"

zipdirname="InVEST-${1}"
zipfilename="$zipdirname-mac.zip"
tempdir="build/mac_zip/$zipdirname"
if [ -d "$tempdir" ]
then
    # Remove the temp dir if it already exists.
    rm -rfd "$tempdir"
fi
mkdir -p "$tempdir"

# Copy the binaries and the html docs into the tempdir
cp -r "$2" "$tempdir/InVEST.app"
cp -r "$3" "$tempdir/documentation"

# copy the release notes (HISTORY.rst) into the directory to be zipped as an HTML doc.
pandoc HISTORY.rst -o "$tempdir/HISTORY.html"

pushd "build/mac_zip"
zip -r "../../dist/$zipfilename" "$zipdirname"
popd
