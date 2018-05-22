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
# TODO: unmount any existing disk images with the same name.

CONFIG_DIR="installer/darwin"
appdirname="InVEST_${1}_unstable"  # the name of the folder the user will drag from the DMG to their applications folder.
title="InVEST ${1}"  # the name of the volume the DMG provides.
finalDMGName="dist/InVEST ${1}.dmg"  # the name of the final DMG file.

# remove temp files that can get in the way
tempdir="build/dmg/$appdirname"
if [ -d "$tempdir" ]
then
    rm -rfd "$tempdir"
fi

# prepare a local temp dir for a filesystem
mkdir -p "$tempdir"

# copy out all the invest shell files and fixup the paths.
# .command extension makes the scripts runnable by the user.
# Shell files without the `invest_` prefix will be left alone.
new_basename='InVEST'
_APPDIR="$tempdir/$new_basename.app"
_MACOSDIR="$_APPDIR/Contents/MacOS"
_RESOURCEDIR="$_APPDIR/Contents/Resources"
mkdir -p "${_MACOSDIR}"
mkdir -p "${_RESOURCEDIR}"
cp -r "$2" "$_MACOSDIR/invest_dist"
new_command_file="$_MACOSDIR/$new_basename"
cp $CONFIG_DIR/invest.icns "$_RESOURCEDIR/invest.icns"

new_plist_file="$_APPDIR/Contents/Info.plist"
cp $CONFIG_DIR/Info.plist "$new_plist_file"

# replace the version and application name strings in the Info.plist file
sed -i '' "s|++NAME++|$new_basename|g" "$new_plist_file"
sed -i '' "s|++VERSION++|${1}|g" "$new_plist_file"

# This is the command that will launch the application.
echo '#!/bin/bash' > $new_command_file
echo '`dirname $0`/invest_dist/invest launcher' >> $new_command_file
chmod a+x $new_command_file

# copy the docs into the dmg
docsdir="$3"
if [ -d $docsdir ]
then
    cp -r $docsdir $tempdir/documentation
fi

# Copy the release notes (HISTORY.rst) into the dmg as an HTML doc.
pandoc HISTORY.rst -o $tempdir/HISTORY.html

dmgbuild -Dinvestdir="$tempdir" -s $CONFIG_DIR/dmgconf.py "$title" "$finalDMGName"
    
