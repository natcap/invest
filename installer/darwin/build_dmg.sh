#!/bin/bash
#
# Arguments:
#  $1 = the version string to use
#  $2 = the path to the binary dir to package.
#
# Script adapted from http://stackoverflow.com/a/1513578/299084
# TODO: unmount any existing disk images with the same name.

appdirname="InVEST_${1}_unstable"  # the name of the folder the user will drag from the DMG to their applications folder.
title="InVEST ${1}"  # the name of the volume the DMG provides.
finalDMGName="InVEST ${1}"  # the name of the final DMG file.

# remove temp files that can get in the way
tempdir=temp/"$appdirname"
rm *.dmg
if [ -d "$tempdir" ]
then
    rm -rfd "$tempdir"
fi

# prepare a local temp dir for a filesystem
mkdir -p "$tempdir"
invest_bindir="$tempdir"/.`basename $2`  # preceeding . hides the folder from users
cp -r "$2" "$invest_bindir"

# copy out all the invest shell files and fixup the paths.
# .command extension makes the scripts runnable by the user.
# Shell files without the `invest_` prefix will be left alone.
find "$invest_bindir" -iname "invest_*.sh" | while read sh_file
do
    new_name=`echo "$sh_file" | sed 's/\.sh$//g'`
    new_basename=`basename $new_name`
    _APPDIR="$tempdir/$new_basename.app"
    _MACOSDIR="$_APPDIR/Contents/MacOS"
    _RESOURCEDIR="$_APPDIR/Contents/Resources"
    mkdir -p "${_MACOSDIR}"
    mkdir -p "${_RESOURCEDIR}"
    new_command_file="$_MACOSDIR/$new_basename"
    mv "$sh_file" "$new_command_file"
    cp invest.icns "$_RESOURCEDIR/invest.icns"

    new_plist_file="$_APPDIR/Contents/Info.plist"
    cp Info.plist "$new_plist_file"

    # replace the version and application name strings in the Info.plist file
    sed -i '' "s|++NAME++|$new_basename|g" "$new_plist_file"
    sed -i '' "s|++VERSION++|${1}|g" "$new_plist_file"

    # Three directories up from `dirname $0` is the InVEST installation folder,
    # which contains invest_dist.
    sed -i '' 's|./invest|`dirname $0`/../../../.invest_dist/invest|g' "$new_command_file"
done

# copy the docs into the dmg
cp -r ../../doc/users-guide/build/html $tempdir/documentation

dmgbuild -Dinvestdir="$tempdir" -s dmgconf.py "$title" "$finalDMGName"
    
