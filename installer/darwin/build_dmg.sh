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
invest_bindir="$tempdir"/`basename $2`
cp -r "$2" "$invest_bindir"

# copy out all the invest shell files and fixup the paths.
# .command extension makes the scripts runnable by the user.
# Shell files without the `invest_` prefix will be left alone.
find "$invest_bindir" -iname "invest_*.sh" | while read sh_file
do
    new_name=`echo "$sh_file" | sed 's/\.sh/.command/g'`
    new_command_file="$tempdir"/`basename "$new_name"`
    mv "$sh_file" "$new_command_file"
    sed -i '' 's/.\/invest/`dirname $0`\/invest_dist\/invest/g' "$new_command_file"
done

chmod u+x "$tempdir"/*.command

source=temp

dir_size_k=`du -k -d 0 $source | awk -F ' ' '{print $1}'`
new_disk_size=`python -c "print $dir_size_k + 1024*10"`
tempdmgname=pack.temp.dmg
hdiutil create -srcfolder "${source}" -volname "${title}" -fs HFS+ \
    -fsargs "-c c=64,a=16,e=16" -format UDRW -size ${new_disk_size}k $tempdmgname

device=$(hdiutil attach -readwrite -noverify -noautoopen "$tempdmgname" | \
    egrep '^/dev/' | sed 1q | awk '{print $1}')
ls -la /Volumes

# UNCOMMENT THESE LINES TO CREATE A BACKGROUND IMAGE
# ALSO, BE SURE TO INCLUDE A SNAZZY BACKGROUND IMAGE.  DO IT RIGHT IF YOU DO IT AT ALL.
# mkdir /Volumes/"${title}"/.background
# cp background.png /Volumes/"${title}"/.background/background.png
# backgroundPictureName='background.png'

applicationName="`basename ${2}`"

echo '
   tell application "Finder"
     tell disk "'${title}'"
           open
           set current view of container window to icon view
           set toolbar visible of container window to false
           set statusbar visible of container window to false
           set the bounds of container window to {400, 100, 885, 430}
           set theViewOptions to the icon view options of container window
           set arrangement of theViewOptions to not arranged
           set icon size of theViewOptions to 72
           set background picture of theViewOptions to file ".background:'${backgroundPictureName}'"
           make new alias file at container window to POSIX file "/Applications" with properties {name:"Applications"}
           set position of item "'${applicationName}'" of container window to {100, 100}
           set position of item "Applications" of container window to {375, 100}
           update without registering applications
           delay 5
           close
     end tell
   end tell
' | osascript

chmod -Rf go-w /Volumes/"${title}"
sync
sync
hdiutil detach ${device}
hdiutil convert "${tempdmgname}" -format UDZO -imagekey zlib-level=9 -o "${finalDMGName}"
rm -f /$tempdmgname
