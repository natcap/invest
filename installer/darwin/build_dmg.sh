#!/bin/bash
#
# Arguments:
#  $1 = the version string to use 
#  $2 = the path to the binary dir to package.
#
# Script adapted from http://stackoverflow.com/a/1513578/299084
# TODO: unmount any existing disk images with the same name.

title="InVEST ${1}"
finalDMGName="InVEST ${1}"

rm *.dmg

# prepare a local temp dir for a filesystem
mkdir temp
cp -r $2 temp
source=temp

size=40000  # ~40 MB
hdiutil create -srcfolder "${source}" -volname "${title}" -fs HFS+ \
    -fsargs "-c c=64,a=16,e=16" -format UDRW -size ${size}k pack.temp.dmg

device=$(hdiutil attach -readwrite -noverify -noautoopen "pack.temp.dmg" | \
    egrep '^/dev/' | sed 1q | awk '{print $1}')
ls -la /Volumes

# UNCOMMENT THESE LINES TO CREATE A BACKGROUND IMAGE
# ALSO, BE SURE TO INCLUDE A SNAZZY BACKGROUND IMAGE.  DO IT RIGHT IF YOU DO IT AT ALL.
mkdir /Volumes/"${title}"/.background
cp background.png /Volumes/"${title}"/.background/background.png
backgroundPictureName='background.png'

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
hdiutil convert "pack.temp.dmg" -format UDZO -imagekey zlib-level=9 -o "${finalDMGName}"
rm -f /pack.temp.dmg
