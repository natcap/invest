#!/usr/bin/env bash

# Create a read-only disk image of the contents of a folder
# This script is derived from the create-dmg utility
# https://github.com/create-dmg/create-dmg

# Bail out on any unhandled errors
set -ex;

VOLUME_NAME=$1
APP_BUNDLE_PATH=$2
DMG_PATH=$3

VOLUME_ICON_FILE=installer/darwin/invest.icns
BACKGROUND_FILE=installer/darwin/background.png
EULA=LICENSE.txt

SRC_FOLDER=$(cd $APP_BUNDLE_PATH > /dev/null; pwd)
DMG_DIRNAME=$(dirname $DMG_PATH)
DMG_DIR=$(cd $DMG_DIRNAME > /dev/null; pwd)
DMG_NAME=$(basename $DMG_PATH)
DMG_TEMP_NAME=$DMG_DIR/tmp.$DMG_NAME

# remove .DS_Store from source folder, if there is one
rm $SRC_FOLDER/.DS_Store || true

# remove intermediate dmg from previous run, if there is one
rm -f $DMG_TEMP_NAME || true

# Use Megabytes since hdiutil fails with very large byte numbers
function blocks_to_megabytes() {
    # Add 1 extra MB, since there's no decimal retention here
    MB_SIZE=$((($1 * 512 / 1000 / 1000) + 1))
    echo $MB_SIZE
}

function get_size() {
    # Get block size in disk
    bytes_size=$(du -s "$1" | sed -e 's/    .*//g')
    echo $(blocks_to_megabytes $bytes_size)
}

# Create the DMG with the hdiutil estimation
hdiutil create \
    -srcfolder $SRC_FOLDER \
    -volname "$VOLUME_NAME" \
    -fs HFS+ `# type of file system to write to the image`\
    -fsargs "-c c=64,a=16,e=16" \
    -format UDRW `# UDIF read/write image, so we can add the license to it`\
    $DMG_TEMP_NAME

# Get the created DMG actual size
DISK_IMAGE_SIZE=$(get_size $DMG_TEMP_NAME)

# Add extra space for additional resources
DISK_IMAGE_SIZE=$(expr $DISK_IMAGE_SIZE + 20)

# Make sure target image size is within limits
MIN_DISK_IMAGE_SIZE=$(hdiutil resize -limits $DMG_TEMP_NAME | awk 'NR=1{print int($1/2048+1)}')
if [ $MIN_DISK_IMAGE_SIZE -gt $DISK_IMAGE_SIZE ]; then
    DISK_IMAGE_SIZE=$MIN_DISK_IMAGE_SIZE
fi

# Resize the image for the extra stuff
hdiutil resize -size ${DISK_IMAGE_SIZE}m $DMG_TEMP_NAME

# Mount the new DMG
MOUNT_DIR="/Volumes/$VOLUME_NAME"

# Unmount leftover dmg if it was mounted previously (e.g. developer mounted dmg, installed app and forgot to unmount it)
if [[ -d $MOUNT_DIR ]]; then
    DEV_NAME=$(hdiutil info | grep -E --color=never '^/dev/' | sed 1q | awk '{print $1}')
    hdiutil detach $DEV_NAME
fi

DEV_NAME=$(hdiutil attach -readwrite -noverify -noautoopen $DMG_TEMP_NAME | grep -E --color=never '^/dev/' | sed 1q | awk '{print $1}')

# Copy background image file into volume
mkdir "$MOUNT_DIR/.background"
cp $BACKGROUND_FILE "$MOUNT_DIR/.background/$(basename $BACKGROUND_FILE)"

# Link Applications into volume
ln -s /Applications "$MOUNT_DIR/Applications"

# Copy icon into volume
cp $VOLUME_ICON_FILE "$MOUNT_DIR/.VolumeIcon.icns"
SetFile -c icnC "$MOUNT_DIR/.VolumeIcon.icns"

sleep 2 # pause to workaround occasional "Can’t get disk" (-1728) issues  
# Run AppleScript to do all the Finder cosmetic stuff
/usr/bin/osascript installer/darwin/customize_dmg.applescript "$VOLUME_NAME"
sleep 4

# Make sure it's not world writeable
chmod -Rf go-w "$MOUNT_DIR" &> /dev/null || true

# Tell the volume that it has a special file attribute
SetFile -a C "$MOUNT_DIR"

# Delete unnecessary file system events log if possible
rm -rf "$MOUNT_DIR/.fseventsd" || true

# Unmount
unmounting_attempts=0
until
  echo "Unmounting disk image..."
  (( unmounting_attempts++ ))
  hdiutil detach $DEV_NAME
    exit_code=$?
    (( exit_code ==  0 )) && break            # nothing goes wrong
    (( exit_code != 16 )) && exit $exit_code  # exit with the original exit code
    # The above statement returns 1 if test failed (exit_code == 16).
    #   It can make the code in the {do... done} block to be executed
do
  (( unmounting_attempts == 3 )) && exit 16  # patience exhausted, exit with code EBUSY
    echo "Wait a moment..."
  sleep $(( 1 * (2 ** unmounting_attempts) ))
done
unset unmounting_attempts

# Compress image
hdiutil convert $DMG_TEMP_NAME \
    -format UDZO `# UDIF zlib-compressed image`\
    -imagekey zlib-level=9 `# zlib compression level`\
    -o $DMG_DIR/$DMG_NAME

rm -f $DMG_TEMP_NAME

# Add EULA resources
EULA_RESOURCES_FILE=$(mktemp -t createdmg.tmp.XXXXXXXXXX)
# Encode the EULA to base64
EULA_DATA=$(base64 -b 52 $EULA | sed s$'/^\(.*\)$/\t\t\t\\1/')
# Fill the template with the custom EULA contents
eval "cat > \"${EULA_RESOURCES_FILE}\" <<EOF$(<installer/darwin/eula_resources_template.xml)EOF"
# Add the license
hdiutil udifrez -xml $EULA_RESOURCES_FILE '' $DMG_DIR/$DMG_NAME
rm EULA_RESOURCES_FILE
