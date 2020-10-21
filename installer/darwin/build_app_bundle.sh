#!/bin/bash -ve
#
# This script is assumed to be executed from the project root.
#
# Arguments:
#  $1 = the version string to use
#  $2 = the path to the binary dir to package.
#  $3 = the path to where the application bundle should be written.

# remove temp files that can get in the way
tempdir=`basename $3`
if [ -d "$tempdir" ]
then
    rm -rfd "$tempdir"
fi

# prepare a local temp dir for a filesystem
mkdir -p "$tempdir"

# copy out all the invest shell files and fixup the paths.
# .command extension makes the scripts runnable by the user.
# Shell files without the `invest_` prefix will be left alone.


# Correct macOS app bundle structure 
# https://developer.apple.com/library/archive/technotes/tn2206/_index.html#//apple_ref/doc/uid/DTS40007919-CH1-TNTAG201
# see https://developer.apple.com/forums/thread/114943?answerId=353975022#353975022
#
# Only Mach-O format files should be signed as code because they have the LC_CODE_SIGNATURE attribute.
# Trying to sign other formats as code results in the signature being stored in an extended attribute.
# Extended attributes can be lost when moving files around so it's not recommended.
# Everything in Resources/ is still signed, but as a resource under the seal of the parent bundle. 
#
#
# InVEST.app/
#   Contents/
#       Frameworks/
#           all .dylibs
#       MacOS/
#           InVEST unix executable
#       Resources/
#           everything else!
#       Info.plist

new_basename='InVEST'
_APPDIR="$3"
_MACOSDIR="$_APPDIR/Contents/MacOS"
_RESOURCEDIR="$_APPDIR/Contents/Resources"
_FRAMEWORKSDIR="$_APPDIR/Contents/Frameworks"
CONFIG_DIR="installer/darwin"
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
echo '`dirname $0`/invest_dist/invest launch' >> $new_command_file
chmod a+x $new_command_file

