#!/bin/bash -ve
#
# This script is assumed to be executed from the project root.
#
# Arguments:
#  $1 = the version string to use
#  $2 = the path to the binary dir to package.
#  $3 = the path to the HTML documentation
#  $4 = the path to where the application bundle should be written.

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
new_basename='InVEST'
_APPDIR="$4"
_MACOSDIR="$_APPDIR/Contents/MacOS"
_RESOURCEDIR="$_APPDIR/Contents/Resources"
_INVEST_DIST_DIR="$_MACOSDIR/invest_dist"
_USERGUIDE_HTML_DIR="$_INVEST_DIST_DIR/documentation"
CONFIG_DIR="installer/darwin"
mkdir -p "${_MACOSDIR}"
mkdir -p "${_RESOURCEDIR}"

cp -r "$2" "$_INVEST_DIST_DIR"

mkdir -p "${_USERGUIDE_HTML_DIR}"
cp -r "$3" "$_USERGUIDE_HTML_DIR"
new_command_file="$_MACOSDIR/$new_basename"
cp $CONFIG_DIR/invest.icns "$_RESOURCEDIR/invest.icns"

new_plist_file="$_APPDIR/Contents/Info.plist"
cp $CONFIG_DIR/Info.plist "$new_plist_file"

# replace the version and application name strings in the Info.plist file
sed -i '' "s|++NAME++|$new_basename|g" "$new_plist_file"
sed -i '' "s|++VERSION++|${1}|g" "$new_plist_file"

# This is the command that will launch the application.
echo '#!/bin/bash' > $new_command_file
echo '#' >> $new_command_file
echo '# the QT_MAC_WANTS_LAYER definition is supposed to have been set by the' >> $new_command_file
echo "# runtime hook, but doesn't seem to be working.  Setting it here allows the" >> $new_command_file
echo "# binary to run on OSX Big Sur." >> $new_command_file
echo "#" >> $new_command_file
echo "# Taken from https://stackoverflow.com/a/246128/299084" >> $new_command_file
echo 'DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"' >> $new_command_file
echo 'QT_MAC_WANTS_LAYER=1 "$DIR/invest_dist/invest" launch' >> $new_command_file
chmod a+x $new_command_file

