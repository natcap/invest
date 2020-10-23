#!/bin/bash
#
# This script prepares the body of text for a github release object to be
# created through the `hub` command-line utility.
#
# This requires the $VERSION environment variable to be declared,
# which must be in the format of an InVEST release version.
#
# This script also requires the pandoc executable to be on the PATH.
VERSION="${1:-$VERSION}"  # Take from arg 1 first, or else from $VERSION.
: "${VERSION:?Need to set VERSION.  Example: VERSION=3.8.0}"

# Format the HISTORY of this release for the release.
# This file represents both the title and the body of the
# release.  A blank line separates the title from the body.
RELEASE_MESSAGE_RST_FILE=release_message.rst
rm -f $RELEASE_MESSAGE_RST_FILE  # remove the file if it exists
echo "$VERSION" >> "$RELEASE_MESSAGE_RST_FILE"
echo "" >> "$RELEASE_MESSAGE_RST_FILE"
echo "This bugfix release includes the following fixes and features:" >> "$RELEASE_MESSAGE_RST_FILE"
echo "" >> "$RELEASE_MESSAGE_RST_FILE"  # extra line to clarify we're starting a bulleted list.

# Read HISTORY from the released tag up until the first
# blank line.
# The tail +3 cuts off the version string and underline of the title.
sed -n "/$VERSION/,/^$/p" HISTORY.rst | tail -n +3 >> $RELEASE_MESSAGE_RST_FILE

# markdown_strict is needed for proper formatting in github issue text.
pandoc \
    --from rst \
    --to markdown_strict \
    $RELEASE_MESSAGE_RST_FILE
