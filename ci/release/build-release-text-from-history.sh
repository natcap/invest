#!/bin/bash
#
# This script prepares the body of text for a github release object to be
# created through the `hub` command-line utility.
#
# This requires the GITHUB_REF environment variable to be declared,
# which must be in the git refs format of a tag (examples: refs/tags/3.8.0)

: "${GITHUB_REF:?Need to set GITHUB_REF to a git ref string.  Example: refs/tags/3.8.0}"

# Format the HISTORY of this release for the release.
# This file represents both the title and the body of the
# release.  A blank line separates the title from the body.
RELEASE_MESSAGE_RST_FILE=release_message.rst
echo "${GITHUB_REF:10}" >> $RELEASE_MESSAGE_RST_FILE
echo "" >> $RELEASE_MESSAGE_RST_FILE

# Read HISTORY from the released tag up until the first
# blank line
sed -n "/${GITHUB_REF:10}/,/^$/p" HISTORY.rst | tail -n +3 >> $RELEASE_MESSAGE_RST_FILE

pandoc \
    --from rst \
    --to markdown \
    $RELEASE_MESSAGE_RST_FILE
