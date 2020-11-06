#!/usr/bin/env bash

VERSION=$1
: "${VERSION:?'The version string is needed as parameter 1.'}"

# Exit the script immediately if any subshell has a nonzero exit code.
set -e

# Members of the natcap software team can push to the autorelease branch on
# natcap/invest; this branch is a special case for our release process.
AUTORELEASE_BRANCH=autorelease/$VERSION
git checkout -b "$AUTORELEASE_BRANCH"
git add Makefile HISTORY.rst
git commit -m "Committing the $VERSION release."
git tag "$VERSION"

echo ""
echo "The release has been committed and tagged."
echo "To push to natcap/invest:"
echo "  $ git push git@github.com:natcap/invest.git $VERSION $AUTORELEASE_BRANCH"

# Figure out the origin repo's username for nicer push messages.
ORIGIN_REPO=$(git config --get remote.origin.url | sed 's|\.git$||g' | grep -E -o '[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$')
if [ "$ORIGIN_REPO" != "natcap/invest" ]
then
    echo ""
    echo "To push to your fork via SSH:"
    echo "  $ git push git@github.com:$ORIGIN_REPO.git $VERSION $AUTORELEASE_BRANCH"
    echo ""
    echo "To push to your fork via HTTPS:
    echo "  $ git push https://github.com/$ORIGIN_REPO.git $VERSION $AUTORELEASE_BRANCH"
fi

echo ""
echo "After pushing, wait for builds to finish before continuing."
echo "    See https://github.com/natcap/invest/wiki/Bugfix-Release-Checklist#wait-for-builds-to-complete"
