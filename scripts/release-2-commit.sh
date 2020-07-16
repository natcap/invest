#!/usr/bin/env bash

: "${VERSION:?'The version string is needed as parameter 1.'}"

set -e

# Members of the natcap software team can push to the autorelease branch on
# natcap/invest; this branch is a special case for our release process.
AUTORELEASE_BRANCH=autorelease/$VERSION
git checkout -b "$AUTORELEASE_BRANCH"
git add Makefile HISTORY.rst
git commit -m "Committing the $VERSION release."
git tag "$VERSION"

echo "The release has been committed and tagged."
echo "To push to natcap/invest:"
echo "  $ git push --tags git@github.com/natcap/invest.git $AUTORELEASE_BRANCH"

# Figure out the origin repo's username for nicer push messages.
ORIGIN_REPO=$(git config --get remote.origin.url | sed 's|\.git$||g' | grep -E -o '[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$')
if [ "$ORIGIN_REPO" != "natcap/invest" ]
then
    echo ""
    echo "To push to your fork:"
    echo "  $ git push --tags git@github.com/$ORIGIN_REPO.git $AUTORELEASE_BRANCH"
fi
