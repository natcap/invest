#!/usr/bin/env bash
#
# Initiate a release on the current commit.
VERSION=$1

: "${VERSION:?'The version string is needed as parameter 1.'}"

set -e

if ! git diff --exit-code > /dev/null  # fail if uncommitted, unstaged changes
then
    echo "You have uncommitted changes."
    echo "Commit or stash your changes and re-run this script."
    exit 1
fi

if ! git diff --exit-code --staged > /dev/null  # fail if uncommitted, staged changes
then
    echo "You have staged changes."
    echo "Commit or stash your staged changes and re-run this script."
    exit 2
fi

if git rev-parse "$VERSION" >/dev/null 2>&1  # fail if tag already exists
then
    echo "The tag $VERSION already exists in this repo."
    echo "Are you sure you're creating the right version?"
    exit 3
fi

python ci/release/increment-userguide-revision.py
python ci/release/update-history.py "$VERSION" "$(date '+%Y-%m-%d')"

echo ""
echo "Changes have been made to the following files:"
echo "  * HISTORY.rst has been updated with the release version and today's date"
echo "  * Makefile has been updated with the latest user's guide revision"
echo ""
echo "To continue with the release:"
echo "  $ ./scripts/release-2-commit.sh $VERSION"
