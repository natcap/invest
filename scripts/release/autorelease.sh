#!/usr/bin/env bash
#
# Initiate a release on the current commit.
#
set -e  # Exit the script immediately if any subshell has a nonzero exit code.

VERSION=$1

: "${VERSION:?'The version string is needed as parameter 1.'}"

: "${GITHUB_TOKEN:?'The GITHUB_TOKEN environment variable must be defined and have repo write permissions.'}"

check_required_programs.sh pandoc twine gsutil gh envsubst

REPO="natcap/invest"

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

python increment-userguide-revision.py
python update-history.py "$VERSION" "$(date '+%Y-%m-%d')"

echo ""
echo "Changes have been made to the following files:"
echo "  * HISTORY.rst has been updated with the release version and today's date"
echo "  * Makefile has been updated with the latest user's guide revision"
echo ""
echo "To continue with the release:"
echo "  $ ./scripts/release-2-commit.sh $VERSION"



# Members of the natcap software team can push to the autorelease branch on
# natcap/invest; this branch is a special case for our release process.
AUTORELEASE_BRANCH=autorelease/$VERSION
git checkout -b "$AUTORELEASE_BRANCH"
git add Makefile HISTORY.rst
git commit -m "Committing the $VERSION release."
git tag "$VERSION"

# Push the tag
git push https://github.com/natcap/invest.git $VERSION $AUTORELEASE_BRANCH


echo ""
echo "After pushing, wait for builds to finish before continuing."
echo "    See https://github.com/natcap/invest/wiki/Bugfix-Release-Checklist#wait-for-builds-to-complete"

RUN_ID=$(gh run list --repo natcap/invest | awk -F '\t' '{ if ($5 == "release/3.13.0") { print $7 } }')

if (( ${#RUN_ID} > 10 ))
then
    echo "Multiple run IDs found: ${RUN_ID}"
    exit 4
fi

gh run watch $RUN_ID

BUCKET="$(make jprint-RELEASES_BUCKET)/invest"

source RELEASE_MANAGER.env


# Using -p here to not fail the command if the directory already exists.
mkdir -p dist build

gsutil cp "$BUCKET/$VERSION/*.zip" dist  # UG, sampledata, mac binaries
gsutil cp "$BUCKET/$VERSION/*.exe" dist  # Windows installer
gsutil cp "$BUCKET/$VERSION/natcap.invest*" dist  # Grab python distributions

RELEASE_MESSAGE_FILE=build/release_message.md
build-release-text-from-history.sh > $RELEASE_MESSAGE_FILE
gh release create $VERSION \
    --repo $REPO \
    --notes-file $RELEASE_MESSAGE_FILE \
    --verify-tag \
    dist/*

PR_MESSAGE_FILE=build/pr_msg_text_$VERSION.txt

# Explicitly setting the environment variables we need in envsubst.  The only
# other alternative is to `export` them all, which then pollutes the shell as a
# side effect.
SOURCE_BRANCH="main" BUGFIX_VERSION="$VERSION" GITHUB_REPOSITORY="$REPO" \
    envsubst < bugfix-autorelease-branch-pr-body.md > "$PR_MESSAGE_FILE"

# Create a pull request from the autorelease branch into main
gh pr create \
    --base "${REPO}:main" \
    --head "${REPO}:autorelease/$VERSION" \
    --title "${VERSION} release" \
    --body-file "$PR_MESSAGE_FILE" \
    --reviewer "@me" \
    --assign "@me" \
    --labels "auto"

twine upload dist/natcap.invest.*
