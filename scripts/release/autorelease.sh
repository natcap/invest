#!/usr/bin/env bash
#
# Initiate a release on the current commit.
#
set -e  # Exit the script immediately if any subshell has a nonzero exit code.

# - Create autorelease branch
# - Update HISTORY.rst
# - Commit the changes to HISTORY.rst
# - Tag the commit
# - Push the tag
# - Build workflows run on push, wait for them to finish
# - Download the build artifacts
# - Create a github release for the tag
#   - Attach release notes
#   - Attach the build artifacts
# - Create a pull request from the autorelease branch into main
# - Create a pypi release
#   - Attach the build artifacts

VERSION=$1

: "${VERSION:?'The version string is needed as parameter 1.'}"

: "${GITHUB_TOKEN:?'The GITHUB_TOKEN environment variable must be defined and have repo write permissions.'}"

check_required_programs.sh pandoc twine gsutil gh envsubst

REPO="natcap/invest"
AUTORELEASE_BRANCH=autorelease/$VERSION
BUCKET="$(make jprint-RELEASES_BUCKET)/invest"
RELEASE_MESSAGE_FILE=build/release_message.md
PR_MESSAGE_FILE=build/pr_msg_text_$VERSION.txt


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

python update-history.py "$VERSION" "$(date '+%Y-%m-%d')"

echo ""
echo "Changes have been made to the following files:"
echo "  * HISTORY.rst has been updated with the release version and today's date"

# Members of the natcap software team can push to the autorelease branch on
# natcap/invest; this branch is a special case for our release process.

git checkout -b "$AUTORELEASE_BRANCH"
git add HISTORY.rst
git commit -m "Committing the $VERSION release."
git tag "$VERSION"

# Push the tag
git push https://github.com/natcap/invest.git $VERSION $AUTORELEASE_BRANCH


echo ""
echo "After pushing, wait for builds to finish before continuing."
echo "    See https://github.com/natcap/invest/wiki/Bugfix-Release-Checklist#wait-for-builds-to-complete"

RUN_ID=$(gh run list --repo natcap/invest | awk -F '\t' '{ if ($5 == "$AUTORELEASE_BRANCH") { print $7 } }')

if (( ${#RUN_ID} > 10 ))
then
    echo "Multiple run IDs found: ${RUN_ID}"
    exit 4
fi

gh --repo $REPO run watch $RUN_ID

# Using -p here to not fail the command if the directory already exists.
mkdir -p dist build

gh --repo $REPO run download --dir dist \
    --name InVEST-Windows-binary.zip \
    --name InVEST-macOS-binary.zip \
    --name Workbench-Windows-binary.zip \
    --name Workbench-macOS-binary.zip \
    --name InVEST-sample-data.zip \
    --name InVEST-user-guide.zip \
    --name "Source distribution.zip" \
    --name "Wheel for *.zip"

gsutil cp "$BUCKET/$VERSION/*.zip" dist  # UG, sampledata, mac binaries
gsutil cp "$BUCKET/$VERSION/*.exe" dist  # Windows installer
gsutil cp "$BUCKET/$VERSION/natcap.invest*" dist  # Grab python distributions

build-release-text-from-history.sh > $RELEASE_MESSAGE_FILE
gh release create $VERSION \
    --repo $REPO \
    --notes-file $RELEASE_MESSAGE_FILE \
    --verify-tag \
    dist/*


# Explicitly setting the environment variables we need in envsubst.  The only
# other alternative is to `export` them all, which then pollutes the shell as a
# side effect.
BUGFIX_VERSION="$VERSION" GITHUB_REPOSITORY="$REPO" \
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
