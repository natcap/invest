#!/usr/bin/env bash
# Initiate a release on the current commit.
# - Create autorelease branch
# - Update HISTORY.rst
# - Commit the changes to HISTORY.rst
# - Tag the commit
# - Push the tag
# - Build workflows run on push, wait for them to finish

set -e  # Exit the script immediately if any subshell has a nonzero exit code.

VERSION=$1
GITHUB_REPO="natcap/invest"
SCRIPT_PATH=$(dirname "$0")

# Validate inputs and environment
: "${VERSION:?'The version string is needed as parameter 1.'}"
$SCRIPT_PATH/check_required_programs.sh pandoc twine gh envsubst

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

if git rev-parse "$VERSION" > /dev/null 2>&1  # fail if tag already exists
then
    echo "The tag $VERSION already exists in this repo."
    echo "Are you sure you're creating the right version?"
    exit 3
fi

# Define variables
AUTORELEASE_BRANCH=autorelease/$VERSION
RELEASE_MESSAGE_FILE=build/release_message.md
PR_MESSAGE_FILE=build/pr_msg_text_$VERSION.txt

# On a new branch, update HISTORY, commit the change, tag it, and push
#
# Members of the natcap software team can push to the autorelease branch on
# natcap/invest; this branch is a special case for our release process.
git checkout -b "$AUTORELEASE_BRANCH"

# Replace
#
# Unreleased Changes
# ------------------
#
# with
#
# ..
#   Unreleased Changes
#   ------------------
#
# X.X.X (XXXX-XX-XX)
# ------------------

HEADER="$VERSION ($(date '+%Y-%m-%d'))"
HEADER_LENGTH=${#HEADER}
UNDERLINE=$(for i in $(seq 1 $HEADER_LENGTH); do echo -n "-"; done)
perl -0777 -i -pe \
    "s/Unreleased Changes\n------------------/..\n  Unreleased Changes\n  ------------------\n\n${HEADER}\n${UNDERLINE}/g" \
    HISTORY.rst

git add HISTORY.rst
git commit -m "Committing the $VERSION release."
git tag "$VERSION"
git push https://github.com/${GITHUB_REPO}.git $VERSION $AUTORELEASE_BRANCH

echo "Waiting for Github Actions run to start..."
sleep 5
# from the list of recent github actions runs,
# extract the run ID corresponding to the release tag
RUN_ID=$(gh --repo $GITHUB_REPO run list --branch $VERSION --json databaseId --jq ".[].databaseId")

if (( ${#RUN_ID} = 0 ))
then
    echo "No matching run found"
    exit 4
fi
if (( ${#RUN_ID} > 10 ))
then
    echo "Multiple run IDs found: ${RUN_ID}"
    exit 4
fi

echo "Wait for Github Actions run to finish:"
echo "https://github.com/$GITHUB_REPO/actions/runs/$RUN_ID"
echo ""
echo "When it's completed successfully, run the second release script:"
echo "./autorelease-step-2.sh $VERSION $RUN_ID"

