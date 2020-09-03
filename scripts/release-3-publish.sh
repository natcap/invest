#!/usr/bin/env bash

VERSION=$1
: "${VERSION:?'The version string is needed as parameter 1.'}"

: "${GITHUB_TOKEN:?'The GITHUB_TOKEN environment variable must be defined and have repo write permissions.'}"

# Exit the script immediately if any subshell has a nonzero exit code.
set -e

# assume a default user of "natcap" if none has been provided.
GH_USER="${2:-'natcap'}"
if [ "$GH_USER" = "natcap" ]
then
    BUCKET="$(make jprint-RELEASES_BUCKET)/invest"
else
    BUCKET="$(make jprint-DEV_BUILD_BUCKET)/invest/$GH_USER"
fi

source ./ci/release/RELEASE_MANAGER.env

./scripts/check_required_programs.sh pandoc twine gsutil hub envsubst

# Using -p here to not fail the command if the directory already exists.
mkdir -p dist build

gsutil cp "$BUCKET/$VERSION/*.zip" dist  # UG, sampledata, mac binaries
gsutil cp "$BUCKET/$VERSION/*.exe" dist  # Windows installer
gsutil cp "$BUCKET/$VERSION/natcap.invest*" dist  # Grab python distributions

RELEASE_MESSAGE_FILE=build/release_message.md
./ci/release/build-release-text-from-history.sh > $RELEASE_MESSAGE_FILE
hub release create \
    --file $RELEASE_MESSAGE_FILE \
    --committish "$VERSION" \
    "$VERSION"

PR_MSG_TEXT=build/pr_msg_text_$VERSION.txt

# Explicitly setting the environment variables we need in envsubst.  The only
# other alternative is to `export` them all, which then pollutes the shell as a
# side effect.
SOURCE_BRANCH="main" BUGFIX_VERSION="$VERSION" GITHUB_REPOSITORY="$GH_USER/invest" \
    envsubst < ci/release/bugfix-autorelease-branch-pr-body.md > "$PR_MSG_TEXT"

hub pull-request \
    --base "natcap/invest:main" \
    --head "natcap/invest:autorelease/$VERSION" \
    --reviewer "$RELEASE_MANAGER" \
    --assign "$RELEASE_MANAGER" \
    --labels "auto" \
    --file "$PR_MSG_TEXT"

twine upload dist/natcap.invest.*
