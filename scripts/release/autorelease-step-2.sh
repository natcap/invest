# Finish the release.
# - Download the build artifacts
# - Create a github release for the tag
#   - Attach release notes
#   - Attach the build artifacts
# - Create a pull request from the autorelease branch into main
# - Create a pypi release
#   - Attach the build artifacts
set -e  # Exit the script immediately if any subshell has a nonzero exit code.

VERSION=$1
GH_RUN_ID=$2
GITHUB_REPO="natcap/invest"
PYPI_REPO="testpypi"
SCRIPT_PATH=$(dirname "$0")

# Using -p here to not fail the command if the directory already exists.
mkdir -p dist build

echo "Downloading artifacts from github actions run: "
echo "https://github.com/$GITHUB_REPO/actions/runs/$RUN_ID"
gh --repo $GITHUB_REPO run download $RUN_ID \
    --dir dist \
    --name InVEST-Windows-binary.zip \
    --name InVEST-macOS-binary.zip \
    --name Workbench-Windows-binary.zip \
    --name Workbench-macOS-binary.zip \
    --name InVEST-sample-data.zip \
    --name InVEST-user-guide.zip \
    --name "Source distribution.zip" \
    --name "Wheel for *.zip"

# Create a release on Github
RELEASE_MESSAGE_FILE=build/release_message.md
$SCRIPT_PATH/build-release-text-from-history.sh > $RELEASE_MESSAGE_FILE  # Create the release message
gh release create $VERSION \
    --repo $GITHUB_REPO \
    --notes-file $RELEASE_MESSAGE_FILE \
    --verify-tag \
    dist/*
rm $RELEASE_MESSAGE_FILE

# Create a pull request from the autorelease branch into main
#
# Create the PR message, substituting in variables
# Use envsubst to avoid polluting the shell
PR_MESSAGE_FILE=build/pr_msg_text_$VERSION.txt
envsubst < bugfix-autorelease-branch-pr-body.md > "$PR_MESSAGE_FILE"
gh pr create \
    --base "${GITHUB_REPO}:main" \
    --head "${GITHUB_REPO}:autorelease/$VERSION" \
    --title "${VERSION} release" \
    --body-file "$PR_MESSAGE_FILE" \
    --reviewer "@me" \
    --assign "@me" \
    --labels "auto"
rm $PR_MESSAGE_FILE

# Create a release on PyPI
# This is the only step that can't be rolled back
twine upload -r $PYPI_REPO dist/natcap.invest.*

echo "Release has been created using testpypi. To release officially:"
echo "twine upload -r pypi dist/natcap.invest.*"
