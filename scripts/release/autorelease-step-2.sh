set -ex  # Exit the script immediately if any subshell has a nonzero exit code.

VERSION=$1
GH_RUN_ID=$2
GITHUB_REPO="emlys/invest-mirror"
PYPI_REPO="testpypi"
SCRIPT_PATH=$(dirname "$0")

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

# Using -p here to not fail the command if the directory already exists.
mkdir -p dist build

echo "Found github actions run: https://github.com/$GITHUB_REPO/actions/runs/$RUN_ID"
echo "Downloading artifacts..."
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
