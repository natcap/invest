Auto: Release $BUGFIX_VERSION and merge into $SOURCE_BRANCH

This PR includes changes needed to perform the $BUGFIX_VERSION release.
When the PR was created, the bugfix release tag $BUGFIX_VERSION should have
been created on `$TARGET_BRANCH`, which should also have triggered builds
for python wheels as well as Windows and Mac binaries.  These binaries
will be automatically uploaded to the draft release object at
https://github.com/$GITHUB_REPOSITORY/releases/tag/$BUGFIX_VERSION.

## If something doesn't look right

1. PR whatever changes are needed into this branch (`$SOURCE_BRANCH`)
2. Continue until things look right.

## If everything looks OK

Approve and merge this branch.  You can delete this PR branch if you like.  It
will be automatically deleted once the PR is merged if it still exists.

## What happens once the PR is merged

When this PR is merged, a workflow will:

1. Remove the branch `$SOURCE_BRANCH` if it still exists.
2. Publish the draft release.
