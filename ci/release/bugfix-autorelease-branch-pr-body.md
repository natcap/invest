Auto: Release $BUGFIX_VERSION and merge into $SOURCE_BRANCH

# Release $BUGFIX_VERSION

This PR includes changes needed to perform the $BUGFIX_VERSION release.
When the PR was created, the bugfix release tag $BUGFIX_VERSION should have
been created on `$TARGET_BRANCH`, which should also have triggered builds
for python wheels as well as Windows and Mac binaries.  These binaries
will be automatically uploaded to the release object at
https://github.com/$GITHUB_REPOSITORY/releases/tag/$BUGFIX_VERSION.

## If something doesn't look right

1. Decline this PR.  Do not delete the `$TARGET_BRANCH` branch.
2. Go to the created release object and delete the binaries that have been
   automatically uploaded there:
     1. Go to https://github.com/$GITHUB_REPOSITORY/releases/tag/$BUGFIX_VERSION.
     2. Click "Edit"
     3. For each binary file, click the "X" button on the right-hand side.
     4. Click the green "Update Release" button at the bottom of the
        release page.
     5. Click the red "Delete" button to delete the release.
3. PR whatever changes are needed into `$TARGET_BRANCH`.
   ```shell
   $ git checkout master
   $ git pull upstream master
   $ git checkout $TARGET_BRANCH
   < make and commit any needed changes >
   $ git push origin $TARGET_BRANCH
   ```
4. Once things look right, tag `$TARGET_BRANCH` with `$BUGFIX_VERSION` and
   push to `$TARGET_BRANCH`.
   ```shell
   $ git checkout $TARGET_BRANCH
   $ git tag --force $BUGFIX_VERSION
   $ git push git@github.com:$GITHUB_REPOSITORY.git $TARGET_BRANCH $BUGFIX_VERSION
   ```
   Re-tagging and pushing the files to `$TARGET_BRANCH` will cause the release
   binaries to be rebuilt and re-uploaded to the release object.
5. Submit a PR from `$GITHUB_REPOSITORY:$TARGET_BRANCH` into `$GITHUB_REPOSITORY:$SOURCE_BRANCH`.


## If everything looks OK

1. Approve and merge this PR.
2. Delete the branch `$TARGET_BRANCH` once the PR is merged.
3. Take a look at the release object and make sure everything is there:
   https://github.com/$GITHUB_REPOSITORY/releases/tag/$BUGFIX_VERSION.
4. Upload python wheels to PyPI.  An issue was created for this,
   so remember to mark progress there as well.
