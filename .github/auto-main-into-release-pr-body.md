Merge `$SOURCE_BRANCH` into `$RELEASE_BRANCH` $PR_MESSAGE

This PR has been triggered in an effort to update `$RELEASE_BRANCH` with the
changes that were just added to `$SOURCE_BRANCH`.

### If there aren't any conflicts and tests pass

Great, merge away!

### If there are merge conflicts

You (as a software team member) have write access to this branch
(`$AUTOPR_BRANCH`) on this repo.  This means you can address any merge
conflicts directly either by:

1. Using the GitHub UI to fix the conflict
2. Doing the work directly on your fork and pushing up to this branch.
   (see below for an example of how to do this)

### If tests fail or there are problems with the merge

This is unlikely, but it can happen.  To fix issues, make the changes you need
to make on your fork and push the changes up to this branch.  See below for an
example of how to do this.

### To push changes to this branch

In your fork, run:

```shell
git fetch upstream $AUTOPR_BRANCH
# Make and commit your changes here.
#
# When changes are ready to go, continue by running this:
git push upstream $AUTOPR_BRANCH
```

All changes pushed in this way will appear in this PR.

<hr>

<sub>

This PR was created because there was a push to `$SOURCE_BRANCH` that triggered
this workflow, which is defined at `.github/workflows/autopr-create.yml`

</sub>
