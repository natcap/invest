Merge `$SOURCE_BRANCH` into `$RELEASE_BRANCH` $PR_MESSAGE

This PR has been triggered in an effort to update `$RELEASE_BRANCH` with the
changes that were just added to `$SOURCE_BRANCH`.

You have write access to the branch `$AUTOPR_BRANCH` on this repo, so feel free
to use this to resolve merge conflicts or make any needed changes before this
merge is completed into `$RELEASE_BRANCH`.

### How to push to this branch

This PR uses a special-case branch that only members of the software team have
write access to.  When this PR is merged, the branch will be automatically deleted.
If you need to manually push changes to this PR, use this process:

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
The workflow defining this PR is located at
`.github/workflows/autopr-create.yml`.  In short, this PR
was created because there was a push to `$SOURCE_BRANCH` that triggered this
workflow.
</sub>
