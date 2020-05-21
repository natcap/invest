AUTO: merge $SOURCE_BRANCH into $RELEASE_BRANCH

This PR was automatically generated in response to a push to `master`,
and is a chance to review any changes that will be included in the release
branch before merging.  Under most circumstances, this PR will probably be
a formality.  However, there are a few cases where we may need to do some
extra work to make sure `$RELEASE_BRANCH` contains what it should after the
merge:

## There is a merge conflict in this PR

1. Leave a comment on this PR about the merge conflict and close the PR.
2. In your fork, make a new `pr-resolution` branch off of `$SOURCE_BRANCH`:
   ```shell
   $ git checkout $SOURCE_BRANCH
   $ git pull upstream $SOURCE_BRANCH  # Include the latest changes on the upstream master
   $ git checkout -b pr-resolution
   $ git merge $RELEASE_BRANCH
   ```
3. Resolve the conflicts locally
4. Commit the changes to `pr-resolution`.
5. Create a PR from `pr-resolution` into `$RELEASE_BRANCH`, and include a link
   to the origin PR in the description.
6. When the PR is complete, delete the `pr-resolution` branch.  That will
   help us avoid confusion and extra work down the road when we do this again.

## This PR contains content that should not be in `$RELEASE_BRANCH`

1. Leave a comment on this PR about the content that should not be included
   and close the PR.
2. In your fork, make a new `pr-resolution` branch off of `$SOURCE_BRANCH`:
   ```shell
   $ git checkout $SOURCE_BRANCH
   $ git pull upstream $SOURCE_BRANCH  # Include the latest changes on the upstream master
   $ git checkout -b pr-resolution
   $ git merge $RELEASE_BRANCH
   ```
3. Handle the content that should not end up in `$RELEASE_BRANCH` however it
   needs to be handled.
4. Commit the updated content to `pr-resolution`.
5. Create a PR from `pr-resolution` into `$RELEASE_BRANCH`, and include a link
   to the origin PR in the description.
6. When the PR is complete, delete the `pr-resolution` branch.  That will
   help us avoid confusion and extra work down the road when we do this again.

## What happens if we accidentally merge something we shouldn't?

There are several possibilities for recovery if we get to such a state.

1. A merge can be undone through the github interface if the error is caught
   directly after the PR is merged.
2. If we're commits in past the erroneous merge, create a branch off of
   `$RELEASE_BRANCH`, back out of the changes or edit files needed to resolve
   the issue, and PR the branch back into `$RELEASE_BRANCH`.

### Why was this PR created?

The workflow defining this PR is located at
`.github/workflows/auto-pr-from-master-into-releases.yml`.  In short, this PR
was created because there was a push to `$SOURCE_BRANCH` that triggered this
workflow.  Some events that can trigger this include:

* Other pull requests being merged into `$SOURCE_BRANCH`
* Automated releases on `$SOURCE_BRANCH`
* Any manual push to `$SOURCE_BRANCH`, if ever that happens (which shouldn't be the
  case given our branch protections)

