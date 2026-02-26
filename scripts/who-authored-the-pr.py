"""Determine who authored the changes in a merge commit.

Suppose we have a commit with SHA __A__ that was created by a PR.  It is useful
for our automation to be able to identify who authored the PR that resulted in
the merge commit.  This script accomplishes this.

Usage:

    $ python scripts/who-authored-the-pr.py 2376d54cc9e7acb446ecc81dc9cd4d6eb8ec3775

Note that logging will be sent to standard out, so this is an example of
capturing the output of this program for use in a shell script:

    $ usernames=$(python scripts/who-authored-the-pr.py 2376d54cc9e7acb446ecc81dc9cd4d6eb8ec3775)
    $ echo $usernames
"""
import argparse
import logging

import requests

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

GITHUB_HEADERS = {
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
}


def main(args=None):
    """Determine who created the PR that created the target commit.

    Args:
        args (list or None): A list of string arguments to provide to argparse,
            or else ``None``.

    Returns:
        A comma-separated list of usernames of users who created the target PR.
    """
    parser = argparse.ArgumentParser(
        prog=__name__,
        description="Given a commit SHA, identify who wrote the PR.",
    )
    parser.add_argument('sha', help=(
        "The git SHA of the commit to consider.  This is expected to be a "
        "commit on the main branch."))
    parser.add_argument('--repo', default='natcap/invest', help=(
        "The github repository to use, in the form <org>/<repo>"))

    args = parser.parse_args(args)

    github_repo = args.repo
    git_sha = args.sha

    # find out if any PRs were associated with this commit specifically.
    resp = requests.get(
        f'https://api.github.com/repos/{github_repo}/commits/{git_sha}/pulls',
        headers=GITHUB_HEADERS)
    resp.raise_for_status()

    # Handling the case where we have an "octopus PR", where a single commit
    # represents multiple merges.  Doable in git, not sure how it might happen
    # in github.
    pr_authors = set()
    for contributing_pr in resp.json():
        pr_num = contributing_pr['number']
        pr_created_by_user = contributing_pr['user']['login']
        pr_authors.add(pr_created_by_user)
        LOGGER.info(f"PR {pr_num} was created by {pr_created_by_user}")

    return ",".join(sorted(pr_authors))


if __name__ == '__main__':
    print(main())
