import argparse
import json
import logging

import requests

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

GITHUB_HEADERS = {
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
}


def github_api(endpoint, payload=None):
    req = requests.get(endpoint, None)
    req.raise_for_status()
    return req.json()


def main(args=None):
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
    resp = requests.get(f'https://api.github.com/repos/{github_repo}/commits/{git_sha}/pulls')
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

    return json.dumps({"authors": list(pr_authors)})


if __name__ == '__main__':
    print(main())
