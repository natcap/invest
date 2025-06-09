import logging
import math
import os

import requests

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(os.path.basename(__file__))
CLA_HOST = 'https://vpejnqubjf.us-east-2.awsapprunner.com/'
GITHUB_HEADERS = {
    "X-GitHub-Api-Version": "2022-11-28",
    "Accept": "application/vnd.github+json",
}


def check_contributor(github_username):
    """Check if a single contributor has signed the CLA.

    Args:
        github_username (str): The string github username to check.

    Returns:
        cla_signed (bool): Whether the user has signed the CLA.
    """
    LOGGER.info(f"Checking CLA status of username {github_username}")
    resp = requests.get(
        f"{CLA_HOST}/contributor",
        params={
            "checkContributor": github_username,
        })
    resp.raise_for_status()
    return bool(resp.json()['isContributor'])


def contributors_to_pr(pr_num, github_org='natcap', github_repo='invest'):
    """Determine who committed to a PR via the git history.

    Args:
        pr_num (int, str): The PR number to query.
        github_org='natcap' (str): The github organization of the repo and PR
            that should be queried.
        github_repo='invest' (str): The github repo within the ``github_org``
            organization that the target PR belongs to.

    Returns:
        committers (set): A set of github usernames that have authored or
            committed commits in this repo.
    """
    resp = requests.get(
        f"https://api.github.com/repos/{github_org}/{github_repo}/pulls/{int(pr_num)}",
        headers=GITHUB_HEADERS,
    )
    pr_data = resp.json()
    n_commits_in_pr = pr_data['commits']

    # The github api only allows 250 commits to be read via a PR, so warn if we
    # exceed this so we can fix it.
    if n_commits_in_pr > 250:
        LOGGER.error(
            "There are over 250 commits in this PR, which this script is not "
            "designed to handle.  Review the github api docs and use the "
            "commits api endpoint instead.")

    # github allows a max of 100 commits per page, so paginate to work around
    # this.
    pr_committers = set()
    page_num = 1
    commits_per_page = 30
    n_pages_to_parse = math.ceil(n_commits_in_pr / commits_per_page)
    for page_num in range(1, n_pages_to_parse+1):
        LOGGER.info(
            f"Reading commits page {page_num} of {n_pages_to_parse}")
        resp = requests.get(
            pr_data['commits_url'],
            params={
                "page": page_num,
                "per_page": commits_per_page,
            }
        )

        commits_data = resp.json()
        for c in commits_data:
            # git tracks the author and committer separately.  These will often
            # be the same person, particularly for smaller projects like ours.
            pr_committers.add(c['author']['login'])
            pr_committers.add(c['committer']['login'])

    # clean up committers that are part of github's web interface
    for invalid_committer in [
            "web-flow",  # The github git committer for web commits
            ]:
        pr_committers.remove(invalid_committer)

    return pr_committers


def main():
    pass


if __name__ == '__main__':
    check_contributor('phargogh')
    print(contributors_to_pr(1268))
