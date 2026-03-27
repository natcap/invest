"""Verify that contributors to a PR have signed the CLA.

This command-line script is intended to be run as a part of a github actions
workflow that takes the output of this script and posts it as a comment on the
relevant PR.  Having said that, there is nothing that is GHA-specific within
this script, so it could be run anywhere, without the GHA context, without any
issue.

The script makes use of the public GitHub REST API, which means that we don't
need to bother with API keys, but there are also rate limits in place.  There
is not currently an option to provide an API key to the github API requests.

To install dependencies, run:
    $ pip install requests

To invoke this script on a PR in the natcap/invest repo, use
    $ python scripts/check-cla.py 1268

To invoke this on a PR that's in a fork of InVEST, use
    $ python scripts/check-cla.py <pr_num> --repo=<user>/invest
"""

import argparse
import datetime
import json
import logging
import math
import os
import re
import sys

import requests

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
LOGGER = logging.getLogger(os.path.basename(__file__))
CLA_HOST = 'https://vpejnqubjf.us-east-2.awsapprunner.com/'
GITHUB_HEADERS = {
    "X-GitHub-Api-Version": "2022-11-28",
    "Accept": "application/vnd.github+json",
}
UNSIGNED_MSG = (
    "Thank you for your pull request and welcome to our community! We require "
    "contributors to sign our short [Contributor License "
    "Agreement](https://natcap.github.io/invest-cla/). "
    "In order to review and merge your code, please follow the link above "
    "and follow the instructions to authenticate with your github account "
    "and agree to the CLA.  If you have questions or received this message "
    "in error, please don't hesitate to mention `@softwareteam` or leave a "
    "comment here in the PR. \n"
    "\n"
    "{committers_message}\n"
    "\n"
    "<!--\n"
    "    METADATA {last_checked_metadata}\n"
    "-->\n"
)


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


def clean_and_format_commit_member_json(commit_member):
    """Return an author or committer json object without timestamps.

    Args:
        commit_member (dict): A commit author or committer dictionary from the
            github api.

    Returns:
        identifiying_string (str): A rendered string identifying the author or
            committer, derived from the input dict.
    """
    exclude_keys = set(['date'])
    output_dict = dict(
        (key, value) for (key, value) in commit_member.items()
        if key not in exclude_keys)
    return "JSON:" + json.dumps(output_dict, sort_keys=True)


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
            },
        )

        commits_data = resp.json()
        for c in commits_data:
            # When a commit is made by github actions, it doesn't have any
            # github login or author information, so we can just look for a
            # similar signature and skip it if it matches.
            if ((c['author'] == c['committer'] == {})
                    and (c['commit']['author'] == c['commit']['committer'])
                    and (c['commit']['author']['name']
                         == c['commit']['committer']['name']
                         == 'GitHub Actions')):
                LOGGER.info(f"Commit {c['sha']} was created by github "
                            "actions. This is a special case, as actions "
                            "would have been written by an author who signed "
                            "the CLA. Skipping.")
                continue

            # git tracks the author and committer separately.  These will often
            # be the same person, particularly for smaller projects like ours.
            #
            # Also, if the author or committer has used local git
            # identification information that is not matched up with a github
            # username, then we will need to figure out a manual process for
            # verification.
            if c['author'] is None:
                LOGGER.warning(
                    "Author was not recognized by github: "
                    f"{c['commit']['author']}")
                pr_committers.add(
                    clean_and_format_commit_member_json(
                        c['commit']['author']))
            else:
                pr_committers.add(c['author']['login'])

            if c['committer'] is None:
                LOGGER.warning(
                    "Committer was not recognized by github: "
                    f"{c['commit']['committer']}")
                pr_committers.add(
                    clean_and_format_commit_member_json(
                        c['commit']['committer']))
            else:
                pr_committers.add(c['committer']['login'])

    # clean up committers that are part of github's web interface
    for invalid_committer in [
            "web-flow",  # The github git committer for web commits
            ]:
        try:
            pr_committers.remove(invalid_committer)
            LOGGER.info(
                f"Removed {invalid_committer} from the list of committers to "
                "this PR")
        except KeyError:
            # When the invalid_committer was not found in the PR.
            pass

    return pr_committers


def have_we_already_commented(
        pr_num, unsigned_committers, github_org='natcap',
        github_repo='invest'):
    """Have we already commented on this PR with the current committer info?

    Queries the PR and issues APIs to determine if we have already commented on
    the PR with a message about signing the CLA.  If a new committer is added
    to the PR's commits, then that counts as us not having posted.

    Args:
        pr_num (int, str): The PR number to query.
        unsigned_committers (list): A list of github usernames of committers
            who have not yet signed the CLA.
        github_org='natcap' (str): The github organization of the repo and PR
            that should be queried.
        github_repo='invest' (str): The github repo within the ``github_org``
            organization that the target PR belongs to.

    """
    resp = requests.get(
        f"https://api.github.com/repos/{github_org}/{github_repo}/pulls/{int(pr_num)}",
        headers=GITHUB_HEADERS,
    )
    pr_data = resp.json()
    n_comments = pr_data['comments']

    comments_per_page = 30  # the github default
    n_pages = math.ceil(n_comments / comments_per_page)
    latest_matching_comment = None
    for page_num in range(1, n_pages+1):
        comments_resp = requests.get(
            f'https://api.github.com/repos/{github_org}/{github_repo}/'
            f'issues/{int(pr_num)}/comments',
            headers=GITHUB_HEADERS,
            params={
                "page": page_num,
                "per_page": comments_per_page,
            },
        )
        comments_resp.raise_for_status()
        pr_comments = comments_resp.json()

        for comment in pr_comments:
            if (comment['user']['login'] == 'github-actions[bot]'
                    and 'METADATA' in comment['body']):
                latest_matching_comment = comment

    # Base case: If no matching comments, we have not commented yet.
    if not latest_matching_comment:
        return False

    # extract metadata
    for line in latest_matching_comment:
        line = line.strip()
        if line.startswith('METADATA'):
            comment_metadata = json.loads(line.replace('METADATA', ''))
            last_unsigned_committers = set(
                comment_metadata['unsigned_committers'])

            if not set(unsigned_committers).issubset(last_unsigned_committers):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description="Check the CLA signing status for a PR")
    parser.add_argument('pr_num', help="The PR number of the current PR")
    parser.add_argument(
        '--repo',
        help=(
            "The github repo the PR belongs to, in the form "
            "username/reponame"),
        default="natcap/invest")

    args = parser.parse_args()

    pr_num = int(args.pr_num)
    username, repo = args.repo.split('/')
    LOGGER.info(f"Checking PR {args.pr_num} on {username}/{repo}")
    pr_committers = contributors_to_pr(pr_num, username, repo)
    signed_committers = set()
    unsigned_committers = set()
    unknown_committers = set()
    for committer in pr_committers:
        # Github usernames are only allowed to contain alphanumeric chars and
        # dashes, so if it doesn't match this regexp then it must represent an
        # non-github git committer.
        if not re.match('^[a-z0-9-]+$', committer):
            unknown_committers.add(committer)
        elif check_contributor(committer):
            signed_committers.add(committer)
        else:
            unsigned_committers.add(committer)

    LOGGER.info(f"Committers who have signed: {signed_committers}")
    LOGGER.info(f"Committers who have not signed: {unsigned_committers}")
    LOGGER.info("Committers who are unknown to github and should be reviewed "
                f"manually: {unknown_committers}")
    if len(unsigned_committers) + len(unknown_committers) == 0:
        LOGGER.info("Looks like everyone has signed the CLA!")
        parser.exit(0)

    cla_messages = []
    if unsigned_committers:
        cla_messages.append(
            "The CLA has not yet been signed by github users: "
            ", ".join(sorted(f"@{c}" for c in unsigned_committers)))

    if unknown_committers:
        message = (
            "\nThe following authors/committers were not recognized by github "
            "and should be verified manually: ")
        for uc in unknown_committers:
            message += (f"\n* `{uc}`")

        cla_messages.append(message)

    committer_metadata = {
        "last_checked": datetime.datetime.now().isoformat(),
        "signed_committers": sorted(signed_committers),
        "unsigned_committers": sorted(unsigned_committers),
    }
    if have_we_already_commented(
            pr_num, committer_metadata['unsigned_committers'],
            github_org=username, github_repo=repo):
        LOGGER.info(
            "Looks like we already commented on the repo and the committers "
            "have not changed. Not commenting again at this time.")
        parser.exit(1)
    else:
        LOGGER.info("Printing formatted message for posting to the PR")
        print(UNSIGNED_MSG.format(
            last_checked_metadata=committer_metadata,
            committers_message='\n'.join(cla_messages),
        ))
        parser.exit(2)

    parser.exit(255)  # This should not execute


if __name__ == '__main__':
    main()
