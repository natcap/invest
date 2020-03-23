# encoding=UTF-8

import sys
import argparse
import requests


def main(args=None):
    """Tag a GitHub repository.

    Accepts command-line arguments and makes the HTTP request to the GitHub
    API to tag the target repository.

    Returns:
        ``None``

    """
    parser = argparse.ArgumentParser(description=(
        "Tag a GitHub-hosted repository with the provided tag string."))

    parser.add_argument('tag', help='The tag name to use.')
    parser.add_argument('sha', help='The SHA object to tag with the tagname.')
    parser.add_argument('oauthtoken', help='The GitHub OAuth2 token')
    parser.add_argument('repo', help=(
        'The GitHub username and repo to use.  Example: "natcap/invest"'))

    parsed_args = parser.parse_args(args)

    r = requests.post(
        'https://api.github.com/repos/{parsed_args.repo}/git/refs/',
        headers={
            'Authorization': f'token {parsed_args.oauthtoken}'
        },
        data={
            'ref': f'refs/tags/{parsed_args.tag}',
            'sha': parsed_args.sha,
        })
    # Raise an exception if there was one.
    r.raise_for_status()


if __name__ == '__main__':
    main(sys.argv[1:])
