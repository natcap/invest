# encoding=UTF-8

import sys

import requests


def read_changes_for_version(version):
    changes_text = []
    in_version_block = False
    with open('HISTORY.rst') as history_file:
        for line in history_file:
            if line.startswith(version):
                in_version_block = True

            if in_version_block:
                line = line.rstrip()  # only strip trailing newline

                # Only read until the first blank line after the version notes
                # block.
                if line.strip() == '':
                    break

                # Track the line and also replace RST double-backticks with MD
                # single-backticks.
                changes_text.append(line.replace('``', '`'))

    return '\n'.join(changes_text)


def create_github_release(args=None):
    parser = argparse.ArgumentParser(description=(
        "Create a release in a GitHub-hosted repository for the target tag. "
        "The body of the release object will be derived from the changes "
        "noted in this release's section in HISTORY.rst. "
        "If the tag does not exist, an error will be raised."
    ))

    parser.add_argument('tag', help='The tag name to use.')
    parser.add_argument('oauthtoken', help='The GitHub OAuth2 token')
    parser.add_argument('repo', help=(
        'The GitHub username and repo to use.  Example: "natcap/invest"'))

    parsed_args = parser.parse_args(args)

    result = requests.post(
        f'https://api.github.com/repos/{parsed_args.repo}/releases',
        headers={
            'Authorization': f'token {parsed_args.oauthtoken}'
        },
        data={
            'tag_name': str(parsed_args.tag),
            'target_commitish': str(parsed_args.tag),
            'name': f'InVEST {parsed_args.tag}',
            'body': read_changes_for_version(parsed_args.tag),
            'draft': False,
            'prerelease': False,
        })
    r.raise_for_status()  # Raises an exception on HTTP error.


if __name__ == '__main__':
    create_github_release(sys.argv[:])
