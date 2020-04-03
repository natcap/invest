# encoding=UTF-8
"""Check if a release exists on a GitHub repository."""
import argparse

import github3
import github3.exceptions


def main(args=None):
    """Parse command-line arguments for this script.

    Parameters:
        args=None (list or None): A list of string parameters to be parsed.

    Returns:
        ``None``

    """
    parser = argparse.ArgumentParser(description=(
        "Check that a release exists on the target github repository."))

    parser.add_argument('repo', help=(
        "The GitHub user/repo string that the target release is attached to. "
        "Example: 'natcap/invest'"))
    parser.add_argument('tagname', help="The tag of the target release.")

    parsed_args = parser.parse_args(args)

    session = github3.GitHub()
    repository = session.repository(*parsed_args.repo.split('/'))
    try:
        _ = repository.release_from_tag(parsed_args.tagname)
    except github3.exceptions.NotFoundError:
        parser.exit(1, 'ERROR: Tag "%s" not found in repo %s\n' % (
            parsed_args.tagname, parsed_args.repo))

    parser.exit(0, 'OK: Tag "%s" present in repo %s\n' % (
        parsed_args.tagname, parsed_args.repo))


if __name__ == '__main__':
    main()
