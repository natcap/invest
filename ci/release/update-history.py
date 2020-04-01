# encoding=UTF-8
import sys
import argparse
import shutil
import os
import datetime


def note_release_section(version, date_string):
    """Update the release section to the target version.

    Parameters:
        version (string): The version being released.
        date_string (string): The date of the release, in the form YYYY-MM-DD.
            Single-digit dates must be zero-padded, so the third date of
            the fourth month of 2020 should be ``2020-04-03``.

    Returns:
        ``None``

    """
    # Parsing the date string here will make sure the date makes sense.
    # If datetime can't parse it, an error will be raised.
    release_date = datetime.datetime.strptime(date_string, '%Y-%m-%d')

    new_history_path = os.path.join('build', 'HISTORY.rst')
    history_path = 'HISTORY.rst'
    if not os.path.exists('build'):
        os.makedirs('build')

    with open(new_history_path, 'w', newline='\n') as new_history:
        with open(history_path) as history:
            for line in history:
                if line.rstrip() == 'Unreleased Changes':
                    # Include the Unreleased Changes section, but commented
                    # out and above the new section title.
                    new_history.write('..\n')
                    new_history.write('  Unreleased Changes\n')
                    new_history.write('  ------------------\n')
                    new_history.write('\n')

                    new_title_string = '%s (%s)' % (
                        version, release_date.strftime(
                            '%Y-%m-%d'))
                    new_history.write(new_title_string + '\n')
                    new_history.write('-' * len(new_title_string) + '\n')
                    _ = next(history)  # discard previous unreleased line
                else:
                    new_history.write(line)

    shutil.copyfile(new_history_path, history_path)

    print('HISTORY has been updated to release version %s on %s' % (
        version, date_string))


def main(args=None):
    """Main function with an argparse interface.

    Parameters:
        args (list or None): A list of command-line args to parse, such as from
            ``sys.argv``.  If ``None``, arguments will be taken from the
            command-line.

    Returns:
        ``None``.

    """
    parser = argparse.ArgumentParser(description=(
        'Update the "Unreleased Changes" section header in HISTORY.rst to '
        'indicate the provided version and release date.'
    ))

    parser.add_argument('version', help='The version string being released.')
    parser.add_argument('datestring', help=(
        'The date of the release, in the form YYYY-MM-DD.'))

    parsed_args = parser.parse_args(args)
    note_release_section(parsed_args.version, parsed_args.datestring)


if __name__ == '__main__':
    main(sys.argv[1:])
