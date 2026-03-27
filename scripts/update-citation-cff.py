import argparse
import codecs
import os

RELEASES_BUCKET = 'http://releases.naturalcapitalproject.org'
PREFIX = '10.60793'


def update_citation_file(citation_filepath, version, date):
    """Rebuild specific lines in the citation file to use the version and date.

    This program will write the updated citation data back to the citation
    file.

    Args:
        citation_filepath (str): The path to the citation file.
        version (str): The version string of the InVEST version.
        date (str): The date of the release, in the form YYYY-MM-DD.

    Returns:
        ``None``.
    """
    with codecs.open(citation_filepath, encoding='utf-8') as citation_file:
        citation_data = citation_file.readlines()

    # This approach allows us to only change the lines we need to change and
    # not worry about PyYAML changing all of the lines in the file because they
    # have a different idea of how the file should be formatted.
    # Plus, this way the script only depends on the standard library.
    lines = []
    for line in citation_data:
        if line.startswith('version'):
            line = f"version: {version}\n"
        elif line.startswith('date-released'):
            line = f"date-released: {date}\n"
        elif line.startswith(f'  {RELEASES_BUCKET}/?prefix='):
            line = f'  {RELEASES_BUCKET}/?prefix=invest/{version}\n'
        elif line.startswith(f'    value: {PREFIX}'):
            line = f'    value: {PREFIX}/natcap-invest-{version}\n'

        lines.append(line)

    with codecs.open(citation_filepath, 'w', encoding='utf-8') as citation_file:
        citation_file.writelines(lines)


def main():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description="A script to update CITATION.cff for an impending release."
    )
    parser.add_argument(
        'citation_cff_file', help="The path to a citation.cff file.")
    parser.add_argument(
        'version', help=("The version string of the version being released."))
    parser.add_argument(
        'date', help=('The date of the release, in the form YYYY-MM-DD'))

    args = parser.parse_args()

    return (args.citation_cff_file, args.version, args.date)


if __name__ == '__main__':
    citation_file_path, version, date = main()
    update_citation_file(citation_file_path, version, date)
