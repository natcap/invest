#!python

import os
import argparse


def main(userguide_dir):
    """Check existence of files referenced in ``natcap.invest.MODEL_METADATA``.

    Parameters:
        userguide_dir (string): path to the local userguide repository.

    Returns:
        None

    Raises:
        OSError if any models reference files that do not exist.
    """
    from natcap.invest.model_metadata import MODEL_METADATA

    missing_files = []
    userguide_dir_source = os.path.join(userguide_dir, 'source')

    for data in MODEL_METADATA.values():
        # html referenced won't exist unless we actually built the UG,
        # so check for the rst with the same basename.
        model_rst = f'{os.path.splitext(data.userguide)[0]}.rst'
        if not os.path.exists(os.path.join(
                userguide_dir_source, model_rst)):
            missing_files.append(data.userguide)
    if missing_files:
        raise ValueError(
            f'the following models do not have a corresponding rst '
            f'file in {userguide_dir_source}.\n'
            f'{missing_files}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Validate all userguide links in "
                    "natcap.invest.MODEL_METADATA.")
    parser.add_argument('userguidedir', type=str)
    args = parser.parse_args()
    main(args.userguidedir)
