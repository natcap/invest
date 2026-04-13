from pathlib import Path

from setuptools.build_meta import *
from setuptools.build_meta import prepare_metadata_for_build_wheel as _prepare_metadata_for_build_wheel


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    # Call the regular setuptools function to build metadata
    metadata = _prepare_metadata_for_build_wheel(metadata_directory, config_settings)

    # Modify the METADATA file to add the additional dependencies
    # by inserting an extra Requires-Dist line
    path = Path(metadata_directory / metadata / 'METADATA')
    lines = path.read_text().splitlines()
    idx = next(i for i, line in enumerate(lines) if line.startswith('Requires-Dist'))
    lines.insert(idx, 'Requires-Dist: gdal==3.10.*\n')
    path.write_text(''.join(lines))

    return metadata
