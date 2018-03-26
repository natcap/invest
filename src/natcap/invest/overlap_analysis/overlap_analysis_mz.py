'''
This is the preperatory class for the management zone portion of overlap
analysis.
'''
from __future__ import absolute_import
import os

from osgeo import gdal

from natcap.invest.overlap_analysis import overlap_analysis_mz_core
from natcap.invest.overlap_analysis import overlap_core
from .. import validation
from .. import utils



def execute(args):
    """Overlap Analysis: Management Zones.

    Parameters:
        args: A python dictionary created by the UI and passed to this
            method. It will contain the following data.
        args['workspace_dir'] (string): The directory in which to place all
            resulting files, will come in as a string. (required)
        args['zone_layer_loc'] (string): A URI pointing to a shapefile with
            the analysis zones on it. (required)
        args['overlap_data_dir_loc'] (string): URI pointing to a directory
            where multiple shapefiles are located. Each shapefile represents
            an activity of interest for the model. (required)

    Returns:
        ``None``
    """

    mz_args = {}

    workspace = args['workspace_dir']
    output_dir = workspace + os.sep + 'output'
    inter_dir = workspace + os.sep + 'intermediate'

    if not (os.path.exists(output_dir)):
        os.makedirs(output_dir)

    if not (os.path.exists(inter_dir)):
        os.makedirs(inter_dir)

    mz_args['workspace_dir'] = args['workspace_dir']

    #We are passing in the AOI shapefile, as well as the dimension that we want
    #the raster pixels to be.
    mz_args['zone_layer_file'] = gdal.OpenEx(args['zone_layer_loc'])

    file_dict = overlap_core.get_files_dict(args['overlap_data_dir_loc'])

    mz_args['over_layer_dict'] = file_dict

    overlap_analysis_mz_core.execute(mz_args)


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate an input dictionary for OA:MZ.

    Parameters:
        args (dict): The args dictionary.
        limit_to=None (str or None): If a string key, only this args parameter
            will be validated.  If ``None``, all args parameters will be
            validated.

    Returns:
        A list of tuples where tuple[0] is an iterable of keys that the error
        message applies to and tuple[1] is the string validation warning.
    """
    warnings = []
    keys_missing = []
    keys_without_value = []
    for required_key in ('workspace_dir', 'zone_layer_loc',
                         'overlap_data_dir_loc'):
        try:
            if args[required_key] in ('', None):
                keys_without_value.append(required_key)
        except KeyError:
            keys_missing.append(required_key)

    if len(keys_missing) > 0:
        raise KeyError('Args is missing these keys: %s'
                       % ', '.join(keys_missing))

    if len(keys_without_value) > 0:
        warnings.append((keys_without_value,
                         'Parameter must have a value.'))

    if limit_to in ('zone_layer_loc', None):
        with utils.capture_gdal_logging():
            vector = gdal.OpenEx(args['zone_layer_loc'])
            if vector is None:
                warnings.append((['zone_layer_loc'],
                                 ('Parameter must be a path to an '
                                  'OGR-compatible file on disk.')))

    if limit_to in ('overlap_data_dir_loc', None):
        if not os.path.isdir(args['overlap_data_dir_loc']):
            warnings.append((['overlap_data_dir_loc'],
                             'Parameter must be a path to a folder on disk.'))

    return warnings
