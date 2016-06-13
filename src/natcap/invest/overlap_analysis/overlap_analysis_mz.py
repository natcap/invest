'''
This is the preperatory class for the management zone portion of overlap
analysis.
'''
import os

from osgeo import ogr

from natcap.invest.overlap_analysis import overlap_analysis_mz_core
from natcap.invest.overlap_analysis import overlap_core


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
    mz_args['zone_layer_file'] = ogr.Open(args['zone_layer_loc'])

    file_dict = overlap_core.get_files_dict(args['overlap_data_dir_loc'])

    mz_args['over_layer_dict'] = file_dict

    overlap_analysis_mz_core.execute(mz_args)
