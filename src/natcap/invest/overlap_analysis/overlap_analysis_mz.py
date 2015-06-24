'''
This is the preperatory class for the management zone portion of overlap
analysis.
'''
import os

from osgeo import ogr

from natcap.invest.overlap_analysis import overlap_analysis_mz_core
from natcap.invest.overlap_analysis import overlap_core


def execute(args):
    '''
    Input:
        args: A python dictionary created by the UI and passed to this method.
            It will contain the following data.
        args['workspace_dir']- The directory in which to place all resulting
            files, will come in as a string.
        args['zone_layer_loc']- A URI pointing to a shapefile with the analysis
            zones on it.
        args['overlap_data_dir_loc']- URI pointing to a directory where
            multiple shapefiles are located. Each shapefile represents an
            activity of interest for the model.
    Output:
        mz_args- The dictionary of all arguments that are needed by the
            overlap_analysis_mz_core.py class. This list of processed inputs
            will be directly passed to the core in order to create model
            outputs.
    '''

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
