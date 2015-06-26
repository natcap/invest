"""Cropland Expansion Tool"""

import os
import logging

import pygeoprocessing

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger(
    'natcap.invest.cropland_expansion.cropland_expansion')

def execute(args):
    """Main entry point for cropland expansion tool model.

        args['workspace_dir'] - (string) output directory for intermediate,
            temporary, and final files
        args['results_suffix'] - (optional) (string) string to append to any
            output files
        args['base_lulc_uri'] - (string)

    """
    #append a _ to the suffix if it's not empty and doens't already have one
    try:
        file_suffix = args['results_suffix']
        if file_suffix != "" and not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''

    #create working directories
    output_dir = os.path.join(args['workspace_dir'], 'output')
    intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')
    tmp_dir = os.path.join(args['workspace_dir'], 'tmp')

    pygeoprocessing.geoprocessing.create_directories(
        [output_dir, intermediate_dir, tmp_dir])

    if args['expand_from_ag']:
        _expand_from_ag(args)

    if args['expand_from_forest_edge']:
        _expand_from_forest_edge(args)

    if args['fragment_forest']:
        _fragment_forest(args)

def _expand_from_ag(args):
    """ """
    pass

def _expand_from_forest_edge(args):
    """ """
    pass


def _fragment_forest(args):
    """ """
    pass