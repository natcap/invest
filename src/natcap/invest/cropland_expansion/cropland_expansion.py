"""Cropland Expansion Tool"""

import os
import logging
import numpy

import gdal
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
        _expand_from_ag(args, intermediate_dir, file_suffix)

    if args['expand_from_forest_edge']:
        _expand_from_forest_edge(args)

    if args['fragment_forest']:
        _fragment_forest(args)

def _expand_from_ag(args, intermediate_dir, file_suffix):
    """ """
    #mask agriculture types from LULC
    ag_mask_uri = os.path.join(intermediate_dir, 'ag_mask%s.tif' % file_suffix)

    lulc_nodata = pygeoprocessing.get_nodata_from_uri(
        args['base_lulc_uri'])
    pixel_size_out = pygeoprocessing.get_cell_size_from_uri(
        args['base_lulc_uri'])
    ag_mask_nodata = 2
    ag_lucode = int(args['agriculture_type'])
    def _mask_ag_op(lulc):
        """create a mask of ag pixels only"""
        ag_mask = (lulc == ag_lucode)
        return numpy.where(lulc == lulc_nodata, ag_mask_nodata, ag_mask)

    pygeoprocessing.vectorize_datasets(
        [args['base_lulc_uri']], _mask_ag_op, ag_mask_uri, gdal.GDT_Byte,
        ag_mask_nodata, pixel_size_out, "intersection", vectorize_op=False)

    #distance transform mask
    distance_from_ag_uri = os.path.join(
        intermediate_dir, 'distance_from_ag%s.tif' % file_suffix)
    pygeoprocessing.distance_transform_edt(ag_mask_uri, distance_from_ag_uri)

    #mask out distance transform for everything that can be converted
    convertable_type_list = numpy.array([
        int(x) for x in args['convertable_landcover_types'].split()])

    convertable_type_nodata = -1
    convertable_distances_uri = os.path.join(
        intermediate_dir, 'convertable_distances%s.tif' % file_suffix)
    def _mask_to_convertable_types(distance_from_ag, lulc):
        """masks out the distance transform to a set of given landcover codes"""
        convertable_mask = numpy.in1d(
            lulc.flatten(), convertable_type_list).reshape(lulc.shape)
        return numpy.where(
            convertable_mask, distance_from_ag, convertable_type_nodata)

    pygeoprocessing.vectorize_datasets(
        [distance_from_ag_uri, args['base_lulc_uri']],
        _mask_to_convertable_types, convertable_distances_uri, gdal.GDT_Float32,
        convertable_type_nodata, pixel_size_out, "intersection",
        vectorize_op=False)


    #disk sort to select the top N pixels to convert

def _expand_from_forest_edge(args):
    """ """
    pass


def _fragment_forest(args):
    """ """
    pass