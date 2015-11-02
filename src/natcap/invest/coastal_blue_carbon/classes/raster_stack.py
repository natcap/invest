"""RasterStack Class."""

import os
import shutil
import logging

try:
    import gdal
    import ogr
    import osr
except ImportError:
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr

import numpy as np
from shapely.geometry import Polygon
import shapely
import pygeoprocessing as pygeo
from pygeoprocessing import geoprocessing as geoprocess

from natcap.invest.coastal_blue_carbon.classes.raster import Raster
from natcap.invest.coastal_blue_carbon.classes.affine import Affine

LOGGER = logging.getLogger('natcap.invest.coastal_blue_carbon.classes.raster_stack')
logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H/%M/%S')


class RasterStack(object):

    """A class for interacting with gdal raster files."""

    def __init__(self, raster_list):
        self.raster_list = raster_list

    def __del__(self):
        for i in self.raster_list:
            del i

    def __str__(self):
        string =  '\nRasterStack'
        for i in self.raster_list:
            string += '\n  ' + i.uri
        return string

    def get_raster_uri_list(self):
        """Get Raster URI List."""
        return [r.uri for r in self.raster_list]

    def assert_same_projection(self):
        """Assert Rasters in Same Projection."""
        return all(self.raster_list[0].get_projection() == r.get_projection() \
            for r in self.raster_list)

    def all_same_projection(self):
        """Check if Rasters in Same Projection."""
        return all(self.raster_list[0].get_projection() == r.get_projection() \
            for r in self.raster_list)

    def assert_same_alignment(self):
        """Assert Rasters in Same Alignment."""
        return all(self.raster_list[0].get_affine() == r.get_affine() \
            for r in self.raster_list)

    def all_aligned(self):
        """Check if All Rasters are Aligned."""
        return all(self.raster_list[0].get_affine() == r.get_affine() \
            for r in self.raster_list)

    def assert_resample_methods_assigned(self):
        try:
            for r in self.raster_list:
                r.get_resample_method()
            return True
        except AttributeError:
            return False

    def align(self):
        """Align Rasters."""
        if not self.assert_resample_methods_assigned():
            raise ValueError("Resample method not assigned to at least one raster.")
        dataset_uri_list = self.get_raster_uri_list()
        dataset_out_uri_list = \
            [geoprocess.temporary_filename() for _ in self.raster_list]
        resample_method_list = \
            [r.get_resample_method() for r in self.raster_list]
        out_pixel_size = geoprocess.get_cell_size_from_uri(
            self.raster_list[0].uri)
        mode = 'dataset'
        dataset_to_align_index = 0

        raster_uri_list = geoprocess.align_dataset_list(
            dataset_uri_list,
            dataset_out_uri_list,
            resample_method_list,
            out_pixel_size,
            mode,
            dataset_to_align_index,
            dataset_to_bound_index=0)

        new_raster_list = [Raster.from_file(fp) for fp in dataset_out_uri_list]
        return RasterStack(new_raster_list)

    def set_standard_nodata(self, nodata_val):
        new_raster_list = []
        for r in self.raster_list:
            new_raster_list.append(r.set_nodata(nodata_val))
        return RasterStack(new_raster_list)
