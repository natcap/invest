"""RasterFactory Class."""

import random

import numpy as np

from natcap.invest.coastal_blue_carbon.classes.affine import Affine
from natcap.invest.coastal_blue_carbon.classes.raster import Raster


class RasterFactory(object):

    def __init__(self, proj, datatype, nodata_val, rows, cols, affine=Affine.identity):
        self.proj = proj
        self.datatype = datatype
        self.nodata_val = nodata_val
        self.rows = rows
        self.cols = cols
        self.affine = affine

    def get_metadata(self):
        meta = {}
        meta['proj'] = self.proj
        meta['datatype'] = self.datatype
        meta['nodata_val'] = self.nodata_val
        meta['rows'] = self.rows
        meta['cols'] = self.cols
        meta['affine'] = self.affine
        return meta

    def _create_raster(self, array):
        return Raster.from_array(
            array, self.affine, self.proj, self.datatype, self.nodata_val)

    def uniform(self, val):
        a = np.ones((self.rows, self.cols)) * val
        return self._create_raster(a)

    def alternating(self, val1, val2):
        a = np.ones((self.rows, self.cols)) * val2
        a[::2, ::2] = val1
        a[1::2, 1::2] = val1
        return self._create_raster(a)

    def random(self):
        a = np.random.rand(self.rows, self.cols)
        return self._create_raster(a)

    def random_from_list(self, l):
        a = np.zeros((self.rows, self.cols))
        for i in xrange(len(a)):
            for j in xrange(len(a[0])):
                a[i, j] = random.choice(l)
        return self._create_raster(a)

    def horizontal_ramp(self, val1, val2):
        a = np.zeros((self.rows, self.cols))
        col_vals = np.linspace(val1, val2, self.cols)
        a[:] = col_vals
        return self._create_raster(a)

    def vertical_ramp(self, val1, val2):
        a = np.zeros((self.cols, self.rows))
        row_vals = np.linspace(val1, val2, self.rows)
        a[:] = row_vals
        a = a.T
        return self._create_raster(a)
