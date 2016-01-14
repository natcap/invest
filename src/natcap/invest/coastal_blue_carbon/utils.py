# -*- coding: utf-8 -*-
"""Utility Functions."""
import os
import shutil

import gdal
import numpy as np
from pygeoprocessing import geoprocessing as geoprocess

# Utils
def reclass(a, reclass_dict, mask_other_vals=True, out_dtype=None):
    """Reclass values in array.

    Should set values not in reclass_dict to NaN. (Currently does not)
    """
    def reclass_op(array):
        a = np.ma.masked_array(np.copy(array))
        if out_dtype:
            a = a.astype(out_dtype)
        if mask_other_vals:
            u = list(np.unique(a.data))
            other_vals = []
            for i in u:
                if i not in reclass_dict:
                    other_vals.append(i)
            for i in other_vals:
                a[array == i] = np.nan #np.ma.masked
        for item in reclass_dict.items():
            a[array == item[0]] = item[1]
        if isinstance(array, np.ma.masked_array) or mask_other_vals:
            return a
        else:
            return a.data
    return reclass_op(a)

def reclass_transition(a_prev, a_next, trans_dict, out_dtype=None):
    """Reclass arrays based on element-wise combinations between two arrays."""
    def reclass_transition_op(a_prev, a_next):
        a = a_prev.flatten()
        b = a_next.flatten()
        c = np.ma.masked_array(np.zeros(a.shape))
        if out_dtype:
            c = c.astype(out_dtype)
        z = zip(a, b)
        for i in range(0, len(z)):
            if z[i] in trans_dict:
                c[i] = trans_dict[z[i]]
            else:
                c[i] = np.ma.masked
        return c.reshape(a_prev.shape)
    return reclass_transition_op(a_prev, a_next)

def write_to_raster(output_raster, array, xoff, yoff):
    ds = gdal.Open(output_raster, gdal.GA_Update)
    band = ds.GetRasterBand(1)
    band.WriteArray(array, xoff, yoff)
    ds.FlushCache()
    ds = None

def read_from_raster(input_raster, offset_block):
    ds = gdal.Open(input_raster)
    band = ds.GetRasterBand(1)
    a = band.ReadAsArray(**offset_block)
    ds.FlushCache()
    ds = None
    return a
