# -*- coding: utf-8 -*-
"""Utility Functions."""
import os
import shutil

import gdal
import numpy as np
from pygeoprocessing import geoprocessing as geoprocess


def reclass(a, reclass_dict, mask_other_vals=False, out_dtype=None):
    """Reclass values in array."""
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
                a[array == i] = np.ma.masked
        for item in reclass_dict.items():
            a[array == item[0]] = item[1]
        if isinstance(array, np.ma.masked_array) or mask_other_vals:
            return a
        else:
            return a.data
    raster_out = local_op_x1(reclass_op, a)
    return raster_out

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
    raster_out = local_op_x2(reclass_transition_op, a_prev, a_next)
    return raster_out

def local_op_x1(local_op, a):
    """Local operation."""
    raster_list = [a]

    nodata_val = geoprocess.get_nodata_from_uri(a)
    pixel_size_out = geoprocess.get_cell_size_from_uri(a)
    bounding_box_mode = 'intersection'

    raster_out = geoprocess.temporary_filename()
    datatype_out = gdal.GDT_Float32
    nodata_out = -9999

    geoprocess.vectorize_datasets(
        raster_list,
        local_op,
        raster_out,
        datatype_out,
        nodata_out,
        pixel_size_out,
        bounding_box_mode,
        assert_datasets_projected=False,
        vectorize_op=False)

    return raster_out

def local_op_x2(local_op, a, b):
    """Local operation."""
    raster_list = [a, b]

    nodata_val = geoprocess.get_nodata_from_uri(a)
    pixel_size_out = geoprocess.get_cell_size_from_uri(a)
    bounding_box_mode = 'intersection'

    raster_out = geoprocess.temporary_filename()
    datatype_out = gdal.GDT_Float32
    nodata_out = -9999

    geoprocess.vectorize_datasets(
        raster_list,
        local_op,
        raster_out,
        datatype_out,
        nodata_out,
        pixel_size_out,
        bounding_box_mode,
        assert_datasets_projected=False,
        vectorize_op=False)

    return raster_out

def local_op_x4(local_op, a, b, c, d):
    """Local operation."""
    raster_list = [a, b, c, d]

    nodata_val = geoprocess.get_nodata_from_uri(a)
    pixel_size_out = geoprocess.get_cell_size_from_uri(a)
    bounding_box_mode = 'intersection'

    raster_out = geoprocess.temporary_filename()
    datatype_out = gdal.GDT_Float32
    nodata_out = -9999

    geoprocess.vectorize_datasets(
        raster_list,
        local_op,
        raster_out,
        datatype_out,
        nodata_out,
        pixel_size_out,
        bounding_box_mode,
        assert_datasets_projected=False,
        vectorize_op=False)

    return raster_out

def add(a, b):
    """Add two rasters together."""
    def add_op(a1, a2):
        a1 = np.ma.masked_array(a1)
        a2 = np.ma.masked_array(a2)
        a1[a1 == -9999] = np.ma.masked
        a2[a2 == -9999] = np.ma.masked
        return a1 + a2
    raster_out = local_op_x2(add_op, a, b)
    return raster_out

def add_inplace(a, b):
    """Add second raster to first raster."""
    def add_inplace_op(a1, a2):
        a1 = np.ma.masked_array(a1)
        a2 = np.ma.masked_array(a2)
        a1[a1 == -9999] = np.ma.masked
        a2[a2 == -9999] = np.ma.masked
        return a1 + a2
    raster_out = local_op_x2(add_inplace_op, a, b)
    os.remove(a)

    return raster_out

def sub(a, b):
    """Subtract second raster from first raster."""
    def sub_op(a1, a2):
        a1 = np.ma.masked_array(a1)
        a2 = np.ma.masked_array(a2)
        a1[a1 == -9999] = np.ma.masked
        a2[a2 == -9999] = np.ma.masked
        return a1 - a2
    raster_out = local_op_x2(sub_op, a, b)
    return raster_out

def mul(a, b):
    """Multiply two rasters together."""
    def mul_op(a1, a2):
        a1 = np.ma.masked_array(a1)
        a2 = np.ma.masked_array(a2)
        a1[a1 == -9999] = np.ma.masked
        a2[a2 == -9999] = np.ma.masked
        return a1 * a2
    raster_out = local_op_x2(mul_op, a, b)
    return raster_out

def mul_scalar(a, scalar):
    """Multiply scalar operation."""
    def mul_scaler_op(a):
        a = np.ma.masked_array(a)
        a[a == -9999] = np.ma.masked
        return a * scalar
    raster_out = local_op_x1(mul_scaler_op, a)
    return raster_out

def div(a, b):
    """Divide first raster by second raster."""
    def div_op(a1, a2):
        a1 = np.ma.masked_array(a1)
        a2 = np.ma.masked_array(a2)
        a1[a1 == -9999] = np.ma.masked
        a2[a2 == -9999] = np.ma.masked
        return a1 / a2
    raster_out = local_op_x2(div_op, a, b)
    return raster_out

def pow(a, b):
    """Raise elements of first raster to power of elements in second
    raster."""
    def pow_op(a1, a2):
        a1 = np.ma.masked_array(a1)
        a2 = np.ma.masked_array(a2)
        a1[a1 == -9999] = np.ma.masked
        a2[a2 == -9999] = np.ma.masked
        return a1 ** a2
    raster_out = local_op_x2(pow_op, a, b)
    return raster_out

def zeros(a, dtype=np.float32):
    """Create a raster of zeros."""
    def zeros_op(a):
        return np.zeros(a.shape).astype(np.float32)
    raster_out = local_op_x1(zeros_op, a)
    return raster_out

def copy(src):
    """Copy raster."""
    dst = geoprocess.temporary_filename()
    shutil.copy(src, dst)
    return dst

def nodata_to_zeros(a):
    """Nodata to zeros."""
    nodata_val = geoprocess.get_nodata_from_uri(a)
    def nodata_to_zeros_op(a):
        a[a == nodata_val] = 0
        return a
    raster_out = local_op_x1(nodata_to_zeros_op, a)
    return raster_out

def compute_E_pt(D_pr, H_pr, offset_t, end_t):
    """Compute emissions."""
    nodata_val = geoprocess.get_nodata_from_uri(D_pr)

    def compute_E_pt_op(D_pr, H_pr):
        D_pr = np.ma.masked_array(D_pr)
        H_pr = np.ma.masked_array(H_pr)
        D_pr[D_pr == -9999] = np.ma.masked
        H_pr[H_pr == -9999] = np.ma.masked
        a = D_pr * ((0.5**((offset_t)/H_pr)) - (0.5**((end_t)/H_pr)))
        a.fill_value = 0
        a = np.ma.fix_invalid(a)
        a[a == a.mask] = 0
        return a

    raster_out = local_op_x2(compute_E_pt_op, D_pr, H_pr)
    return raster_out

def compute_V_t(N_t, price_t):
    """Compute Net Present Value."""
    nodata_val = geoprocess.get_nodata_from_uri(N_t)

    def compute_V_t_op(N_t):
        N_t = np.ma.masked_array(N_t)
        N_t[N_t == -9999] = np.ma.masked
        a = N_t * price_t
        a.fill_value = 0
        a = np.ma.fix_invalid(a)
        a[a == a.mask] = 0
        return a

    raster_out = local_op_x1(compute_V_t_op, N_t)
    return raster_out

def compute_T_b(T_b, N_r, L_s_0, L_s_1):
    """Compute Total Stock."""
    nodata_val = geoprocess.get_nodata_from_uri(T_b)

    def compute_T_b_op(T_b, N_r, L_s_0, L_s_1):
        T_b = np.ma.masked_array(T_b)
        N_r = np.ma.masked_array(N_r)
        L_s_0 = np.ma.masked_array(L_s_0)
        L_s_1 = np.ma.masked_array(L_s_1)
        T_b[T_b == -9999] = np.ma.masked
        N_r[N_r == -9999] = np.ma.masked
        L_s_0[L_s_0 == -9999] = np.ma.masked
        L_s_1[L_s_1 == -9999] = np.ma.masked
        a = T_b + N_r - L_s_0 + L_s_1
        a.fill_value = 0
        a = np.ma.fix_invalid(a)
        a[a == a.mask] = 0
        return a

    raster_out = local_op_x4(compute_T_b_op, T_b, N_r, L_s_0, L_s_1)
    return raster_out
