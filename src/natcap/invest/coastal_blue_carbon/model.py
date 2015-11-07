# -*- coding: utf-8 -*-
"""Coastal Blue Carbon Model.
"""
import os
import shutil
import logging

import numpy as np

from natcap.invest.coastal_blue_carbon import NODATA_INT, NODATA_FLOAT, HA_PER_M2
from natcap.invest.coastal_blue_carbon.utils import reclass, \
    reclass_transition, add, add_inplace, sub, mul, mul_scalar, zeros, copy, \
    compute_E_pt, compute_V_t, compute_T_b

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')
LOGGER = logging.getLogger('natcap.invest.coastal_blue_carbon.model')


def init_SLT(d):
    """Set Initial Conditions: Create S_p0, L_0, N, V, T_0; Save T_0.

    Parameters:
        d (dict): data dictionary
    """
    LOGGER.info("Setting Initial Conditions...")
    d.S_pb.biomass.append(reclass(d.C_s[0], d.lulc_to_Sb, out_dtype=np.float32))
    d.S_pb.soil.append(reclass(d.C_s[0], d.lulc_to_Ss, out_dtype=np.float32))
    d.L_s.append(reclass(d.C_s[0], d.lulc_to_L, out_dtype=np.float32))
    d.T_b.append(copy(d.L_s[0]))
    d.T_b[0] = add_inplace(d.T_b[0], d.S_pb.biomass[0])
    d.T_b[0] = add_inplace(d.T_b[0], d.S_pb.soil[0])
    d.N = zeros(d.C_s[0], dtype=np.float32)
    d.V = zeros(d.C_s[0], dtype=np.float32)


def reclass_C_to_YDH(d, r):
    """Reclass C to Y_pr, D_pr, H_pr.

    Parameters:
        d (dict): data dictionary
        r (int): transition index
    """
    d.Y_pr.biomass.append(
        reclass(d.C_s[r+1], d.lulc_to_Yb, out_dtype=np.float32))
    d.Y_pr.soil.append(
        reclass(d.C_s[r+1], d.lulc_to_Ys, out_dtype=np.float32))
    d.D_pr.biomass.append(
        mul(reclass_transition(
                d.C_s[r], d.C_s[r+1], d.lulc_trans_to_Db, out_dtype=np.float32),
            d.S_pb.biomass[-1]))
    d.D_pr.soil.append(
        mul(reclass_transition(
                d.C_s[r], d.C_s[r+1], d.lulc_trans_to_Ds, out_dtype=np.float32),
            d.S_pb.soil[-1]))
    d.H_pr.biomass.append(
        reclass(d.C_s[r], d.lulc_to_Hb, out_dtype=np.float32))
    d.H_pr.soil.append(
        reclass(d.C_s[r], d.lulc_to_Hs, out_dtype=np.float32))


def compute_AENSLT(d, r):
    """Compute A_pr, E_pr, N_pr; Save A_r, E_r, N_r, T_b.

    Parameters:
        d (dict): data dictionary
        r (int): transition index
    """
    # Compute A_pr
    timesteps = d.border_year_list[r+1] - d.border_year_list[r]
    d.A_pr.biomass.append(mul_scalar(d.Y_pr.biomass[-1], timesteps))
    d.A_pr.soil.append(mul_scalar(d.Y_pr.soil[-1], timesteps))

    # Compute E_pr
    E_total_biomass = zeros(d.C_s[0], dtype=np.float32)
    E_total_soil = zeros(d.C_s[0], dtype=np.float32)
    for i in range(0, r+1):
        offset_t = d.border_year_list[r] - d.border_year_list[i]
        end_t = d.border_year_list[r+1] - d.border_year_list[i]
        E_r_biomass = compute_E_pt(
            d.D_pr.biomass[i], d.H_pr.biomass[i], offset_t, end_t)
        E_r_soil = compute_E_pt(
            d.D_pr.soil[i], d.H_pr.soil[i], offset_t, end_t)
        E_total_biomass = add_inplace(E_total_biomass, E_r_biomass)
        E_total_soil = add_inplace(E_total_soil, E_r_soil)
    d.E_pr.biomass.append(E_total_biomass)
    d.E_pr.soil.append(E_total_soil)

    # Compute N_pr
    d.N_pr.biomass.append(sub(d.A_pr.biomass[-1], d.E_pr.biomass[-1]))
    d.N_pr.soil.append(sub(d.A_pr.soil[-1], d.E_pr.soil[-1]))

    # Compute N_r
    d.N_r.append(add(d.N_pr.biomass[-1], d.N_pr.soil[-1]))

    # Compute S_pb
    d.S_pb.biomass.append(add(d.S_pb.biomass[-1], d.N_pr.biomass[-1]))
    d.S_pb.soil.append(add(d.S_pb.soil[-1], d.N_pr.soil[-1]))

    # Compute L_s
    d.L_s.append(reclass(d.C_s[r+1], d.lulc_to_L, out_dtype=np.float32))

    # Compute T_b
    d.T_b.append(compute_T_b(d.T_b[r], d.N_r[-1], d.L_s[-2], d.L_s[-1]))


def compute_transition(d, r):
    """Compute transition.

    Parameters:
        d (dict): data dictionary
        r (int): transition index
    """
    LOGGER.info("Computing transition %i..." % r)
    reclass_C_to_YDH(d, r)
    compute_AENSLT(d, r)


def compute_timestep_AENV(d, r, t):
    """Compute A_pt, E_pt, N_t, V_t, V.

    Parameters:
        d (dict): data dictionary
        r (int): transition index
        t (int): timestep index
    """
    A_t_total = add(copy(d.Y_pr.biomass[r]), copy(d.Y_pr.soil[r]))

    E_t_total = zeros(A_t_total)
    for i in xrange(0, r+1):
        trans_offset = d.border_year_list[i] - d.border_year_list[0]
        diff = t - trans_offset
        E_pt_biomass_r = compute_E_pt(
            d.D_pr.biomass[i], d.H_pr.biomass[i], diff, diff+1)
        E_pt_soil_r = compute_E_pt(
            d.D_pr.soil[i], d.H_pr.soil[i], diff, diff+1)
        E_t_total = add_inplace(E_t_total, E_pt_biomass_r)
        E_t_total = add_inplace(E_t_total, E_pt_soil_r)

        # Clean up
        del E_pt_biomass_r
        del E_pt_soil_r

    year = d.border_year_list[0] + t
    N_t = sub(A_t_total, E_t_total)
    d.N = add(d.N, N_t)
    V_t = compute_V_t(N_t, d.price_t[year])
    d.V = add(d.V, V_t)

    # Clean up
    del A_t_total
    del E_t_total
    del N_t
    del V_t


def compute_NV_across_timesteps(d, r):
    """Compute N, V across timesteps of transition.

    Parameters:
        d (dict): data dictionary
        r (int): transition index
    """
    LOGGER.info("Computing NV across timesteps...")
    offset = d.border_year_list[r] - d.border_year_list[0]
    diff = d.border_year_list[r+1] - d.border_year_list[r]
    for j in xrange(0, diff):
        LOGGER.info("Computing NV for year %i..." % (d.border_year_list[r] + j))
        compute_timestep_AENV(d, r, j+offset)


def save_TAENV(d):
    """Save T_b, A_r, E_r, N_r, V.

    Parameters:
        d (dict): data dictionary
    """
    LOGGER.info("Saving rasters...")

    # Total Carbon Stock
    for i in range(0, len(d.T_b)):
        fn = 'carbon_stock_at_%s.tif' % d.border_year_list[i]
        dst_filepath = os.path.join(d.outputs_dir, fn)
        shutil.copyfile(d.T_b[i], dst_filepath)

    # Transition Accumulation
    for i in range(0, len(d.A_pr.biomass)):
        A_r = add(d.A_pr.biomass[i], d.A_pr.soil[i])
        fn = 'carbon_accumulation_between_%s_and_%s.tif' % (
            d.border_year_list[i], d.border_year_list[i+1])
        dst_filepath = os.path.join(d.outputs_dir, fn)
        shutil.copyfile(A_r, dst_filepath)

    # Transition Emissions
    for i in range(0, len(d.E_pr.biomass)):
        E_r = add(d.E_pr.biomass[i], d.E_pr.soil[i])
        fn = 'carbon_emissions_between_%s_and_%s.tif' % (
            d.border_year_list[i], d.border_year_list[i+1])
        dst_filepath = os.path.join(d.outputs_dir, fn)
        shutil.copyfile(E_r, dst_filepath)

    # Transition Net Sequestration
    for i in range(0, len(d.N_pr.biomass)):
        N_r = add(d.N_pr.biomass[i], d.N_pr.soil[i])
        fn = 'net_carbon_sequestration_between_%s_and_%s.tif' % (
            d.border_year_list[i], d.border_year_list[i+1])
        dst_filepath = os.path.join(d.outputs_dir, fn)
        shutil.copyfile(N_r, dst_filepath)

    # Net Sequestration from Base Year to Analysis Year
    if d.do_economic_analysis:
        fn = 'net_carbon_sequestration_between_%s_and_%s.tif' % (
            d.border_year_list[0], d.border_year_list[-1])
        dst_filepath = os.path.join(d.outputs_dir, fn)
        shutil.copyfile(d.N, dst_filepath)

        # Total Net Present Value
        fn = 'net_present_value.tif'
        dst_filepath = os.path.join(d.outputs_dir, fn)
        shutil.copyfile(d.V, dst_filepath)

    LOGGER.info("...rasters saved.")


def run(d):
    """Run model.

    Parameters:
        d (dict): data dictionary
    """
    init_SLT(d)
    LOGGER.info("Running Transient Analysis...")
    for r in range(0, len(d.C_s)-1):
        compute_transition(d, r)
        if d.do_economic_analysis:
            compute_NV_across_timesteps(d, r)
    save_TAENV(d)
    return d
