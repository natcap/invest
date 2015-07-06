"""
(About Blue Carbon IO)
"""

import logging
import os
import pprint as pp

from osgeo import gdal, osr

from natcap.invest import raster_utils

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('blue_carbon')


def fetch_args(args):
    '''
    Fetches inputs for blue carbon model and returns dictionary of variables

    Example Args::

        args = {
            'workspace_dir': '/path/to/workspace_dir/',
            'lulc_uri_1': '/path/to/lulc_uri_1',
            'year_1': 2004,
            'lulc_uri_2': '/path/to/lulc_uri_2',
            'year_2': 2050,
            'lulc_uri_3': '/path/to/lulc_uri_3',
            'year_3': 2100,
            'lulc_uri_4': '/path/to/lulc_uri_4',
            'analysis_year': 2150,
            'soil_disturbance_csv_uri': '/path/to/csv',
            'biomass_disturbance_csv_uri': '/path/to/csv',
            'carbon_pools_uri': '/path/to/csv',
            'half_life_csv_uri': '/path/to/csv',
            'transition_matrix_uri': '/path/to/csv',
            'do_private_valuation': True,
            'discount_rate': 5,
            'do_price_table': True,
            'carbon_schedule': '/path/to/csv',
            'carbon_value': 43.00,
            'rate_change': 0,
        }

    Example Returns::

        vars_dict = {
            # ... original args ...

            # Derived Variables
            'lulc_uri_dict': {
                '(lulc_year)': '(raster_uri)',
                ...
            },
            'lulc_years': [2004, 2050, 2100]
            'gdal_type_carbon': gdal.GDT_Float64,
            'cell_size': 1000,
            'nodata_default_int': -1,
            'nodata_default_float': -1,
            'nodata_lulc': 0,

            'half_lives_by_veg_dict': {
                '(id)': {
                    'biomass (years)': (float),
                    'soil (years)': (float),
                    'veg name': (string),
                    'veg type': (int)
                }
            },
            'half_life_field_key': 'veg_type',
            'half_life_field_bio': 'biomass (years)',
            'half_life_field_soil': 'soil (years)',

            'carbon_stock_by_veg_pool_id_dict': {
                (veg_type): {
                    'Above (Mg / ha)': {
                        (Id): (???), ...
                    }, ...
                }, ...
            },
            'veg_type_list': [0, 1, 2],
            'carbon_stock_soil_key': 'Soil (Mg / ha)',
            'carbon_stock_litter_key': 'Litter (Mg / ha)',
            'carbon_stock_biomass_key': 'biomass',

            'transition_accumulation_dict': {
                '(veg_type)': {
                    'biomass_accumulation_dict': {
                        (original_lulc, transition_lulc): (accumulation),
                        ...
                    },
                    'soil_accumulation_dict': {
                        (original_lulc, transition_lulc): (accumulation),
                        ...
                    },
                }, ...
            },

            'transition_disturbance_dict': {
                'biomass_disturbance_dict': {  # <-- derived from biomass_disturbance_csv_uri
                    (original_lulc, transition_lulc): (disturbance),
                    ...
                },
                'soil_disturbance_dict': {  # <-- derived from soil_disturbance_csv_uri
                    (original_lulc, transition_lulc): (disturbance),
                    ...
                }
            },

            # Intermediate Outputs
            'intermediate_dir': '/path/to/intermediate/',
            'veg_stock_bio_uris': '/path/to/%i_veg_%i_stock_biomass.tif',
            'veg_stock_soil_uris': '/path/to/%i_veg_%i_stock_soil.tif',
            'veg_litter_uris': '/path/to/%i_veg_%i_litter.tif',
            'acc_soil_uris': '/path/to/%i_acc_soil.tif',
            'acc_bio_uris': '/path/to/%i_acc_bio.tif',
            'veg_acc_bio_uris': '/path/to/%i_%i_veg_%i_acc_bio.tif',
            'veg_acc_soil_uris': '/path/to/%i_%i_veg_%i_acc_soil.tif',
            'veg_dis_bio_uris': '/path/to/%i_%i_veg_%i_dis_bio.tif',
            'veg_dis_soil_uris': '/path/to/%i_%i_veg_%i_dis_soil.tif',
            'dis_bio_uris': '/path/to/%i_dis_bio.tif',
            'dis_soil_uris': '/path/to/%i_dis_soil.tif',
            'veg_adj_acc_bio_uris': '/path/to/%i_%i_veg_%i_adj_acc_bio.tif',
            'veg_adj_acc_soil_uris': '/path/to/%i_%i_veg_%i_adj_acc_soil.tif',
            'veg_adj_dis_bio_uris': '/path/to/%i_%i_veg_%i_adj_dis_bio.tif',
            'veg_adj_dis_soil_uris': '/path/to/%i_%i_veg_%i_adj_dis_soil.tif',
            'veg_adj_em_dis_bio_uris': '/path/to/%i_%i_veg_%i_adj_em_dis_bio.tif',
            'veg_adj_em_dis_soil_uris': '/path/to/%i_%i_veg_%i_adj_em_dis_soil.tif',
            'veg_em_bio_uris': '/path/to/%i_%i_veg_%i_em_bio.tif',
            'veg_em_soil_uris': '/path/to/%i_%i_veg_%i_em_soil.tif',

            'this_total_acc_soil_uris': '/path/to/%i_%i_soil_acc.tif',
            'this_total_acc_bio_uris': '/path/to/%i_%i_bio_acc.tif',
            'this_total_dis_soil_uris': '/path/to/%i_%i_soil_dis.tif',
            'this_total_dis_bio_uris': '/path/to/%i_%i_bio_dis.tif',

            # Final Outputs
            'extent_uri': '/path/to/extent.shp',  # Bounding area of all input LULC maps
            'carbon_stock_uri': '/path/to/stock_%%i.tif',  # Carbon stock at time t
            'net_sequestration_uri': '/path/to/sequest_%i_%i.tif',  # Carbon sequested over given timeperiod
            'gain_uri': '/path/to/gain_%i_%i.tif',  # Only positive net sequestration
            'loss_uri': '/path/to/loss_%i_%i.tif',  # Only negative net sequestration
            'blue_carbon_csv_uri': '/path/to/sequestration.csv',  # Valuation table
        }

    '''
    vars_dict = dict(args.items())

    # Create Workspace
    workspace_dir = args['workspace_dir']
    intermediate_dir = os.path.join(args['workspace_dir'], "intermediate")
    vars_dict['intermediate_dir'] = intermediate_dir

    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)

    vars_dict = set_constants(vars_dict)

    # Fetch Inputs
    vars_dict = fetch_lulc_rasters(vars_dict)
    vars_dict = read_disturbance_csvs(vars_dict)
    vars_dict = read_half_life_csv(vars_dict)
    vars_dict = read_transition_matrix_csv(vars_dict)
    vars_dict = read_carbon_pools_csv(vars_dict)

    vars_dict = create_transition_matrices(vars_dict)

    # Validate Inputs
    _validate_rasters(vars_dict)

    # Define Outputs
    vars_dict = create_output_uris(vars_dict)

    return vars_dict


def fetch_lulc_rasters(vars_dict):
    '''
    Returns:
        lulc_uri_dict
        lulc_years
        conversion
    '''
    # Create list of LULC rasters
    lulc_list = []
    for i in range(1, 6):
        if "year_%i" % i in vars_dict:
            lulc_list.append(
                {"uri": vars_dict["lulc_uri_%i" % i], "year": vars_dict["year_%i" % i]})
        else:
            break
    lulc_uri_dict = dict([(lulc["year"], lulc["uri"]) for lulc in lulc_list])
    lulc_years = lulc_uri_dict.keys()
    lulc_years.sort()
    lulc_uri_dict[vars_dict["analysis_year"]] = lulc_uri_dict[lulc_years[-1]]
    vars_dict['lulc_uri_dict'] = lulc_uri_dict
    vars_dict['lulc_years'] = lulc_years

    vars_dict['conversion'] = (raster_utils.get_cell_size_from_uri(
        lulc_uri_dict[lulc_years[0]]) ** 2) / 10000.0  # convert to Ha
    LOGGER.debug("Cell size is %s hectacres.", vars_dict['conversion'])

    return vars_dict


def set_constants(vars_dict):
    '''

    Returns:
        gdal_type_carbon
        nodata_default_int
        nodata_default_float
        trans_acc
    '''
    vars_dict['gdal_type_carbon'] = gdal.GDT_Float64
    vars_dict['nodata_default_int'] = -1
    vars_dict['nodata_default_float'] = -1
    vars_dict['trans_acc'] = "Accumulation"

    return vars_dict


def read_disturbance_csvs(vars_dict):
    '''
    Returns:
        biomass_disturbance_dict (dict): descr

        soil_disturbance_dict (dict): descr
    '''
    disturbance_tables_key = "veg type"

    # soil_disturbance_csv_uri - dis_soil
    vars_dict['soil_disturbance_dict'] = raster_utils.get_lookup_from_csv(
        vars_dict["soil_disturbance_csv_uri"], disturbance_tables_key)
    for k in vars_dict['soil_disturbance_dict']:
        vars_dict['soil_disturbance_dict'][k][vars_dict['trans_acc']] = 0.0

    # biomass_disturbance_csv_uri - dis_bio
    vars_dict['biomass_disturbance_dict'] = raster_utils.get_lookup_from_csv(
        vars_dict["biomass_disturbance_csv_uri"], disturbance_tables_key)
    for k in vars_dict['biomass_disturbance_dict']:
        vars_dict['biomass_disturbance_dict'][k][vars_dict['trans_acc']] = 0.0

    return vars_dict


def read_half_life_csv(vars_dict):
    '''
    Returns:
        half_life_field_bio
        half_life_field_soil
        half_lives_by_veg_dict
    '''
    half_life_field_key = "veg type"
    vars_dict['half_life_field_bio'] = "biomass (years)"
    vars_dict['half_life_field_soil'] = "soil (years)"
    vars_dict['half_lives_by_veg_dict'] = raster_utils.get_lookup_from_csv(
        vars_dict['half_life_csv_uri'],
        half_life_field_key)

    return vars_dict


def read_transition_matrix_csv(vars_dict):
    '''
    Returns:
        transition_csv_dict
    '''
    vars_dict['transition_csv_dict'] = raster_utils.get_lookup_from_csv(
        vars_dict['transition_matrix_uri'], "Id")
    return vars_dict


def read_carbon_pools_csv(vars_dict):
    '''

    Returns:
        carbon_pools
        carbon_stock_by_veg_pool_id_dict
        veg_type_list
        carbon_stock_soil_key
        carbon_stock_biomass_key
        carbon_stock_litter_key
        biomass_accumulation_dict
        soil_accumulation_dict
    '''
    vars_dict['carbon_pools'] = raster_utils.get_lookup_from_csv(
        vars_dict['carbon_pools_uri'], "Id")

    carbon_stock_veg = "Veg Type"
    carbon_stock_above = "Above (Mg / ha)"
    carbon_stock_below = "Below (Mg / ha)"
    carbon_acc_soil_field = "Soil_accum_rate (Mg / ha / yr)"
    carbon_acc_bio_field = "Bio_accum_rate (Mg / ha / yr)"
    vars_dict['carbon_stock_litter_key'] = "Litter (Mg / ha)"
    carbon_stock_soil_key = "Soil (Mg / ha)"
    vars_dict['carbon_stock_soil_key'] = carbon_stock_soil_key

    veg_dict = dict([(k, int(vars_dict['carbon_pools'][k][
        carbon_stock_veg])) for k in vars_dict['carbon_pools']])

    veg_type_list = list(set([veg_dict[k] for k in veg_dict]))
    vars_dict['veg_type_list'] = veg_type_list


    class InfiniteDict:
        def __init__(self, k, v):
            self.d = {k: v}

        def __getitem__(self, k):
            try:
                return self.d[k]
            except KeyError:
                return 0.0

        def __repr__(self):
            return repr(self.d)

    # constructing accumulation tables from carbon table
    # soil_accumulation_dict - from carbon
    soil_accumulation_dict = {}
    for k in vars_dict['carbon_pools']:
        soil_accumulation_dict[k] = InfiniteDict(
            vars_dict['trans_acc'],
            vars_dict['carbon_pools'][k][carbon_acc_soil_field])
    vars_dict['soil_accumulation_dict'] = soil_accumulation_dict

    # biomass_accumulation_dict - from carbon
    biomass_accumulation_dict = {}
    for k in vars_dict['carbon_pools']:
        biomass_accumulation_dict[k] = InfiniteDict(
            vars_dict['trans_acc'],
            vars_dict['carbon_pools'][k][carbon_acc_bio_field])
    vars_dict['biomass_accumulation_dict'] = biomass_accumulation_dict

    carbon_stock_by_veg_pool_id_dict = {}
    for veg_type in veg_type_list:
        carbon_stock_by_veg_pool_id_dict[veg_type] = {}
        for field in [carbon_stock_above,
                      carbon_stock_below,
                      vars_dict['carbon_stock_litter_key'],
                      carbon_stock_soil_key]:
            carbon_stock_by_veg_pool_id_dict[veg_type][field] = {}
            for k in vars_dict['carbon_pools']:
                if int(vars_dict['carbon_pools'][k][carbon_stock_veg]) == veg_type:
                    carbon_stock_by_veg_pool_id_dict[veg_type][field][k] = float(
                        vars_dict['carbon_pools'][k][field]) * vars_dict['conversion']
                else:
                    carbon_stock_by_veg_pool_id_dict[veg_type][field][k] = 0.0

    # add biomass to carbon field
    carbon_stock_biomass_key = "biomass"
    for veg_type in veg_type_list:
        carbon_stock_by_veg_pool_id_dict[veg_type][carbon_stock_biomass_key] = {}
        for k in vars_dict['carbon_pools']:
            # sum (below, above) carbon pools together into 'bio'
            carbon_stock_by_veg_pool_id_dict[veg_type][carbon_stock_biomass_key][k] = carbon_stock_by_veg_pool_id_dict[
                veg_type][carbon_stock_below][k] + carbon_stock_by_veg_pool_id_dict[veg_type][
                carbon_stock_above][k]
    vars_dict['carbon_stock_by_veg_pool_id_dict'] = carbon_stock_by_veg_pool_id_dict
    vars_dict['carbon_stock_biomass_key'] = carbon_stock_biomass_key

    return vars_dict


def create_transition_matrices(vars_dict):
    '''
    Returns:
        transition_accumulation_dict
        transition_disturbance_dict
    '''
    carbon_stock_veg = "Veg Type"

    # accumulation
    biomass_accumulation_dict = vars_dict['biomass_accumulation_dict']
    soil_accumulation_dict = vars_dict['soil_accumulation_dict']
    transition_csv_dict = vars_dict['transition_csv_dict']
    transition_accumulation_dict = {}
    for veg_type in vars_dict['veg_type_list']:
        transition_accumulation_dict[veg_type] = {}
        for component, component_dict in [
                ("soil_accumulation_dict", soil_accumulation_dict),
                ("biomass_accumulation_dict", biomass_accumulation_dict)]:
            transition_accumulation_dict[veg_type][component] = {}
            for original_lulc in transition_csv_dict:
                transition_accumulation_dict[veg_type][component][original_lulc] = {}
                for transition_lulc in transition_csv_dict:
                    if int(vars_dict['carbon_pools'][transition_lulc][
                            carbon_stock_veg]) == veg_type:
                        transition_accumulation_dict[veg_type][component][
                            (original_lulc, transition_lulc)] = component_dict[
                                transition_lulc][transition_csv_dict[
                                    original_lulc][str(
                                        transition_lulc)]] * vars_dict['conversion']
                    else:
                        transition_accumulation_dict[veg_type][component][(
                            original_lulc, transition_lulc)] = 0.0
    vars_dict['transition_accumulation_dict'] = transition_accumulation_dict

    # disturbance
    transition_disturbance_dict = {}
    for component, component_dict in [
            ("biomass_disturbance_dict", vars_dict['biomass_disturbance_dict']),
            ("soil_disturbance_dict", vars_dict['soil_disturbance_dict'])]:
        transition_disturbance_dict[component] = {}
        for original_lulc in transition_csv_dict:
            for transition_lulc in transition_csv_dict:
                transition_disturbance_dict[component][(original_lulc, transition_lulc)] = \
                    component_dict[vars_dict['carbon_pools'][original_lulc][
                        carbon_stock_veg]][transition_csv_dict[original_lulc][
                            str(transition_lulc)]]
    vars_dict['transition_disturbance_dict'] = transition_disturbance_dict

    return vars_dict


def create_output_uris(vars_dict):
    '''
    '''
    workspace_dir = vars_dict['workspace_dir']
    intermediate_dir = vars_dict['intermediate_dir']

    # carbon pool file names
    vars_dict['veg_stock_bio_uris'] = os.path.join(
        intermediate_dir, "%i_veg_%i_stock_biomass.tif")
    vars_dict['veg_stock_soil_uris'] = os.path.join(
        intermediate_dir, "%i_veg_%i_stock_soil.tif")

    # carbon litter
    vars_dict['veg_litter_uris'] = os.path.join(
        intermediate_dir, "%i_veg_%i_litter.tif")

    # carbon accumulation file names
    vars_dict['acc_soil_uris'] = os.path.join(
        intermediate_dir, "%i_acc_soil.tif")
    vars_dict['acc_bio_uris'] = os.path.join(
        intermediate_dir, "%i_acc_bio.tif")

    vars_dict['veg_acc_bio_uris'] = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_acc_bio.tif")
    vars_dict['veg_acc_soil_uris'] = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_acc_soil.tif")
    vars_dict['veg_dis_bio_uris'] = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_dis_bio.tif")
    vars_dict['veg_dis_soil_uris'] = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_dis_soil.tif")

    # carbon disturbance file names
    vars_dict['dis_bio_uris'] = os.path.join(
        intermediate_dir, "%i_dis_bio.tif")
    vars_dict['dis_soil_uris'] = os.path.join(
        intermediate_dir, "%i_dis_soil.tif")

    # adjusted carbon file names
    vars_dict['veg_adj_acc_bio_uris'] = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_adj_acc_bio.tif")
    vars_dict['veg_adj_acc_soil_uris'] = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_adj_acc_soil.tif")
    vars_dict['veg_adj_dis_bio_uris'] = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_adj_dis_bio.tif")
    vars_dict['veg_adj_dis_soil_uris'] = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_adj_dis_soil.tif")

    vars_dict['veg_adj_em_dis_bio_uris'] = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_adj_em_dis_bio.tif")
    vars_dict['veg_adj_em_dis_soil_uris'] = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_adj_em_dis_soil.tif")

    # emission file names
    vars_dict['veg_em_bio_uris'] = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_em_bio.tif")
    vars_dict['veg_em_soil_uris'] = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_em_soil.tif")

    # totals
    vars_dict['this_total_acc_soil_uris'] = os.path.join(
        intermediate_dir, "%i_%i_soil_acc.tif")
    vars_dict['this_total_acc_bio_uris'] = os.path.join(
        intermediate_dir, "%i_%i_bio_acc.tif")
    vars_dict['this_total_dis_soil_uris'] = os.path.join(
        intermediate_dir, "%i_%i_soil_dis.tif")
    vars_dict['this_total_dis_bio_uris'] = os.path.join(
        intermediate_dir, "%i_%i_bio_dis.tif")

    # Create Output URIs
    vars_dict['extent_uri'] = os.path.join(
        workspace_dir, "extent.shp")
    vars_dict['blue_carbon_csv_uri'] = os.path.join(
        workspace_dir, "sequestration.csv")
    vars_dict['net_sequestration_uri'] = os.path.join(
        workspace_dir, "sequest_%i_%i.tif")
    vars_dict['gain_uri'] = os.path.join(
        workspace_dir, "gain_%i_%i.tif")
    vars_dict['loss_uri'] = os.path.join(
        workspace_dir, "loss_%i_%i.tif")
    vars_dict['carbon_stock_uri'] = os.path.join(
        workspace_dir, "stock_%i.tif")

    return vars_dict


def _alignment_check_uri(dataset_uri_list):
    '''
    '''
    dataset_uri = dataset_uri_list[0]
    dataset = gdal.Open(dataset_uri)
    srs = osr.SpatialReference()
    srs.SetProjection(dataset.GetProjection())

    base_n_rows = dataset.RasterYSize
    base_n_cols = dataset.RasterXSize
    base_linear_units = srs.GetLinearUnits()
    base_geotransform = dataset.GetGeoTransform()

    dataset = None

    for dataset_uri in dataset_uri_list[1:]:
        dataset = gdal.Open(dataset_uri)
        srs.SetProjection(dataset.GetProjection())

        LOGGER.debug("Checking linear units.")
        if srs.GetLinearUnits() != base_linear_units:
            msg = "Linear unit mismatch."
            LOGGER.error(msg)
            raise ValueError, msg

        LOGGER.debug("Checking origin, cell size, and rotation of pixels.")
        if dataset.GetGeoTransform() != base_geotransform:
            msg = "Geotransform mismatch."
            LOGGER.error(msg)
            raise ValueError, msg

        LOGGER.debug("Checking extents.")
        if dataset.RasterYSize != base_n_rows:
            msg = "Number or rows mismatch."
            LOGGER.error(msg)
            raise ValueError, msg

        if dataset.RasterXSize != base_n_cols:
            msg = "Number of columns mismatch."
            LOGGER.error(msg)
            raise ValueError, msg

        dataset = None

    return True


def _validate_rasters(vars_dict):
    '''
    '''
    # validate disturbance and accumulation tables
    transition_csv_dict = vars_dict['transition_csv_dict']
    lulc_uri_dict = vars_dict['lulc_uri_dict']
    lulc_years = vars_dict['lulc_years']

    change_types = set()
    for k1 in transition_csv_dict:
        for k2 in transition_csv_dict:
            change_types.add(transition_csv_dict[k1][str(k2)])

    # check that all rasters have same nodata value
    vars_dict['nodata_lulc'] = set([raster_utils.get_nodata_from_uri(
        lulc_uri_dict[k]) for k in lulc_uri_dict])
    if len(vars_dict['nodata_lulc']) == 1:
        LOGGER.debug("All rasters have the same nodata value.")
        vars_dict['nodata_lulc'] = vars_dict['nodata_lulc'].pop()
    else:
        msg = "All rasters must have the same nodata value."
        LOGGER.error(msg)
        raise ValueError, msg

    # check that all rasters have same cell size
    vars_dict['cell_size'] = set([raster_utils.get_cell_size_from_uri(
        lulc_uri_dict[k]) for k in lulc_uri_dict])
    if len(vars_dict['cell_size']) == 1:
        LOGGER.debug("All rasters have the same cell size.")
        vars_dict['cell_size'] = vars_dict['cell_size'].pop()
    else:
        msg = "All rasters must have the same cell size."
        LOGGER.error(msg)
        raise ValueError, msg

    # check that all rasters are aligned
    LOGGER.debug("Checking alignment.")
    try:
        _alignment_check_uri([lulc_uri_dict[k] for k in lulc_uri_dict])
    except ValueError, msg:
        LOGGER.error("Alignment check FAILED.")
        LOGGER.error(msg)
        raise ValueError, msg
