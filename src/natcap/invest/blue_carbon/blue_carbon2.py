"""
(About Blue Carbon)
"""

import logging
import os
import operator

import numpy
from osgeo import gdal, ogr, osr

import pygeoprocessing.geoprocessing

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.blue_carbon.blue_carbon2')


def datasource_from_dataset_bounding_box_uri(dataset_uri, datasource_uri):
    """Creates a shapefile with the bounding box from a raster.

    Args:
        dataset_uri (str): The uri for the input raster.
        datasource_uri (str): The uri for the output shapefile.

    Returns:
        None
    """
    LOGGER.debug("Creating extent from: %s", dataset_uri)
    LOGGER.debug("Storing extent in: %s", datasource_uri)

    # getting projection and bounding box information
    bounding_box = pygeoprocessing.geoprocessing.get_bounding_box(dataset_uri)
    upper_left_x, upper_left_y, lower_right_x, lower_right_y = bounding_box

    # loading shapefile drive and opening output for writing
    driver = ogr.GetDriverByName('ESRI Shapefile')

    if os.path.exists(datasource_uri):
        driver.DeleteDataSource(datasource_uri)

    datasource = driver.CreateDataSource(datasource_uri)
    if datasource is None:
        msg = "Could not create %s." % datasource_uri
        LOGGER.error(msg)
        raise IOError, msg

    dataset = gdal.Open(dataset_uri)

    # adding arbitrary attribute data
    field_name = "Id"
    field_value = 1

    # add projection
    srs = osr.SpatialReference()
    srs.ImportFromWkt(dataset.GetProjectionRef())

    # create layer with field definitions
    layer = datasource.CreateLayer("raster", geom_type=ogr.wkbPolygon, srs=srs)
    field_defn = ogr.FieldDefn(field_name, ogr.OFTInteger)
    layer.CreateField(field_defn)

    feature_defn = layer.GetLayerDefn()

    # create polygon
    polygon = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)

    ring.AddPoint(upper_left_x, upper_left_y)
    ring.AddPoint(lower_right_x, upper_left_y)
    ring.AddPoint(lower_right_x, lower_right_y)
    ring.AddPoint(upper_left_x, lower_right_y)
    ring.AddPoint(upper_left_x, upper_left_y)

    ring.CloseRings()
    polygon.AddGeometry(ring)

    # create a new feature
    feature = ogr.Feature(feature_defn)
    feature.SetGeometry(polygon)
    feature.SetField(field_name, field_value)

    layer.CreateFeature(feature)

    # clean up and write to disk
    polygon = None
    feature = None

    datasource = None


def sum_uri(dataset_uri, datasource_uri):
    """
    Wrapper call to pygeoprocessing.geoprocessing.aggregate_raster_values_uri to extract
    total

    Args:
        dataset_uri (str): The uri for the input raster.
        datasource_uri (str): The uri for the input shapefile.

    Returns:
        float: The total of the raster values within the shapefile
    """
    total = pygeoprocessing.geoprocessing.aggregate_raster_values_uri(
        dataset_uri, datasource_uri).total
    return total.__getitem__(total.keys().pop())


def alignment_check_uri(dataset_uri_list):
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


def emissions_interpolation(start_year, end_year, this_year, next_year, alpha):
    """
    returns the proportion of the half-life contained within the subrange
    """
    return ((1 - (0.5 ** ((next_year - start_year)/alpha))) - (1 - (0.5 ** (
        (this_year - start_year)/alpha))))/(1 - (0.5 ** (
            (end_year - start_year)/alpha)))


def execute(args):
    """
    Entry point for the blue carbon model.

    Args:
        workspace_dir (string): the directory to hold output from a particular
            model run
        lulc_uri_1 (string): the land use land cover raster for time 1.
        year_1 (int): the year for the land use land cover raster for time 1.
        lulc_uri_2 (string): the land use land cover raster for time 2.
        year_2 (int): the year for the land use land cover raster for time 2.
        lulc_uri_3 (string): the year for the land use land cover raster for
            time 3.
        year_3 (int): the year for the land use land cover raster for time 3.
        lulc_uri_4 (string): the year for the land use land cover raster for
            time 4.
        year_4 (int): the year for the land use land cover raster for time 4.
        lulc_uri_5 (string): the year for the land use land cover raster for
            time 5.
        year_5 (int): the year for the land use land cover raster for time 5.
        analysis_year (int): analysis end year
        soil_disturbance_csv_uri (string): soil disturbance csv file
        biomass_disturbance_csv_uri (string): biomass disturbance csv file
        carbon_pools_uri (string): Carbon in Metric Tons per Hectacre
            (t ha-1) stored in each of the four fundamental pools for
            each land-use/land-cover class.
        half_life_csv_uri (string): carbon half-lives csv file
        transition_matrix_uri (string): Coefficients for the carbon storage
            rates for the transtion between each of the land-use/land-cover
            classes. Values above 1 indicate an increase, values below 1
            indicate a decrease. Absent transitions are assigned a value
            of 1, representing no change.
        snapshots (boolean): enable snapshots
        start (int): start year
        step (int): years between snapshots
        stop (int): stop year
        private_valuation (boolean): enable private valuation
        discount_rate (int): the discount rate as an integer percent.
        price_table (boolean): enable price table
        carbon_schedule (string): the social cost of carbon table.
        carbon_value (float): the price per unit ton of carbon or C02 as
            defined in the carbon price units.
        rate_change (float): the integer percent increase of the price of
            carbon per year.

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
            'private_valuation': True,
            'discount_rate': 5,
            'price_table': True,
            'carbon_schedule': '/path/to/csv',
            'carbon_value': 43.00,
            'rate_change': 0,
        }

    """
    print args

    # preprocess args for possible ease of adoption of future IUI features
    # this creates a hypothetical IUI element from existing element
    lulc_list = []
    for i in range(1, 6):
        if "year_%i" % i in args:
            lulc_list.append(
                {"uri": args["lulc_uri_%i" % i], "year": args["year_%i" % i]})
        else:
            break

    # create a list of the analysis years and a dictionary of the correspond
    # rasters
    lulc_uri_dict = dict([(lulc["year"], lulc["uri"]) for lulc in lulc_list])
    lulc_years = lulc_uri_dict.keys()
    lulc_years.sort()

    # constants
    gdal_format = "GTiff"
    gdal_type_carbon = gdal.GDT_Float64
    nodata_default_int = -1
    nodata_default_float = -1
    gdal_type_identity_raster = gdal.GDT_Int16

    # inputs parameters
    workspace_dir = args["workspace_dir"]
    analysis_year = args["analysis_year"]

    debug_log_file = open(os.path.join(workspace_dir, "debug.txt"), mode="w")
    debug_log = logging.StreamHandler(debug_log_file)
    debug_log.setFormatter(logging.Formatter(
        fmt='%(asctime)s %(message)s', datefmt="%M:%S"))
    LOGGER.addHandler(debug_log)

    # copy LULC for analysis year
    lulc_uri_dict[analysis_year] = lulc_uri_dict[lulc_years[-1]]

    # carbon schedule
    carbon_schedule_field_key = "Year"
    carbon_schedule_field_rate = "Price"

    # carbon pools table
    carbon_uri = args["carbon_pools_uri"]

    carbon_field_key = "Id"
    carbon_field_veg = "Veg Type"
    carbon_field_above = "Above (Mg / ha)"
    carbon_field_below = "Below (Mg / ha)"
    carbon_field_soil = "Soil (Mg / ha)"
    carbon_field_litter = "Litter (Mg / ha)"
    carbon_acc_bio_field = "Bio_accum_rate (Mg / ha / yr)"
    carbon_acc_soil_field = "Soil_accum_rate (Mg / ha / yr)"

    # transition matrix
    trans_comment_uri = args["transition_matrix_uri"]

    # remove transition comment
    trans_uri = pygeoprocessing.geoprocessing.temporary_filename()
    trans_file = open(trans_uri, 'w')
    trans_comment_table = open(trans_comment_uri).readlines()
    row_count = len(trans_comment_table[0].strip().strip(",").split(","))
    trans_file.write("".join(trans_comment_table[:row_count-1]))
    trans_file.close()

    trans_field_key = "Id"
    trans_acc = "Accumulation"

    # disturbance table
    dis_bio_csv_uri = args["biomass_disturbance_csv_uri"]
    dis_soil_csv_uri = args["soil_disturbance_csv_uri"]

    dis_field_key = "veg type"

    # half-life table
    half_life_csv_uri = args["half_life_csv_uri"]
    half_life_field_key = "veg type"
    half_life_field_bio = "biomass (years)"
    half_life_field_soil = "soil (years)"

    # outputs
    extent_name = "extent.shp"
    blue_carbon_csv_name = "sequestration.csv"
    intermediate_dir = "intermediate"

    if not os.path.exists(os.path.join(workspace_dir, intermediate_dir)):
        os.makedirs(os.path.join(workspace_dir, intermediate_dir))

    # carbon pool file names
    carbon_name = "stock_%i.tif"

    veg_stock_bio_name = os.path.join(
        intermediate_dir, "%i_veg_%i_stock_biomass.tif")
    veg_stock_soil_name = os.path.join(
        intermediate_dir, "%i_veg_%i_stock_soil.tif")

    # carbon litter
    veg_litter_name = os.path.join(intermediate_dir, "%i_veg_%i_litter.tif")

    # carbon accumulation file names
    acc_soil_name = os.path.join(intermediate_dir, "%i_acc_soil.tif")
    acc_bio_name = os.path.join(intermediate_dir, "%i_acc_bio.tif")

    veg_acc_bio_name = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_acc_bio.tif")
    veg_acc_soil_name = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_acc_soil.tif")
    veg_dis_bio_name = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_dis_bio.tif")
    veg_dis_soil_name = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_dis_soil.tif")

    # carbon disturbance file names
    dis_bio_name = os.path.join(intermediate_dir, "%i_dis_bio.tif")
    dis_soil_name = os.path.join(intermediate_dir, "%i_dis_soil.tif")

    # adjusted carbon file names
    veg_adj_acc_bio_name = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_adj_acc_bio.tif")
    veg_adj_acc_soil_name = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_adj_acc_soil.tif")
    veg_adj_dis_bio_name = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_adj_dis_bio.tif")
    veg_adj_dis_soil_name = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_adj_dis_soil.tif")

    veg_adj_em_dis_bio_name = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_adj_em_dis_bio.tif")
    veg_adj_em_dis_soil_name = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_adj_em_dis_soil.tif")

    # emission file names
    veg_em_bio_name = os.path.join(intermediate_dir, "%i_%i_veg_%i_em_bio.tif")
    veg_em_soil_name = os.path.join(
        intermediate_dir, "%i_%i_veg_%i_em_soil.tif")

    # totals
    this_total_acc_soil_name = os.path.join(
        intermediate_dir, "%i_%i_soil_acc.tif")
    this_total_acc_bio_name = os.path.join(
        intermediate_dir, "%i_%i_bio_acc.tif")
    this_total_dis_soil_name = os.path.join(
        intermediate_dir, "%i_%i_soil_dis.tif")
    this_total_dis_bio_name = os.path.join(
        intermediate_dir, "%i_%i_bio_dis.tif")
    net_sequestration_name = "sequest_%i_%i.tif"
    gain_name = "gain_%i_%i.tif"
    loss_name = "loss_%i_%i.tif"

    # uri
    extent_uri = os.path.join(workspace_dir, extent_name)
    blue_carbon_csv_uri = os.path.join(workspace_dir, blue_carbon_csv_name)

    # process inputs
    dis_bio = pygeoprocessing.geoprocessing.get_lookup_from_csv(dis_bio_csv_uri, dis_field_key)

    # adding accumulation value to disturbance table
    for k in dis_bio:
        dis_bio[k][trans_acc] = 0.0

    dis_soil = pygeoprocessing.geoprocessing.get_lookup_from_csv(
        dis_soil_csv_uri, dis_field_key)
    # adding accumulation values to disturbance table
    for k in dis_soil:
        dis_soil[k][trans_acc] = 0.0

    trans = pygeoprocessing.geoprocessing.get_lookup_from_csv(trans_uri, trans_field_key)
    carbon = pygeoprocessing.geoprocessing.get_lookup_from_csv(carbon_uri, carbon_field_key)

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
    acc_soil = {}
    for k in carbon:
        acc_soil[k] = InfiniteDict(trans_acc, carbon[k][carbon_acc_soil_field])

    acc_bio = {}
    for k in carbon:
        acc_bio[k] = InfiniteDict(trans_acc, carbon[k][carbon_acc_bio_field])

    half_life = pygeoprocessing.geoprocessing.get_lookup_from_csv(
        half_life_csv_uri, half_life_field_key)

    # validate disturbance and accumulation tables
    change_types = set()
    for k1 in trans:
        for k2 in trans:
            change_types.add(trans[k1][str(k2)])

    # validating data
    nodata_lulc = set([pygeoprocessing.geoprocessing.get_nodata_from_uri(
        lulc_uri_dict[k]) for k in lulc_uri_dict])
    if len(nodata_lulc) == 1:
        LOGGER.debug("All rasters have the same nodata value.")
        nodata_lulc = nodata_lulc.pop()
    else:
        msg = "All rasters must have the same nodata value."
        LOGGER.error(msg)
        raise ValueError, msg

    cell_size = set([pygeoprocessing.geoprocessing.get_cell_size_from_uri(
        lulc_uri_dict[k]) for k in lulc_uri_dict])
    if len(cell_size) == 1:
        LOGGER.debug("All rasters have the same cell size.")
        cell_size = cell_size.pop()
    else:
        msg = "All rasters must have the same cell size."
        LOGGER.error(msg)
        raise ValueError, msg

    LOGGER.debug("Checking alignment.")
    try:
        alignment_check_uri([lulc_uri_dict[k] for k in lulc_uri_dict])
    except ValueError, msg:
        LOGGER.error("Alignment check FAILED.")
        LOGGER.error(msg)
        raise ValueError, msg

    # construct dictionaries for single parameter lookups
    conversion = (pygeoprocessing.geoprocessing.get_cell_size_from_uri(
        lulc_uri_dict[lulc_years[0]]) ** 2) / 10000.0  # convert to Ha

    LOGGER.debug("Cell size is %s hectacres.", conversion)

    veg_dict = dict([(k, int(carbon[k][carbon_field_veg])) for k in carbon])

    veg_type_list = list(set([veg_dict[k] for k in veg_dict]))

    # create carbon field dictionary
    veg_field_dict = {}
    for veg_type in veg_type_list:
        veg_field_dict[veg_type] = {}
        for field in [carbon_field_above, carbon_field_below, carbon_field_litter, carbon_field_soil]:
            veg_field_dict[veg_type][field] = {}
            for k in carbon:
                if int(carbon[k][carbon_field_veg]) == veg_type:
                    veg_field_dict[veg_type][field][k] = float(
                        carbon[k][field]) * conversion
                else:
                    veg_field_dict[veg_type][field][k] = 0.0

    # add biomass to carbon field
    carbon_field_bio = "bio"
    for veg_type in veg_type_list:
        veg_field_dict[veg_type][carbon_field_bio] = {}
        for k in carbon:
            veg_field_dict[veg_type][carbon_field_bio][k] = veg_field_dict[
                veg_type][carbon_field_below][k] + veg_field_dict[veg_type][
                carbon_field_above][k]

    # create transition field dictionary
    acc_soil_name = "acc_soil"
    acc_bio_name = "acc_bio"
    dis_bio_name = "dis_bio"
    dis_soil_name = "dis_soil"

    # accumulation
    veg_trans_acc_dict = {}
    for veg_type in veg_type_list:
        veg_trans_acc_dict[veg_type] = {}
        for component, component_dict in [(acc_soil_name, acc_soil),
                                          (acc_bio_name, acc_bio)]:
            veg_trans_acc_dict[veg_type][component] = {}
            for original_lulc in trans:
                veg_trans_acc_dict[veg_type][component][original_lulc] = {}
                for transition_lulc in trans:
                    if int(carbon[transition_lulc][carbon_field_veg]) == veg_type:
                        veg_trans_acc_dict[veg_type][component][(original_lulc, transition_lulc)] = component_dict[transition_lulc][trans[original_lulc][str(transition_lulc)]] * conversion
                    else:
                        veg_trans_acc_dict[veg_type][component][(original_lulc, transition_lulc)] = 0.0

    # disturbance
    trans_dis_dict = {}
    for component, component_dict in [(dis_bio_name, dis_bio),
                                      (dis_soil_name, dis_soil)]:
        trans_dis_dict[component] = {}
        for original_lulc in trans:
            for transition_lulc in trans:
                trans_dis_dict[component][(original_lulc, transition_lulc)] = component_dict[carbon[original_lulc][carbon_field_veg]][trans[original_lulc][str(transition_lulc)]]

    # vectorize datasets operations
    # standard ops
    def add_op(*values):
        if nodata_default_int in values:
            return nodata_default_int
        return reduce(operator.add, values)

    def sub_op(*values):
        if nodata_default_int in values:
            return nodata_default_int
        return reduce(operator.sub, values)

    def mul_op(*values):
        if nodata_default_int in values:
            return nodata_default_int
        return reduce(operator.mul, values)

    # custom ops
    def acc_bio_op_closure(veg_type, t):
        def acc_bio_co_op(original_lulc, transition_lulc):
            if nodata_lulc in [original_lulc, transition_lulc]:
                return nodata_default_float
            return veg_trans_acc_dict[veg_type][acc_bio_name][(
                int(original_lulc), int(transition_lulc))] * t

        return acc_bio_co_op

    def acc_soil_op_closure(veg_type, t):
        def acc_soil_co_op(original_lulc, transition_lulc):
            if nodata_lulc in [original_lulc, transition_lulc]:
                return nodata_default_float
            return veg_trans_acc_dict[veg_type][acc_soil_name][(
                int(original_lulc), int(transition_lulc))] * t

        return acc_soil_co_op

    def dis_bio_op(carbon_base, original_lulc, transition_lulc):
        if nodata_lulc in [carbon_base, original_lulc, transition_lulc]:
            return nodata_default_float
        return carbon_base * trans_dis_dict[dis_bio_name][(
            int(original_lulc), int(transition_lulc))]

    def dis_soil_op(carbon_base, original_lulc, transition_lulc):
        if nodata_lulc in [carbon_base, original_lulc, transition_lulc]:
            return nodata_default_float
        return carbon_base * trans_dis_dict[dis_soil_name][(
            int(original_lulc), int(transition_lulc))]

    def adj_op(base, acc, dis):
        if nodata_default_float in [base, acc, dis]:
            return nodata_default_float
        else:
            return base + acc - dis

    def net_sequestration_op(bio_acc, bio_dis, soil_acc, soil_dis):
        if nodata_default_float in [bio_acc, bio_dis, soil_acc, soil_dis]:
            return nodata_default_float
        else:
            return ((bio_acc + soil_acc) - (bio_dis + soil_dis))

    def veg_adj_op(base, adj, mask):
        if nodata_default_float in [base, adj, mask]:
            return nodata_default_float
        else:
            return base + (adj * mask)

    def vectorize_carbon_datasets(
            dataset_uri_list, dataset_pixel_op, dataset_out_uri):
        pygeoprocessing.geoprocessing.vectorize_datasets(
            dataset_uri_list,
            dataset_pixel_op,
            dataset_out_uri,
            gdal_type_carbon,
            nodata_default_float,
            cell_size,
            "union")

    def half_life_op_closure(veg_type, half_life_field, alpha_t):
        def h_l_op(c):
            if c is nodata_default_float:
                return c
            alpha = half_life[veg_type][half_life_field]

            try:
                h_l = alpha_t/float(alpha)
                resid = 0.5 ** h_l
                coeff = 1 - resid
                r = coeff * c
                return r

            except ValueError:
                # return 0 if alpha is None
                return 0

        return h_l_op

    LOGGER.info("Running analysis.")
    # calculate stock carbon values
    this_year = lulc_years[0]
    this_uri = lulc_uri_dict[this_year]

    LOGGER.info("Calculating stock carbon values.")

    # create vegetation specific stock values for biomass and soil
    base_veg_acc_bio = "base_veg_acc_bio"
    base_veg_acc_soil = "base_veg_acc_soil"
    base_veg_dis_bio = "base_veg_dis_bio"
    base_veg_dis_soil = "base_veg_dis_soil"
    veg_base_uri_dict = {}

    # creating zero-fill raster for initial disturbed carbon
    zero_raster_uri = os.path.join(workspace_dir, os.path.join(
        intermediate_dir, "zeros.tif"))
    pygeoprocessing.geoprocessing.new_raster_from_base_uri(
        this_uri,
        zero_raster_uri,
        gdal_format,
        nodata_default_int,
        gdal_type_identity_raster,
        fill_value=0)

    for veg_type in veg_type_list:
        veg_base_uri_dict[veg_type] = {}

        this_veg_stock_soil_uri = os.path.join(
            workspace_dir, veg_stock_soil_name % (this_year, veg_type))
        this_veg_stock_bio_uri = os.path.join(
            workspace_dir, veg_stock_bio_name % (this_year, veg_type))

        pygeoprocessing.geoprocessing.reclassify_dataset_uri(
            this_uri,
            veg_field_dict[veg_type][carbon_field_bio],
            this_veg_stock_bio_uri,
            gdal_type_carbon,
            nodata_default_float,
            exception_flag="values_required")

        pygeoprocessing.geoprocessing.reclassify_dataset_uri(
            this_uri,
            veg_field_dict[veg_type][carbon_field_soil],
            this_veg_stock_soil_uri,
            gdal_type_carbon,
            nodata_default_float,
            exception_flag="values_required")

        veg_base_uri_dict[veg_type][base_veg_acc_bio] = this_veg_stock_bio_uri
        veg_base_uri_dict[veg_type][
            base_veg_acc_soil] = this_veg_stock_soil_uri
        veg_base_uri_dict[veg_type][base_veg_dis_bio] = zero_raster_uri
        veg_base_uri_dict[veg_type][base_veg_dis_soil] = zero_raster_uri

    # loop over lulc years

    # create extent shapefile
    datasource_from_dataset_bounding_box_uri(this_uri, extent_uri)

    totals = {}
    stock_uri_dict = {}
    for this_year, next_year in zip(
            lulc_years, lulc_years[1:]+[analysis_year]):
        this_total_carbon_uri = os.path.join(
            workspace_dir, carbon_name % this_year)
        this_total_carbon_uri_list = []

        this_total_acc_soil_uri = os.path.join(
            workspace_dir, this_total_acc_soil_name % (this_year, next_year))
        this_total_acc_bio_uri = os.path.join(
            workspace_dir, this_total_acc_bio_name % (this_year, next_year))
        this_total_dis_soil_uri = os.path.join(
            workspace_dir, this_total_dis_soil_name % (this_year, next_year))
        this_total_dis_bio_uri = os.path.join(
            workspace_dir, this_total_dis_bio_name % (this_year, next_year))

        veg_acc_bio_uri_list = []
        veg_acc_soil_uri_list = []
        veg_dis_bio_uri_list = []
        veg_dis_soil_uri_list = []

        totals[this_year] = {}

        LOGGER.info("Transition from %i to %i.", this_year, next_year)
        this_uri = lulc_uri_dict[this_year]
        next_uri = lulc_uri_dict[next_year]

        t = next_year - this_year

        for veg_type in veg_type_list:
            totals[this_year][veg_type] = {}

            LOGGER.info("Processing vegetation type %i.", veg_type)
            # litter URI's
            this_veg_litter_uri = os.path.join(
                workspace_dir, veg_litter_name % (this_year, veg_type))

            # disturbance and accumulation URI's
            this_veg_acc_bio_uri = os.path.join(
                workspace_dir, veg_acc_bio_name % (
                    this_year, next_year, veg_type))
            this_veg_acc_soil_uri = os.path.join(
                workspace_dir, veg_acc_soil_name % (
                    this_year, next_year, veg_type))
            this_veg_dis_bio_uri = os.path.join(
                workspace_dir, veg_dis_bio_name % (
                    this_year, next_year, veg_type))
            this_veg_dis_soil_uri = os.path.join(
                workspace_dir, veg_dis_soil_name % (
                    this_year, next_year, veg_type))

            # transition adjusted URI's
            this_veg_adj_acc_bio_uri = os.path.join(
                workspace_dir, veg_adj_acc_bio_name % (
                    this_year, next_year, veg_type))
            this_veg_adj_acc_soil_uri = os.path.join(
                workspace_dir, veg_adj_acc_soil_name % (
                    this_year, next_year, veg_type))
            this_veg_adj_dis_bio_uri = os.path.join(
                workspace_dir, veg_adj_dis_bio_name % (
                    this_year, next_year, veg_type))
            this_veg_adj_dis_soil_uri = os.path.join(
                workspace_dir, veg_adj_dis_soil_name % (
                    this_year, next_year, veg_type))

            # emission URI's
            this_veg_em_bio_uri = os.path.join(
                workspace_dir, veg_em_bio_name % (
                    this_year, next_year, veg_type))
            this_veg_em_soil_uri = os.path.join(
                workspace_dir, veg_em_soil_name % (
                    this_year, next_year, veg_type))

            # emission adjusted URI's
            this_veg_adj_em_dis_bio_uri = os.path.join(
                workspace_dir, veg_adj_em_dis_bio_name % (
                    this_year, next_year, veg_type))
            this_veg_adj_em_dis_soil_uri = os.path.join(
                workspace_dir, veg_adj_em_dis_soil_name % (
                    this_year, next_year, veg_type))

            # litter
            pygeoprocessing.geoprocessing.reclassify_dataset_uri(
                this_uri,
                veg_field_dict[veg_type][carbon_field_litter],
                this_veg_litter_uri,
                gdal_type_carbon,
                nodata_default_float,
                exception_flag="values_required")

            # accumulation
            # biomass accumulation
            vectorize_carbon_datasets(
                [this_uri, next_uri],
                acc_bio_op_closure(veg_type, t),
                this_veg_acc_bio_uri)

            # soil accumulation
            vectorize_carbon_datasets(
                [this_uri, next_uri],
                acc_soil_op_closure(veg_type, t),
                this_veg_acc_soil_uri)

            # disturbance
            # biomass disturbance
            vectorize_carbon_datasets(
                [veg_base_uri_dict[veg_type][
                    base_veg_acc_bio], this_uri, next_uri],
                dis_bio_op,
                this_veg_dis_bio_uri)

            # soil disturbance
            vectorize_carbon_datasets(
                [veg_base_uri_dict[veg_type][
                    base_veg_acc_soil], this_uri, next_uri],
                dis_soil_op,
                this_veg_dis_soil_uri)

            # transition adjustments
            # transition adjusted undisturbed biomass
            vectorize_carbon_datasets(
                [veg_base_uri_dict[veg_type][
                    base_veg_acc_bio],
                    this_veg_acc_bio_uri, this_veg_dis_bio_uri],
                adj_op,
                this_veg_adj_acc_bio_uri)

            # transition adjusted undisturbed soil
            vectorize_carbon_datasets(
                [veg_base_uri_dict[veg_type][
                    base_veg_acc_soil], this_veg_acc_soil_uri,
                    this_veg_dis_soil_uri],
                adj_op,
                this_veg_adj_acc_soil_uri)

            # transition adjusted disturbed biomass
            vectorize_carbon_datasets(
                [veg_base_uri_dict[veg_type][
                    base_veg_dis_bio], this_veg_dis_bio_uri],
                add_op,
                this_veg_adj_dis_bio_uri)

            # transition adjusted disturbed soil
            vectorize_carbon_datasets(
                [veg_base_uri_dict[veg_type][
                    base_veg_dis_soil], this_veg_dis_soil_uri],
                add_op,
                this_veg_adj_dis_soil_uri)

            # emissions
            # biomass emissions
            vectorize_carbon_datasets(
                [this_veg_adj_dis_bio_uri],
                half_life_op_closure(veg_type, half_life_field_bio, t),
                this_veg_em_bio_uri)

            # soil emissions
            vectorize_carbon_datasets(
                [this_veg_adj_dis_soil_uri],
                half_life_op_closure(veg_type, half_life_field_soil, t),
                this_veg_em_soil_uri)

            # emissions adjustment
            # emissions adjusted disturbed biomass
            vectorize_carbon_datasets(
                [this_veg_adj_dis_bio_uri, this_veg_em_bio_uri],
                sub_op,
                this_veg_adj_em_dis_bio_uri)

            # emissions adjusted disturbed soil
            vectorize_carbon_datasets(
                [this_veg_adj_dis_soil_uri, this_veg_em_soil_uri],
                sub_op,
                this_veg_adj_em_dis_soil_uri)

            veg_acc_bio_uri_list.append(this_veg_acc_bio_uri)
            veg_acc_soil_uri_list.append(this_veg_acc_soil_uri)
            veg_dis_bio_uri_list.append(this_veg_dis_bio_uri)
            veg_dis_soil_uri_list.append(this_veg_dis_soil_uri)

            for name, uri in [
                (veg_acc_bio_name, this_veg_acc_bio_uri),
                (veg_acc_soil_name, this_veg_acc_soil_uri),
                (veg_dis_bio_name, this_veg_dis_bio_uri),
                (veg_dis_soil_name, this_veg_dis_soil_uri),
                (veg_adj_acc_bio_name, this_veg_adj_acc_bio_uri),
                (veg_adj_acc_soil_name, this_veg_adj_acc_soil_uri),
                (veg_adj_dis_bio_name, this_veg_adj_dis_bio_uri),
                (veg_adj_dis_soil_name, this_veg_adj_dis_soil_uri),
                (veg_em_bio_name, this_veg_em_bio_uri),
                (veg_em_soil_name, this_veg_em_soil_uri),
                (veg_adj_em_dis_bio_name, this_veg_adj_em_dis_bio_uri),
                    (veg_adj_em_dis_soil_name, this_veg_adj_em_dis_soil_uri)]:
                    totals[this_year][veg_type][name] = sum_uri(
                        uri, extent_uri)

            # switch base carbon rasters
            this_total_carbon_uri_list.append(
                veg_base_uri_dict[veg_type][base_veg_acc_bio])
            this_total_carbon_uri_list.append(
                veg_base_uri_dict[veg_type][base_veg_acc_soil])
            this_total_carbon_uri_list.append(
                this_veg_litter_uri)

            veg_base_uri_dict[veg_type][
                base_veg_acc_bio] = this_veg_adj_acc_bio_uri
            veg_base_uri_dict[veg_type][
                base_veg_acc_soil] = this_veg_adj_acc_soil_uri
            veg_base_uri_dict[veg_type][
                base_veg_dis_bio] = this_veg_adj_em_dis_bio_uri
            veg_base_uri_dict[veg_type][
                base_veg_dis_soil] = this_veg_adj_em_dis_soil_uri

        vectorize_carbon_datasets(this_total_carbon_uri_list,
                                  add_op,
                                  this_total_carbon_uri)

        stock_uri_dict[this_year] = this_total_carbon_uri

        vectorize_carbon_datasets(veg_acc_bio_uri_list,
                                  add_op,
                                  this_total_acc_bio_uri)

        vectorize_carbon_datasets(veg_acc_soil_uri_list,
                                  add_op,
                                  this_total_acc_soil_uri)

        vectorize_carbon_datasets(veg_dis_bio_uri_list,
                                  add_op,
                                  this_total_dis_bio_uri)

        vectorize_carbon_datasets(veg_dis_soil_uri_list,
                                  add_op,
                                  this_total_dis_soil_uri)

    # analysis year raster
    this_total_carbon_uri = os.path.join(
        workspace_dir, carbon_name % analysis_year)
    this_total_carbon_uri_list = []
    for veg_type in veg_type_list:
        this_veg_litter_uri = os.path.join(workspace_dir,
                                           veg_litter_name % (lulc_years[-1],
                                                              veg_type))

        this_veg_adj_acc_bio_uri = os.path.join(
            workspace_dir,
            veg_adj_acc_bio_name % (lulc_years[-1], analysis_year, veg_type))
        this_veg_adj_acc_soil_uri = os.path.join(
            workspace_dir,
            veg_adj_acc_soil_name % (lulc_years[-1], analysis_year, veg_type))

        this_total_carbon_uri_list.append(this_veg_litter_uri)
        this_total_carbon_uri_list.append(this_veg_adj_acc_bio_uri)
        this_total_carbon_uri_list.append(this_veg_adj_acc_soil_uri)

    vectorize_carbon_datasets(this_total_carbon_uri_list,
                              add_op,
                              this_total_carbon_uri)

    stock_uri_dict[analysis_year] = this_total_carbon_uri

    def pos_op(v):
        if v is nodata_default_float:
            return v
        elif v >= 0:
            return v
        else:
            return 0

    def neg_op(v):
        if v is nodata_default_float:
            return v
        elif v < 0:
            return v * -1
        else:
            return 0

    for i, this_year in enumerate(lulc_years):
        for next_year in (lulc_years+[analysis_year])[i+1:]:
            LOGGER.info("Calculating sequestration from %i to %i.", this_year, next_year)
            total_seq_uri = os.path.join(
                workspace_dir, net_sequestration_name % (this_year, next_year))
            gain_uri = os.path.join(
                workspace_dir, gain_name % (this_year, next_year))
            loss_uri = os.path.join(
                workspace_dir, loss_name % (this_year, next_year))

            stock_uri_list = [stock_uri_dict[next_year],
                              stock_uri_dict[this_year]]
            vectorize_carbon_datasets(stock_uri_list,
                                      sub_op,
                                      total_seq_uri)

            vectorize_carbon_datasets([total_seq_uri],
                                      pos_op,
                                      gain_uri)

            vectorize_carbon_datasets([total_seq_uri],
                                      neg_op,
                                      loss_uri)

    # Valuation component starts here
    if args["private_valuation"]:
        # tabulate results in new csv file
        csv = open(blue_carbon_csv_uri, 'w')

        header = ["Start Year", "End Year", "Accumulation"]
        # header += ["Veg %i Bio Emissions" % i for i in veg_type_list]
        # header += ["Veg %i Soil Emissions" % i for i in veg_type_list]
        header += ["Total Emissions", "Sequestration", "Value",
                   "Discount Factor", "Cost"]

        csv.write(",".join(header))

        if not args["price_table"]:
            # If no price table, create carbon schedule
            carbon_schedule = {}
            for year in range(lulc_years[0], analysis_year + 1):
                carbon_schedule[year] = {carbon_schedule_field_rate: float(
                    args["carbon_value"]) * ((
                        1 + (float(args["rate_change"]) / float(100))) ** (
                        year - lulc_years[0]))}
        else:
            # Fetch carbon schedule from provided price table
            carbon_schedule = pygeoprocessing.geoprocessing.get_lookup_from_csv(
                args["carbon_schedule"], carbon_schedule_field_key)

        period_op_dict = {}
        for start_year, end_year in zip(lulc_years, (
                lulc_years + [analysis_year])[1:]):
            period_op_dict[start_year] = {}
            period_op_dict[start_year][
                "accumulation_divisor"] = end_year - start_year
            period_op_dict[start_year][
                "biomass_half_life"] = {}
            period_op_dict[start_year][
                "soil_half_life"] = {}
            period_op_dict[start_year][
                "price"] = {}
            period_op_dict[start_year][
                "discount_factor"] = {}

            for this_year, next_year in zip(
                    range(start_year, end_year),
                    range(start_year + 1, end_year + 1)):
                LOGGER.debug(
                    "Interpolating from %i to %i.", this_year, next_year)

                row = [str(this_year), str(next_year)]
                accumulation = 0
                emissions = 0

                for source in [veg_acc_bio_name,
                               veg_acc_soil_name]:
                    for veg_type in veg_type_list:
                        accumulation += totals[start_year][veg_type][
                            source] / period_op_dict[start_year][
                            "accumulation_divisor"]

                row.append(str(accumulation))

                period_op_dict[start_year]["biomass_half_life"][this_year] = {}
                for veg_type in veg_type_list:
                    try:
                        c = emissions_interpolation(
                            start_year,
                            end_year,
                            this_year,
                            next_year,
                            float(half_life[veg_type][half_life_field_bio]))
                    except ValueError:
                        c = 0

                    period_op_dict[start_year]["biomass_half_life"][
                        this_year][veg_type] = c

                    emissions += totals[start_year][veg_type][
                        veg_em_bio_name] * c

                period_op_dict[start_year]["soil_half_life"][this_year] = {}
                for veg_type in veg_type_list:
                    try:
                        c = emissions_interpolation(
                            start_year,
                            end_year,
                            this_year,
                            next_year,
                            float(half_life[veg_type][half_life_field_soil]))
                    except ValueError:
                        c = 0

                    period_op_dict[start_year]["soil_half_life"][
                        this_year][veg_type] = c
