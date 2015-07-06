"""
(About Blue Carbon Model)
"""

import logging
import os
import operator
import pprint as pp

from osgeo import gdal, ogr, osr

from natcap.invest import raster_utils

import rasterio as rio

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('blue_carbon')


def run_biophysical(vars_dict):
    '''
    Runs the biophysical component of the Blue Carbon model
    '''
    # Setup Operations
    gdal_type_carbon = vars_dict['gdal_type_carbon']
    nodata_default_int = vars_dict['nodata_default_int']
    nodata_default_float = vars_dict['nodata_default_float']
    nodata_lulc = vars_dict['nodata_lulc']
    cell_size = vars_dict['cell_size']
    transition_disturbance_dict = vars_dict['transition_disturbance_dict']
    transition_accumulation_dict = vars_dict['transition_accumulation_dict']
    half_lives_by_veg_dict = vars_dict['half_lives_by_veg_dict']

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
            return transition_accumulation_dict[veg_type]["biomass_accumulation_dict"][(
                int(original_lulc), int(transition_lulc))] * t
        return acc_bio_co_op

    def acc_soil_op_closure(veg_type, t):
        def acc_soil_co_op(original_lulc, transition_lulc):
            if nodata_lulc in [original_lulc, transition_lulc]:
                return nodata_default_float
            return transition_accumulation_dict[veg_type]["soil_accumulation_dict"][(
                int(original_lulc), int(transition_lulc))] * t

        return acc_soil_co_op

    def dis_bio_op(carbon_base, original_lulc, transition_lulc):
        if nodata_lulc in [carbon_base, original_lulc, transition_lulc]:
            return nodata_default_float
        return carbon_base * transition_disturbance_dict['biomass_disturbance_dict'][(
            int(original_lulc), int(transition_lulc))]

    def dis_soil_op(carbon_base, original_lulc, transition_lulc):
        if nodata_lulc in [carbon_base, original_lulc, transition_lulc]:
            return nodata_default_float
        return carbon_base * transition_disturbance_dict['soil_disturbance_dict'][(
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

    def half_life_op_closure(veg_type, half_life_field, alpha_t):
        def h_l_op(c):
            if c is nodata_default_float:
                return c
            alpha = half_lives_by_veg_dict[veg_type][half_life_field]

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

    def carbon_raster_calculation(
            dataset_uri_list, dataset_pixel_op, dataset_out_uri):
        raster_utils.vectorize_datasets(
            dataset_uri_list,
            dataset_pixel_op,
            dataset_out_uri,
            gdal_type_carbon,
            nodata_default_float,
            cell_size,
            "union")

    # Run Biophysical Model
    lulc_years = vars_dict['lulc_years']
    lulc_uri_dict = vars_dict['lulc_uri_dict']
    veg_type_list = vars_dict['veg_type_list']
    carbon_stock_by_veg_pool_id_dict = vars_dict['carbon_stock_by_veg_pool_id_dict']
    workspace_dir = vars_dict['workspace_dir']
    extent_uri = vars_dict['extent_uri']
    intermediate_dir = vars_dict['intermediate_dir']
    carbon_stock_biomass_key = vars_dict['carbon_stock_biomass_key']
    carbon_stock_soil_key = vars_dict['carbon_stock_soil_key']
    carbon_stock_litter_key = vars_dict['carbon_stock_litter_key']
    analysis_year = vars_dict['analysis_year']
    veg_litter_uris = vars_dict['veg_litter_uris']

    # create vegetation specific stock values for biomass and soil
    base_veg_acc_bio = "base_veg_acc_bio"
    base_veg_acc_soil = "base_veg_acc_soil"
    base_veg_dis_bio = "base_veg_dis_bio"
    base_veg_dis_soil = "base_veg_dis_soil"
    veg_base_uri_dict = {}

    LOGGER.info("Running analysis.")
    LOGGER.info("Calculating stock carbon values.")
    # calculate stock carbon values
    this_year = lulc_years[0]
    this_year_uri = lulc_uri_dict[this_year]
    zero_raster_uri = os.path.join(intermediate_dir, "zeros.tif")
    # creating zero-fill raster for initial disturbed carbon
    raster_utils.new_raster_from_base_uri(
        this_year_uri,
        zero_raster_uri,
        "GTiff",
        nodata_default_int,
        gdal.GDT_Int16,
        fill_value=0)

    for veg_type in veg_type_list:
        veg_base_uri_dict[veg_type] = {}

        this_veg_stock_soil_uri = vars_dict['veg_stock_soil_uris'] % (
            this_year, veg_type)
        this_veg_stock_bio_uri = vars_dict['veg_stock_bio_uris'] % (
            this_year, veg_type)

        raster_utils.reclassify_dataset_uri(
            this_year_uri,
            carbon_stock_by_veg_pool_id_dict[veg_type][carbon_stock_biomass_key],
            this_veg_stock_bio_uri,
            gdal_type_carbon,
            nodata_default_float,
            exception_flag="values_required")

        raster_utils.reclassify_dataset_uri(
            this_year_uri,
            carbon_stock_by_veg_pool_id_dict[veg_type][carbon_stock_soil_key],
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
    _datasource_from_dataset_bounding_box_uri(this_year_uri, extent_uri)

    totals = {}
    stock_uri_dict = {}
    for this_year, next_year in zip(
            lulc_years, lulc_years[1:]+[analysis_year]):
        this_total_carbon_uri = vars_dict['carbon_stock_uri'] % this_year
        this_total_carbon_uri_list = []

        this_total_acc_soil_uri = vars_dict['this_total_acc_soil_uris'] % (
            this_year, next_year)
        this_total_acc_bio_uri = vars_dict['this_total_acc_bio_uris'] % (
            this_year, next_year)
        this_total_dis_soil_uri = vars_dict['this_total_dis_soil_uris'] % (
            this_year, next_year)
        this_total_dis_bio_uri = vars_dict['this_total_dis_bio_uris'] % (
            this_year, next_year)

        veg_acc_bio_uri_list = []
        veg_acc_soil_uri_list = []
        veg_dis_bio_uri_list = []
        veg_dis_soil_uri_list = []

        totals[this_year] = {}

        LOGGER.info("Transition from %i to %i.", this_year, next_year)
        this_year_uri = lulc_uri_dict[this_year]
        next_year_uri = lulc_uri_dict[next_year]

        t = next_year - this_year

        for veg_type in veg_type_list:
            totals[this_year][veg_type] = {}

            LOGGER.info("Processing vegetation type %i.", veg_type)
            # litter URI's
            this_veg_litter_uri = vars_dict[
                'veg_litter_uris'] % (this_year, veg_type)

            raster_utils.reclassify_dataset_uri(
                this_year_uri,
                carbon_stock_by_veg_pool_id_dict[veg_type][carbon_stock_litter_key],
                this_veg_litter_uri,
                gdal_type_carbon,
                nodata_default_float,
                exception_flag="values_required")

            # biomass accumulation
            this_veg_acc_bio_uri = vars_dict['veg_acc_bio_uris'] % (
                this_year, next_year, veg_type)
            carbon_raster_calculation(
                [this_year_uri, next_year_uri],
                acc_bio_op_closure(veg_type, t),
                this_veg_acc_bio_uri)

            # soil accumulation
            this_veg_acc_soil_uri = vars_dict['veg_acc_soil_uris'] % (
                this_year, next_year, veg_type)
            carbon_raster_calculation(
                [this_year_uri, next_year_uri],
                acc_soil_op_closure(veg_type, t),
                this_veg_acc_soil_uri)

            # biomass disturbance
            this_veg_dis_bio_uri = vars_dict['veg_dis_bio_uris'] % (
                this_year, next_year, veg_type)
            carbon_raster_calculation(
                [veg_base_uri_dict[veg_type][
                    base_veg_acc_bio], this_year_uri, next_year_uri],
                dis_bio_op,
                this_veg_dis_bio_uri)

            # soil disturbance
            this_veg_dis_soil_uri = vars_dict['veg_dis_soil_uris'] % (
                this_year, next_year, veg_type)
            carbon_raster_calculation(
                [veg_base_uri_dict[veg_type][
                    base_veg_acc_soil], this_year_uri, next_year_uri],
                dis_soil_op,
                this_veg_dis_soil_uri)

            # Transition

            # transition adjusted undisturbed biomass
            this_veg_adj_acc_bio_uri = vars_dict['veg_adj_acc_bio_uris'] % (
                this_year, next_year, veg_type)
            carbon_raster_calculation(
                [veg_base_uri_dict[veg_type][
                    base_veg_acc_bio],
                    this_veg_acc_bio_uri, this_veg_dis_bio_uri],
                adj_op,
                this_veg_adj_acc_bio_uri)

            # transition adjusted undisturbed soil
            this_veg_adj_acc_soil_uri = vars_dict['veg_adj_acc_soil_uris'] % (
                this_year, next_year, veg_type)
            carbon_raster_calculation(
                [veg_base_uri_dict[veg_type][
                    base_veg_acc_soil], this_veg_acc_soil_uri,
                    this_veg_dis_soil_uri],
                adj_op,
                this_veg_adj_acc_soil_uri)

            # transition adjusted disturbed biomass
            this_veg_adj_dis_bio_uri = vars_dict['veg_adj_dis_bio_uris'] % (
                this_year, next_year, veg_type)
            carbon_raster_calculation(
                [veg_base_uri_dict[veg_type][
                    base_veg_dis_bio], this_veg_dis_bio_uri],
                add_op,
                this_veg_adj_dis_bio_uri)

            # transition adjusted disturbed soil
            this_veg_adj_dis_soil_uri = vars_dict['veg_adj_dis_soil_uris'] % (
                this_year, next_year, veg_type)
            carbon_raster_calculation(
                [veg_base_uri_dict[veg_type][
                    base_veg_dis_soil], this_veg_dis_soil_uri],
                add_op,
                this_veg_adj_dis_soil_uri)

            # Emissions
            # biomass emissions
            this_veg_em_bio_uri = vars_dict['veg_em_bio_uris'] % (
                this_year, next_year, veg_type)
            carbon_raster_calculation(
                [this_veg_adj_dis_bio_uri],
                half_life_op_closure(
                    veg_type, vars_dict['half_life_field_bio'], t),
                this_veg_em_bio_uri)

            # soil emissions
            this_veg_em_soil_uri = vars_dict['veg_em_soil_uris'] % (
                this_year, next_year, veg_type)
            carbon_raster_calculation(
                [this_veg_adj_dis_soil_uri],
                half_life_op_closure(
                    veg_type, vars_dict['half_life_field_soil'], t),
                this_veg_em_soil_uri)

            # Emissions Adjustment
            # emissions adjusted disturbed biomass
            this_veg_adj_em_dis_bio_uri = vars_dict[
                'veg_adj_em_dis_bio_uris'] % (this_year, next_year, veg_type)
            carbon_raster_calculation(
                [this_veg_adj_dis_bio_uri, this_veg_em_bio_uri],
                sub_op,
                this_veg_adj_em_dis_bio_uri)

            # emissions adjusted disturbed soil
            this_veg_adj_em_dis_soil_uri = vars_dict[
                'veg_adj_em_dis_soil_uris'] % (this_year, next_year, veg_type)
            carbon_raster_calculation(
                [this_veg_adj_dis_soil_uri, this_veg_em_soil_uri],
                sub_op,
                this_veg_adj_em_dis_soil_uri)

            veg_acc_bio_uri_list.append(this_veg_acc_bio_uri)
            veg_acc_soil_uri_list.append(this_veg_acc_soil_uri)
            veg_dis_bio_uri_list.append(this_veg_dis_bio_uri)
            veg_dis_soil_uri_list.append(this_veg_dis_soil_uri)

            for name, uri in [
                (vars_dict['veg_acc_bio_uris'],
                    this_veg_acc_bio_uri),
                (vars_dict['veg_acc_soil_uris'],
                    this_veg_acc_soil_uri),
                (vars_dict['veg_dis_bio_uris'],
                    this_veg_dis_bio_uri),
                (vars_dict['veg_dis_soil_uris'],
                    this_veg_dis_soil_uri),
                (vars_dict['veg_adj_acc_bio_uris'],
                    this_veg_adj_acc_bio_uri),
                (vars_dict['veg_adj_acc_soil_uris'],
                    this_veg_adj_acc_soil_uri),
                (vars_dict['veg_adj_dis_bio_uris'],
                    this_veg_adj_dis_bio_uri),
                (vars_dict['veg_adj_dis_soil_uris'],
                    this_veg_adj_dis_soil_uri),
                (vars_dict['veg_em_bio_uris'],
                    this_veg_em_bio_uri),
                (vars_dict['veg_em_soil_uris'],
                    this_veg_em_soil_uri),
                (vars_dict['veg_adj_em_dis_bio_uris'],
                    this_veg_adj_em_dis_bio_uri),
                (vars_dict['veg_adj_em_dis_soil_uris'],
                    this_veg_adj_em_dis_soil_uri)]:
                    totals[this_year][veg_type][name] = _sum_uri(
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

            # DEBUG
            _print_raster("Current Year (%i) LULC Raster" % this_year, this_year_uri)
            _print_raster("Next Year (%i) LULC Raster" % next_year, next_year_uri)
            _print_raster("Veg %i accumulated carbon in biomass" % veg_type, this_veg_acc_bio_uri)
            _print_raster("Veg %i accumulated carbon in soil" % veg_type, this_veg_acc_soil_uri)
            # _print_raster("Veg %i disturbed carbon in biomass" % veg_type, this_veg_dis_bio_uri)
            # _print_raster("Veg %i disturbed carbon in soil" % veg_type, this_veg_dis_soil_uri)
            # _print_raster("Veg %i adjusted accumulated carbon in biomass" % veg_type, this_veg_adj_acc_bio_uri)
            # _print_raster("Veg %i adjusted accumulated carbon in soil" % veg_type, this_veg_adj_acc_soil_uri)
            # _print_raster("Veg %i adjusted disturbed carbon in biomass" % veg_type, this_veg_adj_dis_bio_uri)
            # _print_raster("Veg %i adjusted disturbed carbon in soil" % veg_type, this_veg_adj_dis_soil_uri)
            # _print_raster("Veg %i emissions from biomass" % veg_type, this_veg_em_bio_uri)
            # _print_raster("Veg %i emissions from soil" % veg_type, this_veg_em_soil_uri)
            # _print_raster("Veg %i adjusted emissions from biomass" % veg_type, this_veg_adj_em_dis_bio_uri)
            # _print_raster("Veg %i adjusted emissions from soil" % veg_type, this_veg_adj_em_dis_soil_uri)

        carbon_raster_calculation(
            this_total_carbon_uri_list,
            add_op,
            this_total_carbon_uri)

        stock_uri_dict[this_year] = this_total_carbon_uri

        carbon_raster_calculation(
            veg_acc_bio_uri_list,
            add_op,
            this_total_acc_bio_uri)

        carbon_raster_calculation(
            veg_acc_soil_uri_list,
            add_op,
            this_total_acc_soil_uri)

        carbon_raster_calculation(
            veg_dis_bio_uri_list,
            add_op,
            this_total_dis_bio_uri)

        carbon_raster_calculation(
            veg_dis_soil_uri_list,
            add_op,
            this_total_dis_soil_uri)

        # DEBUG
        _print_raster("Total Carbon Stock for Year %i" % this_year, this_total_carbon_uri)
        _print_raster("Carbon accumulated in biomass", this_total_acc_bio_uri)
        _print_raster("Carbon accumulated in soil", this_total_acc_soil_uri)
        _print_raster("Carbon disturbed in biomass", this_total_dis_bio_uri)
        _print_raster("Carbon disturbed in soil", this_total_dis_soil_uri)

    # analysis year raster
    this_total_carbon_uri = vars_dict['carbon_stock_uri'] % analysis_year
    this_total_carbon_uri_list = []
    for veg_type in veg_type_list:
        this_veg_litter_uri = os.path.join(
            workspace_dir, veg_litter_uris % (lulc_years[-1], veg_type))

        this_veg_adj_acc_bio_uri = os.path.join(
            workspace_dir,
            vars_dict['veg_adj_acc_bio_uris'] % (
                lulc_years[-1], analysis_year, veg_type))
        this_veg_adj_acc_soil_uri = os.path.join(
            workspace_dir,
            vars_dict['veg_adj_acc_soil_uris'] % (
                lulc_years[-1], analysis_year, veg_type))

        this_total_carbon_uri_list.append(this_veg_litter_uri)
        this_total_carbon_uri_list.append(this_veg_adj_acc_bio_uri)
        this_total_carbon_uri_list.append(this_veg_adj_acc_soil_uri)

    carbon_raster_calculation(
        this_total_carbon_uri_list,
        add_op,
        this_total_carbon_uri)

    _print_raster("Total Carbon Stock for Analysis Year", this_total_carbon_uri)

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
            LOGGER.info(
                "Calculating sequestration from %i to %i.",
                this_year,
                next_year)
            total_seq_uri = vars_dict[
                'net_sequestration_uri'] % (this_year, next_year)
            gain_uri = vars_dict['gain_uri'] % (this_year, next_year)
            loss_uri = vars_dict['loss_uri'] % (this_year, next_year)

            stock_uri_list = [stock_uri_dict[next_year],
                              stock_uri_dict[this_year]]
            carbon_raster_calculation(
                stock_uri_list,
                sub_op,
                total_seq_uri)

            carbon_raster_calculation(
                [total_seq_uri],
                pos_op,
                gain_uri)

            carbon_raster_calculation(
                [total_seq_uri],
                neg_op,
                loss_uri)

            _print_raster("Next Sequestration from %i to %i" % (this_year, next_year), total_seq_uri)
            _print_raster("Total Gain from %i to %i" % (this_year, next_year), gain_uri)
            _print_raster("Total Loss from %i to %i" % (this_year, next_year), loss_uri)

    vars_dict['totals'] = totals

    return vars_dict


def _datasource_from_dataset_bounding_box_uri(dataset_uri, datasource_uri):
    """
    Creates a shapefile with the bounding box from a raster.

    Args:
        dataset_uri (str): The uri for the input raster.
        datasource_uri (str): The uri for the output shapefile.

    Returns:
        None
    """
    LOGGER.debug("Creating extent from: %s", dataset_uri)
    LOGGER.debug("Storing extent in: %s", datasource_uri)

    # getting projection and bounding box information
    bounding_box = raster_utils.get_bounding_box(dataset_uri)
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


def _sum_uri(dataset_uri, datasource_uri):
    """
    Wrapper call to raster_utils.aggregate_raster_values_uri to extract
    total

    Args:
        dataset_uri (str): The uri for the input raster.
        datasource_uri (str): The uri for the input shapefile.

    Returns:
        float: The total of the raster values within the shapefile
    """
    total = raster_utils.aggregate_raster_values_uri(
        dataset_uri, datasource_uri).total
    return total.__getitem__(total.keys().pop())


def run_valuation(vars_dict):
    '''
    Runs the valuation component of the Blue Carbon model
    '''
    blue_carbon_csv_uri = vars_dict['blue_carbon_csv_uri']
    veg_acc_bio_uris = vars_dict['veg_acc_bio_uris']
    veg_acc_soil_uris = vars_dict['veg_acc_soil_uris']
    veg_type_list = vars_dict['veg_type_list']
    lulc_years = vars_dict['lulc_years']
    analysis_year = vars_dict['analysis_year']
    totals = vars_dict['totals']
    half_lives_by_veg_dict = vars_dict['half_lives_by_veg_dict']
    half_life_field_bio = vars_dict['half_life_field_bio']
    half_life_field_soil = vars_dict['half_life_field_soil']
    veg_em_bio_uris = vars_dict['veg_em_bio_uris']

    # tabulate results in new csv file
    csv = open(blue_carbon_csv_uri, 'w')

    header = ["Start Year", "End Year", "Accumulation"]
    # header += ["Veg %i Bio Emissions" % i for i in veg_type_list]
    # header += ["Veg %i Soil Emissions" % i for i in veg_type_list]
    header += ["Total Emissions", "Sequestration", "Value",
               "Discount Factor", "Cost"]

    csv.write(",".join(header))

    if not vars_dict["do_price_table"]:
        # If no price table, create carbon schedule
        carbon_schedule = {}
        for year in range(lulc_years[0], analysis_year + 1):
            carbon_schedule[year] = {"Price": float(
                vars_dict["carbon_value"]) * ((
                    1 + (float(vars_dict["rate_change"]) / float(100))) ** (
                    year - lulc_years[0]))}
    else:
        # Fetch carbon schedule from provided price table
        carbon_schedule = raster_utils.get_lookup_from_csv(
            vars_dict["carbon_schedule"], "Year")

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

            for source in [veg_acc_bio_uris,
                           veg_acc_soil_uris]:
                for veg_type in veg_type_list:
                    accumulation += totals[start_year][veg_type][
                        source] / period_op_dict[start_year][
                        "accumulation_divisor"]

            row.append(str(accumulation))

            period_op_dict[start_year]["biomass_half_life"][this_year] = {}
            for veg_type in veg_type_list:
                try:
                    c = _emissions_interpolation(
                        start_year,
                        end_year,
                        this_year,
                        next_year,
                        float(half_lives_by_veg_dict[veg_type][half_life_field_bio]))
                except ValueError:
                    c = 0

                period_op_dict[start_year]["biomass_half_life"][
                    this_year][veg_type] = c

                emissions += totals[start_year][veg_type][
                    veg_em_bio_uris] * c

            period_op_dict[start_year]["soil_half_life"][this_year] = {}
            for veg_type in veg_type_list:
                try:
                    c = _emissions_interpolation(
                        start_year,
                        end_year,
                        this_year,
                        next_year,
                        float(half_lives_by_veg_dict[veg_type][half_life_field_soil]))
                except ValueError:
                    c = 0

#####################################################################

                period_op_dict[start_year]["soil_half_life"][
                    this_year][veg_type] = c
                emissions += totals[start_year][veg_type][veg_em_soil_name] * c

                sequestration = accumulation - emissions

                row.append(str(emissions))
                row.append(str(sequestration))

                price = float(carbon_schedule[this_year][carbon_schedule_field_rate])
                period_op_dict[start_year]["price"][this_year] = price

                discount = (1 + (float(args["discount_rate"])/float(100))) ** (this_year-lulc_years[0])

                period_op_dict[start_year]["discount_factor"][this_year] = discount

                row.append(str(price))
                row.append(str(discount))
                row.append(str(sequestration * price / discount))

                csv.write("\n" + ",".join(row))

        csv.close()

        # generate value rasters
        value_uri_list = []
        for this_year, next_year in zip(
                lulc_years, lulc_years[1:]+[analysis_year]):
            LOGGER.info("Generating valuation rasters for era %i to %i.",
                        this_year,
                        next_year)

            # converting period variables into era variables
            accumulation_factor = 0
            LOGGER.debug("Accumulation factor: %s" % str(accumulation_factor))
            for period in period_op_dict[this_year]["price"]:
                price = period_op_dict[this_year]["price"][period]
                discount = period_op_dict[this_year]["discount_factor"][period]
                acc_fraction = 1 / float(period_op_dict[
                    this_year]["accumulation_divisor"])

                accumulation_factor += acc_fraction * price / float(discount)
                LOGGER.debug("Accumulation factor: %s" % str(
                    accumulation_factor))

            emission_biomass_veg_factor_dict = {}
            emission_soil_veg_factor_dict = {}
            for veg_type in veg_type_list:
                emission_biomass_veg_factor_dict[veg_type] = 0
                emission_soil_veg_factor_dict[veg_type] = 0

                for period in period_op_dict[this_year]["price"]:
                    price = period_op_dict[this_year]["price"][period]
                    discount = period_op_dict[this_year][
                        "discount_factor"][period]
                    bio_fract = period_op_dict[this_year][
                        "biomass_half_life"][period][veg_type]
                    soil_fract = period_op_dict[this_year][
                        "soil_half_life"][period][veg_type]

                    emission_biomass_veg_factor_dict[
                        veg_type] += bio_fract * price / float(discount)
                    emission_soil_veg_factor_dict[
                        veg_type] += soil_fract * price / float(discount)

            # create value rasters
            LOGGER.debug("Calculating accumulation value raster.")
            def acc_price_op(*values):
                if nodata_default_float in values:
                    return nodata_default_float
                else:
                    return sum([v * accumulation_factor for v in values])

            acc_uri_list = []
            for veg_type in veg_type_list:
                acc_uri_list.append(os.path.join(
                    workspace_dir, veg_acc_bio_name % (
                        this_year, next_year, veg_type)))
                acc_uri_list.append(os.path.join(
                    workspace_dir, veg_acc_soil_name % (
                        this_year, next_year, veg_type)))

            acc_value_uri = os.path.join(
                workspace_dir, acc_value_name  % (this_year, next_year))
            vectorize_carbon_datasets(
                acc_uri_list,
                acc_price_op,
                acc_value_uri)

            LOGGER.debug("Calculating biomass emissions value raster.")
            def em_op_closure(indecies, em_dict):
                def em_op(*values):
                    if nodata_default_float in values:
                        return nodata_default_float
                    else:
                        return sum([v * em_dict[i]  for v, i in zip(values, indecies)])
                return em_op

            em_uri_list = []
            em_uri_veg_type_list = []
            for veg_type in veg_type_list:
                em_uri_list.append(os.path.join(
                    workspace_dir, veg_em_bio_name  % (
                        this_year, next_year, veg_type)))
                em_uri_veg_type_list.append(veg_type)

            em_bio_value_uri = os.path.join(
                workspace_dir, em_bio_value_name  % (
                    this_year, next_year))
            vectorize_carbon_datasets(
                em_uri_list,
                em_op_closure(
                    em_uri_veg_type_list, emission_biomass_veg_factor_dict),
                em_bio_value_uri)

            LOGGER.debug("Calculating soil emissions value raster.")
            em_uri_list = []
            em_uri_veg_type_list = []
            for veg_type in veg_type_list:
                em_uri_list.append(os.path.join(workspace_dir, veg_em_soil_name  % (this_year, next_year, veg_type)))
                em_uri_veg_type_list.append(veg_type)

            em_soil_value_uri = os.path.join(
                workspace_dir, em_soil_value_name  % (
                    this_year, next_year))
            vectorize_carbon_datasets(
                em_uri_list,
                em_op_closure(em_uri_veg_type_list, emission_soil_veg_factor_dict),
                em_soil_value_uri)

            LOGGER.debug("Calculating sequestration value raster.")
            value_uri = os.path.join(workspace_dir, value_name  % (this_year, next_year))
            vectorize_carbon_datasets(
                [acc_value_uri,
                 em_bio_value_uri,
                 em_soil_value_uri],
                 sub_op,
                 value_uri)
            value_uri_list.append(value_uri)

        total_value_uri = os.path.join(
            workspace_dir, value_name % (lulc_years[0], analysis_year))
        vectorize_carbon_datasets(
            value_uri_list,
            add_op,
            total_value_uri)

    # open report
    report = open(report_uri, 'w')
    report.write("<HTML><TITLE>InVEST - Blue Carbon Report</TITLE><BODY>")

    # totals
    report.write("<B>Output Tables</B>")

    ## gains and losses
    report.write("<P><B>Carbon Gained/Lost</B>")
    column_name_list = ["Start-End Year",
                        "Gain",
                        "Loss",
                        "Net<BR>(Sequestration)"]

    report.write("\n<TABLE BORDER=1><TR><TD ALIGN=\"CENTER\"><B>%s</B></TD></TR>" % "</B></TD><TD ALIGN=\"CENTER\"><B>".join(column_name_list))

    for this_year, next_year in zip(
            lulc_years, (lulc_years+[analysis_year])[1:]):
        row = ["%i-%i" % (this_year, next_year)]

        total_seq_uri = os.path.join(
            workspace_dir, net_sequestration_name % (
                this_year, next_year))
        gain_uri = os.path.join(
            workspace_dir, gain_name % (
                this_year, next_year))
        loss_uri = os.path.join(
            workspace_dir, loss_name % (
                this_year, next_year))

        gain = sum_uri(gain_uri, extent_uri)
        loss = sum_uri(loss_uri, extent_uri)
        total_seq = sum_uri(total_seq_uri, extent_uri)

        row.append(str(int(gain)))
        row.append(str(int(loss)))
        row.append(str(int(total_seq)))

        report.write("\n<TR><TD ALIGN=\"RIGHT\">%s</TD></TR>" % "</TD><TD ALIGN=\"RIGHT\">".join(row))

    report.write("\n</TABLE>")

    # accumulation and disturbance
    report.write("<P><B>Carbon Accumulation/Disturbance</B>")
    column_name_list = ["Start-End Year",
                        "Biomass Accumulation",
                        "Soil Accumulation",
                        "Biomass Disturbance",
                        "Soil Disturbance",
                        "Emissions<BR>(Biomass)",
                        "Emissions<BR>(Soil)"]

    report.write("\n<TABLE BORDER=1><TR><TD ALIGN=\"CENTER\"><B>%s</B></TD></TR>" % "</B></TD><TD ALIGN=\"CENTER\"><B>".join(column_name_list))

    for i, this_year in enumerate(lulc_years):
        row = ["%i-%i" % (this_year, (lulc_years+[analysis_year])[i+1])]

        for name in [veg_acc_bio_name,
                     veg_dis_bio_name,
                     veg_acc_soil_name,
                     veg_dis_soil_name,
                     veg_em_bio_name,
                     veg_em_soil_name]:

            total = 0
            for veg_type in veg_type_list:
                total += totals[this_year][veg_type][name]

            row.append(total)
        row.append(row[1]+row[3]-row[5]-row[6])

        report.write("<TR><TD ALIGN=\"RIGHT\">%s</TD></TR>" % "</TD><TD ALIGN=\"RIGHT\">".join(
            [str(value) for value in [row[0],
             int(row[1]),
             int(row[3]),
             int(row[2]),
             int(row[4]),
             int(row[5]),
             int(row[6])]]))

    report.write("\n</TABLE>")

    # valuation
    if args["do_private_valuation"]:
        report.write("\n<P><P><B>Valuation (in specified valuation currency)</B>")
        column_name_list = ["Start-End Year",
                            "Accumulation",
                            "Biomass Emission",
                            "Soil Emission",
                            "Sequestration"]

        report.write("\n<TABLE BORDER=1><TR><TD ALIGN=\"CENTER\"><B>%s</B></TD></TR>" % "</B></TD><TD ALIGN=\"CENTER\"><B>".join(column_name_list))

        for this_year, next_year in zip(lulc_years, lulc_years[1:]+[analysis_year]):
            row = ["%i-%i" % (this_year, next_year)]

            acc_value_uri = os.path.join(workspace_dir, acc_value_name  % (this_year, next_year))
            em_bio_value_uri = os.path.join(workspace_dir, em_bio_value_name  % (this_year, next_year))
            em_soil_value_uri = os.path.join(workspace_dir, em_soil_value_name  % (this_year, next_year))
            value_uri = os.path.join(workspace_dir, value_name  % (this_year, next_year))

            for uri in [acc_value_uri,
                        em_bio_value_uri,
                        em_soil_value_uri,
                        value_uri]:

                row.append(str(int(sum_uri(uri, extent_uri))))

            report.write("<TR><TD ALIGN=\"RIGHT\">" + "</TD><TD ALIGN=\"RIGHT\">".join(row) + "</TR></TD>")

        report.write("\n</TABLE>")

    # input CSVs
    report.write("<P><P><B>Input Tables</B><P><P>")

    for csv_uri, name in [(carbon_uri, "Stock Carbon"),
                          (half_life_csv_uri, "Decay Rates (Half-Life)")]:
        csv = open(csv_uri)
        table = "\n<TABLE BORDER=1><TR><TD><B>"
        table += "</B></TD><TD><B>".join([td.replace(" (","<BR>(",1) for td in csv.readline().strip().split(",")])
        table += "</B></TD></TR>\n"
        for line in csv:
            table += "<TR><TD>" + line.strip().replace(",","</TD><TD>") + "</TD></TR>\n"
        table += "</TABLE>"

        csv.close()

        report.write("<P><P><B>%s</B>" % name)
        report.write(table)

    csv_uri = trans_uri
    name = "Transition Matrix"

    csv = open(csv_uri)
    table = "\n<TABLE BORDER=1><TR><TD><B>"
    table += csv.readline().strip().replace(",","</B></TD><TD><B>")
    table += "</B></TD></TR>\n"
    for line in csv:
        table += "<TR><TD>" + line.strip().replace(",","</TD><TD>") + "</TD></TR>\n"
    table += "</TABLE>"

    report.write("<P><P><B>%s</B>" % name)
    report.write(table)

    csv.close()

    for csv_uri, name in [(dis_bio_csv_uri, "Biomass Disturbance"),
                          (dis_soil_csv_uri, "Soil Disturbance")]:
        csv = open(csv_uri)
        table = "\n<TABLE BORDER=1><TR><TD><B>"
        table += csv.readline().strip().replace(",","</B></TD><TD><B>")
        table += "</B></TD></TR>\n"
        for line in csv:
            line = line.strip().split(",")
            line = line[:2] + [str(float(v) * 100)+"%" for v in line[2:]]
            table += "<TR><TD>" + ",".join(line).replace(",","</TD><TD>") + "</TD></TR>\n"
        table += "</TABLE>"

        report.write("<P><P><B>%s</B>" % name)
        report.write(table)

        csv.close()

    # close report
    report.write("\n</BODY></HTML>")
    report.close()

##
##    ##clean up
    driver = gdal.GetDriverByName('GTiff')
    driver.Delete(zero_raster_uri)


def _emissions_interpolation(start_year, end_year, this_year, next_year, alpha):
    """
    returns the proportion of the half-life contained within the subrange
    """
    return ((1 - (0.5 ** ((next_year - start_year)/alpha))) - (1 - (0.5 ** (
        (this_year - start_year)/alpha))))/(1 - (0.5 ** (
            (end_year - start_year)/alpha)))


def _print_raster(title, uri):
    '''
    Debugging function that prints raster sample to screen and/or writes to file
    '''
    with rio.open(uri) as src:
        a = src.read_band(1)  # [0:100:5, 0:100:5]
        print a
        print "^^^", title
        with open('rasters.txt', 'a') as f:
            pp.pprint(title, stream=f)
            pp.pprint(a[0:100:5, 0:100:5], stream=f)
            f.write('\n')
    # pass
Hide details
Change log
cc7ac2865ffd by Will Bierbower <wbierbower> on Mar 17, 2015   Diff
Update Issue 3285

Adding missing code in valuation function
Go to: 	
Project members, sign in to write a code review
Older revisions
 7b2a0b62f07c by Will Bierbower <wbierbower> on Mar 2, 2015   Diff 
 891a22642d92 by Will Bierbower <wbierbower> on Mar 2, 2015   Diff 
 430c7de62834 by Will Bierbower <wbierbower> on Feb 27, 2015   Diff 
All revisions of this file
File info
Size: 41241 bytes, 1075 lines
View raw file
