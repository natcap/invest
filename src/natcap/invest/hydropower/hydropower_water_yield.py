"""InVEST Hydropower Water Yield model."""
from __future__ import absolute_import

import shutil
import logging
import os
import math
import tempfile

import numpy
from osgeo import gdal
from osgeo import ogr

import pygeoprocessing
from .. import validation
from .. import utils

LOGGER = logging.getLogger('natcap.invest.hydropower.hydropower_water_yield')


def execute(args):
    """Annual Water Yield: Reservoir Hydropower Production.

    Executes the hydropower/water_yield model

    Parameters:
        args['workspace_dir'] (string): a uri to the directory that will write
            output and other temporary files during calculation. (required)

        args['lulc_uri'] (string): a uri to a land use/land cover raster whose
            LULC indexes correspond to indexes in the biophysical table input.
            Used for determining soil retention and other biophysical
            properties of the landscape. (required)

        args['depth_to_root_rest_layer_uri'] (string): a uri to an input
            raster describing the depth of "good" soil before reaching this
            restrictive layer (required)

        args['precipitation_uri'] (string): a uri to an input raster
            describing the average annual precipitation value for each cell
            (mm) (required)

        args['pawc_uri'] (string): a uri to an input raster describing the
            plant available water content value for each cell. Plant Available
            Water Content fraction (PAWC) is the fraction of water that can be
            stored in the soil profile that is available for plants' use.
            PAWC is a fraction from 0 to 1 (required)

        args['eto_uri'] (string): a uri to an input raster describing the
            annual average evapotranspiration value for each cell. Potential
            evapotranspiration is the potential loss of water from soil by
            both evaporation from the soil and transpiration by healthy
            Alfalfa (or grass) if sufficient water is available (mm)
            (required)

        args['watersheds_uri'] (string): a uri to an input shapefile of the
            watersheds of interest as polygons. (required)

        args['sub_watersheds_uri'] (string): a uri to an input shapefile of
            the subwatersheds of interest that are contained in the
            ``args['watersheds_uri']`` shape provided as input. (optional)

        args['biophysical_table_uri'] (string): a uri to an input CSV table of
            land use/land cover classes, containing data on biophysical
            coefficients such as root_depth (mm) and Kc, which are required.
            A column with header LULC_veg is also required which should
            have values of 1 or 0, 1 indicating a land cover type of
            vegetation, a 0 indicating non vegetation or wetland, water.
            NOTE: these data are attributes of each LULC class rather than
            attributes of individual cells in the raster map (required)

        args['seasonality_constant'] (float): floating point value between
            1 and 10 corresponding to the seasonal distribution of
            precipitation (required)

        args['results_suffix'] (string): a string that will be concatenated
            onto the end of file names (optional)

        args['calculate_water_scarcity'] (bool): if True, run water scarcity
            calculation using `args['demand_table_uri']`.

        args['demand_table_uri'] (string): (optional) if a non-empty string,
            a path to an input CSV
            table of LULC classes, showing consumptive water use for each
            landuse / land-cover type (cubic meters per year) to calculate
            water scarcity.

        args['valuation_table_uri'] (string): (optional) if a non-empty
            string, a path to an input CSV table of
            hydropower stations with the following fields to calculate
            valuation:
                ('ws_id', 'time_span', 'discount', 'efficiency', 'fraction',
                'cost', 'height', 'kw_price')

    Returns:
        None

    """
    LOGGER.info('Validating arguments')
    invalid_parameters = validate(args)
    if invalid_parameters:
        raise ValueError("Invalid parameters passed: %s" % invalid_parameters)

    # Construct folder paths
    workspace_dir = args['workspace_dir']
    output_dir = os.path.join(workspace_dir, 'output')
    per_pixel_output_dir = os.path.join(output_dir, 'per_pixel')
    utils.make_directories([workspace_dir, output_dir, per_pixel_output_dir])

    temp_dir = tempfile.mkdtemp(dir=workspace_dir)

    clipped_lulc_uri = os.path.join(temp_dir, 'clipped_lulc.tif')
    eto_uri = os.path.join(temp_dir, 'eto.tif')
    precip_uri = os.path.join(temp_dir, 'precip.tif')
    depth_to_root_rest_layer_uri = os.path.join(
        temp_dir, 'depth_to_root_rest_layer.tif')
    pawc_uri = os.path.join(temp_dir, 'pawc.tif')

    sheds_uri = args['watersheds_uri']
    seasonality_constant = float(args['seasonality_constant'])

    base_raster_path_list = [
        args['eto_uri'], args['precipitation_uri'],
        args['depth_to_root_rest_layer_uri'], args['pawc_uri'],
        args['lulc_uri']]

    aligned_raster_path_list = [
        eto_uri, precip_uri, depth_to_root_rest_layer_uri, pawc_uri,
        clipped_lulc_uri]

    target_pixel_size = pygeoprocessing.get_raster_info(
        args['lulc_uri'])['pixel_size']
    pygeoprocessing.align_and_resize_raster_stack(
        base_raster_path_list, aligned_raster_path_list,
        ['near'] * len(base_raster_path_list), target_pixel_size,
        'intersection', raster_align_index=4,
        base_vector_path_list=[sheds_uri])

    sub_sheds_uri = None
    # If subwatersheds was input get the URI
    if 'sub_watersheds_uri' in args and args['sub_watersheds_uri'] != '':
        sub_sheds_uri = args['sub_watersheds_uri']

    # Open/read in the csv file into a dictionary and add to arguments
    bio_dict = utils.build_lookup_from_csv(
        args['biophysical_table_uri'], 'lucode', to_lower=True)

    # Append a _ to the suffix if it's not empty and doens't already have one
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    # Paths for clipping the fractp/wyield raster to watershed polygons
    fractp_clipped_path = os.path.join(
        per_pixel_output_dir, 'fractp%s.tif' % file_suffix)
    wyield_clipped_path = os.path.join(
        per_pixel_output_dir, 'wyield%s.tif' % file_suffix)

    # Paths for the actual evapotranspiration rasters
    aet_path = os.path.join(per_pixel_output_dir, 'aet%s.tif' % file_suffix)

    # Paths for the watershed and subwatershed tables
    watershed_results_csv_uri = os.path.join(
        output_dir, 'watershed_results_wyield%s.csv' % file_suffix)
    subwatershed_results_csv_uri = os.path.join(
        output_dir, 'subwatershed_results_wyield%s.csv' % file_suffix)

    # The nodata value that will be used for created output rasters
    out_nodata = - 1.0

    # Break the bio_dict into three separate dictionaries based on
    # Kc, root_depth, and LULC_veg fields to use for reclassifying
    Kc_dict = {}
    root_dict = {}
    vegetated_dict = {}

    for lulc_code in bio_dict:
        Kc_dict[lulc_code] = bio_dict[lulc_code]['kc']
        vegetated_dict[lulc_code] = bio_dict[lulc_code]['lulc_veg']
        # If LULC_veg value is 1 get root depth value
        if vegetated_dict[lulc_code] == 1.0:
            root_dict[lulc_code] = bio_dict[lulc_code]['root_depth']
        # If LULC_veg value is 0 then we do not care about root
        # depth value so will just substitute in a 1.0 . This
        # value will not end up being used.
        else:
            root_dict[lulc_code] = 1.0

    # Create Kc raster from table values to use in future calculations
    LOGGER.info("Reclassifying temp_Kc raster")
    tmp_Kc_raster_uri = os.path.join(temp_dir, 'kc_raster.tif')
    pygeoprocessing.reclassify_raster(
        (clipped_lulc_uri, 1), Kc_dict, tmp_Kc_raster_uri, gdal.GDT_Float64,
        out_nodata)

    # Create root raster from table values to use in future calculations
    LOGGER.info("Reclassifying tmp_root raster")
    tmp_root_raster_uri = os.path.join(
        temp_dir, 'root_depth.tif')

    pygeoprocessing.reclassify_raster(
        (clipped_lulc_uri, 1), root_dict, tmp_root_raster_uri,
        gdal.GDT_Float64, out_nodata)

    # Create veg raster from table values to use in future calculations
    # of determining which AET equation to use
    LOGGER.info("Reclassifying tmp_veg raster")
    tmp_veg_raster_uri = os.path.join(temp_dir, 'veg.tif')
    pygeoprocessing.reclassify_raster(
        (clipped_lulc_uri, 1), vegetated_dict, tmp_veg_raster_uri,
        gdal.GDT_Float64, out_nodata)

    # Get out_nodata values so that we can avoid any issues when running
    # operations
    Kc_nodata = pygeoprocessing.get_raster_info(tmp_Kc_raster_uri)['nodata'][0]
    root_nodata = pygeoprocessing.get_raster_info(
        tmp_root_raster_uri)['nodata'][0]
    veg_nodata = pygeoprocessing.get_raster_info(
        tmp_veg_raster_uri)['nodata'][0]
    precip_nodata = pygeoprocessing.get_raster_info(precip_uri)['nodata'][0]
    eto_nodata = pygeoprocessing.get_raster_info(eto_uri)['nodata'][0]
    root_rest_layer_nodata = pygeoprocessing.get_raster_info(
        depth_to_root_rest_layer_uri)['nodata'][0]
    pawc_nodata = pygeoprocessing.get_raster_info(pawc_uri)['nodata'][0]

    def pet_op(eto_pix, Kc_pix):
        """Calculate the plant potential evapotranspiration.

        eto_pix (numpy.ndarray): a numpy array of ETo
        Kc_pix (numpy.ndarray): a numpy array of  Kc coefficient

        Returns:
            PET.

        """
        return numpy.where(
            (eto_pix == eto_nodata) | (Kc_pix == Kc_nodata),
            out_nodata, eto_pix * Kc_pix)

    # Get pixel size from tmp_Kc_raster_uri which should be the same resolution
    # as LULC raster
    tmp_pet_uri = os.path.join(temp_dir, 'pet.tif')
    LOGGER.info('Calculate PET from Ref Evap times Kc')
    pygeoprocessing.raster_calculator(
        [(eto_uri, 1), (tmp_Kc_raster_uri, 1)], pet_op, tmp_pet_uri,
        gdal.GDT_Float64, out_nodata)

    def fractp_op(Kc, eto, precip, root, soil, pawc, veg):
        """Calculate actual evapotranspiration fraction of precipitation.

        Parameters:
            Kc (numpy.ndarray): Kc (plant evapotranspiration
              coefficient) raster values
            eto (numpy.ndarray): potential evapotranspiration raster
              values (mm)
            precip (numpy.ndarray): precipitation raster values (mm)
            root (numpy.ndarray): root depth (maximum root depth for
               vegetated land use classes) raster values (mm)
            soil (numpy.ndarray): depth to root restricted layer raster
                values (mm)
            pawc (numpy.ndarray): plant available water content raster
               values
            veg (numpy.ndarray): 1 or 0 where 1 depicts the land type as
                vegetation and 0 depicts the land type as non
                vegetation (wetlands, urban, water, etc...). If 1 use
                regular AET equation if 0 use: AET = Kc * ETo

        Returns:
            fractp.

        """
        valid_mask = (
            (Kc != Kc_nodata) & (eto != eto_nodata) &
            (precip != precip_nodata) & (root != root_nodata) &
            (soil != root_rest_layer_nodata) & (pawc != pawc_nodata) &
            (veg != veg_nodata) & (precip != 0.0))

        # Compute Budyko Dryness index
        # Use the original AET equation if the land cover type is vegetation
        # If not vegetation (wetlands, urban, water, etc...) use
        # Alternative equation Kc * Eto
        phi = (Kc[valid_mask] * eto[valid_mask]) / precip[valid_mask]
        pet = Kc[valid_mask] * eto[valid_mask]

        # Calculate plant available water content (mm) using the minimum
        # of soil depth and root depth
        awc = numpy.where(
            root[valid_mask] < soil[valid_mask], root[valid_mask],
            soil[valid_mask]) * pawc[valid_mask]
        climate_w = (
            (awc / precip[valid_mask]) * seasonality_constant) + 1.25
        # Capping to 5.0 to set to upper limit if exceeded
        climate_w = numpy.where(climate_w > 5.0, 5.0, climate_w)

        # Compute evapotranspiration partition of the water balance
        aet_p = (
            1.0 + (pet / precip[valid_mask])) - (
                (1.0 + (pet / precip[valid_mask]) ** climate_w) ** (
                    1.0 / climate_w))

        # We take the minimum of the following values (phi, aet_p)
        # to determine the evapotranspiration partition of the
        # water balance (see users guide)
        veg_result = numpy.where(phi < aet_p, phi, aet_p)
        # Take the minimum of precip and Kc * ETo to avoid x / p > 1.0
        nonveg_result = numpy.where(
            precip[valid_mask] < Kc[valid_mask] * eto[valid_mask],
            precip[valid_mask],
            Kc[valid_mask] * eto[valid_mask]) / precip[valid_mask]
        # If veg is 1.0 use the result for vegetated areas else use result
        # for non veg areas
        result = numpy.where(
            veg[valid_mask] == 1.0,
            veg_result, nonveg_result)

        fractp = numpy.empty(valid_mask.shape)
        fractp[:] = out_nodata
        fractp[valid_mask] = result
        return fractp

    # List of rasters to pass into the vectorized fractp operation
    raster_list = [
        tmp_Kc_raster_uri, eto_uri, precip_uri, tmp_root_raster_uri,
        depth_to_root_rest_layer_uri, pawc_uri, tmp_veg_raster_uri]

    LOGGER.debug('Performing fractp operation')
    # Create clipped fractp_clipped raster
    pygeoprocessing.raster_calculator(
        [(x, 1) for x in raster_list], fractp_op, fractp_clipped_path,
        gdal.GDT_Float64, out_nodata)

    def wyield_op(fractp, precip):
        """Calculate water yield.

        Parameters:
           fractp (numpy.ndarray): fractp raster values
           precip (numpy.ndarray): precipitation raster values (mm)

        Returns:
            numpy.ndarray of water yield value (mm).

        """
        return numpy.where(
            (fractp == out_nodata) | (precip == precip_nodata),
            out_nodata, (1.0 - fractp) * precip)

    LOGGER.info('Performing wyield operation')
    # Create clipped wyield_clipped raster
    pygeoprocessing.raster_calculator(
        [(fractp_clipped_path, 1), (precip_uri, 1)], wyield_op,
        wyield_clipped_path, gdal.GDT_Float64, out_nodata)

    # Making a copy of watershed and sub-watershed to add water yield outputs
    # to
    watershed_results_uri = os.path.join(
        output_dir, 'watershed_results_wyield%s.shp' % file_suffix)
    esri_shapefile_driver = gdal.GetDriverByName('ESRI Shapefile')
    watershed_vector = gdal.OpenEx(sheds_uri, gdal.OF_VECTOR)
    esri_shapefile_driver.CreateCopy(watershed_results_uri, watershed_vector)
    watershed_vector = None

    if sub_sheds_uri is not None:
        subwatershed_results_uri = os.path.join(
            output_dir, 'subwatershed_results_wyield%s.shp' % file_suffix)
        subwatershed_vector = gdal.OpenEx(sub_sheds_uri, gdal.OF_VECTOR)
        esri_shapefile_driver.CreateCopy(
            subwatershed_results_uri, subwatershed_vector)
        subwatershed_vector = None

    def aet_op(fractp, precip, veg):
        """Compute actual evapotranspiration values.

        Parameters:
            fractp (numpy.ndarray): fractp raster values
            precip (numpy.ndarray): precipitation raster values (mm)
            veg (numpy.ndarray): value which AET equation was used in
                calculations of fractp. Value of 1.0 indicates original
                equation was used, value of 0.0 indicates the alternate
                version was used (AET = Kc * ETo)

        Returns:
            numpy.ndarray of actual evapotranspiration values (mm).

        """
        # checking if fractp >= 0 because it's a value that's between 0 and 1
        # and the nodata value is a large negative number.
        return numpy.where(
            (fractp >= 0) & (precip != precip_nodata),
            fractp * precip, out_nodata)

    LOGGER.debug('Performing aet operation')
    # Create clipped aet raster
    pygeoprocessing.raster_calculator(
        [(x, 1) for x in [
            fractp_clipped_path, precip_uri, tmp_veg_raster_uri]],
        aet_op, aet_path, gdal.GDT_Float64, out_nodata)

    if sub_sheds_uri is not None:
        # Create a list of tuples that pair up field names and raster uris so
        # that we can nicely do operations below
        sws_tuple_names_uris = [
            ('precip_mn', precip_uri),
            ('PET_mn', tmp_pet_uri),
            ('AET_mn', aet_path)]

        for key_name, rast_uri in sws_tuple_names_uris:
            # Aggregrate mean over the sub-watersheds for each uri listed in
            # 'sws_tuple_names_uri'
            sub_ws_stat_dict = pygeoprocessing.zonal_statistics(
                (rast_uri, 1), sub_sheds_uri, 'subws_id',
                ignore_nodata=False)

            # Add aggregated values to sub-watershed shapefile under new field
            # 'key_name'
            _add_zonal_stats_dict_to_shape(
                subwatershed_results_uri, sub_ws_stat_dict, key_name,
                'subws_id', 'mean')

        # Aggregate values for the water yield raster under the sub-watershed
        agg_wyield_stat_dict = pygeoprocessing.zonal_statistics(
            (wyield_clipped_path, 1), sub_sheds_uri, 'subws_id',
            ignore_nodata=False)
        # Add the wyield mean and number of pixels to the shapefile
        _add_zonal_stats_dict_to_shape(
            subwatershed_results_uri, agg_wyield_stat_dict, 'wyield_mn',
            'subws_id', 'mean')

        # Compute the water yield volume and water yield volume per hectare.
        # The values per sub-watershed will be added as fields in the
        # sub-watersheds shapefile.
        compute_water_yield_volume(subwatershed_results_uri)

        # List of wanted fields to output in the subwatershed CSV table
        field_list_sws = [
            'subws_id', 'num_pixels', 'precip_mn', 'PET_mn', 'AET_mn',
            'wyield_mn', 'wyield_vol']

        # Get a dictionary from the sub-watershed shapefiles attributes based
        # on the fields to be outputted to the CSV table
        wyield_values_sws = _extract_vector_table_by_key(
            subwatershed_results_uri, 'subws_id')

        wyield_value_dict_sws = filter_dictionary(
            wyield_values_sws, field_list_sws)

        # Write sub-watershed CSV table
        _write_table(subwatershed_results_csv_uri, wyield_value_dict_sws)

    # Create a list of tuples that pair up field names and raster uris so that
    # we can nicely do operations below
    ws_tuple_names_uris = [
        ('precip_mn', precip_uri), ('PET_mn', tmp_pet_uri),
        ('AET_mn', aet_path)]

    for key_name, rast_uri in ws_tuple_names_uris:
        # Aggregrate mean over the watersheds for each uri listed in
        # 'ws_tuple_names_uri'
        ws_stats_dict = pygeoprocessing.zonal_statistics(
            (rast_uri, 1), sheds_uri, 'ws_id', ignore_nodata=False)
        # Add aggregated values to watershed shapefile under new field
        # 'key_name'
        _add_zonal_stats_dict_to_shape(
            watershed_results_uri, ws_stats_dict, key_name, 'ws_id', 'mean')

    # Aggregate values for the water yield raster under the watershed
    wyield_stats_dict = pygeoprocessing.zonal_statistics(
        (wyield_clipped_path, 1), sheds_uri, 'ws_id', ignore_nodata=False)
    # Add the wyield mean and number of pixels to the shapefile
    _add_zonal_stats_dict_to_shape(
        watershed_results_uri, wyield_stats_dict, 'wyield_mn', 'ws_id',
        'mean')

    compute_water_yield_volume(watershed_results_uri)

    # List of wanted fields to output in the watershed CSV table
    field_list_ws = [
        'ws_id', 'num_pixels', 'precip_mn', 'PET_mn', 'AET_mn',
        'wyield_mn', 'wyield_vol']

    # Get a dictionary from the watershed shapefiles attributes based on the
    # fields to be outputted to the CSV table
    wyield_values_ws = _extract_vector_table_by_key(
        watershed_results_uri, 'ws_id')

    wyield_value_dict_ws = filter_dictionary(wyield_values_ws, field_list_ws)

    # removing temporary files
    for temp_path in [
            tmp_Kc_raster_uri, tmp_root_raster_uri, tmp_pet_uri,
            tmp_veg_raster_uri]:
        try:
            os.remove(temp_path)
        except OSError:
            LOGGER.warn("could not delete temporary files in %s", temp_path)

    # Check to see if Water Scarcity was selected to run
    if ('calculate_water_scarcity' not in args or
            not args['calculate_water_scarcity']):
        # Since Scarcity and Valuation are not selected write out
        # the CSV table
        _write_table(watershed_results_csv_uri, wyield_value_dict_ws)
        # The rest of the function is water scarcity and valuation, so we can
        # quit now
        try:
            shutil.rmtree(temp_dir)
        except OSError:
            LOGGER.warn("could not delete temporary directory %s", temp_dir)
        return

    LOGGER.info('Starting Water Scarcity')
    demand_dict = utils.build_lookup_from_csv(
        args['demand_table_uri'], 'lucode')
    demand_reclassify_dict = dict(
        [(lucode, demand_dict[lucode]['demand']) for lucode in demand_dict])

    # Create demand raster from table values to use in future calculations
    LOGGER.info("Reclassifying demand raster")
    tmp_demand_uri = os.path.join(temp_dir, 'demand.tif')
    pygeoprocessing.reclassify_raster(
        (clipped_lulc_uri, 1), demand_reclassify_dict, tmp_demand_uri,
        gdal.GDT_Float64, out_nodata)

    # Aggregate the consumption volume over sheds using the
    # reclassfied demand raster
    LOGGER.info('Aggregating Consumption Volume and Mean')

    consump_ws_stats_dict = pygeoprocessing.zonal_statistics(
        (tmp_demand_uri, 1), sheds_uri, 'ws_id', ignore_nodata=False)

    # Add aggregated consumption to sheds shapefiles
    _add_zonal_stats_dict_to_shape(
        watershed_results_uri, consump_ws_stats_dict, 'consum_vol', 'ws_id',
        'sum')

    # Add aggregated consumption means to sheds shapefiles
    _add_zonal_stats_dict_to_shape(
        watershed_results_uri, consump_ws_stats_dict, 'consum_mn', 'ws_id',
        'mean')

    # Calculate the realised water supply after consumption
    LOGGER.info('Calculating RSUPPLY')
    compute_rsupply_volume(watershed_results_uri)

    # List of wanted fields to output in the watershed CSV table
    scarcity_field_list_ws = [
        'ws_id', 'consum_vol', 'consum_mn', 'rsupply_vl', 'rsupply_mn']

    # Aggregate water yield and water scarcity fields, where we exclude the
    # first field in the scarcity list because they are duplicates already
    # in the water yield list
    field_list_ws = field_list_ws + scarcity_field_list_ws[1:]

    # Get a dictionary from the watershed shapefiles attributes based on the
    # fields to be outputted to the CSV table
    watershed_values = _extract_vector_table_by_key(
        watershed_results_uri, 'ws_id')

    watershed_dict = filter_dictionary(watershed_values, field_list_ws)

    try:
        shutil.rmtree(temp_dir)
    except OSError:
        LOGGER.warn("Could not remove temporary directory %s", temp_dir)

    # Check to see if Valuation was selected to run
    if 'valuation_container' in args:
        valuation_checked = args['valuation_container']
    else:
        valuation_checked = False

    if not valuation_checked:
        LOGGER.debug('Valuation Not Selected')
        # Since Valuation are not selected write out
        # the CSV table
        _write_table(watershed_results_csv_uri, watershed_dict)
        # The rest of the function is valuation, so we can quit now
        return

    LOGGER.info('Starting Valuation Calculation')

    # Open/read in valuation parameters from CSV file
    valuation_params = utils.build_lookup_from_csv(
        args['valuation_table_uri'], 'ws_id')

    # Compute NPV and Energy for the watersheds
    LOGGER.info('Calculating NPV/ENERGY for Sheds')
    compute_watershed_valuation(watershed_results_uri, valuation_params)

    # List of fields for the valuation run
    val_field_list_ws = ['ws_id', 'hp_energy', 'hp_val']

    # Aggregate water yield, water scarcity, and valuation fields, where we
    # exclude the first field in the list because they are duplicates
    field_list_ws = field_list_ws + val_field_list_ws[1:]

    # Get a dictionary from the watershed shapefiles attributes based on the
    # fields to be outputted to the CSV table
    watershed_values_ws = _extract_vector_table_by_key(
        watershed_results_uri, 'ws_id')

    watershed_dict_ws = filter_dictionary(watershed_values_ws, field_list_ws)

    # Write out the CSV Table
    _write_table(watershed_results_csv_uri, watershed_dict_ws)


def compute_watershed_valuation(watersheds_uri, val_dict):
    """Compute net present value and energy for the watersheds.

    Parameters:
        watersheds_uri (string): - a URI path to an OGR shapefile for the
            watershed results. Where the results will be added.

        val_dict (mappable): - a python dictionary that has all the valuation
            parameters for each watershed

    Returns:
        None.

    """
    LOGGER.debug(val_dict)
    ws_ds = gdal.OpenEx(watersheds_uri, 1)
    ws_layer = ws_ds.GetLayer()

    # The field names for the new attributes
    energy_field = 'hp_energy'
    npv_field = 'hp_val'

    # Add the new fields to the shapefile
    for new_field in [energy_field, npv_field]:
        field_defn = ogr.FieldDefn(new_field, ogr.OFTReal)
        field_defn.SetWidth(24)
        field_defn.SetPrecision(11)
        ws_layer.CreateField(field_defn)

    num_features = ws_layer.GetFeatureCount()
    # Iterate over the number of features (polygons)
    for feat_id in xrange(num_features):
        ws_feat = ws_layer.GetFeature(feat_id)
        # Get the indices for the output fields
        energy_id = ws_feat.GetFieldIndex(energy_field)
        npv_id = ws_feat.GetFieldIndex(npv_field)

        # Get the watershed ID to index into the valuation parameter dictionary
        ws_index = ws_feat.GetFieldIndex('ws_id')
        ws_id = ws_feat.GetField(ws_index)
        # Get the rsupply volume for the watershed
        rsupply_vl_id = ws_feat.GetFieldIndex('rsupply_vl')
        rsupply_vl = ws_feat.GetField(rsupply_vl_id)

        # Get the valuation parameters for watershed 'ws_id'
        val_row = val_dict[ws_id]

        # Compute hydropower energy production (KWH)
        # This is from the equation given in the Users' Guide
        energy = (
            val_row['efficiency'] * val_row['fraction'] * val_row['height'] *
            rsupply_vl * 0.00272)

        dsum = 0.
        # Divide by 100 because it is input at a percent and we need
        # decimal value
        disc = val_row['discount'] / 100.0
        # To calculate the summation of the discount rate term over the life
        # span of the dam we can use a geometric series
        ratio = 1. / (1. + disc)
        if ratio != 1.:
            dsum = (1. - math.pow(ratio, val_row['time_span'])) / (1. - ratio)

        npv = ((val_row['kw_price'] * energy) - val_row['cost']) * dsum

        # Get the volume field index and add value
        ws_feat.SetField(energy_id, energy)
        ws_feat.SetField(npv_id, npv)

        ws_layer.SetFeature(ws_feat)


def compute_rsupply_volume(watershed_results_uri):
    """Calculate the total realized water supply volume.

     and the mean realized
        water supply volume per hectare for the given sheds. Output units in
        cubic meters and cubic meters per hectare respectively.

    Parameters:
        watershed_results_uri (string): a path to a vector that contains
            fields 'rsupply_vl' and 'rsupply_mn' to caluclate water supply
            volumne per hectare and cubic meters.

    Returns:
        None.

    """
    ws_ds = gdal.OpenEx(watershed_results_uri, 1)
    ws_layer = ws_ds.GetLayer()

    # The field names for the new attributes
    rsupply_vol_name = 'rsupply_vl'
    rsupply_mn_name = 'rsupply_mn'

    # Add the new fields to the shapefile
    for new_field in [rsupply_vol_name, rsupply_mn_name]:
        field_defn = ogr.FieldDefn(new_field, ogr.OFTReal)
        field_defn.SetWidth(24)
        field_defn.SetPrecision(11)
        ws_layer.CreateField(field_defn)

    num_features = ws_layer.GetFeatureCount()
    # Iterate over the number of features (polygons)
    for feat_id in xrange(num_features):
        ws_feat = ws_layer.GetFeature(feat_id)
        # Get mean water yield value
        wyield_mn_id = ws_feat.GetFieldIndex('wyield_mn')
        wyield_mn = ws_feat.GetField(wyield_mn_id)

        # Get water demand/consumption values
        wyield_id = ws_feat.GetFieldIndex('wyield_vol')
        wyield = ws_feat.GetField(wyield_id)

        consump_vol_id = ws_feat.GetFieldIndex('consum_vol')
        consump_vol = ws_feat.GetField(consump_vol_id)
        consump_mn_id = ws_feat.GetFieldIndex('consum_mn')
        consump_mn = ws_feat.GetField(consump_mn_id)

        # Calculate realized supply
        rsupply_vol = wyield - consump_vol
        rsupply_mn = wyield_mn - consump_mn

        # Get the indices for the output fields and set their values
        rsupply_vol_index = ws_feat.GetFieldIndex(rsupply_vol_name)
        ws_feat.SetField(rsupply_vol_index, rsupply_vol)
        rsupply_mn_index = ws_feat.GetFieldIndex(rsupply_mn_name)
        ws_feat.SetField(rsupply_mn_index, rsupply_mn)

        ws_layer.SetFeature(ws_feat)


def filter_dictionary(dict_data, values):
    """
    Create a subset of a dictionary given keys found in a list.

    The incoming dictionary should have keys that point to dictionary's.
        Create a subset of that dictionary by using the same outer keys
        but only using the inner key:val pair if that inner key is found
        in the values list.

    Parameters:
        dict_data (mappable): A dictionary containing values that are also
            dictionaries.
        values (list): a list of keys to copy from the second level
            dictionaries in `dict_data`.

    Returns:
        a dictionary that's a copy of `dict_data` with `values` removed from
        it.

    """
    new_dict = {}

    for key, val in dict_data.iteritems():
        new_dict[key] = {}
        for sub_key, sub_val in val.iteritems():
            if sub_key in values:
                new_dict[key][sub_key] = sub_val

    return new_dict


def _write_table(target_path, data_row_map):
    """Create a csv table from a dictionary.

    Parameters:
        target_path (string): a file path for the new table, if 'ws_id' is
            contained in the field names it will be placed as the first
            column in the table, otherwise columns output in alphabetical
            order.

        data_row_map (mappable): a mapping of row number to a mappable of
            column name to value. The column names should be identical for
            all rows. Example:

            data_row_map = {
                0 : {'id':1, 'precip':43, 'total': 65},
                1 : {'id':2, 'precip':65, 'total': 94}}

    Returns:
        None.

    """
    #  Sort the keys so that the rows are written in order
    sorted_row_index_list = sorted(data_row_map.keys())
    sorted_column_names = sorted(data_row_map.itervalues().next().keys())
    if 'ws_id' in sorted_column_names:
        ws_index = sorted_column_names.index('ws_id')
        sorted_column_names = (
            [sorted_column_names[ws_index]] +
            sorted_column_names[:ws_index] +
            sorted_column_names[ws_index+1:])
    with open(target_path, 'wb') as csv_file:
        #  Write the columns as the first row in the table
        csv_file.write(','.join(sorted_column_names))
        csv_file.write('\n')

        # Write the rows from the dictionary
        for row_index in sorted_row_index_list:
            csv_file.write(','.join(
                [str(data_row_map[row_index][key])
                 for key in sorted_column_names]))
            csv_file.write('\n')
    csv_file.close()


def compute_water_yield_volume(shape_uri):
    """Calculate the water yield volume per sub-watershed or watershed.

        shape_uri - a URI path a vector for the sub-watershed
            or watershed shapefile. This shapefiles features should have a
            'wyield_mn' attribute. Results are added to a 'wyield_vol' field
            in `shape_uri` whose units are in cubic meters.

    Returns:
        None.

    """
    shape = gdal.OpenEx(shape_uri, 1)
    layer = shape.GetLayer()

    # The field names for the new attributes
    vol_name = 'wyield_vol'

    # Add the new field to the shapefile
    field_defn = ogr.FieldDefn(vol_name, ogr.OFTReal)
    field_defn.SetWidth(24)
    field_defn.SetPrecision(11)
    layer.CreateField(field_defn)

    num_features = layer.GetFeatureCount()
    # Iterate over the number of features (polygons) and compute volume
    for feat_id in xrange(num_features):
        feat = layer.GetFeature(feat_id)
        wyield_mn_id = feat.GetFieldIndex('wyield_mn')
        wyield_mn = feat.GetField(wyield_mn_id)
        geom = feat.GetGeometryRef()
        # Calculate water yield volume,
        # 1000 is for converting the mm of wyield to meters
        vol = wyield_mn * geom.Area() / 1000.0
        # Get the volume field index and add value
        vol_index = feat.GetFieldIndex(vol_name)
        feat.SetField(vol_index, vol)

        layer.SetFeature(feat)


def _add_zonal_stats_dict_to_shape(
        shape_path, stats_map, field_name, key, aggregate_field_id):
    """Add a new field to a shapefile with values from a dictionary.

        The dictionaries keys should match to the values of a unique fields
        values in the shapefile

        shape_path (string): a path to a vector with a unique field
            `key`. The field `key` should have values that map to the keys
            of  `stats_map`.

        stats_map (mappable): a dictionary in the format generated by
            pygeoprocessing.zonal_statistics that contains at least the key
            value of `aggregate_field_id` per feature id.

        field_name (str): a string for the name of the new field to add to
            the target vector.

        key (str): a string for the field name in the `shape_path` vector
            a unique features per record.

        aggregate_field_id (string): one of 'min' 'max' 'sum' 'mean' 'count'
            or 'nodata_count' as defined by pygeoprocessing.zonal_statistics.

    Returns:
        None

    """
    vector = gdal.OpenEx(shape_path, gdal.OF_VECTOR | gdal.GA_Update)
    layer = vector.GetLayer()

    # Create the new field
    field_defn = ogr.FieldDefn(field_name, ogr.OFTReal)
    field_defn.SetWidth(24)
    field_defn.SetPrecision(11)
    layer.CreateField(field_defn)

    # Get the number of features (polygons) and iterate through each
    num_features = layer.GetFeatureCount()
    for feat_id in xrange(num_features):
        feat = layer.GetFeature(feat_id)

        # Get the index for the unique field
        ws_id = feat.GetFieldIndex(key)

        # Get the unique value that will index into the dictionary as a key
        ws_val = feat.GetField(ws_id)

        # Using the unique value from the field of the feature, index into the
        # dictionary to get the corresponding value
        if aggregate_field_id == 'mean':
            field_val = float(
                stats_map[ws_val]['sum']) / stats_map[ws_val]['count']
        else:
            field_val = float(stats_map[ws_val][aggregate_field_id])

        # Get the new fields index and set the new value for the field
        field_index = feat.GetFieldIndex(field_name)
        feat.SetField(field_index, field_val)

        layer.SetFeature(feat)


def _extract_vector_table_by_key(vector_path, key_field):
    """Return vector attribute table of first layer as dictionary.

    Create a dictionary lookup table of the features in the attribute table
    of the vector referenced by vector_path.

    Parameters:
        vector_path (string): a path to an OGR vector
        key_field: a field in vector_path that refers to a key value
            for each row such as a polygon id.

    Returns:
        attribute_dictionary (dict): returns a dictionary of the
            form {key_field_0: {field_0: value0, field_1: value1}...}

    """
    # Pull apart the vector
    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()
    layer_def = layer.GetLayerDefn()

    # Build up a list of field names for the vector table
    field_names = []
    for field_id in xrange(layer_def.GetFieldCount()):
        field_def = layer_def.GetFieldDefn(field_id)
        field_names.append(field_def.GetName())

    # Loop through each feature and build up the dictionary representing the
    # attribute table
    attribute_dictionary = {}
    for feature in layer:
        feature_fields = {}
        for field_name in field_names:
            feature_fields[field_name] = feature.GetField(field_name)
        key_value = feature.GetField(key_field)
        attribute_dictionary[key_value] = feature_fields

    layer.ResetReading()
    # Explictly clean up the layers so the files close
    layer = None
    vector = None
    return attribute_dictionary


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to `execute`'s contract.

    Parameters:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in `execute` docstring.
        limit_to (str): (optional) if not None indicates that validation
            should only occur on the args[limit_to] value. The intent that
            individual key validation could be significantly less expensive
            than validating the entire `args` dictionary.

    Returns:
        list of ([invalid key_a, invalid_keyb, ...], 'warning/error message')
            tuples. Where an entry indicates that the invalid keys caused
            the error message in the second part of the tuple. This should
            be an empty list if validation succeeds.

    """
    missing_key_list = []
    no_value_list = []
    validation_error_list = []

    required_keys = [
        'workspace_dir',
        'precipitation_uri',
        'eto_uri',
        'depth_to_root_rest_layer_uri',
        'pawc_uri',
        'lulc_uri',
        'watersheds_uri',
        'biophysical_table_uri',
        'seasonality_constant']

    for key in required_keys:
        if limit_to is None or limit_to == key:
            if key not in args:
                missing_key_list.append(key)
            elif args[key] in ['', None]:
                no_value_list.append(key)

    if len(missing_key_list) > 0:
        # if there are missing keys, we have raise KeyError to stop hard
        raise KeyError(
            "The following keys were expected in `args` but were missing " +
            ', '.join(missing_key_list))

    if len(no_value_list) > 0:
        validation_error_list.append(
            (no_value_list, 'parameter has no value'))

    file_type_list = [
        ('lulc_uri', 'raster'),
        ('eto_uri', 'raster'),
        ('precipitation_uri', 'raster'),
        ('depth_to_root_rest_layer_uri', 'raster'),
        ('pawc_uri', 'raster'),
        ('watersheds_uri', 'vector'),
        ('biophysical_table_uri', 'table'),
        ('demand_table_uri', 'table'),
        ('valuation_table_uri', 'table'),
        ]

    if ('sub_watersheds_uri' in args and
            args['sub_watersheds_uri'] != ''):
        file_type_list.append(('sub_watersheds_uri', 'vector'))

    # check that existing/optional files are the correct types
    with utils.capture_gdal_logging():
        for key, key_type in file_type_list:
            if (limit_to is None or limit_to == key) and key in args:
                if not os.path.exists(args[key]):
                    validation_error_list.append(
                        ([key], 'not found on disk'))
                    continue
                if key_type == 'raster':
                    raster = gdal.OpenEx(args[key], gdal.OF_RASTER)
                    if raster is None:
                        validation_error_list.append(
                            ([key], 'not a raster'))
                    del raster
                elif key_type == 'vector':
                    vector = gdal.OpenEx(args[key], gdal.OF_VECTOR)
                    if vector is None:
                        validation_error_list.append(
                            ([key], 'not a vector'))
                    del vector

    return validation_error_list
