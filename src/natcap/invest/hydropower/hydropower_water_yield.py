"""Module that contains the core computational components for the hydropower
    model including the water yield, water scarcity, and valuation functions"""

import logging
import os
import csv
import math

import numpy
from osgeo import gdal
from osgeo import ogr

import pygeoprocessing.geoprocessing

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
            both evaporation from the soil and transpiration by healthy Alfalfa
            (or grass) if sufficient water is available (mm) (required)

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

        args['demand_table_uri'] (string): a uri to an input CSV table of
            LULC classes, showing consumptive water use for each landuse /
            land-cover type (cubic meters per year) (required for water
            scarcity)

        args['valuation_table_uri'] (string): a uri to an input CSV table of
            hydropower stations with the following fields (required for
            valuation):
            ('ws_id', 'time_span', 'discount', 'efficiency', 'fraction',
            'cost', 'height', 'kw_price')

    Returns:
        None"""

    LOGGER.info('Starting Water Yield Core Calculations')

    # Construct folder paths
    workspace = args['workspace_dir']
    output_dir = os.path.join(workspace, 'output')
    per_pixel_output_dir = os.path.join(output_dir, 'per_pixel')
    pygeoprocessing.geoprocessing.create_directories([
        workspace, output_dir, per_pixel_output_dir])

    clipped_lulc_uri = pygeoprocessing.geoprocessing.temporary_filename()
    eto_uri = pygeoprocessing.geoprocessing.temporary_filename()
    precip_uri = pygeoprocessing.geoprocessing.temporary_filename()
    depth_to_root_rest_layer_uri = (
        pygeoprocessing.geoprocessing.temporary_filename())
    pawc_uri = pygeoprocessing.geoprocessing.temporary_filename()

    sheds_uri = args['watersheds_uri']
    seasonality_constant = float(args['seasonality_constant'])

    original_raster_uris = [
        args['eto_uri'], args['precipitation_uri'],
        args['depth_to_root_rest_layer_uri'], args['pawc_uri'],
        args['lulc_uri']]

    aligned_raster_uris = [
        eto_uri, precip_uri, depth_to_root_rest_layer_uri, pawc_uri,
        clipped_lulc_uri]

    pixel_size_out = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
        args['lulc_uri'])
    pygeoprocessing.geoprocessing.align_dataset_list(
        original_raster_uris, aligned_raster_uris,
        ['nearest'] * len(original_raster_uris),
        pixel_size_out, 'intersection', 4,
        aoi_uri=sheds_uri)

    sub_sheds_uri = None
    # If subwatersheds was input get the URI
    if 'sub_watersheds_uri' in args and args['sub_watersheds_uri'] != '':
        sub_sheds_uri = args['sub_watersheds_uri']

    # Open/read in the csv file into a dictionary and add to arguments
    bio_dict = {}
    biophysical_table_file = open(args['biophysical_table_uri'], 'rU')
    reader = csv.DictReader(biophysical_table_file)
    for row in reader:
        bio_dict[int(row['lucode'])] = {
            'Kc':float(row['Kc']), 'root_depth':float(row['root_depth']),
            'LULC_veg':float(row['LULC_veg'])
            }

    biophysical_table_file.close()

    # Append a _ to the suffix if it's not empty and doens't already have one
    try:
        file_suffix = args['results_suffix']
        if file_suffix != "" and not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''

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
        Kc_dict[lulc_code] = bio_dict[lulc_code]['Kc']
        vegetated_dict[lulc_code] = bio_dict[lulc_code]['LULC_veg']
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
    tmp_Kc_raster_uri = pygeoprocessing.geoprocessing.temporary_filename()

    pygeoprocessing.geoprocessing.reclassify_dataset_uri(
            clipped_lulc_uri, Kc_dict, tmp_Kc_raster_uri, gdal.GDT_Float64,
            out_nodata)

    # Create root raster from table values to use in future calculations
    LOGGER.info("Reclassifying tmp_root raster")
    tmp_root_raster_uri = pygeoprocessing.geoprocessing.temporary_filename()

    pygeoprocessing.geoprocessing.reclassify_dataset_uri(
            clipped_lulc_uri, root_dict, tmp_root_raster_uri, gdal.GDT_Float64,
            out_nodata)

    # Create veg raster from table values to use in future calculations
    # of determining which AET equation to use
    LOGGER.info("Reclassifying tmp_veg raster")
    tmp_veg_raster_uri = pygeoprocessing.geoprocessing.temporary_filename()

    pygeoprocessing.geoprocessing.reclassify_dataset_uri(
            clipped_lulc_uri, vegetated_dict, tmp_veg_raster_uri, gdal.GDT_Float64,
            out_nodata)

    # Get out_nodata values so that we can avoid any issues when running
    # operations
    Kc_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
        tmp_Kc_raster_uri)
    root_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
        tmp_root_raster_uri)
    veg_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
        tmp_veg_raster_uri)
    precip_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
        precip_uri)
    eto_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
        eto_uri)
    root_rest_layer_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
        depth_to_root_rest_layer_uri)
    pawc_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
        pawc_uri)

    def pet_op(eto_pix, Kc_pix):
        """Vectorize operation for calculating the plant potential
            evapotranspiration

            eto_pix - a float value for ETo
            Kc_pix - a float value for Kc coefficient

            returns - a float value for pet"""
        return numpy.where(
                (eto_pix == eto_nodata) | (Kc_pix == Kc_nodata),
                out_nodata, eto_pix * Kc_pix)

    # Get pixel size from tmp_Kc_raster_uri which should be the same resolution
    # as LULC raster
    pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(tmp_Kc_raster_uri)
    tmp_pet_uri = pygeoprocessing.geoprocessing.temporary_filename()

    LOGGER.debug('Calculate PET from Ref Evap times Kc')
    pygeoprocessing.geoprocessing.vectorize_datasets(
            [eto_uri, tmp_Kc_raster_uri], pet_op, tmp_pet_uri, gdal.GDT_Float64,
            out_nodata, pixel_size, 'intersection', aoi_uri=sheds_uri,
            vectorize_op=False)

    # Dictionary of out_nodata values corresponding to values for fractp_op
    # that will help avoid any out_nodata calculation issues
    fractp_nodata_dict = {
        'Kc':Kc_nodata,
        'eto':eto_nodata,
        'precip':precip_nodata,
        'root':root_nodata,
        'veg':veg_nodata,
        'soil':root_rest_layer_nodata,
        'pawc':pawc_nodata,
        }
    def fractp_op(Kc, eto, precip, root, soil, pawc, veg):
        """Function that calculates the fractp (actual evapotranspiration
            fraction of precipitation) raster

        Kc - numpy array with the Kc (plant evapotranspiration
              coefficient) raster values
        eto - numpy array with the potential evapotranspiration raster
              values (mm)
        precip - numpy array with the precipitation raster values (mm)
        root - numpy array with the root depth (maximum root depth for
               vegetated land use classes) raster values (mm)
        soil - numpy array with the depth to root restricted layer raster
            values (mm)
        pawc - numpy array with the plant available water content raster
               values
        veg - numpy array with a 1 or 0 where 1 depicts the land type as
                vegetation and 0 depicts the land type as non
                vegetation (wetlands, urban, water, etc...). If 1 use
                regular AET equation if 0 use: AET = Kc * ETo

        returns - fractp value
        """

        valid_mask = (
            (Kc != Kc_nodata) & (eto != eto_nodata) &
            (precip != precip_nodata) & (root != root_nodata) &
            (soil != root_rest_layer_nodata) & (pawc != pawc_nodata) &
            (veg != veg_nodata) & (precip != 0.0) & (Kc != 0.0) &
            (eto != 0.0))

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
    LOGGER.debug(fractp_nodata_dict)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        raster_list, fractp_op, fractp_clipped_path, gdal.GDT_Float64,
        out_nodata, pixel_size, 'intersection', aoi_uri=sheds_uri,
        vectorize_op=False)

    def wyield_op(fractp, precip):
        """Function that calculates the water yeild raster

           fractp - numpy array with the fractp raster values
           precip - numpy array with the precipitation raster values (mm)

           returns - water yield value (mm)"""
        return numpy.where(
                (fractp == out_nodata) | (precip == precip_nodata),
                out_nodata, (1.0 - fractp) * precip)

    LOGGER.debug('Performing wyield operation')
    # Create clipped wyield_clipped raster
    pygeoprocessing.geoprocessing.vectorize_datasets(
            [fractp_clipped_path, precip_uri], wyield_op, wyield_clipped_path,
            gdal.GDT_Float64, out_nodata, pixel_size, 'intersection',
            aoi_uri=sheds_uri, vectorize_op=False)

    # Making a copy of watershed and sub-watershed to add water yield outputs
    # to
    watershed_results_uri = os.path.join(
            output_dir, 'watershed_results_wyield%s.shp' % file_suffix)
    pygeoprocessing.geoprocessing.copy_datasource_uri(sheds_uri, watershed_results_uri)

    if sub_sheds_uri is not None:
        subwatershed_results_uri = os.path.join(
                output_dir, 'subwatershed_results_wyield%s.shp' % file_suffix)
        pygeoprocessing.geoprocessing.copy_datasource_uri(sub_sheds_uri, subwatershed_results_uri)

    def aet_op(fractp, precip, veg):
        """Function to compute the actual evapotranspiration values

            fractp - numpy array with the fractp raster values
            precip - numpy array with the precipitation raster values (mm)
            veg - numpy array that depicts which AET equation was used in
                calculations of fractp. Value of 1.0 indicates original
                equation was used, value of 0.0 indicates the alternate
                version was used (AET = Kc * ETo)

            returns - actual evapotranspiration values (mm)"""

        # checking if fractp >= 0 because it's a value that's between 0 and 1
        # and the nodata value is a large negative number.
        return numpy.where(
                (fractp >= 0) & (precip != precip_nodata),
                fractp * precip, out_nodata)

    LOGGER.debug('Performing aet operation')
    # Create clipped aet raster
    pygeoprocessing.geoprocessing.vectorize_datasets(
            [fractp_clipped_path, precip_uri, tmp_veg_raster_uri], aet_op, aet_path,
            gdal.GDT_Float64, out_nodata, pixel_size, 'intersection',
            aoi_uri=sheds_uri, vectorize_op=False)

    # Get the area of the pixel to use in later calculations for volume
    wyield_pixel_area = pygeoprocessing.geoprocessing.get_cell_size_from_uri(wyield_clipped_path) ** 2

    if sub_sheds_uri is not None:
        # Create a list of tuples that pair up field names and raster uris so
        # that we can nicely do operations below
        sws_tuple_names_uris = [
                ('precip_mn', precip_uri),('PET_mn', tmp_pet_uri),
                ('AET_mn', aet_path)]

        for key_name, rast_uri in sws_tuple_names_uris:
            # Aggregrate mean over the sub-watersheds for each uri listed in
            # 'sws_tuple_names_uri'
            key_dict = pygeoprocessing.geoprocessing.aggregate_raster_values_uri(
                rast_uri, sub_sheds_uri, 'subws_id',
                ignore_nodata=False).pixel_mean
            # Add aggregated values to sub-watershed shapefile under new field
            # 'key_name'
            add_dict_to_shape(
                    subwatershed_results_uri, key_dict, key_name, 'subws_id')

        # Aggregate values for the water yield raster under the sub-watershed
        agg_wyield_tup = pygeoprocessing.geoprocessing.aggregate_raster_values_uri(
                wyield_clipped_path, sub_sheds_uri, 'subws_id',
                ignore_nodata=False)
        # Get the pixel mean for aggregated for water yield and the number of
        # pixels in which it aggregated over
        wyield_mean_dict = agg_wyield_tup.pixel_mean
        hectare_mean_dict = agg_wyield_tup.hectare_mean
        pixel_count_dict = agg_wyield_tup.n_pixels
        # Add the wyield mean and number of pixels to the shapefile
        add_dict_to_shape(
                subwatershed_results_uri, wyield_mean_dict, 'wyield_mn', 'subws_id')
        add_dict_to_shape(
                subwatershed_results_uri, pixel_count_dict, 'num_pixels',
                'subws_id')

        # Compute the water yield volume and water yield volume per hectare. The
        # values per sub-watershed will be added as fields in the sub-watersheds
        # shapefile
        compute_water_yield_volume(subwatershed_results_uri, wyield_pixel_area)

        # List of wanted fields to output in the subwatershed CSV table
        field_list_sws = [
                'subws_id', 'num_pixels', 'precip_mn', 'PET_mn', 'AET_mn',
                'wyield_mn', 'wyield_vol']

        # Get a dictionary from the sub-watershed shapefiles attributes based
        # on the fields to be outputted to the CSV table
        wyield_values_sws = pygeoprocessing.geoprocessing.extract_datasource_table_by_key(
                subwatershed_results_uri, 'subws_id')

        wyield_value_dict_sws = filter_dictionary(wyield_values_sws, field_list_sws)

        LOGGER.debug('wyield_value_dict_sws : %s', wyield_value_dict_sws)

        # Write sub-watershed CSV table
        write_new_table(
                subwatershed_results_csv_uri, field_list_sws, wyield_value_dict_sws)

    # Create a list of tuples that pair up field names and raster uris so that
    # we can nicely do operations below
    ws_tuple_names_uris = [
            ('precip_mn', precip_uri),('PET_mn', tmp_pet_uri),
            ('AET_mn', aet_path)]

    for key_name, rast_uri in ws_tuple_names_uris:
        # Aggregrate mean over the watersheds for each uri listed in
        # 'ws_tuple_names_uri'
        key_dict = pygeoprocessing.geoprocessing.aggregate_raster_values_uri(
            rast_uri, sheds_uri, 'ws_id', ignore_nodata=False).pixel_mean
        # Add aggregated values to watershed shapefile under new field
        # 'key_name'
        add_dict_to_shape(watershed_results_uri, key_dict, key_name, 'ws_id')

    # Aggregate values for the water yield raster under the watershed
    agg_wyield_tup = pygeoprocessing.geoprocessing.aggregate_raster_values_uri(
            wyield_clipped_path, sheds_uri, 'ws_id', ignore_nodata=False)
    # Get the pixel mean for aggregated for water yield and the number of
    # pixels in which it aggregated over
    wyield_mean_dict = agg_wyield_tup.pixel_mean
    pixel_count_dict = agg_wyield_tup.n_pixels
    # Add the wyield mean and number of pixels to the shapefile
    add_dict_to_shape(
            watershed_results_uri, wyield_mean_dict, 'wyield_mn', 'ws_id')
    add_dict_to_shape(
            watershed_results_uri, pixel_count_dict, 'num_pixels', 'ws_id')

    compute_water_yield_volume(watershed_results_uri, wyield_pixel_area)

    # List of wanted fields to output in the watershed CSV table
    field_list_ws = [
        'ws_id', 'num_pixels', 'precip_mn', 'PET_mn', 'AET_mn',
        'wyield_mn', 'wyield_vol']

    # Get a dictionary from the watershed shapefiles attributes based on the
    # fields to be outputted to the CSV table
    wyield_values_ws = pygeoprocessing.geoprocessing.extract_datasource_table_by_key(
            watershed_results_uri, 'ws_id')

    wyield_value_dict_ws = filter_dictionary(wyield_values_ws, field_list_ws)

    LOGGER.debug('wyield_value_dict_ws : %s', wyield_value_dict_ws)

    #clear out the temporary filenames, doing this because a giant run of
    #hydropower water yield chews up all available disk space
    for tmp_uri in [
        tmp_Kc_raster_uri, tmp_root_raster_uri, tmp_pet_uri, tmp_veg_raster_uri]:
        os.remove(tmp_uri)

    # Check to see if Water Scarcity was selected to run
    if 'water_scarcity_container' in args:
        water_scarcity_checked = args['water_scarcity_container']
    else:
        water_scarcity_checked = False

    if not water_scarcity_checked:
        LOGGER.debug('Water Scarcity Not Selected')
        # Since Scarcity and Valuation are not selected write out
        # the CSV table
        write_new_table(watershed_results_csv_uri, field_list_ws, wyield_value_dict_ws)
        # The rest of the function is water scarcity and valuation, so we can
        # quit now
        os.remove(clipped_lulc_uri)
        return

    LOGGER.info('Starting Water Scarcity')

    # Open/read in the demand csv file into a dictionary
    demand_dict = {}
    demand_table_file = open(args['demand_table_uri'], 'rU')
    reader = csv.DictReader(demand_table_file)
    for row in reader:
        demand_dict[int(row['lucode'])] = float(row['demand'])

    LOGGER.debug('Demand_Dict : %s', demand_dict)
    demand_table_file.close()

    # Create demand raster from table values to use in future calculations
    LOGGER.info("Reclassifying demand raster")
    tmp_demand_uri = pygeoprocessing.geoprocessing.temporary_filename()
    pygeoprocessing.geoprocessing.reclassify_dataset_uri(
            clipped_lulc_uri, demand_dict, tmp_demand_uri, gdal.GDT_Float64,
            out_nodata)

    # Aggregate the consumption volume over sheds using the
    # reclassfied demand raster
    LOGGER.info('Aggregating Consumption Volume and Mean')

    consump_ws = pygeoprocessing.geoprocessing.aggregate_raster_values_uri(
        tmp_demand_uri, sheds_uri, 'ws_id', ignore_nodata=False)
    consump_vol_dict_ws = consump_ws.total
    consump_mn_dict_ws = consump_ws.pixel_mean

    # Add aggregated consumption to sheds shapefiles
    add_dict_to_shape(
            watershed_results_uri, consump_vol_dict_ws, 'consum_vol', 'ws_id')

    # Add aggregated consumption means to sheds shapefiles
    add_dict_to_shape(
            watershed_results_uri, consump_mn_dict_ws, 'consum_mn', 'ws_id')

    # Calculate the realised water supply after consumption
    LOGGER.info('Calculating RSUPPLY')
    compute_rsupply_volume(watershed_results_uri)

    # List of wanted fields to output in the watershed CSV table
    scarcity_field_list_ws = [
            'ws_id', 'consum_vol', 'consum_mn', 'rsupply_vl',
            'rsupply_mn']

    # Aggregate water yield and water scarcity fields, where we exclude the
    # first field in the scarcity list because they are duplicates already
    # in the water yield list
    field_list_ws = field_list_ws + scarcity_field_list_ws[1:]

    # Get a dictionary from the watershed shapefiles attributes based on the
    # fields to be outputted to the CSV table
    watershed_values = pygeoprocessing.geoprocessing.extract_datasource_table_by_key(
            watershed_results_uri, 'ws_id')

    watershed_dict = filter_dictionary(watershed_values, field_list_ws)

    #Don't need this anymore
    os.remove(tmp_demand_uri)
    os.remove(clipped_lulc_uri)

    for raster_uri in aligned_raster_uris:
        try:
            os.remove(raster_uri)
        except OSError:
            #might have deleted earlier
            pass


    # Check to see if Valuation was selected to run
    if 'valuation_container' in args:
        valuation_checked = args['valuation_container']
    else:
        valuation_checked = False

    if not valuation_checked:
        LOGGER.debug('Valuation Not Selected')
        # Since Valuation are not selected write out
        # the CSV table
        write_new_table(
                watershed_results_csv_uri, field_list_ws, watershed_dict)
        # The rest of the function is valuation, so we can quit now
        return

    LOGGER.info('Starting Valuation Calculation')

    # Open/read in valuation parameters from CSV file
    valuation_params = {}
    valuation_table_file = open(args['valuation_table_uri'], 'rU')
    reader = csv.DictReader(valuation_table_file)
    for row in reader:
        for key, val in row.iteritems():
            try:
                row[key] = float(val)
            except ValueError:
                pass

        valuation_params[int(row['ws_id'])] = row

    valuation_table_file.close()

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
    watershed_values_ws = pygeoprocessing.geoprocessing.extract_datasource_table_by_key(
            watershed_results_uri, 'ws_id')

    watershed_dict_ws = filter_dictionary(watershed_values_ws, field_list_ws)

    # Write out the CSV Table
    write_new_table(
            watershed_results_csv_uri, field_list_ws, watershed_dict_ws)

def compute_watershed_valuation(watersheds_uri, val_dict):
    """Computes and adds the net present value and energy for the watersheds to
        an output shapefile.

        watersheds_uri - a URI path to an OGR shapefile for the
            watershed results. Where the results will be added.

        val_dict - a python dictionary that has all the valuation parameters for
            each watershed

        returns - Nothing
    """
    ws_ds = ogr.Open(watersheds_uri, 1)
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
        energy = val_row['efficiency'] * val_row['fraction'] * val_row['height'] * rsupply_vl * 0.00272

        dsum = 0
        # Divide by 100 because it is input at a percent and we need
        # decimal value
        disc = val_row['discount'] / 100
        # To calculate the summation of the discount rate term over the life
        # span of the dam we can use a geometric series
        ratio = 1 / (1 + disc)
        dsum = (1 - math.pow(ratio, val_row['time_span'])) / (1 - ratio)

        npv = ((val_row['kw_price'] * energy) - val_row['cost']) * dsum

        # Get the volume field index and add value
        ws_feat.SetField(energy_id, energy)
        ws_feat.SetField(npv_id, npv)

        ws_layer.SetFeature(ws_feat)

def compute_rsupply_volume(watershed_results_uri):
    """Calculate the total realized water supply volume and the mean realized
        water supply volume per hectare for the given sheds. Output units in
        cubic meters and cubic meters per hectare respectively.

        watershed_results_uri - a URI path to an OGR shapefile to get water yield
            values from

        returns - Nothing"""
    ws_ds = ogr.Open(watershed_results_uri, 1)
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
        dict_data (dictionary): a dictionary that has keys which point to
            dictionary's.
        values (list): a list of keys to keep from the inner dictionary's
            of 'dict_data'

    Returns:
        a dictionary
    """
    new_dict = {}

    for key, val in dict_data.iteritems():
        new_dict[key] = {}
        for sub_key, sub_val in val.iteritems():
            if sub_key in values:
                new_dict[key][sub_key] = sub_val

    return new_dict

def write_new_table(filename, fields, data):
    """Create a new csv table from a dictionary

        filename - a URI path for the new table to be written to disk

        fields - a python list of the column names. The order of the fields in
            the list will be the order in how they are written. ex:
            ['id', 'precip', 'total']

        data - a python dictionary representing the table. The dictionary
            should be constructed with unique numerical keys that point to a
            dictionary which represents a row in the table:
            data = {0 : {'id':1, 'precip':43, 'total': 65},
                    1 : {'id':2, 'precip':65, 'total': 94}}

        returns - nothing
    """
    csv_file = open(filename, 'wb')

    #  Sort the keys so that the rows are written in order
    row_keys = data.keys()
    row_keys.sort()

    csv_writer = csv.DictWriter(csv_file, fields)
    #  Write the columns as the first row in the table
    csv_writer.writerow(dict((fn, fn) for fn in fields))

    # Write the rows from the dictionary
    for index in row_keys:
        csv_writer.writerow(data[index])

    csv_file.close()

def compute_water_yield_volume(shape_uri, pixel_area):
    """Calculate the water yield volume per sub-watershed or watershed.
        Add results to shape_uri, units are cubic meters

        shape_uri - a URI path to an ogr datasource for the sub-watershed
            or watershed shapefile. This shapefiles features should have a
            'wyield_mn' attribute, which calculations are derived from

        pixel_area - the area in meters squared of a pixel from the wyield
            raster.

        returns - Nothing"""
    shape = ogr.Open(shape_uri, 1)
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
        pixel_count_id = feat.GetFieldIndex('num_pixels')
        pixel_count = feat.GetField(pixel_count_id)

        # Calculate water yield volume,
        #1000 is for converting the mm of wyield to meters
        vol = wyield_mn * pixel_area * pixel_count / 1000.0
        # Get the volume field index and add value
        vol_index = feat.GetFieldIndex(vol_name)
        feat.SetField(vol_index, vol)

        layer.SetFeature(feat)

def add_dict_to_shape(shape_uri, field_dict, field_name, key):
    """Add a new field to a shapefile with values from a dictionary.
        The dictionaries keys should match to the values of a unique fields
        values in the shapefile

        shape_uri - a URI path to a ogr datasource on disk with a unique field
            'key'. The field 'key' should have values that
            correspond to the keys of 'field_dict'

        field_dict - a python dictionary with keys mapping to values. These
            values will be what is filled in for the new field

        field_name - a string for the name of the new field to add

        key - a string for the field name in 'shape_uri' that represents
            the unique features

        returns - nothing"""

    shape = ogr.Open(shape_uri, 1)
    layer = shape.GetLayer()

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
        field_val = float(field_dict[ws_val])

        # Get the new fields index and set the new value for the field
        field_index = feat.GetFieldIndex(field_name)
        feat.SetField(field_index, field_val)

        layer.SetFeature(feat)
