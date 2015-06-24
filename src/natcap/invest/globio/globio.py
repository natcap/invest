"""GLOBIO InVEST Model"""

import os
import logging

import gdal
import osr
import numpy
import pygeoprocessing

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.globio.globio')

def execute(args):
    """main execute entry point"""

    #append a _ to the suffix if it's not empty and doens't already have one
    try:
        file_suffix = args['results_suffix']
        if file_suffix != "" and not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''

    pygeoprocessing.geoprocessing.create_directories([args['workspace_dir']])

    if not args['predefined_globio']:
        out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
            args['lulc_uri'])
    else:
        out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
            args['globio_lulc_uri'])

    if not args['predefined_globio']:
        #reclassify the landcover map
        lulc_to_globio_table = pygeoprocessing.geoprocessing.get_lookup_from_table(
            args['lulc_to_globio_table_uri'], 'lucode')

        lulc_to_globio = dict(
            [(lulc_code, int(table['globio_lucode'])) for
             (lulc_code, table) in lulc_to_globio_table.items()])

        intermediate_globio_lulc_uri = os.path.join(
            args['workspace_dir'], 'intermediate_globio_lulc%s.tif' % file_suffix)
        globio_nodata = -1
        pygeoprocessing.geoprocessing.reclassify_dataset_uri(
            args['lulc_uri'], lulc_to_globio, intermediate_globio_lulc_uri,
            gdal.GDT_Int32, globio_nodata, exception_flag='values_required')

        globio_lulc_uri = os.path.join(
            args['workspace_dir'], 'globio_lulc%s.tif' % file_suffix)

        sum_yieldgap_uri = args['sum_yieldgap_uri']
        potential_vegetation_uri = args['potential_vegetation_uri']
        pasture_uri = args['pasture_uri']

        #smoothed natural areas are natural areas run through a gaussian filter
        natural_areas_uri = os.path.join(
            args['workspace_dir'], 'natural_areas%s.tif' % file_suffix)
        natural_areas_nodata = -1

        def natural_area_mask_op(lulc_array):
            """masking out natural areas"""
            nodata_mask = lulc_array == globio_nodata
            result = (
                (lulc_array == 130) | (lulc_array == 1))
            return numpy.where(nodata_mask, natural_areas_nodata, result)

        LOGGER.info("create mask of natural areas")
        pygeoprocessing.geoprocessing.vectorize_datasets(
            [intermediate_globio_lulc_uri], natural_area_mask_op,
            natural_areas_uri, gdal.GDT_Int32, natural_areas_nodata,
            out_pixel_size, "intersection", dataset_to_align_index=0,
            assert_datasets_projected=False, vectorize_op=False)

        LOGGER.info('gaussian filter natural areas')
        sigma = 9.0
        gaussian_kernel_uri = os.path.join(
            args['workspace_dir'], 'gaussian_kernel%s.tif' % file_suffix)
        make_gaussian_kernel_uri(sigma, gaussian_kernel_uri)
        smoothed_natural_areas_uri = os.path.join(
            args['workspace_dir'], 'smoothed_natural_areas%s.tif' % file_suffix)
        pygeoprocessing.geoprocessing.convolve_2d_uri(
            natural_areas_uri, gaussian_kernel_uri, smoothed_natural_areas_uri)

        ffqi_uri = os.path.join(
            args['workspace_dir'], 'ffqi%s.tif' % file_suffix)

        def ffqi_op(natural_areas_array, smoothed_natural_areas):
            """mask out ffqi only where there's an ffqi"""
            return numpy.where(
                natural_areas_array != natural_areas_nodata,
                natural_areas_array * smoothed_natural_areas,
                natural_areas_nodata)

        LOGGER.info('calculate ffqi')
        pygeoprocessing.geoprocessing.vectorize_datasets(
            [natural_areas_uri, smoothed_natural_areas_uri], ffqi_op,
            ffqi_uri, gdal.GDT_Float32, natural_areas_nodata,
            out_pixel_size, "intersection", dataset_to_align_index=0,
            assert_datasets_projected=False, vectorize_op=False)


        #remap globio lulc to an internal lulc based on ag and yield gaps
        #these came from the 'expansion_scenarios.py' script as numbers Justin
        #provided way back on the unilever project.
        high_intensity_agriculture_threshold = float(args['high_intensity_agriculture_threshold'])
        pasture_threshold = float(args['pasture_threshold'])
        yieldgap_threshold = float(args['yieldgap_threshold'])
        primary_threshold = float(args['primary_threshold'])

        sum_yieldgap_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
            args['sum_yieldgap_uri'])

        potential_vegetation_nodata = (
            pygeoprocessing.geoprocessing.get_nodata_from_uri(
                args['potential_vegetation_uri']))
        pasture_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
            args['pasture_uri'])

        def create_globio_lulc(
                lulc_array, sum_yieldgap, potential_vegetation_array, pasture_array,
                ffqi):

            #Step 1.2b: Assign high/low according to threshold based on yieldgap.
            nodata_mask = lulc_array == globio_nodata
            high_low_intensity_agriculture = numpy.where(
                sum_yieldgap < yieldgap_threshold *
                high_intensity_agriculture_threshold, 9.0, 8.0)

            #Step 1.2c: Stamp ag_split classes onto input LULC
            lulc_ag_split = numpy.where(
                lulc_array == 132.0, high_low_intensity_agriculture, lulc_array)
            nodata_mask = nodata_mask | (lulc_array == globio_nodata)

            #Step 1.3a: Split Scrublands and grasslands into pristine
            #vegetations, livestock grazing areas, and man-made pastures.
            three_types_of_scrubland = numpy.where(
                (potential_vegetation_array <= 8) & (lulc_ag_split == 131), 6.0,
                5.0)

            three_types_of_scrubland = numpy.where(
                (three_types_of_scrubland == 5.0) &
                (pasture_array < pasture_threshold), 1.0,
                three_types_of_scrubland)

            #Step 1.3b: Stamp ag_split classes onto input LULC
            broad_lulc_shrub_split = numpy.where(
                lulc_ag_split == 131, three_types_of_scrubland, lulc_ag_split)

            #Step 1.4a: Split Forests into Primary, Secondary
            four_types_of_forest = numpy.empty(lulc_array.shape)
            #1.0 is primary forest
            four_types_of_forest[(ffqi >= primary_threshold)] = 1
            #3 is secondary forest
            four_types_of_forest[(ffqi < primary_threshold)] = 3

            #Step 1.4b: Stamp ag_split classes onto input LULC
            globio_lulc = numpy.where(
                broad_lulc_shrub_split == 130, four_types_of_forest,
                broad_lulc_shrub_split) #stamp primary vegetation

            return numpy.where(nodata_mask, globio_nodata, globio_lulc)

        LOGGER.info('create the globio lulc')
        pygeoprocessing.geoprocessing.vectorize_datasets(
            [intermediate_globio_lulc_uri, sum_yieldgap_uri,
             potential_vegetation_uri, pasture_uri, ffqi_uri],
            create_globio_lulc, globio_lulc_uri, gdal.GDT_Int32, globio_nodata,
            out_pixel_size, "intersection", dataset_to_align_index=0,
            assert_datasets_projected=False, vectorize_op=False)
    else:
        LOGGER.info('no need to calcualte GLOBIO LULC because it is passed in')
        globio_lulc_uri = args['globio_lulc_uri']
        globio_nodata = pygeoprocessing.get_nodata_from_uri(globio_lulc_uri)


    """This is from Justin's old code:
    #Step 1.2b: Assign high/low according to threshold based on yieldgap.
    #high_low_intensity_agriculture_uri = args["export_folder"]+"high_low_intensity_agriculture_"+args['run_id']+".tif"
    high_intensity_agriculture_threshold = 1 #hardcode for now until UI is determined. Eventually this is a user input. Do I bring it into the ARGS dict?
    high_low_intensity_agriculture = numpy.where(sum_yieldgap < float(args['yieldgap_threshold']*high_intensity_agriculture_threshold), 9.0, 8.0) #45. = average yieldgap on global cells with nonzero yieldgap.


    #Step 1.2c: Stamp ag_split classes onto input LULC
    broad_lulc_ag_split = numpy.where(broad_lulc_array==132.0, high_low_intensity_agriculture, broad_lulc_array)

    #Step 1.3a: Split Scrublands and grasslands into pristine vegetations,
    #livestock grazing areas, and man-made pastures.
    three_types_of_scrubland = numpy.zeros(scenario_lulc_array.shape)
    potential_vegetation_array = geotiff_to_array(aligned_agriculture_uris[0])
    three_types_of_scrubland = numpy.where((potential_vegetation_array <= 8) & (broad_lulc_ag_split== 131), 6.0, 5.0) # < 8 min potential veg means should have been forest, 131 in broad  is grass, so 1.0 implies man made pasture
    pasture_array = geotiff_to_array(aligned_agriculture_uris[1])
    three_types_of_scrubland = numpy.where((three_types_of_scrubland == 5.0) & (pasture_array < args['pasture_threshold']), 1.0, three_types_of_scrubland)

    #Step 1.3b: Stamp ag_split classes onto input LULC
    broad_lulc_shrub_split = numpy.where(broad_lulc_ag_split==131, three_types_of_scrubland, broad_lulc_ag_split)

    #Step 1.4a: Split Forests into Primary, Secondary, Lightly Used and Plantation.
    sigma = 9
    primary_threshold = args['primary_threshold']
    secondary_threshold = args['secondary_threshold']
    is_natural = (broad_lulc_shrub_split == 130) | (broad_lulc_shrub_split == 1)
    blurred = scipy.ndimage.filters.gaussian_filter(is_natural.astype(float), sigma, mode='constant', cval=0.0)
    ffqi = blurred * is_natural

    four_types_of_forest = numpy.empty(scenario_lulc_array.shape)
    four_types_of_forest[(ffqi >= primary_threshold)] = 1.0
    four_types_of_forest[(ffqi < primary_threshold) & (ffqi >= secondary_threshold)] = 3.0
    four_types_of_forest[(ffqi < secondary_threshold)] = 4.0

    #Step 1.4b: Stamp ag_split classes onto input LULC
    globio_lulc = numpy.where(broad_lulc_shrub_split == 130 ,four_types_of_forest, broad_lulc_shrub_split) #stamp primary vegetation

    return globio_lulc"""

    #load the infrastructure layers from disk
    infrastructure_filenames = []
    infrastructure_nodata_list = []
    for root_directory, _, filename_list in os.walk(
            args['infrastructure_dir']):

        for filename in filename_list:
            LOGGER.debug(filename)
            if filename.lower().endswith(".tif"):
                LOGGER.debug("tiff added %s", filename)
                infrastructure_filenames.append(
                    os.path.join(root_directory, filename))
                infrastructure_nodata_list.append(
                    pygeoprocessing.geoprocessing.get_nodata_from_uri(
                        infrastructure_filenames[-1]))
            if filename.lower().endswith(".shp"):
                LOGGER.debug("shape added %s", filename)
                infrastructure_tmp_raster = (
                   os.path.join(args['workspace_dir'], os.path.basename(filename.lower() + ".tif")))
                pygeoprocessing.geoprocessing.new_raster_from_base_uri(
                    globio_lulc_uri, infrastructure_tmp_raster,
                    'GTiff', -1.0, gdal.GDT_Int32, fill_value=0)
                pygeoprocessing.geoprocessing.rasterize_layer_uri(
                    infrastructure_tmp_raster,
                    os.path.join(root_directory, filename), burn_values=[1],
                    option_list=["ALL_TOUCHED=TRUE"])
                infrastructure_filenames.append(infrastructure_tmp_raster)
                infrastructure_nodata_list.append(
                    pygeoprocessing.geoprocessing.get_nodata_from_uri(
                        infrastructure_filenames[-1]))

    if len(infrastructure_filenames) == 0:
        raise ValueError(
            "infrastructure directory didn't have any GeoTIFFS or "
            "Shapefiles at %s", args['infrastructure_dir'])

    infrastructure_nodata = -1
    infrastructure_uri = os.path.join(
        args['workspace_dir'], 'combined_infrastructure%s.tif' % file_suffix)

    def collapse_infrastructure_op(*infrastructure_array_list):
        """Combines all input infrastructure into a single map where if any
            pixel on the stack is 1 gets passed through, any nodata pixel
            masks out all of them"""
        nodata_mask = (
            infrastructure_array_list[0] == infrastructure_nodata_list[0])
        infrastructure_result = infrastructure_array_list[0] > 0
        for index in range(1, len(infrastructure_array_list)):
            current_nodata = (
                infrastructure_array_list[index] ==
                infrastructure_nodata_list[index])

            infrastructure_result = (
                infrastructure_result |
                ((infrastructure_array_list[index] > 0) & ~current_nodata))

            nodata_mask = (
                nodata_mask & current_nodata)

        return numpy.where(
            nodata_mask, infrastructure_nodata, infrastructure_result)

    LOGGER.info('collapse infrastructure into one raster')
    pygeoprocessing.geoprocessing.vectorize_datasets(
        infrastructure_filenames, collapse_infrastructure_op,
        infrastructure_uri, gdal.GDT_Byte, infrastructure_nodata,
        out_pixel_size, "intersection", dataset_to_align_index=0,
        assert_datasets_projected=False, vectorize_op=False)

    #calc_msa_f
    primary_veg_mask_uri = os.path.join(
        args['workspace_dir'], 'primary_veg_mask%s.tif' % file_suffix)
    primary_veg_mask_nodata = -1

    def primary_veg_mask_op(lulc_array):
        """masking out natural areas"""
        nodata_mask = lulc_array == globio_nodata
        result = (lulc_array == 1)
        return numpy.where(nodata_mask, primary_veg_mask_nodata, result)

    LOGGER.info("create mask of primary veg areas")
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [globio_lulc_uri], primary_veg_mask_op,
        primary_veg_mask_uri, gdal.GDT_Int32, primary_veg_mask_nodata,
        out_pixel_size, "intersection", dataset_to_align_index=0,
        assert_datasets_projected=False, vectorize_op=False)

    LOGGER.info('gaussian filter primary veg')
    sigma = 9.0
    gaussian_kernel_uri = os.path.join(
        args['workspace_dir'], 'gaussian_kernel%s.tif' % file_suffix)
    make_gaussian_kernel_uri(sigma, gaussian_kernel_uri)
    smoothed_primary_veg_mask_uri = os.path.join(
        args['workspace_dir'], 'smoothed_primary_veg_mask%s.tif' % file_suffix)
    pygeoprocessing.geoprocessing.convolve_2d_uri(
        primary_veg_mask_uri, gaussian_kernel_uri, smoothed_primary_veg_mask_uri)

    primary_veg_smooth_uri = os.path.join(
        args['workspace_dir'], 'ffqi%s.tif' % file_suffix)

    def primary_veg_smooth_op(primary_veg_mask_array, smoothed_primary_veg_mask):
        """mask out ffqi only where there's an ffqi"""
        return numpy.where(
            primary_veg_mask_array != primary_veg_mask_nodata,
            primary_veg_mask_array * smoothed_primary_veg_mask,
            primary_veg_mask_nodata)

    LOGGER.info('calculate primary_veg_smooth')
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [primary_veg_mask_uri, smoothed_primary_veg_mask_uri],
        primary_veg_smooth_op, primary_veg_smooth_uri, gdal.GDT_Float32,
        primary_veg_mask_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, assert_datasets_projected=False,
        vectorize_op=False)

    msa_nodata = -1
    def msa_f_op(primary_veg_smooth):
        """calcualte msa fragmentation"""
        nodata_mask = primary_veg_mask_nodata == primary_veg_smooth

        msa_f = numpy.empty(primary_veg_smooth.shape)
        msa_f[:] = 1.0
        #These thresholds come from FFQI from Justin's code; I don't
        #know where they otherwise came from.
        msa_f[(primary_veg_smooth > .9825) & (primary_veg_smooth <= .9984)] = 0.95
        msa_f[(primary_veg_smooth > .89771) & (primary_veg_smooth <= .9825)] = 0.90
        msa_f[(primary_veg_smooth > .578512) & (primary_veg_smooth <= .89771)] = 0.7
        msa_f[(primary_veg_smooth > .42877) & (primary_veg_smooth <= .578512)] = 0.6
        msa_f[(primary_veg_smooth <= .42877)] = 0.3
        msa_f[nodata_mask] = msa_nodata

        return msa_f

    LOGGER.info('calculate msa_f')
    msa_f_uri = os.path.join(args['workspace_dir'], 'msa_f%s.tif' % file_suffix)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [primary_veg_smooth_uri], msa_f_op, msa_f_uri, gdal.GDT_Float32,
        msa_nodata, out_pixel_size, "intersection", dataset_to_align_index=0,
        assert_datasets_projected=False, vectorize_op=False)

    #calc_msa_i
    infrastructure_impact_zones = {
        'no impact': 1.0,
        'low impact': 0.9,
        'medium impact': 0.8,
        'high impact': 0.4
    }

    def msa_i_op(lulc_array, distance_to_infrastructure):
        """calculate msa infrastructure"""
        msa_i_tropical_forest = numpy.empty(lulc_array.shape)
        distance_to_infrastructure *= out_pixel_size #convert to meters
        msa_i_tropical_forest[:] = infrastructure_impact_zones['no impact']
        msa_i_tropical_forest[(distance_to_infrastructure > 4000.0) & (distance_to_infrastructure <= 14000.0)] = infrastructure_impact_zones['low impact']
        msa_i_tropical_forest[(distance_to_infrastructure > 1000.0) & (distance_to_infrastructure <= 4000.0)] = infrastructure_impact_zones['medium impact']
        msa_i_tropical_forest[(distance_to_infrastructure <= 1000.0)] = infrastructure_impact_zones['high impact']

        msa_i_temperate_and_boreal_forest = numpy.empty(lulc_array.shape)
        msa_i_temperate_and_boreal_forest[:] = infrastructure_impact_zones['no impact']
        msa_i_temperate_and_boreal_forest[(distance_to_infrastructure > 1200.0) & (distance_to_infrastructure <= 4200.0)] = infrastructure_impact_zones['low impact']
        msa_i_temperate_and_boreal_forest[(distance_to_infrastructure > 300.0) & (distance_to_infrastructure <= 1200.0)] = infrastructure_impact_zones['medium impact']
        msa_i_temperate_and_boreal_forest[(distance_to_infrastructure <= 300.0)] = infrastructure_impact_zones['high impact']

        msa_i_cropland_and_grassland = numpy.empty(lulc_array.shape)
        msa_i_cropland_and_grassland[:] = infrastructure_impact_zones['no impact']
        msa_i_cropland_and_grassland[(distance_to_infrastructure > 2000.0) & (distance_to_infrastructure <= 7000.0)] = infrastructure_impact_zones['low impact']
        msa_i_cropland_and_grassland[(distance_to_infrastructure > 500.0) & (distance_to_infrastructure <= 2000.0)] = infrastructure_impact_zones['medium impact']
        msa_i_cropland_and_grassland[(distance_to_infrastructure <= 500.0)] = infrastructure_impact_zones['high impact']

        msa_i = numpy.where((lulc_array >= 1) & (lulc_array <= 5), msa_i_temperate_and_boreal_forest, infrastructure_impact_zones['no impact'])
        msa_i = numpy.where((lulc_array >= 6) & (lulc_array <= 12), msa_i_cropland_and_grassland, msa_i)

        return msa_i

    LOGGER.info('calculate msa_i')
    distance_to_infrastructure_uri = os.path.join(
        args['workspace_dir'], 'distance_to_infrastructure%s.tif' % file_suffix)
    pygeoprocessing.geoprocessing.distance_transform_edt(
        infrastructure_uri, distance_to_infrastructure_uri)
    msa_i_uri = os.path.join(args['workspace_dir'], 'msa_i%s.tif' % file_suffix)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [globio_lulc_uri, distance_to_infrastructure_uri], msa_i_op, msa_i_uri,
        gdal.GDT_Float32, msa_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, assert_datasets_projected=False,
        vectorize_op=False)


    #calc_msa_lu
    lu_msa_lookup = {
        0.0: 0.0, #map 0 to 0
        1.0: 1.0, #primary veg
        2.0: 0.7, #lightly used natural forest
        3.0: 0.5, #secondary forest
        4.0: 0.2, #forest plantation
        5.0: 0.7, #livestock grazing
        6.0: 0.1, #man-made pastures
        7.0: 0.5, #agroforesty
        8.0: 0.3, #low-input agriculture
        9.0: 0.1, #intenstive agriculture
        10.0: 0.05, #built-up areas
    }
    msa_lu_uri = os.path.join(
        args['workspace_dir'], 'msa_lu%s.tif' % file_suffix)
    LOGGER.info('calculate msa_lu')
    pygeoprocessing.geoprocessing.reclassify_dataset_uri(
        globio_lulc_uri, lu_msa_lookup, msa_lu_uri,
        gdal.GDT_Float32, globio_nodata, exception_flag='values_required')

    LOGGER.info('calculate msa')
    msa_uri = os.path.join(
        args['workspace_dir'], 'msa%s.tif' % file_suffix)
    def msa_op(msa_f, msa_lu, msa_i):
        return numpy.where(
            msa_f != globio_nodata, msa_f* msa_lu * msa_i, globio_nodata)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [msa_f_uri, msa_lu_uri, msa_i_uri], msa_op, msa_uri,
        gdal.GDT_Float32, msa_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, assert_datasets_projected=False,
        vectorize_op=False)
    #calc msa msa = msa_f[tail_type] * msa_lu[tail_type] * msa_i[tail_type]



def make_gaussian_kernel_uri(sigma, kernel_uri):
    """create a gaussian kernel raster"""
    max_distance = sigma * 5
    kernel_size = int(numpy.round(max_distance * 2 + 1))

    driver = gdal.GetDriverByName('GTiff')
    kernel_dataset = driver.Create(
        kernel_uri.encode('utf-8'), kernel_size, kernel_size, 1,
        gdal.GDT_Float32, options=['BIGTIFF=IF_SAFER'])

    #Make some kind of geotransform, it doesn't matter what but
    #will make GIS libraries behave better if it's all defined
    kernel_dataset.SetGeoTransform([444720, 30, 0, 3751320, 0, -30])
    srs = osr.SpatialReference()
    srs.SetUTM(11, 1)
    srs.SetWellKnownGeogCS('NAD27')
    kernel_dataset.SetProjection(srs.ExportToWkt())

    kernel_band = kernel_dataset.GetRasterBand(1)
    kernel_band.SetNoDataValue(-9999)

    col_index = numpy.array(xrange(kernel_size))
    integration = 0.0
    for row_index in xrange(kernel_size):
        kernel = numpy.exp(
            -((row_index - max_distance)**2 +
                (col_index - max_distance) ** 2)/(2.0*sigma**2)).reshape(
                    1, kernel_size)

        integration += numpy.sum(kernel)
        kernel_band.WriteArray(kernel, xoff=0, yoff=row_index)

    for row_index in xrange(kernel_size):
        kernel_row = kernel_band.ReadAsArray(
            xoff=0, yoff=row_index, win_xsize=kernel_size, win_ysize=1)
        kernel_row /= integration
        kernel_band.WriteArray(kernel_row, 0, row_index)
