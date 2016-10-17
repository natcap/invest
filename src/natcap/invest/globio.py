"""GLOBIO InVEST Model."""

import os
import logging
import collections
import csv
import uuid

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy
import pygeoprocessing

from . import utils

LOGGER = logging.getLogger('natcap.invest.globio')

# this value of sigma == 9.0 was derived by Justin Johnson as a good
# approximation to use as a gaussian filter to replace the connectivity index.
# I don't have any other documentation than his original code base.
SIGMA = 9.0


def execute(args):
    """GLOBIO.

    The model operates in two modes.  Mode (a) generates a landcover map
    based on a base landcover map and information about crop yields,
    infrastructure, and more.  Mode (b) assumes the globio landcover
    map is generated.  These modes are used below to describe input
    parameters.

    Parameters:

        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['predefined_globio'] (boolean): if True then "mode (b)" else
            "mode (a)"
        args['results_suffix'] (string): (optional) string to append to any
            output files
        args['lulc_uri'] (string): used in "mode (a)" path to a base landcover
            map with integer codes
        args['lulc_to_globio_table_uri'] (string): used in "mode (a)" path to
            table that translates the land-cover args['lulc_uri'] to
            intermediate GLOBIO classes, from which they will be further
            differentiated using the additional data in the model.  Contains
            at least the following fields:

            * 'lucode': Land use and land cover class code of the dataset
              used. LULC codes match the 'values' column in the LULC
              raster of mode (b) and must be numeric and unique.
            * 'globio_lucode': The LULC code corresponding to the GLOBIO class
              to which it should be converted, using intermediate codes
              described in the example below.

        args['infrastructure_dir'] (string): used in "mode (a) and (b)" a path
            to a folder containing maps of either gdal compatible rasters or
            OGR compatible shapefiles.  These data will be used in the
            infrastructure to calculation of MSA.
        args['pasture_uri'] (string): used in "mode (a)" path to pasture raster
        args['potential_vegetation_uri'] (string): used in "mode (a)" path to
            potential vegetation raster
        args['pasture_threshold'] (float): used in "mode (a)"
        args['intensification_fraction'] (float): used in "mode (a)"; a value
            between 0 and 1 denoting proportion of total agriculture that
            should be classified as 'high input'
        args['primary_threshold'] (float): used in "mode (a)"
        args['msa_parameters_uri'] (string): path to MSA classification
            parameters
        args['aoi_uri'] (string): (optional) if it exists then final MSA raster
            is summarized by AOI
        args['globio_lulc_uri'] (string): used in "mode (b)" path to predefined
            globio raster.

    Returns:
        None
    """
    msa_parameter_table = load_msa_parameter_table(
        args['msa_parameters_uri'], float(args['intensification_fraction']))
    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    output_dir = os.path.join(args['workspace_dir'])
    intermediate_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    tmp_dir = os.path.join(args['workspace_dir'], 'tmp')

    pygeoprocessing.geoprocessing.create_directories(
        [output_dir, intermediate_dir, tmp_dir])

    # cell size should be based on the landcover map
    if not args['predefined_globio']:
        out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
            args['lulc_uri'])
        globio_lulc_uri = _calculate_globio_lulc_map(
            args['lulc_to_globio_table_uri'], args['lulc_uri'],
            args['potential_vegetation_uri'], args['pasture_uri'],
            float(args['pasture_threshold']), float(args['primary_threshold']),
            file_suffix, intermediate_dir, tmp_dir, out_pixel_size)
    else:
        out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
            args['globio_lulc_uri'])
        LOGGER.info('no need to calculate GLOBIO LULC because it is passed in')
        globio_lulc_uri = args['globio_lulc_uri']

    globio_nodata = pygeoprocessing.get_nodata_from_uri(globio_lulc_uri)

    infrastructure_uri = os.path.join(
        intermediate_dir, 'combined_infrastructure%s.tif' % file_suffix)
    _collapse_infrastructure_layers(
        args['infrastructure_dir'], globio_lulc_uri, infrastructure_uri)

    # calc_msa_f
    primary_veg_mask_uri = os.path.join(
        tmp_dir, 'primary_veg_mask%s.tif' % file_suffix)
    primary_veg_mask_nodata = -1

    def _primary_veg_mask_op(lulc_array):
        """Masking out natural areas."""
        nodata_mask = lulc_array == globio_nodata
        # landcover type 1 in the GLOBIO schema represents primary vegetation
        result = (lulc_array == 1)
        return numpy.where(nodata_mask, primary_veg_mask_nodata, result)

    LOGGER.info("create mask of primary veg areas")
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [globio_lulc_uri], _primary_veg_mask_op,
        primary_veg_mask_uri, gdal.GDT_Int32, primary_veg_mask_nodata,
        out_pixel_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)

    LOGGER.info('gaussian filter primary veg')
    gaussian_kernel_uri = os.path.join(
        tmp_dir, 'gaussian_kernel%s.tif' % file_suffix)
    make_gaussian_kernel_uri(SIGMA, gaussian_kernel_uri)
    smoothed_primary_veg_mask_uri = os.path.join(
        tmp_dir, 'smoothed_primary_veg_mask%s.tif' % file_suffix)
    pygeoprocessing.geoprocessing.convolve_2d_uri(
        primary_veg_mask_uri, gaussian_kernel_uri,
        smoothed_primary_veg_mask_uri)

    primary_veg_smooth_uri = os.path.join(
        intermediate_dir, 'primary_veg_smooth%s.tif' % file_suffix)

    def _primary_veg_smooth_op(
            primary_veg_mask_array, smoothed_primary_veg_mask):
        """Mask out ffqi only where there's an ffqi."""
        return numpy.where(
            primary_veg_mask_array != primary_veg_mask_nodata,
            primary_veg_mask_array * smoothed_primary_veg_mask,
            primary_veg_mask_nodata)

    LOGGER.info('calculate primary_veg_smooth')
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [primary_veg_mask_uri, smoothed_primary_veg_mask_uri],
        _primary_veg_smooth_op, primary_veg_smooth_uri, gdal.GDT_Float32,
        primary_veg_mask_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    msa_nodata = -1

    msa_f_table = msa_parameter_table['msa_f']
    msa_f_values = sorted(msa_f_table)

    def _msa_f_op(primary_veg_smooth):
        """Calculate msa fragmentation."""
        nodata_mask = primary_veg_mask_nodata == primary_veg_smooth

        msa_f = numpy.empty(primary_veg_smooth.shape)

        for value in reversed(msa_f_values):
            # special case if it's a > or < value
            if value == '>':
                msa_f[primary_veg_smooth > msa_f_table['>'][0]] = (
                    msa_f_table['>'][1])
            elif value == '<':
                continue
            else:
                msa_f[primary_veg_smooth <= value] = msa_f_table[value]

        if '<' in msa_f_table:
            msa_f[primary_veg_smooth < msa_f_table['<'][0]] = (
                msa_f_table['<'][1])

        msa_f[nodata_mask] = msa_nodata

        return msa_f

    LOGGER.info('calculate msa_f')
    msa_f_uri = os.path.join(output_dir, 'msa_f%s.tif' % file_suffix)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [primary_veg_smooth_uri], _msa_f_op, msa_f_uri, gdal.GDT_Float32,
        msa_nodata, out_pixel_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)

    # calc_msa_i
    msa_f_values = sorted(msa_f_table)
    msa_i_other_table = msa_parameter_table['msa_i_other']
    msa_i_primary_table = msa_parameter_table['msa_i_primary']
    msa_i_other_values = sorted(msa_i_other_table)
    msa_i_primary_values = sorted(msa_i_primary_table)

    def _msa_i_op(lulc_array, distance_to_infrastructure):
        """Calculate msa infrastructure."""
        distance_to_infrastructure *= out_pixel_size  # convert to meters
        msa_i_primary = numpy.empty(lulc_array.shape)
        msa_i_other = numpy.empty(lulc_array.shape)

        for value in reversed(msa_i_primary_values):
            # special case if it's a > or < value
            if value == '>':
                msa_i_primary[distance_to_infrastructure >
                              msa_i_primary_table['>'][0]] = (
                                  msa_i_primary_table['>'][1])
            elif value == '<':
                continue
            else:
                msa_i_primary[distance_to_infrastructure <= value] = (
                    msa_i_primary_table[value])

        if '<' in msa_i_primary_table:
            msa_i_primary[distance_to_infrastructure <
                          msa_i_primary_table['<'][0]] = (
                              msa_i_primary_table['<'][1])

        for value in reversed(msa_i_other_values):
            # special case if it's a > or < value
            if value == '>':
                msa_i_other[distance_to_infrastructure >
                            msa_i_other_table['>'][0]] = (
                                msa_i_other_table['>'][1])
            elif value == '<':
                continue
            else:
                msa_i_other[distance_to_infrastructure <= value] = (
                    msa_i_other_table[value])

        if '<' in msa_i_other_table:
            msa_i_other[distance_to_infrastructure <
                        msa_i_other_table['<'][0]] = (
                            msa_i_other_table['<'][1])

        msa_i = numpy.where(lulc_array == 1, msa_i_primary, msa_i_other)
        return msa_i

    LOGGER.info('calculate msa_i')
    distance_to_infrastructure_uri = os.path.join(
        intermediate_dir, 'distance_to_infrastructure%s.tif' % file_suffix)
    pygeoprocessing.geoprocessing.distance_transform_edt(
        infrastructure_uri, distance_to_infrastructure_uri)
    msa_i_uri = os.path.join(output_dir, 'msa_i%s.tif' % file_suffix)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [globio_lulc_uri, distance_to_infrastructure_uri], _msa_i_op,
        msa_i_uri, gdal.GDT_Float32, msa_nodata, out_pixel_size,
        "intersection", dataset_to_align_index=0, vectorize_op=False)

    # calc_msa_lu
    msa_lu_uri = os.path.join(
        output_dir, 'msa_lu%s.tif' % file_suffix)
    LOGGER.info('calculate msa_lu')
    pygeoprocessing.geoprocessing.reclassify_dataset_uri(
        globio_lulc_uri, msa_parameter_table['msa_lu'], msa_lu_uri,
        gdal.GDT_Float32, globio_nodata, exception_flag='values_required')

    LOGGER.info('calculate msa')
    msa_uri = os.path.join(
        output_dir, 'msa%s.tif' % file_suffix)

    def _msa_op(msa_f, msa_lu, msa_i):
        """Calculate the MSA which is the product of the sub MSAs."""
        return numpy.where(
            msa_f != globio_nodata, msa_f * msa_lu * msa_i, globio_nodata)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [msa_f_uri, msa_lu_uri, msa_i_uri], _msa_op, msa_uri,
        gdal.GDT_Float32, msa_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    # ensure that aoi_uri is defined and it's not an empty string
    if 'aoi_uri' in args and len(args['aoi_uri']) > 0:
        # copy the aoi to an output shapefile
        original_datasource = ogr.Open(args['aoi_uri'])
        summary_aoi_uri = os.path.join(
            output_dir, 'aoi_summary%s.shp' % file_suffix)
        # Delete if existing shapefile with the same name
        if os.path.isfile(summary_aoi_uri):
            os.remove(summary_aoi_uri)
        # Copy the input shapefile into the designated output folder
        esri_driver = ogr.GetDriverByName('ESRI Shapefile')
        datasource_copy = esri_driver.CopyDataSource(
            original_datasource, summary_aoi_uri)
        layer = datasource_copy.GetLayer()
        msa_summary_field_def = ogr.FieldDefn('msa_mean', ogr.OFTReal)
        layer.CreateField(msa_summary_field_def)

        # make an identifying id per polygon that can be used for aggregation
        layer_defn = layer.GetLayerDefn()
        while True:
            # last 8 characters because shapefile fields are limited to 8
            poly_id_field = str(uuid.uuid4())[-8:]
            if layer_defn.GetFieldIndex(poly_id_field) == -1:
                break
        layer_id_field = ogr.FieldDefn(poly_id_field, ogr.OFTInteger)
        layer.CreateField(layer_id_field)
        for poly_index, poly_feat in enumerate(layer):
            poly_feat.SetField(poly_id_field, poly_index)
            layer.SetFeature(poly_feat)
        layer.SyncToDisk()

        # aggregate by ID
        msa_summary = pygeoprocessing.aggregate_raster_values_uri(
            msa_uri, summary_aoi_uri, shapefile_field=poly_id_field)

        # add new column to output file
        for feature_id in xrange(layer.GetFeatureCount()):
            feature = layer.GetFeature(feature_id)
            key_value = feature.GetFieldAsInteger(poly_id_field)
            feature.SetField(
                'msa_mean', float(msa_summary.pixel_mean[key_value]))
            layer.SetFeature(feature)

        # don't need a random poly id anymore
        layer.DeleteField(layer_defn.GetFieldIndex(poly_id_field))

    for root_dir, _, files in os.walk(tmp_dir):
        for filename in files:
            try:
                os.remove(os.path.join(root_dir, filename))
            except OSError:
                LOGGER.warn("couldn't remove temporary file %s", filename)


def make_gaussian_kernel_uri(sigma, kernel_uri):
    """Create a gaussian kernel raster."""
    max_distance = sigma * 5
    kernel_size = int(numpy.round(max_distance * 2 + 1))

    driver = gdal.GetDriverByName('GTiff')
    kernel_dataset = driver.Create(
        kernel_uri.encode('utf-8'), kernel_size, kernel_size, 1,
        gdal.GDT_Float32, options=['BIGTIFF=IF_SAFER'])

    # Make some kind of geotransform, it doesn't matter what but
    # will make GIS libraries behave better if it's all defined
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


def load_msa_parameter_table(
        msa_parameter_table_filename, intensification_fraction):
    """Load parameter table to a dict that to define the MSA ranges.

    Parameters:
        msa_parameter_table_filename (string): path to msa csv table
        intensification_fraction (float): a number between 0 and 1 indicating
            what level between msa_lu 8 and 9 to define the general GLOBIO
            code "12" to.


        returns a dictionary of the form
            {
                'msa_f': {
                    valuea: msa_f_value, ...
                    valueb: ...
                    '<': (bound, msa_f_value),
                    '>': (bound, msa_f_value)}
                'msa_i_other_table': {
                    valuea: msa_i_value, ...
                    valueb: ...
                    '<': (bound, msa_i_other_value),
                    '>': (bound, msa_i_other_value)}
                'msa_i_primary': {
                    valuea: msa_i_primary_value, ...
                    valueb: ...
                    '<': (bound, msa_i_primary_value),
                    '>': (bound, msa_i_primary_value)}
                'msa_lu': {
                    valuea: msa_lu_value, ...
                    valueb: ...
                    '<': (bound, msa_lu_value),
                    '>': (bound, msa_lu_value)
                    12: (msa_lu_8 * (1.0 - intensification_fraction) +
                         msa_lu_9 * intensification_fraction}
            }
    """
    with open(msa_parameter_table_filename, 'rb') as msa_parameter_table_file:
        reader = csv.DictReader(msa_parameter_table_file)
        msa_dict = collections.defaultdict(dict)
        for line in reader:
            if line['Value'][0] in ['<', '>']:
                # put the limit and the MSA value in a tub
                value = line['Value'][0]
                msa_dict[line['MSA_type']][value] = (
                    float(line['Value'][1:]), float(line['MSA_x']))
                continue
            elif '-' in line['Value']:
                value = float(line['Value'].split('-')[1])
            else:
                value = float(line['Value'])
            msa_dict[line['MSA_type']][value] = float(line['MSA_x'])
    # cast back to a regular dict so we get keyerrors on non-existant keys
    msa_dict['msa_lu'][12] = (
        msa_dict['msa_lu'][8] * (1.0 - intensification_fraction) +
        msa_dict['msa_lu'][9] * intensification_fraction)
    return dict(msa_dict)


def _calculate_globio_lulc_map(
        lulc_to_globio_table_uri, lulc_uri, potential_vegetation_uri,
        pasture_uri, pasture_threshold, primary_threshold, file_suffix,
        intermediate_dir, tmp_dir, out_pixel_size):
    """Translate a general landcover map into a GLOBIO version.

    Parameters:
        lulc_to_globio_table_uri (string): a table that maps arbitrary
            landcover values to globio equivalents.
        lulc_uri (string): path to the raw landcover map.
        potential_vegetation_uri (string): a landcover map that indicates what
            the vegetation types would be if left to revert to natural state
        pasture_uri (string): a path to a raster that indicates the percent
            of pasture contained in the pixel.  used to classify forest types
            from scrubland.
        pasture_threshold (float): the threshold to classify pixels in pasture
            as potential forest or scrub
        primary_threshold (float): the threshold to classify the calculated
            FFQI pixels into core forest or secondary
        file_suffix - (string) to append on output file
        intermediate_dir - (string) path to location for temporary files of
            which the following files are created
                'intermediate_globio_lulc.tif': reclassified landcover map to
                    globio landcover codes
                'ffqi.tif': index of fragmentation due to infrastructure and
                    original values of landscape
                'globio_lulc.tif': primary output of the function, starts
                    with intermeidate globio and modifies based on the other
                    biophysical parameters to the function as described in the
                    GLOBIO process

        tmp_dir - (string) path to location for temporary files
        out_pixel_size - (float) pixel size of output globio map

    Returns:
        a (string) filename to the generated globio GeoTIFF map
    """
    lulc_to_globio_table = pygeoprocessing.get_lookup_from_table(
        lulc_to_globio_table_uri, 'lucode')

    lulc_to_globio = dict(
        [(lulc_code, int(table['globio_lucode'])) for
         (lulc_code, table) in lulc_to_globio_table.items()])

    intermediate_globio_lulc_uri = os.path.join(
        tmp_dir, 'intermediate_globio_lulc%s.tif' % file_suffix)
    globio_nodata = -1
    pygeoprocessing.geoprocessing.reclassify_dataset_uri(
        lulc_uri, lulc_to_globio, intermediate_globio_lulc_uri,
        gdal.GDT_Int32, globio_nodata, exception_flag='values_required')

    globio_lulc_uri = os.path.join(
        intermediate_dir, 'globio_lulc%s.tif' % file_suffix)

    # smoothed natural areas are natural areas run through a gaussian filter
    forest_areas_uri = os.path.join(
        tmp_dir, 'forest_areas%s.tif' % file_suffix)
    forest_areas_nodata = -1

    def _forest_area_mask_op(lulc_array):
        """Masking out forest areas."""
        nodata_mask = lulc_array == globio_nodata
        # landcover code 130 represents all MODIS forest codes which originate
        # as 1-5
        result = (lulc_array == 130)
        return numpy.where(nodata_mask, forest_areas_nodata, result)

    LOGGER.info("create mask of natural areas")
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [intermediate_globio_lulc_uri], _forest_area_mask_op,
        forest_areas_uri, gdal.GDT_Int32, forest_areas_nodata,
        out_pixel_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)

    LOGGER.info('gaussian filter natural areas')
    gaussian_kernel_uri = os.path.join(
        tmp_dir, 'gaussian_kernel%s.tif' % file_suffix)
    make_gaussian_kernel_uri(SIGMA, gaussian_kernel_uri)
    smoothed_forest_areas_uri = os.path.join(
        tmp_dir, 'smoothed_forest_areas%s.tif' % file_suffix)
    pygeoprocessing.geoprocessing.convolve_2d_uri(
        forest_areas_uri, gaussian_kernel_uri, smoothed_forest_areas_uri)

    ffqi_uri = os.path.join(
        intermediate_dir, 'ffqi%s.tif' % file_suffix)

    def _ffqi_op(forest_areas_array, smoothed_forest_areas):
        """Mask out ffqi only where there's an ffqi."""
        return numpy.where(
            forest_areas_array != forest_areas_nodata,
            forest_areas_array * smoothed_forest_areas,
            forest_areas_nodata)

    LOGGER.info('calculate ffqi')
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [forest_areas_uri, smoothed_forest_areas_uri], _ffqi_op,
        ffqi_uri, gdal.GDT_Float32, forest_areas_nodata,
        out_pixel_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)

    # remap globio lulc to an internal lulc based on ag and intensification
    # proportion these came from the 'expansion_scenarios.py'
    def _create_globio_lulc(
            lulc_array, potential_vegetation_array, pasture_array,
            ffqi):
        """Construct GLOBIO lulc given relevant biophysical parameters."""
        # Step 1.2b: Assign high/low according to threshold based on yieldgap.
        nodata_mask = lulc_array == globio_nodata

        # Step 1.2c: Classify all ag classes as a new LULC value "12" per our
        # custom design of agriculture
        # landcover 132 represents agriculture landcover types in the GLOBIO
        # classification scheme
        lulc_ag_split = numpy.where(
            lulc_array == 132, 12, lulc_array)
        nodata_mask = nodata_mask | (lulc_array == globio_nodata)

        # Step 1.3a: Split Scrublands and grasslands into pristine
        # vegetations, livestock grazing areas, and man-made pastures.
        # landcover 131 represents grassland/shrubland in the GLOBIO
        # classification
        three_types_of_scrubland = numpy.where(
            (potential_vegetation_array <= 8) & (lulc_ag_split == 131), 6.0,
            5.0)

        three_types_of_scrubland = numpy.where(
            (three_types_of_scrubland == 5.0) &
            (pasture_array < pasture_threshold), 1.0,
            three_types_of_scrubland)

        # Step 1.3b: Stamp ag_split classes onto input LULC
        # landcover 131 represents grassland/shrubland in the GLOBIO
        # classification
        broad_lulc_shrub_split = numpy.where(
            lulc_ag_split == 131, three_types_of_scrubland, lulc_ag_split)

        # Step 1.4a: Split Forests into Primary, Secondary
        four_types_of_forest = numpy.empty(lulc_array.shape)
        # 1 is primary forest
        four_types_of_forest[(ffqi >= primary_threshold)] = 1
        # 3 is secondary forest
        four_types_of_forest[(ffqi < primary_threshold)] = 3

        # Step 1.4b: Stamp ag_split classes onto input LULC
        # landcover code 130 represents all MODIS forest codes which originate
        # as 1-5
        globio_lulc = numpy.where(
            broad_lulc_shrub_split == 130, four_types_of_forest,
            broad_lulc_shrub_split)  # stamp primary vegetation

        return numpy.where(nodata_mask, globio_nodata, globio_lulc)

    LOGGER.info('create the globio lulc')
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [intermediate_globio_lulc_uri, potential_vegetation_uri, pasture_uri,
         ffqi_uri], _create_globio_lulc, globio_lulc_uri, gdal.GDT_Int32,
        globio_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    return globio_lulc_uri


def _collapse_infrastructure_layers(
        infrastructure_dir, base_raster_uri, infrastructure_uri):
    """Collapse all GIS infrastructure layers to one raster.

    Gathers all the GIS layers in the given directory and collapses them
    to a single byte raster mask where 1 indicates a pixel overlapping with
    one of the original infrastructure layers, 0 does not, and nodata
    indicates a region that has no layers that overlap but are still contained
    in the bounding box.

    Parameters:
        infrastructure_dir (string): path to a directory containing maps of
            either gdal compatible rasters or OGR compatible shapefiles.
        base_raster_uri (string): a path to a file that has the dimensions and
            projection of the desired output infrastructure file.
        infrastructure_uri (string): (output) path to a file that will be a
            byte raster with 1s everywhere there was a GIS layer present in
            the GIS layers in `infrastructure_dir`.

    Returns:
        None
    """
    # load the infrastructure layers from disk
    infrastructure_filenames = []
    infrastructure_nodata_list = []
    infrastructure_tmp_filenames = []
    for root_directory, _, filename_list in os.walk(infrastructure_dir):
        for filename in filename_list:
            if filename.lower().endswith(".tif"):
                infrastructure_filenames.append(
                    os.path.join(root_directory, filename))
                infrastructure_nodata_list.append(
                    pygeoprocessing.geoprocessing.get_nodata_from_uri(
                        infrastructure_filenames[-1]))
            if filename.lower().endswith(".shp"):
                infrastructure_tmp_raster = (
                    pygeoprocessing.temporary_filename())
                pygeoprocessing.geoprocessing.new_raster_from_base_uri(
                    base_raster_uri, infrastructure_tmp_raster,
                    'GTiff', -1.0, gdal.GDT_Int32, fill_value=0)
                pygeoprocessing.geoprocessing.rasterize_layer_uri(
                    infrastructure_tmp_raster,
                    os.path.join(root_directory, filename), burn_values=[1],
                    option_list=["ALL_TOUCHED=TRUE"])
                infrastructure_filenames.append(infrastructure_tmp_raster)
                infrastructure_tmp_filenames.append(infrastructure_tmp_raster)
                infrastructure_nodata_list.append(
                    pygeoprocessing.geoprocessing.get_nodata_from_uri(
                        infrastructure_filenames[-1]))

    if len(infrastructure_filenames) == 0:
        raise ValueError(
            "infrastructure directory didn't have any GeoTIFFS or "
            "Shapefiles at %s", infrastructure_dir)

    infrastructure_nodata = -1

    def _collapse_infrastructure_op(*infrastructure_array_list):
        """For each pixel, create mask 1 if all valid, else set to nodata."""
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
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
        base_raster_uri)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        infrastructure_filenames, _collapse_infrastructure_op,
        infrastructure_uri, gdal.GDT_Byte, infrastructure_nodata,
        out_pixel_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)

    # clean up the temporary filenames
    for filename in infrastructure_tmp_filenames:
        os.remove(filename)
