import os
import sys
import math

import numpy
import scipy.stats
from bisect import bisect

import shutil
import logging

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from pygeoprocessing import geoprocessing
import natcap.invest.utils

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.scenic_quality.scenic_quality')

_OUTPUT_BASE_FILES = {
    'viewshed_valuation_path': 'vshed.tif',
    'viewshed_path': 'viewshed_counts.tif',
    'viewshed_quality_path': 'vshed_qual.tif',
    'pop_stats_path': 'populationStats.html',
    'overlap_path': 'vp_overlap.shp'
    }

_TMP_BASE_FILES = {
    'aoi_proj_dem_path' : 'aoi_proj_to_dem.shp',
    'aoi_proj_pop_path' : 'aoi_proj_to_pop.shp',
    'clipped_dem_path' : 'dem_clipped.tif',
    'viewshed_dem_reclass_path' : 'dem_vs_re.tif',
    'clipped_pop_path' : 'pop_clipped.tif',
    'pop_proj_path' : 'pop_proj.tif',
    'pop_vs_path' : 'pop_vs.tif',
    'viewshed_reclass_path' : 'vshed_bool.tif',
    'viewshed_polygon_path' : 'vshed.shp',
    'dem_below_sea_path' : 'dem_below_sea_lvl.tif'
    }


def execute(args):
    """Run the Scenic Quality Model.

    Parameters:
        args['workspace_dir'] (string):
        args['aoi_path'] (string):
        args['cell_size'] (int):
        args['structure_path'] (string):
        args['dem_path'] (string):
        args['refraction'] (float):
        args['population_path'] (string):
        args['overlap_path'] (string):
        args['suffix] (string):
        args['valuation_function'] (string):
        args['a_coef'] (string):
        args['b_coef'] (string):
        args['c_coef'] (string):
        args['d_coef'] (string):
        args['max_valuation_radius'] (string):

    """
    LOGGER.info("Start Scenic Quality Model")

    dem_cell_size = geoprocessing.get_cell_size_from_uri(args['dem_path'])
    curvature_correction = float(args['refraction'])

    output_dir = os.path.join(args['workspace_dir'], 'output')
    geoprocessing.create_directories([output_dir])

    files_suffix = utils.make_suffix_string(args, 'suffix')

    LOGGER.info('Building file registry')
    file_registry = natcap.invest.utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_TMP_BASE_FILES, output_dir)], file_suffix)

    # Clip DEM by AOI and reclass
    dem_wkt = geoprocessing.get_dataset_projection_wkt_uri(args['dem_path'])
    geoprocessing.reproject_datasource_uri(
        args['aoi_path'], dem_wkt, file_registry['aoi_proj_dem_path'])

    dem_nodata = geoprocessing.get_nodata_from_uri(args['dem_uri'])

    def below_sea_level_op(dem_value):
        """Set any value lower than 0 to 0."""
        valid_mask = (dem_value != dem_nodata)
        below_sea_lvl = numpy.where(dem_value < 0, 0, dem_value)

        result = numpy.empty(valid_mask.shape)
        result[:] = dem_nodata
        result[valid_mask] = below_sea_lvl
        return result

    dem_data_type = geoprocessing.get_datatype_from_uri(args['dem_path'])
    # Set DEM values below zero to zero
    geoprocessing.vectorize_datasets(
        [args['dem_path']], below_sea_level_op, file_registry['dem_below_sea_path'],
        dem_data_type, dem_nodata, dem_cell_size, 'intersection',
        vectorize_op=False, aoi_uri=file_registry['aoi_proj_dem_path'])

    viewshed_dir = geoprocessing.get_temporary_directory()
    LOGGER.info("Calculating viewshed.")
    # Call to James's impending viewshed alg
    geoprocessing.viewshed(
        file_registry['dem_below_sea_path'], args['structure_path'],
        file_registry['viewshed_path'], curved_earth=True,
        refractivity=curvature_correction, temp_dir=viewshed_dir,
        block_size=1024, max_radius=float('inf'), skip_over_nodata=True)

    # Read in valuation coefficients
    coef_a = float(args['a_coef'])
    coef_b = float(args['b_coef'])
    coef_c = float(args['c_coef'])
    coef_d = float(args['d_coef'])
    max_val_radius = float(args['max_valuation_radius'])

    def polynomial_val(dist_val):
        """Third Degree Polynomial Valuation function."""

        valid_mask = (dist != dist_nodata)
        max_dist_mask = (dist > max_val_radius)

        dist_one = numpy.where(
            dist[valid_mask] >= 1000,
            (coef_a + coef_b * dist[valid_mask] +
             coef_c * dist[valid_mask]**2 + coef_d * dist[valid_mask]**3),
            ((coef_a + coef_b * 1000 + coef_c * 1000**2 + coef_d * 1000**3) -
             (coef_b + 2 * coef_c * 1000 + 3 * coef_d * 1000**2) *
             (1000 - dist[valid_mask]))
        )

        dist_final = numpy.empty(valid_mask.shape)
        dist_final[:] = dist_nodata
        dist_final[valid_mask] = dist_one
        dist_final[max_dist_mask] = 0.0
        return dist_final

    def log_val(dist):
        """Logarithmic Valuation function."""

        valid_mask = (dist != dist_nodata)
        max_dist_mask = dist > max_valuation_radius

        dist_one = numpy.where(
            dist[valid_mask] >= 1000,
            coef_a + coef_b * numpy.log(dist[valid_mask]),
            (coef_a + coef_b * numpy.log(1000) - (coef_b / 1000) *
             (1000 - dist[valid_mask]))
        )

        dist_final = numpy.empty(valid_mask.shape)
        dist_final[:] = dist_nodata
        dist_final[valid_mask] = dist_one
        dist_final[max_dist_mask] = 0.0
        return dist_final

    def multiply_op(val, view):
        """Multiply the feature valuation by the viewshed weights."""
        return numpy.where(
            (val != nodata and view != nodata), val * view, nodata)

    def add_op(*raster):
        """Aggregate valuation matrices.

        Sums all non-nodata values.

        Parameters:
            *raster (list of numpy arrays): List of numpy matrices.

        Returns:
            numpy.array where the pixel value represents the combined
                pixel values found across all matrices."""

        accumulator = numpy.zeros(raster[0].shape)
        for array in raster:
            array[array == nodata] = 0
            accumulator = numpy.add(accumulator, array)
        return accumulator

    val_raster_list = []
    if args["valuation_function"] == "polynomial":
        val_op = polynomial_val
    else:
        val_op = log_val

    nodata = geoprocessing.get_nodata_from_uri(file_registry['viewshed_path'])

    for viewshed_path in viewshed_dir.listdir():
        # do a distance transform on each feature raster
        dist_path = geoprocessing.get_temporary_filename()
        # What is the pixel of the point? Does it have a unique value?
        distance_transform_edt(
                viewshed_path, dist_path, process_pool=None)

        vshed_val_path = geoprocessing.get_temporary_filename()
        # run valuation equation on distance raster
        geoprocessing.vectorize_datasets(
            [dist_path], val_op, vshed_val_path, gdal.GDT_Float32, nodata,
            dem_cell_size, 'intersection', vectorize_op=False)

        vshed_val_final_path = geoprocessing.get_temporary_filename()
        geoprocessing.vectorize_datasets(
            [vshed_val_path, viewshed_path], multiply_op, vshed_val_final_path,
            gdal.GDT_Float32, nodata, dem_cell_size, 'intersection',
            vectorize_op=False)

        val_raster_list.append(vshed_val_final_path)
        os.path.remove(dist_path)
        os.path.remove(vshed_val_path)

    geoprocessing.vectorize_datasets(
        [val_raster_list], add_op, file_registry['viewshed_valuation_path'],
        gdal.GDT_Float32, nodata, dem_cell_size, 'intersection',
        vectorize_op=False, datasets_are_pre_aligned=True)

    # remove all intermediate val rasters
    for raster_path in val_raster_list:
        os.remove(raster_path)

    # Do quantiles on viewshed_uri
    percentile_list = [25, 50, 75, 100]

    def raster_percentile(band):
        """Operation to use in vectorize_datasets that takes
            the pixels of 'band' and groups them together based on
            their percentile ranges.

            band - A gdal raster band

            returns - An integer that places each pixel into a group
        """
        return bisect(percentiles, band)

    # Get the percentile values for each percentile
    percentiles = calculate_percentiles_from_raster(
        file_registry['viewshed_valuation_path'], percentile_list)

    LOGGER.debug('percentiles_list : %s', percentiles)

    # Add the start_value to the beginning of the percentiles so that any value
    # before the start value is set to nodata
    percentiles.insert(0, int(start_value))

    # Set nodata to a very small negative number
    nodata = -9999919
    pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
        file_registry['viewshed_valuation_path'])

    # Classify the pixels of raster_dataset into groups and write
    # them to output
    pygeoprocessing.geoprocessing.vectorize_datasets(
            [file_registry['viewshed_valuation_path']], raster_percentile,
            file_registry['viewshed_quality_path'], gdal.GDT_Int32, nodata,
            pixel_size, 'intersection', assert_datasets_projected=False)

    # population html stuff
    if 'pop_uri' in args:


        #tabulate population impact
        nodata_pop = geoprocessing.get_nodata_from_uri(args["pop_uri"])
        nodata_viewshed = geoprocessing.get_nodata_from_uri(viewshed_uri)

        #clip population
        pop_wkt = geoprocessing.get_dataset_projection_wkt_uri(args['pop_uri'])
        geoprocessing.reproject_datasource_uri(
            args['aoi_uri'], pop_wkt, aoi_pop_uri)

        geoprocessing.clip_dataset_uri(
            args['pop_uri'], aoi_pop_uri, pop_clip_uri, False)

        #reproject clipped population
        LOGGER.debug("Reprojecting clipped population raster.")
        vs_wkt = geoprocessing.get_dataset_projection_wkt_uri(viewshed_uri)
        pop_cell_size = geoprocessing.get_cell_size_from_uri(pop_clip_uri)
        geoprocessing.reproject_dataset_uri(
            pop_clip_uri, pop_cell_size, vs_wkt, 'bilinear', pop_prj_uri)

        def pop_affected_op(pop, view):
            valid_mask = ((pop != pop_nodata) & (view != view_nodata))

            pop_places = numpy.where(view[valid_mask] > 0, pop[valid_mask], 0)
            pop_final = numpy.empty(valid_mask.shape)
            pop_final[:] = pop_nodata
            pop_final[valid_mask] = pop_places
            return pop_final

        def pop_unaffected_op(pop, view):
            valid_mask = ((pop != pop_nodata) & (view != view_nodata))

            pop_places = numpy.where(view[valid_mask] == 0, pop[valid_mask], 0)
            pop_final = numpy.empty(valid_mask.shape)
            pop_final[:] = pop_nodata
            pop_final[valid_mask] = pop_places
            return pop_final

        viewshed_cell_size = geoprocessing.get_cell_size_from_uri(
            file_registry['viewshed_path'])

        geoprocessing.vectorize_datasets(
            [file_registry['pop_proj_path'], file_registry['viewshed_path']],
            pop_affected_op, file_registry['pop_affected_path'],
            gdal.GDT_Float32, pop_nodata, viewshed_cell_size,
            "intersection", resample_method_list=None,
            dataset_to_align_index=1)

        geoprocessing.vectorize_datasets(
            [file_registry['pop_proj_path'], file_registry['viewshed_path']],
            pop_unaffected_op, file_registry['pop_unaffected_path'],
            gdal.GDT_Float32, pop_nodata, viewshed_cell_size,
            "intersection", resample_method_list=None,
            dataset_to_align_index=1)

        affected_sum = 0
        affected_count = 0
        for _, block in pygeoprocessing.geoprocessing.iterblocks(
            file_registry['pop_affected_path']):

            valid_mask = (block != pop_nodata)
            affected_count += numpy.sum(valid_mask)
            affected_sum += numpy.sum(block[valid_mask])

        unaffected_sum = 0
        unaffected_count = 0
        for _, block in pygeoprocessing.geoprocessing.iterblocks(
            file_registry['pop_unaffected_path']):

            valid_mask = (block != pop_nodata)
            unaffected_count += numpy.sum(valid_mask)
            unaffected_sum += numpy.sum(block[valid_mask])

        if args['pop_type'] == "Density":
            affected_sum = affected_sum * (affected_count * cell_area)
            unaffected_sum = unaffected_sum * (unaffected_count * cell_area)
        else:
            affected_sum = affected_sum / resample_factor
            unaffected_sum = unaffected_sum / resample_factor

    if "overlap_uri" in args:
        geoprocessing.copy_datasource_uri(args["overlap_uri"], file_registry['overlap_uri'])

        LOGGER.debug("Adding id field to overlap features.")
        id_name = "investID"
        add_id_feature_set_uri(overlap_uri, id_name)

        LOGGER.debug("Add area field to overlap features.")
        area_name = "overlap"
        add_field_feature_set_uri(overlap_uri, area_name, ogr.OFTReal)

        LOGGER.debug("Count overlapping pixels per area.")
        values = geoprocessing.aggregate_raster_values_uri(
            viewshed_reclass_uri, overlap_uri, id_name, ignore_nodata=True).total

        def calculate_percent(feature):
            if feature.GetFieldAsInteger(id_name) in values:
                return (values[feature.GetFieldAsInteger(id_name)] * \
                aq_args["cell_size"]) / feature.GetGeometryRef().GetArea()
            else:
                return 0

        LOGGER.debug("Set area field values.")
        set_field_by_op_feature_set_uri(overlap_uri, area_name, calculate_percent)


    LOGGER.info('deleting temporary files')
    for file_id in _TMP_BASE_FILES:
        try:
            if isinstance(file_registry[file_id], basestring):
                os.remove(file_registry[file_id])
            elif isinstance(file_registry[file_id], list):
                for index in xrange(len(file_registry[file_id])):
                    os.remove(file_registry[file_id][index])
        except OSError:
            # Let it go.
            pass

def calculate_percentiles_from_raster(raster_uri, percentiles):
    """Does a memory efficient sort to determine the percentiles
        of a raster. Percentile algorithm currently used is the
        nearest rank method.

        raster_uri - a uri to a gdal raster on disk
        percentiles - a list of desired percentiles to lookup
            ex: [25,50,75,90]

        returns - a list of values corresponding to the percentiles
            from the percentiles list
    """
    raster = gdal.Open(raster_uri, gdal.GA_ReadOnly)

    def numbers_from_file(fle):
        """Generates an iterator from a file by loading all the numbers
            and yielding

            fle = file object
        """
        arr = np.load(fle)
        for num in arr:
            yield num

    # List to hold the generated iterators
    iters = []

    band = raster.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    n_rows = raster.RasterYSize
    n_cols = raster.RasterXSize

    # Variable to count the total number of elements to compute percentile
    # from. This leaves out nodata values
    n_elements = 0

    #Set the row strides to be something reasonable, like 256MB blocks
    row_strides = max(int(2**28 / (4 * n_cols)), 1)

    for row_index in xrange(0, n_rows, row_strides):
        #It's possible we're on the last set of rows and the stride
        #is too big, update if so
        if row_index + row_strides >= n_rows:
            row_strides = n_rows - row_index

        # Read in raster chunk as array
        arr = band.ReadAsArray(0, row_index, n_cols, row_strides)

        tmp_uri = pygeoprocessing.geoprocessing.temporary_filename()
        tmp_file = open(tmp_uri, 'wb')
        # Make array one dimensional for sorting and saving
        arr = arr.flatten()
        # Remove nodata values from array and thus percentile calculation
        arr = np.delete(arr, np.where(arr == nodata))
        # Tally the number of values relevant for calculating percentiles
        n_elements += len(arr)
        # Sort array before saving
        arr = np.sort(arr)

        np.save(tmp_file, arr)
        tmp_file.close()
        tmp_file = open(tmp_uri, 'rb')
        tmp_file.seek(0)
        iters.append(numbers_from_file(tmp_file))
        arr = None

    # List to store the rank/index where each percentile will be found
    rank_list = []
    # For each percentile calculate nearest rank
    for perc in percentiles:
        rank = math.ceil(perc/100.0 * n_elements)
        rank_list.append(int(rank))

    # A variable to burn through when doing heapq merge sort over the
    # iterators. Variable is used to check if we've iterated to a
    # specified rank spot, to grab percentile value
    counter = 0
    # Setup a list of zeros to replace with percentile results
    results = [0] * len(rank_list)

    LOGGER.debug('Percentile Rank List: %s', rank_list)

    for num in heapq.merge(*iters):
        # If a percentile rank has been hit, grab percentile value
        if counter in rank_list:
            LOGGER.debug('percentile value is : %s', num)
            results[rank_list.index(counter)] = int(num)
        counter += 1

    band = None
    raster = None
    return results

















'''

 #determining best data type for viewshed
    features = get_count_feature_set_uri(args['structure_uri'])
    if features < 2 ** 16:
        viewshed_type = gdal.GDT_UInt16
        viewshed_nodata = (2 ** 16) - 1
    elif features < 2 ** 32:
        viewshed_type = gdal.GDT_UInt32
        viewshed_nodata = (2 ** 32) - 1
    else:
        raise ValueError, "Too many structures."


def reclassify_quantile_dataset_uri(
    dataset_uri, quantile_list, dataset_out_uri, datatype_out, nodata_out):
    """Create a Raster based on quantiles.
    """
    nodata_ds = geoprocessing.get_nodata_from_uri(dataset_uri)

    memory_file_uri = geoprocessing.temporary_filename()
    memory_array = geoprocessing.load_memory_mapped_array(
        dataset_uri, memory_file_uri)
    memory_array_flat = memory_array.reshape((-1,))

    quantile_breaks = [0]
    for quantile in quantile_list:
        quantile_breaks.append(scipy.stats.scoreatpercentile(
                memory_array_flat, quantile, (0.0, np.amax(memory_array_flat))))
        LOGGER.debug('quantile %f: %f', quantile, quantile_breaks[-1])

    def reclass(value):
        if value == nodata_ds:
            return nodata_out
        else:
            for new_value,quantile_break in enumerate(quantile_breaks):
                if value <= quantile_break:
                    return new_value
        raise ValueError, "Value was not within quantiles."

    cell_size = geoprocessing.get_cell_size_from_uri(dataset_uri)

    geoprocessing.vectorize_datasets(
        [dataset_uri], reclass, dataset_out_uri, datatype_out, nodata_out,
        cell_size, "union", dataset_to_align_index=0)

    geoprocessing.calculate_raster_stats_uri(dataset_out_uri)

def compute_viewshed_uri(in_dem_uri, out_viewshed_uri, in_structure_uri,
    curvature_correction, refr_coeff, args):
    """ Compute the viewshed as it is defined in ArcGIS where the inputs are:

        -in_dem_uri: URI to input surface raster
        -out_viewshed_uri: URI to the output raster
        -in_structure_uri: URI to a point shapefile that contains the location
        of the observers and the viewshed radius in (negative) meters
        -curvature_correction: flag for the curvature of the earth. Either
        FLAT_EARTH or CURVED_EARTH. Not used yet.
        -refraction: refraction index between 0 (max effect) and 1 (no effect).
        Default is 0.13."""

    # Extract cell size from input DEM
    cell_size = geoprocessing.get_cell_size_from_uri(in_dem_uri)

    # Extract nodata
    nodata = geoprocessing.get_nodata_from_uri(in_dem_uri)

    ## Build I and J arrays, and save them to disk
    rows, cols = geoprocessing.get_row_col_from_uri(in_dem_uri)
    I, J = np.meshgrid(range(rows), range(cols), indexing = 'ij')
    # Base path uri
    base_uri = os.path.split(out_viewshed_uri)[0]
    I_uri = os.path.join(base_uri, 'I.tif')
    J_uri = os.path.join(base_uri, 'J.tif')
    #I_uri = geoprocessing.temporary_filename()
    #J_uri = geoprocessing.temporary_filename()
    geoprocessing.new_raster_from_base_uri(in_dem_uri, I_uri, 'GTiff', \
        -32768., gdal.GDT_Float32, fill_value = -32768.)
    I_raster = gdal.Open(I_uri, gdal.GA_Update)
    I_band = I_raster.GetRasterBand(1)
    I_band.WriteArray(I)
    I_band = None
    I_raster = None
    geoprocessing.new_raster_from_base_uri(in_dem_uri, J_uri, 'GTiff', \
        -32768., gdal.GDT_Float32, fill_value = -32768.)
    J_raster = gdal.Open(J_uri, gdal.GA_Update)
    J_band = J_raster.GetRasterBand(1)
    J_band.WriteArray(J)
    J_band = None
    J_raster = None
    # Extract the input raster geotransform
    GT = geoprocessing.get_geotransform_uri(in_dem_uri)

    # Open the input URI and extract the numpy array
    input_raster = gdal.Open(in_dem_uri)
    input_array = input_raster.GetRasterBand(1).ReadAsArray()
    input_raster = None

    # Create a raster from base before passing it to viewshed
    visibility_uri = out_viewshed_uri #geoprocessing.temporary_filename()
    geoprocessing.new_raster_from_base_uri(in_dem_uri, visibility_uri, 'GTiff', \
        255, gdal.GDT_Byte, fill_value = 255)

    # Call the non-uri version of viewshed.
    #compute_viewshed(in_dem_uri, visibility_uri, in_structure_uri,
    compute_viewshed(input_array, visibility_uri, in_structure_uri,
    cell_size, rows, cols, nodata, GT, I_uri, J_uri, curvature_correction,
    refr_coeff, args)


#def compute_viewshed(in_dem_uri, visibility_uri, in_structure_uri, \
def compute_viewshed(input_array, visibility_uri, in_structure_uri, \
    cell_size, rows, cols, nodata, GT, I_uri, J_uri, curvature_correction, \
    refr_coeff, args):
    """ array-based function that computes the viewshed as is defined in ArcGIS
    """
    # default parameter values that are not passed to this function but that
    # scenic_quality_core.viewshed needs
    obs_elev = 1.0 # Observator's elevation in meters
    tgt_elev = 0.0  # Extra elevation applied to all the DEM
    max_dist = -1.0 # max. viewing distance(m). Distance is infinite if negative
    coefficient = 1.0 # Used to weight the importance of individual viewsheds
    height = 0.0 # Per viewpoint height offset--updated as we read file info

    #input_raster = gdal.Open(in_dem_uri)
    #input_band = input_raster.GetRasterBand(1)
    #input_array = input_band.ReadAsArray()
    #input_band = None
    #input_raster = None

    # Compute the distance for each point
    def compute_distance(vi, vj, cell_size):
        def compute(i, j, v):
            if v == 1:
                return ((vi - i)**2 + (vj - j)**2)**.5 * cell_size
            else:
                return -1.
        return compute

    # Apply the valuation functions to the distance
    def polynomial(a, b, c, d, max_valuation_radius):
        def compute(x, v):
            if v==1:
                if x < 1000:
                    return a + b*1000 + c*1000**2 + d*1000**3 - \
                        (b + 2*c*1000 + 3*d*1000**2)*(1000-x)
                elif x <= max_valuation_radius:
                    return a + b*x + c*x**2 + d*x**3
                else:
                    return 0.
            else:
                return 0.
        return compute

    def logarithmic(a, b, max_valuation_radius):
        def compute(x, v):
            if v==1:
                if x < 1000:
                    return a + b*math.log(1000) - (b/1000)*(1000-x)
                elif x <= max_valuation_radius:
                    return a + b*math.log(x)
                else:
                    return 0.
            else:
                return 0.
        return compute

    # Multiply a value by a constant
    def multiply(c):
        def compute(x):
            return x*c
        return compute


    # Setup valuation function
    a = args["a_coefficient"]
    b = args["b_coefficient"]
    c = args["c_coefficient"]
    d = args["d_coefficient"]

    valuation_function = None
    max_valuation_radius = args['max_valuation_radius']
    if "polynomial" in args["valuation_function"]:
        print("Polynomial")
        valuation_function = polynomial(a, b, c, d, max_valuation_radius)
    elif "logarithmic" in args['valuation_function']:
        print("logarithmic")
        valuation_function = logarithmic(a, b, max_valuation_radius)

    assert valuation_function is not None

    # Make sure the values don't become too small at max_valuation_radius:
    edge_value = valuation_function(max_valuation_radius, 1)
    message = "Valuation function can't be negative if evaluated at " + \
    str(max_valuation_radius) + " meters (value is " + str(edge_value) + ")"
    assert edge_value >= 0., message

    # Base path uri
    base_uri = os.path.split(visibility_uri)[0]

    # Temporary files that will be used
    distance_uri = geoprocessing.temporary_filename()
    viewshed_uri = geoprocessing.temporary_filename()


    # The model extracts each viewpoint from the shapefile
    point_list = []
    shapefile = ogr.Open(in_structure_uri)
    assert shapefile is not None
    layer = shapefile.GetLayer(0)
    assert layer is not None
    iGT = gdal.InvGeoTransform(GT)[1]
    feature_count = layer.GetFeatureCount()
    viewshed_uri_list = []
    print('Number of viewpoints: ' + str(feature_count))
    for f in range(feature_count):
        print("feature " + str(f))
        feature = layer.GetFeature(f)
        field_count = feature.GetFieldCount()
        # Check for feature information (radius, coeff, height)
        for field in range(field_count):
            field_def = feature.GetFieldDefnRef(field)
            field_name = field_def.GetNameRef()
            if (field_name.upper() == 'RADIUS2') or \
                (field_name.upper() == 'RADIUS'):
                max_dist = abs(int(feature.GetField(field)))
                assert max_dist is not None, "max distance can't be None"
                if max_dist < args['max_valuation_radius']:
                    LOGGER.warning( \
                        'Valuation radius > maximum visibility distance: ' + \
                        '(' + str(args['max_valuation_radius']) + ' < ' + \
                        str(max_dist) + ')')
                    LOGGER.warning( \
                        'The valuation is performed beyond what is visible')
                max_dist = int(max_dist/cell_size)
            if field_name.lower() == 'coeff':
                coefficient = float(feature.GetField(field))
                assert coefficient is not None, "feature coeff can't be None"
            if field_name.lower() == 'OFFSETA':
                obs_elev = float(feature.GetField(field))
                assert obs_elev is not None, "OFFSETA can't be None"
            if field_name.lower() == 'OFFSETB':
                tgt_elev = float(feature.GetField(field))
                assert tgt_elev is not None, "OFFSETB can't be None"

        geometry = feature.GetGeometryRef()
        assert geometry is not None
        message = 'geometry type is ' + str(geometry.GetGeometryName()) + \
        ' point is "POINT"'
        assert geometry.GetGeometryName() == 'POINT', message
        x = geometry.GetX()
        y = geometry.GetY()
        j = int((iGT[0] + x*iGT[1] + y*iGT[2]))
        i = int((iGT[3] + x*iGT[4] + y*iGT[5]))

        array_shape = (rows, cols)

        #tmp_visibility_uri = geoprocessing.temporary_filename()
        tmp_visibility_uri = os.path.join(base_uri, 'visibility_' + str(f) + '.tif')
        geoprocessing.new_raster_from_base_uri( \
            visibility_uri, tmp_visibility_uri, 'GTiff', \
            255, gdal.GDT_Float64, fill_value=255)
        scenic_quality_core.viewshed(input_array, cell_size, \
        array_shape, nodata, tmp_visibility_uri, (i,j), obs_elev, tgt_elev, \
        max_dist, refr_coeff)

        # Compute the distance
        #tmp_distance_uri = geoprocessing.temporary_filename()
        tmp_distance_uri = os.path.join(base_uri, 'distance_' + str(f) + '.tif')
        geoprocessing.new_raster_from_base_uri(visibility_uri, \
        tmp_distance_uri, 'GTiff', \
        255, gdal.GDT_Byte, fill_value = 255)
        distance_fn = compute_distance(i,j, cell_size)
        geoprocessing.vectorize_datasets([I_uri, J_uri, tmp_visibility_uri], \
        distance_fn, tmp_distance_uri, gdal.GDT_Float64, -1., cell_size, "union")
        # Apply the valuation function
        #tmp_viewshed_uri = geoprocessing.temporary_filename()
        tmp_viewshed_uri = os.path.join(base_uri, 'viewshed_' + str(f) + '.tif')

        geoprocessing.vectorize_datasets(
            [tmp_distance_uri, tmp_visibility_uri],
            valuation_function, tmp_viewshed_uri, gdal.GDT_Float64, -9999.0, cell_size,
            "union")


        # Multiply the viewshed by its coefficient
        scaled_viewshed_uri = geoprocessing.temporary_filename()
        #os.path.join(base_uri, 'vshed_' + str(f) + '.tif') #geoprocessing.temporary_filename()
        apply_coefficient = multiply(coefficient)
        geoprocessing.vectorize_datasets([tmp_viewshed_uri], apply_coefficient, \
        scaled_viewshed_uri, gdal.GDT_Float64, 0., cell_size, "union")
        viewshed_uri_list.append(scaled_viewshed_uri)

    layer = None
    shapefile = None
    # Accumulate result to combined raster
    def sum_rasters(*x):
        return np.sum(x, axis = 0)
    LOGGER.debug('Summing up everything using vectorize_datasets...')
    LOGGER.debug('visibility_uri' + visibility_uri)
    LOGGER.debug('viewshed_uri_list: ' + str(viewshed_uri_list))
    geoprocessing.vectorize_datasets( \
        viewshed_uri_list, sum_rasters, \
        visibility_uri, gdal.GDT_Float64, -1., cell_size, "union", vectorize_op=False)

def add_field_feature_set_uri(fs_uri, field_name, field_type):
    shapefile = ogr.Open(fs_uri, 1)
    layer = shapefile.GetLayer()
    new_field = ogr.FieldDefn(field_name, field_type)
    layer.CreateField(new_field)
    shapefile = None

def add_id_feature_set_uri(fs_uri, id_name):
    shapefile = ogr.Open(fs_uri, 1)
    message = "Failed to open " + fs_uri + ": can't add new field."
    assert shapefile is not None, message
    layer = shapefile.GetLayer()
    new_field = ogr.FieldDefn(id_name, ogr.OFTInteger)
    layer.CreateField(new_field)

    for feature_id in xrange(layer.GetFeatureCount()):
        feature = layer.GetFeature(feature_id)
        feature.SetField(id_name, feature_id)
        layer.SetFeature(feature)
    shapefile = None

def set_field_by_op_feature_set_uri(fs_uri, value_field_name, op):
    shapefile = ogr.Open(fs_uri, 1)
    layer = shapefile.GetLayer()

    for feature_id in xrange(layer.GetFeatureCount()):
        feature = layer.GetFeature(feature_id)
        feature.SetField(value_field_name, op(feature))
        layer.SetFeature(feature)
    shapefile = None

def get_count_feature_set_uri(fs_uri):
    shapefile = ogr.Open(fs_uri)
    layer = shapefile.GetLayer()
    count = layer.GetFeatureCount()
    shapefile = None

    return count



    #compute_viewshed_uri(viewshed_dem_reclass_uri,
    #         viewshed_uri,
    #         aq_args['structure_uri'],
    #         curvature_correction,
    #         aq_args['refraction'],
    #         aq_args)

    # Do Valuation on viewshed_uri

    # Do quantiles on viewshed_uri
    LOGGER.info("Ranking viewshed.")
    #rank viewshed
    quantile_list = [25,50,75,100]
    LOGGER.debug('reclassify input %s', viewshed_uri)
    LOGGER.debug('reclassify output %s', viewshed_quality_uri)
    reclassify_quantile_dataset_uri(
        viewshed_uri, quantile_list, viewshed_quality_uri, viewshed_type,
        viewshed_nodata)

    # Do population results
    if "pop_uri" in args:
        #tabulate population impact
        LOGGER.info("Tabulating population impact.")
        LOGGER.debug("Tabulating unaffected population.")
        nodata_pop = geoprocessing.get_nodata_from_uri(args["pop_uri"])
        LOGGER.debug("The no data value for the population raster is %s.",
            str(nodata_pop))
        nodata_viewshed = geoprocessing.get_nodata_from_uri(viewshed_uri)
        LOGGER.debug("The no data value for the viewshed raster is %s.",
            str(nodata_viewshed))

        #clip population
        LOGGER.debug("Projecting AOI for population raster clip.")
        pop_wkt = geoprocessing.get_dataset_projection_wkt_uri(args['pop_uri'])
        geoprocessing.reproject_datasource_uri(
            args['aoi_uri'], pop_wkt, aoi_pop_uri)

        LOGGER.debug("Clipping population raster by projected AOI.")
        geoprocessing.clip_dataset_uri(
            args['pop_uri'], aoi_pop_uri, pop_clip_uri, False)

        #reproject clipped population
        LOGGER.debug("Reprojecting clipped population raster.")
        vs_wkt = geoprocessing.get_dataset_projection_wkt_uri(viewshed_uri)
        pop_cell_size = geoprocessing.get_cell_size_from_uri(pop_clip_uri)
        geoprocessing.reproject_dataset_uri(
            pop_clip_uri, pop_cell_size, vs_wkt, 'bilinear', pop_prj_uri)

        #align and resample population
        def copy(value1, value2):
            if value2 == nodata_viewshed:
                return nodata_pop
            else:
                return value1

        LOGGER.debug("Resampling and aligning population raster.")
        pop_prj_datatype = geoprocessing.get_datatype_from_uri(pop_prj_uri)
        geoprocessing.vectorize_datasets(
            [pop_prj_uri, viewshed_uri], copy, pop_vs_uri,
            pop_prj_datatype, nodata_pop, args["cell_size"],
            "intersection", ["bilinear", "bilinear"], 1)

        pop = gdal.Open(pop_vs_uri)
        pop_band = pop.GetRasterBand(1)
        vs = gdal.Open(viewshed_uri)
        vs_band = vs.GetRasterBand(1)

        affected_pop = 0
        unaffected_pop = 0
        for row_index in range(vs_band.YSize):
            pop_row = pop_band.ReadAsArray(0, row_index, pop_band.XSize, 1)
            vs_row = vs_band.ReadAsArray(0, row_index, vs_band.XSize, 1).astype(np.float64)

            pop_row[pop_row == nodata_pop]=0.0
            vs_row[vs_row == nodata_viewshed]=-1

            affected_pop += np.sum(pop_row[vs_row > 0])
            unaffected_pop += np.sum(pop_row[vs_row == 0])

        pop_band = None
        pop = None
        vs_band = None
        vs = None

        table="""
        <html>
        <title>Marine InVEST</title>
        <center><H1>Scenic Quality Model</H1><H2>(Visual Impact from Objects)</H2></center>
        <br><br><HR><br>
        <H2>Population Statistics</H2>

        <table border="1", cellpadding="0">
        <tr><td align="center"><b>Number of Features Visible</b></td><td align="center"><b>Population (estimate)</b></td></tr>
        <tr><td align="center">None visible<br> (unaffected)</td><td align="center">%i</td>
        <tr><td align="center">1 or more<br>visible</td><td align="center">%i</td>
        </table>
        </html>
        """

        outfile = open(pop_stats_uri, 'w')
        outfile.write(table % (unaffected_pop, affected_pop))
        outfile.close()

    # Do overlap / polygon percentages if provided

    #perform overlap analysis
    LOGGER.info("Performing overlap analysis.")

    LOGGER.debug("Reclassifying viewshed")

    nodata_vs_bool = 0
    def non_zeros(value):
        if value == nodata_vs_bool:
            return nodata_vs_bool
        elif value > 0:
            return 1
        else:
            return nodata_vs_bool

    geoprocessing.vectorize_datasets(
        [viewshed_uri], non_zeros, viewshed_reclass_uri, gdal.GDT_Byte,
        nodata_vs_bool, args["cell_size"], "union")

    if "overlap_uri" in args:
        LOGGER.debug("Copying overlap analysis features.")
        geoprocessing.copy_datasource_uri(args["overlap_uri"], overlap_uri)

        LOGGER.debug("Adding id field to overlap features.")
        id_name = "investID"
        add_id_feature_set_uri(overlap_uri, id_name)

        LOGGER.debug("Add area field to overlap features.")
        area_name = "overlap"
        add_field_feature_set_uri(overlap_uri, area_name, ogr.OFTReal)

        LOGGER.debug("Count overlapping pixels per area.")
        values = geoprocessing.aggregate_raster_values_uri(
            viewshed_reclass_uri, overlap_uri, id_name, ignore_nodata=True).total

        def calculate_percent(feature):
            if feature.GetFieldAsInteger(id_name) in values:
                return (values[feature.GetFieldAsInteger(id_name)] * \
                aq_args["cell_size"]) / feature.GetGeometryRef().GetArea()
            else:
                return 0

        LOGGER.debug("Set area field values.")
        set_field_by_op_feature_set_uri(overlap_uri, area_name, calculate_percent)
'''