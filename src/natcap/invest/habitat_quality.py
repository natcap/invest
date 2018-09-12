"""InVEST Habitat Quality model."""
from __future__ import absolute_import
import collections
import os
import logging
import tempfile

import numpy
from osgeo import gdal
from osgeo import osr
import pygeoprocessing

from . import utils
from . import validation

LOGGER = logging.getLogger('natcap.invest.habitat_quality')


def execute(args):
    """Habitat Quality.

    Open files necessary for the portion of the habitat_quality
    model.

    Args:
        workspace_dir (string): a uri to the directory that will write output
            and other temporary files during calculation (required)
        landuse_cur_uri (string): a uri to an input land use/land cover raster
            (required)
        landuse_fut_uri (string): a uri to an input land use/land cover raster
            (optional)
        landuse_bas_uri (string): a uri to an input land use/land cover raster
            (optional, but required for rarity calculations)
        threat_folder (string): a uri to the directory that will contain all
            threat rasters (required)
        threats_uri (string): a uri to an input CSV containing data
            of all the considered threats. Each row is a degradation source
            and each column a different attribute of the source with the
            following names: 'THREAT','MAX_DIST','WEIGHT' (required).
        access_uri (string): a uri to an input polygon shapefile containing
            data on the relative protection against threats (optional)
        sensitivity_uri (string): a uri to an input CSV file of LULC types,
            whether they are considered habitat, and their sensitivity to each
            threat (required)
        half_saturation_constant (float): a python float that determines
            the spread and central tendency of habitat quality scores
            (required)
        suffix (string): a python string that will be inserted into all
            raster uri paths just before the file extension.

    Example Args Dictionary::

        {
            'workspace_dir': 'path/to/workspace_dir',
            'landuse_cur_uri': 'path/to/landuse_cur_raster',
            'landuse_fut_uri': 'path/to/landuse_fut_raster',
            'landuse_bas_uri': 'path/to/landuse_bas_raster',
            'threat_raster_folder': 'path/to/threat_rasters/',
            'threats_uri': 'path/to/threats_csv',
            'access_uri': 'path/to/access_shapefile',
            'sensitivity_uri': 'path/to/sensitivity_csv',
            'half_saturation_constant': 0.5,
            'suffix': '_results',
        }

    Returns:
        None
    """
    workspace = args['workspace_dir']

    # Append a _ to the suffix if it's not empty and doesn't already have one
    suffix = utils.make_suffix_string(args, 'suffix')

    # Check to see if each of the workspace folders exists.  If not, create the
    # folder in the filesystem.
    inter_dir = os.path.join(workspace, 'intermediate')
    out_dir = os.path.join(workspace, 'output')
    utils.make_directories([inter_dir, out_dir])

    # get a handle on the folder with the threat rasters
    threat_raster_dir = args['threat_raster_folder']

    threat_dict = utils.build_lookup_from_csv(
        args['threats_uri'], 'THREAT', to_lower=False)
    sensitivity_dict = utils.build_lookup_from_csv(
        args['sensitivity_uri'], 'LULC', to_lower=False)

    # check that the threat names in the threats table match with the threats
    # columns in the sensitivity table. Raise exception if they don't.
    sens_row = sensitivity_dict.itervalues().next()
    for threat in threat_dict:
        if 'L_' + threat not in sens_row:
            raise ValueError(
                'Threat "L_%s" does not match any column in the sensitivity '
                'table. Possible columns: %s' % (
                    threat, str(sens_row.keys())))

    # get the half saturation constant
    half_saturation = float(args['half_saturation_constant'])

    # Determine which land cover scenarios we should run, and append the
    # appropriate suffix to the landuser_scenarios list as necessary for the
    # scenario.
    landuse_scenarios = {'cur': '_c'}
    scenario_constants = [('landuse_fut_uri', 'fut', '_f'),
                          ('landuse_bas_uri', 'bas', '_b')]
    for lu_uri, lu_time, lu_ext in scenario_constants:
        if lu_uri in args:
            landuse_scenarios[lu_time] = lu_ext

    # declare dictionaries to store the land cover rasters and the density
    # rasters pertaining to the different threats
    landuse_uri_dict = {}
    density_uri_dict = {}

    # for each possible land cover that was provided try opening the raster and
    # adding it to the dictionary. Also compile all the threat/density rasters
    # associated with the land cover
    for scenario, ext in landuse_scenarios.iteritems():
        landuse_uri_dict[ext] = args['landuse_' + scenario + '_uri']

        # add a key to the density dictionary that associates all density/threat
        # rasters with this land cover
        density_uri_dict['density' + ext] = {}

        # for each threat given in the CSV file try opening the associated
        # raster which should be found in threat_raster_folder
        for threat in threat_dict:
            if ext == '_b':
                density_uri_dict['density' + ext][threat] = (
                    resolve_ambiguous_raster_path(
                        os.path.join(threat_raster_dir, threat + ext),
                        raise_error=False))
            else:
                density_uri_dict['density' + ext][threat] = (
                    resolve_ambiguous_raster_path(
                        os.path.join(threat_raster_dir, threat + ext)))

    # checking to make sure the land covers have the same projections, and
    # printing warnings in case projections are not identical
    landuse_info_list = [
        pygeoprocessing.get_raster_info(path) for path in
        landuse_uri_dict.values()]
    projection_set = set([x['projection'] for x in landuse_info_list])
    if len(projection_set) != 1:
        LOGGER.warn(
            "Projections are not identical. Here's the projections: %s" %
            str([(path, x['projection']) for path, x in zip(
                landuse_uri_dict.values(), landuse_info_list)]))

    LOGGER.debug('Starting habitat_quality biophysical calculations')

    cur_landuse_uri = landuse_uri_dict['_c']

    out_nodata = -1.0

    # If access_lyr: convert to raster, if value is null set to 1,
    # else set to value
    try:
        LOGGER.debug('Handling Access Shape')
        access_dataset_path = os.path.join(
            inter_dir, 'access_layer%s.tif' % suffix)
        pygeoprocessing.new_raster_from_base(
            cur_landuse_uri, access_dataset_path, gdal.GDT_Float32,
            [out_nodata], fill_value_list=[1.0])
        # Fill raster to all 1's (fully accessible) in case polygons do not
        # cover land area

        pygeoprocessing.rasterize(
            args['access_uri'], access_dataset_path, burn_values=None,
            option_list=['ATTRIBUTE=ACCESS'])

    except KeyError:
        LOGGER.debug(
            'No Access Shape Provided, access raster filled with 1s.')

    # calculate the weight sum which is the sum of all the threats weights
    weight_sum = 0.0
    for threat_data in threat_dict.itervalues():
        # Sum weight of threats
        weight_sum = weight_sum + float(threat_data['WEIGHT'])

    LOGGER.debug('landuse_uri_dict : %s', landuse_uri_dict)

    # for each land cover raster provided compute habitat quality
    for lulc_key, lulc_ds_uri in landuse_uri_dict.iteritems():
        LOGGER.debug('Calculating results for landuse : %s', lulc_key)

        # Create raster of habitat based on habitat field
        habitat_path = os.path.join(
            inter_dir, 'habitat_%s%s.tif' % (lulc_key, suffix))

        map_raster_to_dict_values(
            lulc_ds_uri, habitat_path, sensitivity_dict, 'HABITAT', out_nodata,
            'none')

        # initialize a list that will store all the density/threat rasters
        # after they have been adjusted for distance, weight, and access
        degradation_raster_list = []

        # a list to keep track of the normalized weight for each threat
        weight_list = numpy.array([])

        # variable to indicate whether we should break out of calculations
        # for a land cover because a threat raster was not found
        exit_landcover = False

        # adjust each density/threat raster for distance, weight, and access
        for threat, threat_data in threat_dict.iteritems():

            LOGGER.debug('Calculating threat : %s', threat)
            LOGGER.debug('Threat Data : %s', threat_data)

            # get the density raster for the specific threat
            threat_dataset_path = density_uri_dict['density' + lulc_key][threat]
            LOGGER.debug('threat_dataset_path %s', threat_dataset_path)
            if threat_dataset_path is None:
                LOGGER.info(
                    'A certain threat raster could not be found for the '
                    'Baseline Land Cover. Skipping Habitat Quality '
                    'calculation for this land cover.')
                exit_landcover = True
                break

            # get the cell size from LULC to use for intermediate / output
            # rasters
            lulc_cell_size = (
                pygeoprocessing.get_raster_info(
                    args['landuse_cur_uri']))['pixel_size']

            # need the cell size for the threat raster so we can create
            # an appropriate kernel for convolution
            threat_cell_size = pygeoprocessing.get_raster_info(
                threat_dataset_path)['mean_pixel_size']

            # convert max distance (given in KM) to meters
            dr_max = float(threat_data['MAX_DIST']) * 1000.0

            # convert max distance from meters to the number of pixels that
            # represents on the raster
            dr_pixel = dr_max / threat_cell_size
            LOGGER.debug('Max distance in pixels: %f', dr_pixel)

            filtered_threat_path = os.path.join(
                inter_dir, threat + '_filtered_%s%s.tif' % (lulc_key, suffix))

            # blur the threat raster based on the effect of the threat over
            # distance
            decay_type = threat_data['DECAY']
            kernel_handle, kernel_path = tempfile.mkstemp()
            os.close(kernel_handle)
            if decay_type == 'linear':
                make_linear_decay_kernel_path(dr_pixel, kernel_path)
            elif decay_type == 'exponential':
                utils.exponential_decay_kernel_raster(dr_pixel, kernel_path)
            else:
                raise ValueError(
                    "Unknown type of decay in biophysical table, should be "
                    "either 'linear' or 'exponential' input was %s" % (
                        decay_type))
            pygeoprocessing.convolve_2d(
                (threat_dataset_path, 1), (kernel_path, 1), filtered_threat_path)
            os.remove(kernel_path)
            # create sensitivity raster based on threat
            sens_uri = os.path.join(
                inter_dir, 'sens_' + threat + lulc_key + suffix + '.tif')

            map_raster_to_dict_values(
                lulc_ds_uri, sens_uri, sensitivity_dict,
                'L_' + threat, out_nodata, 'values_required')

            # get the normalized weight for each threat
            weight_avg = float(threat_data['WEIGHT']) / weight_sum

            # add the threat raster adjusted by distance and the raster
            # representing sensitivity to the list to be past to
            # vectorized_rasters below
            degradation_raster_list.append(filtered_threat_path)
            degradation_raster_list.append(sens_uri)

            # store the normalized weight for each threat in a list that
            # will be used below in total_degradation
            weight_list = numpy.append(weight_list, weight_avg)

        # check to see if we got here because a threat raster was missing
        # and if so then we want to skip to the next landcover
        if exit_landcover:
            continue

        def total_degradation(*raster):
            """A vectorized function that computes the degradation value for
                each pixel based on each threat and then sums them together

                *rasters - a list of floats depicting the adjusted threat
                    value per pixel based on distance and sensitivity.
                    The values are in pairs so that the values for each threat
                    can be tracked:
                    [filtered_val_threat1, sens_val_threat1,
                     filtered_val_threat2, sens_val_threat2, ...]
                    There is an optional last value in the list which is the
                    access_raster value, but it is only present if
                    access_raster is not None.

                returns - the total degradation score for the pixel"""

            # we can not be certain how many threats the user will enter,
            # so we handle each filtered threat and sensitivity raster
            # in pairs
            sum_degradation = numpy.zeros(raster[0].shape)
            for index in range(len(raster) / 2):
                step = index * 2
                sum_degradation += (
                    raster[step] * raster[step + 1] * weight_list[index])

            nodata_mask = numpy.empty(raster[0].shape, dtype=numpy.int8)
            nodata_mask[:] = 0
            for array in raster:
                nodata_mask = nodata_mask | (array == out_nodata)

            # the last element in raster is access
            return numpy.where(
                    nodata_mask, out_nodata, sum_degradation * raster[-1])

        # add the access_raster onto the end of the collected raster list. The
        # access_raster will be values from the shapefile if provided or a
        # raster filled with all 1's if not
        degradation_raster_list.append(access_dataset_path)

        deg_sum_path = os.path.join(
            out_dir, 'deg_sum_out' + lulc_key + suffix + '.tif')

        LOGGER.debug('Starting aligning and resizing degradation rasters')

        aligned_degradation_raster_list = [
            path.replace('.tif', '_aligned.tif') for path in
            degradation_raster_list]
        pygeoprocessing.align_and_resize_raster_stack(
            degradation_raster_list, aligned_degradation_raster_list,
            ['near']*len(degradation_raster_list), lulc_cell_size,
            'intersection')

        LOGGER.debug('Finished aligning and resizing degradation rasters')

        LOGGER.debug('Starting raster calculation on total_degradation')

        degradation_raster_band_list = [
            (path, 1) for path in aligned_degradation_raster_list]
        pygeoprocessing.raster_calculator(
            degradation_raster_band_list, total_degradation, deg_sum_path,
            gdal.GDT_Float32, out_nodata)

        LOGGER.debug('Finished raster calculation on total_degradation')

        # Compute habitat quality
        # scaling_param is a scaling parameter set to 2.5 as noted in the users
        # guide
        scaling_param = 2.5

        # a term used below to compute habitat quality
        ksq = half_saturation**scaling_param

        def quality_op(degradation, habitat):
            """Vectorized function that computes habitat quality given
                a degradation and habitat value.

                degradation - a float from the created degradation
                    raster above.
                habitat - a float indicating habitat suitability from
                    from the habitat raster created above.

                returns - a float representing the habitat quality
                    score for a pixel
            """
            degredataion_clamped = numpy.where(degradation < 0, 0, degradation)

            return numpy.where(
                    (degradation == out_nodata) | (habitat == out_nodata),
                    out_nodata,
                    (habitat * (1.0 - ((degredataion_clamped**scaling_param) /
                     (degredataion_clamped**scaling_param + ksq)))))

        quality_path = os.path.join(
            out_dir, 'quality_out' + lulc_key + suffix + '.tif')

        LOGGER.debug('Starting aligning and resizing degradation and habitat \
                     rasters.')

        deg_hab_raster_list = [deg_sum_path, habitat_path]
        aligned_deg_hab_raster_list = [
            path.replace('.tif', '_aligned.tif') for path in
            deg_hab_raster_list]
        pygeoprocessing.align_and_resize_raster_stack(
            deg_hab_raster_list, aligned_deg_hab_raster_list,
            ['near']*len(deg_hab_raster_list), lulc_cell_size,
            'intersection')

        LOGGER.debug('Finished aligning and resizing degradation and habitat \
                     rasters.')

        LOGGER.debug('Starting raster calculation on quality_op')

        deg_hab_raster_band_list = [
            (path, 1) for path in aligned_deg_hab_raster_list]
        pygeoprocessing.raster_calculator(
            deg_hab_raster_band_list, quality_op, quality_path,
            gdal.GDT_Float32, out_nodata)

        LOGGER.debug('Finished raster calculation on quality_op')

    # Compute Rarity if user supplied baseline raster
    if '_b' not in landuse_uri_dict:
        LOGGER.info('Baseline not provided to compute Rarity')
    else:
        lulc_base_uri = landuse_uri_dict['_b']

        # get the area of a base pixel to use for computing rarity where the
        # pixel sizes are different between base and cur/fut rasters
        base_area = pygeoprocessing.get_raster_info(
            lulc_base_uri)['mean_pixel_size'] ** 2
        base_nodata = pygeoprocessing.get_raster_info(
            lulc_base_uri)['nodata'][0]
        rarity_nodata = -64329.0

        lulc_code_count_b = raster_pixel_count(lulc_base_uri)

        # compute rarity for current landscape and future (if provided)
        lulc_nodata = None
        for lulc_cover in ['_c', '_f']:
            if lulc_cover not in landuse_uri_dict:
                continue
            lulc_cover_path = landuse_uri_dict[lulc_cover]

            if lulc_cover == '_c':
                lulc_time = 'current'
            elif lulc_cover == '_f':
                lulc_time = 'future'

            # get the area of a cur/fut pixel
            lulc_area = pygeoprocessing.get_raster_info(
                lulc_cover_path)['mean_pixel_size'] ** 2
            lulc_nodata = pygeoprocessing.get_raster_info(
                lulc_cover_path)['nodata'][0]

            def trim_op(base, cover_x):
                """Trim cover_x to the mask of base.

                Parameters:
                    base (numpy.ndarray): base raster from 'lulc_base'
                    cover_x (numpy.ndarray): either future or current land
                        cover raster from 'lulc_cover_path' above

                Returns:
                    out_nodata where either array has nodata, otherwise
                    cover_x.
                """
                return numpy.where(
                    (base == base_nodata) | (cover_x == lulc_nodata),
                    base_nodata, cover_x)

            LOGGER.debug('Create new cover for %s', lulc_cover)

            new_cover_path = os.path.join(
                inter_dir, 'new_cover' + lulc_cover + suffix + '.tif')

            # set the current/future land cover to be masked to the base
            # land cover

            LOGGER.debug('Starting aligning and resizing %s and base lulc \
                rasters.' % lulc_time)

            lulc_raster_list = [lulc_base_uri, lulc_cover_path]
            aligned_lulc_raster_list = [
                path.replace('.tif', '_aligned.tif') for path in
                lulc_raster_list]
            pygeoprocessing.align_and_resize_raster_stack(
                lulc_raster_list, aligned_lulc_raster_list,
                ['near']*len(lulc_raster_list), lulc_cell_size,
                'intersection')

            LOGGER.debug('Finished aligning and resizing %s and base lulc \
                rasters.' % lulc_time)

            LOGGER.debug('Starting masking %s land cover to base land cover'
                         % lulc_time)

            lulc_raster_band_list = [
                (path, 1) for path in aligned_lulc_raster_list]
            pygeoprocessing.raster_calculator(
                lulc_raster_band_list, trim_op, new_cover_path,
                gdal.GDT_Float32, out_nodata)

            LOGGER.debug('Finished masking %s land cover to base land cover'
                         % lulc_time)

            lulc_code_count_x = raster_pixel_count(new_cover_path)

            # a dictionary to map LULC types to a number that depicts how
            # rare they are considered
            code_index = {}

            # compute rarity index for each lulc code
            # define 0.0 if an lulc code is found in the cur/fut landcover
            # but not the baseline
            for code in lulc_code_count_x.iterkeys():
                if code in lulc_code_count_b:
                    numerator = float(lulc_code_count_x[code] * lulc_area)
                    denominator = float(
                        lulc_code_count_b[code] * base_area)
                    ratio = 1.0 - (numerator / denominator)
                    code_index[code] = ratio
                else:
                    code_index[code] = 0.0

            rarity_uri = os.path.join(
                out_dir, 'rarity' + lulc_cover + suffix + '.tif')

            pygeoprocessing.reclassify_raster(
                (new_cover_path, 1), code_index, rarity_uri, gdal.GDT_Float32,
                rarity_nodata)

            LOGGER.debug('Finished vectorize on map_ratio')
    LOGGER.debug('Finished habitat_quality biophysical calculations')


def resolve_ambiguous_raster_path(path, raise_error=True):
    """Determine real path when we don't know true path extension.

    Parameters:
        path (string): file path that includes the name of the file but not
            its extension

        raise_error (boolean): if True then function will raise an
            ValueError if a valid raster file could not be found.

    Return:
        the full path, plus extension, to the valid raster.
    """
    # Turning on exceptions so that if an error occurs when trying to open a
    # file path we can catch it and handle it properly

    def _error_handler(*_):
        """A dummy error handler that raises a ValueError."""
        raise ValueError()
    gdal.PushErrorHandler(_error_handler)

    # a list of possible suffixes for raster datasets. We currently can handle
    # .tif and directory paths
    possible_suffixes = ['', '.tif', '.img']

    # initialize dataset to None in the case that all paths do not exist
    dataset = None
    for suffix in possible_suffixes:
        full_path = path + suffix
        if not os.path.exists(full_path):
            continue
        try:
            dataset = gdal.OpenEx(full_path, gdal.GA_ReadOnly)
            break
        except ValueError:
            # If GDAL can't open the raster, our GDAL error handler will be
            # executed and ValueError raised.
            continue

    gdal.PopErrorHandler()

    # If a dataset comes back None, then it could not be found / opened and we
    # should fail gracefully
    if dataset is None:
        if raise_error:
            raise ValueError(
                'There was an Error locating a threat raster in the input '
                'folder. One of the threat names in the CSV table does not '
                'match to a threat raster in the input folder. Please check '
                'that the names correspond. The threat raster that could not '
                'be found is: %s' % path)
        else:
            full_path = None

    dataset = None
    return full_path


def raster_pixel_count(raster_path):
    """Count unique pixel values in raster.

    Parameters:
        raster_path (string): path to a raster

    Returns:
        dict of pixel values to frequency.
    """
    nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
    counts = collections.defaultdict(int)
    for _, raster_block in pygeoprocessing.iterblocks(raster_path):
        for value, count in zip(
                *numpy.unique(raster_block, return_counts=True)):
            if value == nodata:
                continue
            counts[value] += count
    return counts


def map_raster_to_dict_values(
        key_raster_uri, out_uri, attr_dict, field, out_nodata, raise_error):
    """Creates a new raster from 'key_raster' where the pixel values from
       'key_raster' are the keys to a dictionary 'attr_dict'. The values
       corresponding to those keys is what is written to the new raster. If a
       value from 'key_raster' does not appear as a key in 'attr_dict' then
       raise an Exception if 'raise_error' is True, otherwise return a
       'out_nodata'

       key_raster_uri - a GDAL raster uri dataset whose pixel values relate to
                     the keys in 'attr_dict'
       out_uri - a string for the output path of the created raster
       attr_dict - a dictionary representing a table of values we are interested
                   in making into a raster
       field - a string of which field in the table or key in the dictionary
               to use as the new raster pixel values
       out_nodata - a floating point value that is the nodata value.
       raise_error - a string that decides how to handle the case where the
           value from 'key_raster' is not found in 'attr_dict'. If 'raise_error'
           is 'values_required', raise Exception, if 'none', return 'out_nodata'

       returns - a GDAL raster, or raises an Exception and fail if:
           1) raise_error is True and
           2) the value from 'key_raster' is not a key in 'attr_dict'
    """

    LOGGER.debug('Starting map_raster_to_dict_values')
    int_attr_dict = {}
    for key in attr_dict:
        int_attr_dict[int(key)] = float(attr_dict[key][field])

    pygeoprocessing.reclassify_raster(
        (key_raster_uri, 1), int_attr_dict, out_uri, gdal.GDT_Float32,
        out_nodata)


def make_linear_decay_kernel_path(max_distance, kernel_path):
    """Create a linear decay kernel as a raster.

    Pixels in raster are equal to d / max_distance where d is the distance to
    the center of the raster in number of pixels.

    Parameters:
        max_distance (int): number of pixels out until the decay is 0.
        kernel_path (string): path to output raster whose values are in (0,1)
            representing distance to edge.  Size is (max_distance * 2 + 1)^2

    Returns:
        None
    """
    kernel_size = int(numpy.round(max_distance * 2 + 1))

    driver = gdal.GetDriverByName('GTiff')
    kernel_dataset = driver.Create(
        kernel_path.encode('utf-8'), kernel_size, kernel_size, 1,
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
        distance_kernel_row = numpy.sqrt(
            (row_index - max_distance) ** 2 +
            (col_index - max_distance) ** 2).reshape(1, kernel_size)
        kernel = numpy.where(
            distance_kernel_row > max_distance, 0.0,
            (max_distance - distance_kernel_row) / max_distance)
        integration += numpy.sum(kernel)
        kernel_band.WriteArray(kernel, xoff=0, yoff=row_index)

    for row_index in xrange(kernel_size):
        kernel_row = kernel_band.ReadAsArray(
            xoff=0, yoff=row_index, win_xsize=kernel_size, win_ysize=1)
        kernel_row /= integration
        kernel_band.WriteArray(kernel_row, 0, row_index)


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

    # required args
    for key in [
            'workspace_dir',
            'landuse_cur_uri',
            'threat_raster_folder',
            'threats_uri',
            'sensitivity_uri',
            'half_saturation_constant']:
        if limit_to is None or limit_to == key:
            if key not in args:
                missing_key_list.append(key)
            elif args[key] in ['', None]:
                no_value_list.append(key)

    if len(missing_key_list) > 0:
        # if there are missing keys, we have raise KeyError to stop hard
        raise KeyError(
            "The following keys were expected in `args` but were missing" +
            ', '.join(missing_key_list))

    # check for required file existence:
    for key in [
            'landuse_cur_uri',
            'threat_raster_folder',
            'threats_uri',
            'sensitivity_uri']:
        if (limit_to is None or limit_to == key) and (
                not os.path.exists(args[key])):
            validation_error_list.append(
                ([key], 'not found on disk'))

    # check that existing/optional files are the correct types
    with utils.capture_gdal_logging():
        for key, key_type in [
                ('landuse_cur_uri', 'raster'),
                ('landuse_fut_uri', 'raster'),
                ('landuse_bas_uri', 'raster'),
                ('access_uri', 'vector')]:
            if (limit_to is None or limit_to == key) and key in args:
                if not os.path.exists(args[key]):
                    validation_error_list.append(
                        ([key], 'not found on disk'))
                    continue
                if key_type == 'raster':
                    raster = gdal.OpenEx(args[key])
                    if raster is None:
                        validation_error_list.append(
                            ([key], 'not a raster'))
                    del raster
                elif key_type == 'vector':
                    vector = gdal.OpenEx(args[key])
                    if vector is None:
                        validation_error_list.append(
                            ([key], 'not a vector'))
                    del vector

    return validation_error_list
