"""InVEST Habitat Quality model"""

import os.path
import logging
import numpy
import csv

from osgeo import gdal
from osgeo import osr
import numpy as np
import pygeoprocessing.geoprocessing

from .. import utils

logging.basicConfig(format='%(asctime)s %(name)-18s %(levelname)-8s \
     %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.habitat_quality.habitat_quality')


def execute(args):
    """
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
        none
    """

    workspace = args['workspace_dir']

    # Append a _ to the suffix if it's not empty and doesn't already have one
    try:
        suffix = args['suffix']
        if suffix != "" and not suffix.startswith('_'):
            suffix = '_' + suffix
    except KeyError:
        suffix = ''

    # Check to see if each of the workspace folders exists.  If not, create the
    # folder in the filesystem.
    inter_dir = os.path.join(workspace, 'intermediate')
    out_dir = os.path.join(workspace, 'output')
    pygeoprocessing.geoprocessing.create_directories([inter_dir, out_dir])

    # get a handle on the folder with the threat rasters
    threat_raster_dir = args['threat_raster_folder']

    threat_dict = make_dictionary_from_csv(args['threats_uri'],'THREAT')
    sensitivity_dict = make_dictionary_from_csv(
        args['sensitivity_uri'], 'LULC')

    # check that the threat names in the threats table match with the threats
    # columns in the sensitivity table. Raise exception if they don't.
    if not threat_names_match(threat_dict, sensitivity_dict, 'L_'):
        raise Exception(
            'The threat names in the threat table do '
            'not match the columns in the sensitivity table')

    # get the half saturation constant
    half_saturation = float(args['half_saturation_constant'])

    # Determine which land cover scenarios we should run, and append the
    # appropriate suffix to the landuser_scenarios list as necessary for the
    # scenario.
    landuse_scenarios = {'cur':'_c'}
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
            try:
                if ext == '_b':
                    density_uri_dict['density' + ext][threat] = resolve_ambiguous_raster_path(
                            os.path.join(threat_raster_dir, threat + ext),
                            raise_error=False)
                else:
                    density_uri_dict['density' + ext][threat] = resolve_ambiguous_raster_path(
                            os.path.join(threat_raster_dir, threat + ext))
            except:
                raise Exception('Error: Failed to open raster for the '
                    'following threat : %s . Please make sure the threat '
                    'names in the CSV table correspond to threat rasters '
                    'in the input folder.'
                    % os.path.join(threat_raster_dir, threat + ext))

    # checking to make sure the land covers have the same projections and are
    # projected in meters. We pass in 1.0 because that is the unit for meters
    if not check_projections(landuse_uri_dict, 1.0):
        raise Exception('Land cover projections are not the same or are '
                        'not projected in meters')

    LOGGER.debug('Starting habitat_quality biophysical calculations')

    cur_landuse_uri = landuse_uri_dict['_c']

    out_nodata = -1.0

    # If access_lyr: convert to raster, if value is null set to 1,
    # else set to value
    try:
        LOGGER.debug('Handling Access Shape')
        access_dataset_uri = os.path.join(
            inter_dir, 'access_layer%s.tif' % suffix)
        pygeoprocessing.geoprocessing.new_raster_from_base_uri(
            cur_landuse_uri, access_dataset_uri, 'GTiff', out_nodata,
            gdal.GDT_Float32, fill_value=1.0)
        #Fill raster to all 1's (fully accessible) incase polygons do not cover
        #land area

        pygeoprocessing.geoprocessing.rasterize_layer_uri(
            access_dataset_uri, args['access_uri'],
            option_list=['ATTRIBUTE=ACCESS'])

    except KeyError:
        LOGGER.debug('No Access Shape Provided, access raster filled with 1s.')

    # calculate the weight sum which is the sum of all the threats weights
    weight_sum = 0.0
    for threat_data in threat_dict.itervalues():
        #Sum weight of threats
        weight_sum = weight_sum + float(threat_data['WEIGHT'])

    LOGGER.debug('landuse_uri_dict : %s', landuse_uri_dict)

    # for each land cover raster provided compute habitat quality
    for lulc_key, lulc_ds_uri in landuse_uri_dict.iteritems():
        LOGGER.debug('Calculating results for landuse : %s', lulc_key)

        #Create raster of habitat based on habitat field
        habitat_uri = os.path.join(
            inter_dir, 'habitat_%s%s.tif' % (lulc_key, suffix))

        map_raster_to_dict_values(
            lulc_ds_uri, habitat_uri, sensitivity_dict, 'HABITAT', out_nodata,
            'none')

        # initialize a list that will store all the density/threat rasters
        # after they have been adjusted for distance, weight, and access
        degradation_rasters = []

        # a list to keep track of the normalized weight for each threat
        weight_list = np.array([])

        # variable to indicate whether we should break out of calculations
        # for a land cover because a threat raster was not found
        exit_landcover = False

        # adjust each density/threat raster for distance, weight, and access
        for threat, threat_data in threat_dict.iteritems():

            LOGGER.debug('Calculating threat : %s', threat)
            LOGGER.debug('Threat Data : %s', threat_data)

            # get the density raster for the specific threat
            threat_dataset_uri = density_uri_dict['density' + lulc_key][threat]
            LOGGER.debug('threat_dataset_uri %s' % threat_dataset_uri)
            if threat_dataset_uri == None:
                LOGGER.info(
                'A certain threat raster could not be found for the '
                'Baseline Land Cover. Skipping Habitat Quality '
                'calculation for this land cover.')
                exit_landcover = True
                break

            # get the cell size from LULC to use for intermediate / output
            # rasters
            lulc_cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                args['landuse_cur_uri'])

            # need the cell size for the threat raster so we can create
            # an appropriate kernel for convolution
            threat_cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                threat_dataset_uri)

            # convert max distance (given in KM) to meters
            dr_max = float(threat_data['MAX_DIST']) * 1000.0

            # convert max distance from meters to the number of pixels that
            # represents on the raster
            dr_pixel = dr_max / threat_cell_size
            LOGGER.debug('Max distance in pixels: %f', dr_pixel)

            filtered_threat_uri = os.path.join(
                inter_dir, threat + '_filtered_%s%s.tif' % (lulc_key, suffix))

            # blur the threat raster based on the effect of the threat over
            # distance
            decay_type = threat_data['DECAY']
            kernel_uri = pygeoprocessing.geoprocessing.temporary_filename()
            if decay_type == 'linear':
                make_linear_decay_kernel_uri(dr_pixel, kernel_uri)
            elif decay_type == 'exponential':
                utils.exponential_decay_kernel_raster(dr_pixel, kernel_uri)
            else:
                raise TypeError("Unknown type of decay in biophysical table, should be either 'linear' or 'exponential' input was %s" % (decay_type))
            pygeoprocessing.geoprocessing.convolve_2d_uri(threat_dataset_uri, kernel_uri, filtered_threat_uri)
            os.remove(kernel_uri)
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
            degradation_rasters.append(filtered_threat_uri)
            degradation_rasters.append(sens_uri)

            # store the normalized weight for each threat in a list that
            # will be used below in total_degradation
            weight_list = np.append(weight_list, weight_avg)

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
            sum_degradation = np.zeros(raster[0].shape)
            for index in range(len(raster) / 2):
                step = index * 2
                sum_degradation += (
                    raster[step] * raster[step + 1] * weight_list[index])

            nodata_mask = np.empty(raster[0].shape, dtype=np.int8)
            nodata_mask[:] = 0
            for array in raster:
                nodata_mask = nodata_mask | (array == out_nodata)

            #the last element in raster is access
            return np.where(
                    nodata_mask, out_nodata, sum_degradation * raster[-1])

        # add the access_raster onto the end of the collected raster list. The
        # access_raster will be values from the shapefile if provided or a
        # raster filled with all 1's if not
        degradation_rasters.append(access_dataset_uri)

        deg_sum_uri = os.path.join(
            out_dir, 'deg_sum_out' + lulc_key + suffix + '.tif')

        LOGGER.debug('Starting vectorize on total_degradation')

        pygeoprocessing.geoprocessing.vectorize_datasets(
            degradation_rasters, total_degradation, deg_sum_uri,
            gdal.GDT_Float32, out_nodata, lulc_cell_size, "intersection",
            vectorize_op=False)

        LOGGER.debug('Finished vectorize on total_degradation')

        #Compute habitat quality
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

            return np.where(
                    (degradation == out_nodata) | (habitat == out_nodata),
                    out_nodata,
                    (habitat * (1.0 - ((degredataion_clamped**scaling_param) /
                        (degredataion_clamped**scaling_param + ksq)))))

        quality_uri = os.path.join(
            out_dir, 'quality_out' + lulc_key + suffix + '.tif')

        LOGGER.debug('Starting vectorize on quality_op')

        pygeoprocessing.geoprocessing.vectorize_datasets(
            [deg_sum_uri, habitat_uri], quality_op, quality_uri,
            gdal.GDT_Float32, out_nodata, lulc_cell_size, "intersection",
            vectorize_op = False)

        LOGGER.debug('Finished vectorize on quality_op')

    #Compute Rarity if user supplied baseline raster
    try:
        # will throw a KeyError exception if no base raster is provided
        lulc_base_uri = landuse_uri_dict['_b']

        # get the area of a base pixel to use for computing rarity where the
        # pixel sizes are different between base and cur/fut rasters
        base_area = pygeoprocessing.geoprocessing.get_cell_size_from_uri(lulc_base_uri) ** 2
        base_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(lulc_base_uri)
        rarity_nodata = -64329.0

        lulc_code_count_b = raster_pixel_count(lulc_base_uri)

        # compute rarity for current landscape and future (if provided)
        for lulc_cover in ['_c', '_f']:
            try:
                lulc_x = landuse_uri_dict[lulc_cover]

                # get the area of a cur/fut pixel
                lulc_area = pygeoprocessing.geoprocessing.get_cell_size_from_uri(lulc_x) ** 2
                lulc_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(lulc_x)

                LOGGER.debug('Base and Cover NODATA : %s : %s', base_nodata,
                        lulc_nodata)

                def trim_op(base, cover_x):
                    """Vectorized function used in vectorized_rasters. Trim
                        cover_x to the mask of base

                        base - the base raster from 'lulc_base' above
                        cover_x - either the future or current land cover raster
                            from 'lulc_x' above

                        return - out_nodata if base or cover_x is equal to their
                            nodata values or the cover_x value
                        """
                    return np.where(
                            (base == base_nodata) | (cover_x == lulc_nodata),
                            base_nodata, cover_x)

                LOGGER.debug('Create new cover for %s', lulc_cover)

                new_cover_uri = os.path.join(
                    inter_dir, 'new_cover' + lulc_cover + suffix + '.tif')

                LOGGER.debug('Starting vectorize on trim_op')

                # set the current/future land cover to be masked to the base
                # land cover

                pygeoprocessing.geoprocessing.vectorize_datasets(
                    [lulc_base_uri, lulc_x], trim_op, new_cover_uri,
                    gdal.GDT_Int32, base_nodata, lulc_cell_size, "intersection",
                    vectorize_op = False)

                LOGGER.debug('Finished vectorize on trim_op')

                lulc_code_count_x = raster_pixel_count(new_cover_uri)

                # a dictionary to map LULC types to a number that depicts how
                # rare they are considered
                code_index = {}

                # compute the ratio or rarity index for each lulc code where we
                # return 0.0 if an lulc code is found in the cur/fut land cover
                # but not the baseline
                for code in lulc_code_count_x.iterkeys():
                    try:
                        numerator = float(lulc_code_count_x[code] * lulc_area)
                        denominator = float(
                            lulc_code_count_b[code] * base_area)
                        ratio = 1.0 - (numerator / denominator)
                        code_index[code] = ratio
                    except KeyError:
                        code_index[code] = 0.0

                rarity_uri = os.path.join(
                    out_dir, 'rarity' + lulc_cover + suffix + '.tif')

                LOGGER.debug('Starting vectorize on map_ratio')

                pygeoprocessing.geoprocessing.reclassify_dataset_uri(
                        new_cover_uri, code_index, rarity_uri, gdal.GDT_Float32,
                        rarity_nodata)

                LOGGER.debug('Finished vectorize on map_ratio')

            except KeyError:
                continue

    except KeyError:
        LOGGER.info('Baseline not provided to compute Rarity')

    LOGGER.debug('Finished habitat_quality biophysical calculations')


def resolve_ambiguous_raster_path(uri, raise_error=True):
    """Get the real uri for a raster when we don't know the extension of how
        the raster may be represented.

        uri - a python string of the file path that includes the name of the
              file but not its extension

        raise_error - a Boolean that indicates whether the function should
            raise an error if a raster file could not be opened.

        return - the resolved uri to the rasster"""

    # Turning on exceptions so that if an error occurs when trying to open a
    # file path we can catch it and handle it properly
    gdal.UseExceptions()

    # a list of possible suffixes for raster datasets. We currently can handle
    # .tif and directory paths
    possible_suffixes = ['', '.tif', '.img']

    # initialize dataset to None in the case that all paths do not exist
    dataset = None
    for suffix in possible_suffixes:
        full_uri = uri + suffix
        if not os.path.exists(full_uri):
            continue
        try:
            dataset = gdal.Open(full_uri, gdal.GA_ReadOnly)
        except:
            dataset = None

        # return as soon as a valid gdal dataset is found
        if dataset is not None:
            break

    # Turning off exceptions because there is a known bug that will hide
    # certain issues we care about later
    gdal.DontUseExceptions()

    # If a dataset comes back None, then it could not be found / opened and we
    # should fail gracefully
    if dataset is None and raise_error:
        raise Exception('There was an Error locating a threat raster in the '
        'input folder. One of the threat names in the CSV table does not match'
        ' to a threat raster in the input folder. Please check that the names '
        'correspond. The threat raster that could not be found is : %s', uri)

    if dataset is None:
        full_uri = None

    dataset = None
    return full_uri


def make_dictionary_from_csv(csv_uri, key_field):
    """Make a basic dictionary representing a CSV file, where the
       keys are a unique field from the CSV file and the values are
       a dictionary representing each row

       csv_uri - a string for the path to the csv file
       key_field - a string representing which field is to be used
                   from the csv file as the key in the dictionary

       returns - a python dictionary
    """
    out_dict = {}
    csv_file = open(csv_uri, 'rU')
    reader = csv.DictReader(csv_file)
    for row in reader:
        # create new dictionary to hold modified key values for case
        # handling
        row_upper = {}
        for key in row.keys():
            # if the key / column header begins with 'L_' we can not
            # set that to uppercase, since it will interfere with
            # matching the threat names to the sensitivity table
            if key.startswith('L_'):
                row_upper[key] = row[key]
            else:
                row_upper[key.upper()] = row[key]

	out_dict[row_upper[key_field]] = row_upper
    csv_file.close()
    return out_dict


def check_projections(ds_uri_dict, proj_unit):
    """Check that a group of gdal datasets are projected and that they are
        projected in a certain unit.

        ds_uri_dict - a dictionary of uris to gdal datasets
        proj_unit - a float that specifies what units the projection should be
            in. ex: 1.0 is meters.

        returns - False if one of the datasets is not projected or not in the
            correct projection type, otherwise returns True if datasets are
            properly projected
    """
    # a list to hold the projection types to compare later
    projections = []

    for dataset_uri in ds_uri_dict.itervalues():
        dataset = gdal.Open(dataset_uri)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(dataset.GetProjection())
        if not srs.IsProjected():
            LOGGER.debug('A Raster is Not Projected')
            return False
        if srs.GetLinearUnits() != proj_unit:
            LOGGER.debug('Proj units do not match %s:%s', \
                         proj_unit, srs.GetLinearUnits())
            return False
        projections.append(srs.GetAttrValue("PROJECTION"))

    # check that all the datasets have the same projection type
    for index in range(len(projections)):
        if projections[0] != projections[index]:
            LOGGER.debug('Projections are not the same')
            return False

    return True

def threat_names_match(threat_dict, sens_dict, prefix):
    """Check that the threat names in the threat table match the columns in the
        sensitivity table that represent the sensitivity of each threat on a
        lulc.

        threat_dict - a dictionary representing the threat table:
            {'crp':{'THREAT':'crp','MAX_DIST':'8.0','WEIGHT':'0.7'},
             'urb':{'THREAT':'urb','MAX_DIST':'5.0','WEIGHT':'0.3'},
             ... }
        sens_dict - a dictionary representing the sensitivity table:
            {'1':{'LULC':'1', 'NAME':'Residential', 'HABITAT':'1',
                  'L_crp':'0.4', 'L_urb':'0.45'...},
             '11':{'LULC':'11', 'NAME':'Urban', 'HABITAT':'1',
                   'L_crp':'0.6', 'L_urb':'0.3'...},
             ...}

        prefix - a string that specifies the prefix to the threat names that is
            found in the sensitivity table

        returns - False if there is a mismatch in threat names or True if
            everything passes"""

    # get a representation of a row from the sensitivity table where 'sens_row'
    # will be a dictionary with the column headers as the keys
    sens_row = sens_dict[sens_dict.keys()[0]]

    for threat in threat_dict:
        sens_key = prefix + threat
        if not sens_key in sens_row:
            return False
    return True


def raster_pixel_count(dataset_uri):
    """Determine how many of each unique pixel lies in the dataset (dataset)

        dataset_uri - a GDAL raster dataset

        returns -  a dictionary whose keys are the unique pixel values and
                   whose values are the number of occurrences
    """
    LOGGER.debug('Entering raster_pixel_count')
    dataset = gdal.Open(dataset_uri)
    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    counts = {}
    for row_index in range(band.YSize):
        cur_array = band.ReadAsArray(0, row_index, band.XSize, 1)
        for val in np.unique(cur_array):
            if val == nodata:
                continue
            if val in counts:
                counts[val] = \
                    float(counts[val] + cur_array[cur_array==val].size)
            else:
                counts[val] = float(cur_array[cur_array==val].size)

    LOGGER.debug('Leaving raster_pixel_count')
    return counts


def map_raster_to_dict_values(key_raster_uri, out_uri, attr_dict, field, \
        out_nodata, raise_error):
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

    pygeoprocessing.geoprocessing.reclassify_dataset_uri(
        key_raster_uri, int_attr_dict, out_uri, gdal.GDT_Float32, out_nodata,
        exception_flag=raise_error)


def make_linear_decay_kernel_uri(max_distance, kernel_uri):
    kernel_size = int(numpy.round(max_distance * 2 + 1))

    driver = gdal.GetDriverByName('GTiff')
    kernel_dataset = driver.Create(
        kernel_uri.encode('utf-8'), kernel_size, kernel_size, 1, gdal.GDT_Float32,
        options=['BIGTIFF=IF_SAFER'])

    #Make some kind of geotransform, it doesn't matter what but
    #will make GIS libraries behave better if it's all defined
    kernel_dataset.SetGeoTransform( [ 444720, 30, 0, 3751320, 0, -30 ] )
    srs = osr.SpatialReference()
    srs.SetUTM( 11, 1 )
    srs.SetWellKnownGeogCS( 'NAD27' )
    kernel_dataset.SetProjection( srs.ExportToWkt() )

    kernel_band = kernel_dataset.GetRasterBand(1)
    kernel_band.SetNoDataValue(-9999)

    col_index = numpy.array(xrange(kernel_size))
    integration = 0.0
    for row_index in xrange(kernel_size):
        distance_kernel_row = numpy.sqrt(
            (row_index - max_distance) ** 2 + (col_index - max_distance) ** 2).reshape(1,kernel_size)
        kernel = numpy.where(
            distance_kernel_row > max_distance, 0.0, (max_distance - distance_kernel_row) / max_distance)
        integration += numpy.sum(kernel)
        kernel_band.WriteArray(kernel, xoff=0, yoff=row_index)

    for row_index in xrange(kernel_size):
        kernel_row = kernel_band.ReadAsArray(
            xoff=0, yoff=row_index, win_xsize=kernel_size, win_ysize=1)
        kernel_row /= integration
        kernel_band.WriteArray(kernel_row, 0, row_index)
