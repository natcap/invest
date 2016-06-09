"""Invest overlap analysis filehandler for data passed in through UI"""

import os
import csv
import logging
import shutil
import fnmatch

import numpy
from osgeo import ogr
import pygeoprocessing.geoprocessing
from osgeo import gdal
from scipy import ndimage


LOGGER = logging.getLogger('natcap.invest.overlap_analysis.overlap_analysis')
logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')


def execute(args):
    """Overlap Analysis.

    This function will take care of preparing files passed into the overlap
    analysis model. It will handle all files/inputs associated with
    calculations and manipulations. It may write log, warning, or error
    messages to stdout.

    Parameters:
        args: A python dictionary created by the UI and passed to this method.
            It will contain the following data.
        args['workspace_dir'] (string): The directory in which to place all
            resulting files, will come in as a string. (required)
        args['zone_layer_uri'] (string): A URI pointing to a shapefile with
            the analysis zones on it. (required)
        args['grid_size'] (int): This is an int specifying how large the
            gridded squares over the shapefile should be. (required)
        args['overlap_data_dir_uri'] (string): URI pointing to a directory
            where multiple shapefiles are located. Each shapefile represents
            an activity of interest for the model. (required)
        args['do-inter'] (bool): Boolean that indicates whether or not
            inter-activity weighting is desired. This will decide if
            the overlap table will be created. (required)
        args['do_intra'] (bool): Boolean which indicates whether or not
            intra-activity weighting is desired. This will will pull
            attributes from shapefiles passed in in 'zone_layer_uri'.
            (required)
        args['do_hubs'] (bool): Boolean which indicates if human use hubs are
            desired. (required)
        args['overlap_layer_tbl'] (string): URI to a CSV file that holds
            relational data and identifier data for all layers being passed
            in within the overlap analysis directory. (optional)
        args['intra_name'] (string): string which corresponds to a field
            within the layers being passed in within overlap analysis
            directory. This is the intra-activity importance for each
            activity. (optional)
        args['hubs_uri'] (string): The location of the shapefile containing
            points for human use hub calculations. (optional)
        args['decay_amt'] (float): A double representing the decay rate of
            value from the human use hubs. (optional)

    Returns:
        ``None``"""

    workspace = args['workspace_dir']
    output_dir = os.path.join(workspace, 'output')
    intermediate_dir = os.path.join(workspace, 'intermediate')
    pygeoprocessing.geoprocessing.create_directories([output_dir, intermediate_dir])

    overlap_uris = map(
        lambda x: os.path.join(args['overlap_data_dir_uri'], x),
        os.listdir(args['overlap_data_dir_uri']))
    overlap_shape_uris = fnmatch.filter(overlap_uris, '*.shp')
    LOGGER.debug(overlap_shape_uris)

    #No need to format the table if no inter-activity weighting is desired.
    if args['do_inter']:
        args['over_layer_dict'] = format_over_table(args['overlap_layer_tbl'])
    if args['do_intra']:
        args['intra_name'] = args['intra_name']
    if args['do_hubs']:
        args['decay'] = float(args['decay_amt'])

    #Create the unweighted rasters, since that will be one of the outputs
    #regardless. However, after they are created, there will be two calls-
    #one to the combine unweighted function, and then the option call for the
    #weighted raster combination that uses the unweighted pre-created rasters.

    aoi_dataset_uri = os.path.join(intermediate_dir, 'AOI_dataset.tif')
    grid_size = float(args['grid_size'])
    pygeoprocessing.geoprocessing.create_raster_from_vector_extents_uri(
        args['zone_layer_uri'], grid_size, gdal.GDT_Int32, 0,
        aoi_dataset_uri)

    pygeoprocessing.geoprocessing.rasterize_layer_uri(
        aoi_dataset_uri, args['zone_layer_uri'], burn_values=[1])

    #Want to get each interest layer, and rasterize them, then combine them all
    #at the end. Could do a list of the filenames that we are creating within
    #the intermediate directory, so that we can access later.
    raster_uris, raster_names = make_indiv_rasters(
        intermediate_dir, overlap_shape_uris, aoi_dataset_uri)

    create_unweighted_raster(output_dir, aoi_dataset_uri, raster_uris)

    #Want to make sure we're passing the open hubs raster to the combining
    #weighted raster file
    if args['do_hubs']:
        hubs_out_uri = os.path.join(intermediate_dir, "hubs_raster.tif")
        create_hubs_raster(
            args['hubs_uri'], args['decay'], aoi_dataset_uri, hubs_out_uri)
        hubs_rast = gdal.Open(hubs_out_uri)
    else:
        hubs_rast = None
        hubs_out_uri = None

    #Need to set up dummy var for when inter or intra are available without the
    #other so that all parameters can be filled in.
    if (args['do_inter'] or args['do_intra'] or args['do_hubs']):

        layer_dict = args['over_layer_dict'] if args['do_inter'] else None
        intra_name = args['intra_name'] if args['do_intra'] else None

        #Want some place to put weighted rasters so we aren't blasting over the
        #unweighted rasters
        weighted_dir = os.path.join(intermediate_dir, 'Weighted')

        if not (os.path.exists(weighted_dir)):
            os.makedirs(weighted_dir)

        #Now we want to create a second raster that includes all of the
        #weighting information
        create_weighted_raster(
            output_dir, weighted_dir, aoi_dataset_uri,
            layer_dict, overlap_shape_uris,
            intra_name, args['do_inter'],
            args['do_intra'], args['do_hubs'],
            hubs_out_uri, raster_uris, raster_names)


def format_over_table(over_tbl):
    '''
    This CSV file contains a string which can be used to uniquely identify a
    .shp file to which the values in that string's row will correspond. This
    string, therefore, should be used as the key for the ovlap_analysis
    dictionary, so that we can get all corresponding values for a shapefile at
    once by knowing its name.

        Input:
            over_tbl- A CSV that contains a list of each interest shapefile,
                and the inter activity weights corresponding to those layers.

        Returns:
            over_dict- The analysis layer dictionary that maps the unique name
                of each layer to the optional parameter of inter-activity
                weight. For each entry, the key will be the string name of the
                layer that it represents, and the value will be the
                inter-activity weight for that layer.
    '''
    over_layer_file = open(over_tbl)
    reader = csv.DictReader(over_layer_file)

    over_dict = {}

    #USING EXPLICIT STRING CALLS to the layers table (these should not be unique
    #to the type of table, but rather, are items that ALL layers tables should
    #contain). I am casting both of the optional values to floats, since both
    #will be used for later calculations.
    for row in reader:
        LOGGER.debug(row)

        #Setting the default values for inter-activity weight and buffer, since
        #they are not actually required to be filled in.

        #NEED TO FIGURE OUT IF THESE SHOULD BE 0 OR 1
        inter_act = 1

        for key in row:
            if 'Inter-Activity' in key and row[key] != '':
                inter_act = float(row[key])

            name = row['LIST OF HUMAN USES']

        over_dict[name] = inter_act

    return over_dict


def create_hubs_raster(hubs_shape_uri, decay, aoi_raster_uri, hubs_out_uri):
    '''
    This will create a rasterized version of the hubs shapefile where each
    pixel on the raster will be set accourding to the decay function from the
    point values themselves. We will rasterize the shapefile so that all land
    is 0, and nodata is the distance from the closest point.

        Input:
            hubs_shape_uri - Open point shapefile containing the hub locations
                as points.
            decay - Double representing the rate at which the hub importance
                depreciates relative to the distance from the location.
            aoi_raster_uri - The URI to the area interest raster on which we
                want to base our new hubs raster.
            hubs_out_uri - The URI location at which the new hubs raster should
                be placed.

        Output:
            This creates a raster within hubs_out_uri whose data will be a
            function of the decay around points provided from hubs shape.

        Returns nothing. '''

    #In this case, want to change the nodata value to 1, and the points
    #themselves to 0, since this is what the distance tranform function expects.
    nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(aoi_raster_uri)
    pygeoprocessing.geoprocessing.new_raster_from_base_uri(
        aoi_raster_uri, hubs_out_uri, 'GTiff', -1, gdal.GDT_Float32,
        fill_value=1)

    pygeoprocessing.geoprocessing.rasterize_layer_uri(
        hubs_out_uri, hubs_shape_uri, burn_values=[0])

    dataset = gdal.Open(hubs_out_uri, gdal.GA_Update)
    band = dataset.GetRasterBand(1)
    matrix = band.ReadAsArray()
    cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(aoi_raster_uri)
    decay_matrix = numpy.exp(
        -decay * ndimage.distance_transform_edt(matrix, sampling=cell_size))
    band.WriteArray(decay_matrix)


def create_unweighted_raster(output_dir, aoi_raster_uri, raster_files_uri):
    '''This will create the set of unweighted rasters- both the AOI and
    individual rasterizations of the activity layers. These will all be
    combined to output a final raster displaying unweighted activity frequency
    within the area of interest.

    Input:
        output_dir- This is the directory in which the final frequency raster
            will be placed. That file will be named 'hu_freq.tif'.
        aoi_raster_uri- The uri to the rasterized version of the AOI file
            passed in with args['zone_layer_file']. We will use this within
            the combination function to determine where to place nodata values.
        raster_files_uri - The uris to the rasterized version of the files
            passed in through args['over_layer_dict']. Each raster file shows
            the presence or absence of the activity that it represents.
    Output:
        A raster file named ['workspace_dir']/output/hu_freq.tif. This depicts
        the unweighted frequency of activity within a gridded area or
        management zone.

    Returns nothing.
    '''

    aoi_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(aoi_raster_uri)
    aoi_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(aoi_raster_uri)

    #When we go to actually burn, should have a "0" where there is AOI, not
    #same as nodata. Need the 0 for later combination function.
    activities_uri = os.path.join(output_dir, 'hu_freq.tif')

    def get_raster_sum(*activity_pixels):
        '''
        For any given pixel, if the AOI covers the pixel, we want to ignore
        nodata value activities, and sum all other activities happening on that
        pixel.

        Input:
            *activity_pixels- This expands into a dynamic list of single
                variables. The first will always be the AOI pixels. Those
                following will be a pixel from the overlap rasters that we are
                looking to combine.

        Returns:
            sum_pixel- This is either the aoi_nodata value if the AOI is not
                turned on in that area, or, if the AOI does cover this pixel,
                this is the sum of all activities that are taking place in that
                area.
        '''
        #We have pre-decided that nodata for the activity pixel will produce a
        #different result from the "no activities within that AOI area" result
        #of 0.

        aoi_pixel_vector = activity_pixels[0]
        aoi_nodata_mask = aoi_pixel_vector == aoi_nodata

        sum_pixel = numpy.zeros(aoi_pixel_vector.shape)

        for activ in activity_pixels[1::]:
            sum_pixel[activ == 1] += 1

        return numpy.where(aoi_nodata_mask, aoi_nodata, sum_pixel)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        raster_files_uri, get_raster_sum, activities_uri, gdal.GDT_Int32,
        aoi_nodata, aoi_pixel_size, "intersection", vectorize_op=False)


def create_weighted_raster(
    out_dir, intermediate_dir, aoi_raster_uri, inter_weights_dict, layers_dict,
    intra_name, do_inter, do_intra, do_hubs, hubs_raster_uri,
        raster_uris, raster_names):
    '''This function will create an output raster that takes into account both
    inter-activity weighting and intra-activity weighting. This will produce a
    map that looks both at where activities are occurring, and how much people
    value those activities and areas.

    Input:
        out_dir- This is the directory into which our completed raster file
            should be placed when completed.
        intermediate_dir- The directory in which the weighted raster files can
            be stored.
        inter_weights_dict- The dictionary that holds the mappings from layer
            names to the inter-activity weights passed in by CSV. The
            dictionary key is the string name of each shapefile, minus the .shp
            extension. This ID maps to a double representing ther
            inter-activity weight of each activity layer.
       layers_dict- This dictionary contains all the activity layers that are
           included in the particular model run. This maps the name of the
           shapefile (excluding the .shp extension) to the open datasource
           itself.
        intra_name- A string which represents the desired field name in our
            shapefiles. This field should contain the intra-activity weight for
            that particular shape.
        do_inter- A boolean that indicates whether inter-activity weighting is
            desired.
        do_intra- A boolean that indicates whether intra-activity weighting is
            desired.
        aoi_raster_uri - The uri to the dataset for our Area Of Interest.
            This will be the base map for all following datasets.
        raster_uris - A list of uris to the open unweighted raster files
            created by make_indiv_rasters that begins with the AOI raster. This
            will be used when intra-activity weighting is not desired.
        raster_names- A list of file names that goes along with the unweighted
            raster files. These strings can be used as keys to the other
            ID-based dictionaries, and will be in the same order as the
            'raster_files' list.
    Output:
        weighted_raster- A raster file output that takes into account both
            inter-activity weights and intra-activity weights.

    Returns nothing.
    '''
    ''' The equation that we are given to work with is:
            IS = (1/n) * SUM (U{i,j}*I{j}
        Where:
            IS = Importance Score
            n = Number of human use activities included
            U{i,j}:
                If do_intra:
                    U{i,j} = X{i,j} / X{max}
                        X {i,j} = intra-activity weight of activity j in
                            grid cell i
                        X{max} = The max potential intra-activity weight for all
                            cells where activity j occurs.
                Else:
                    U{i,j} = 1 if activity exists, or 0 if it doesn't.
            I{j}:
                If do_inter:
                    I{j} = Y{j} / Y{max}
                        Y{j} = inter-activity weight of an activity
                        Y{max} = max inter-activity weight of an activity weight
                            for all activities.
                Else:
                    I{j} = 1'''

    #Want to set up vars that will be universal across all pixels first.
    #n should NOT include the AOI, since it is not an interest layer
    n = len(layers_dict)
    outgoing_uri = os.path.join(out_dir, 'hu_impscore.tif')
    aoi_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(aoi_raster_uri)
    pixel_size_out = pygeoprocessing.geoprocessing.get_cell_size_from_uri(aoi_raster_uri)

    #If intra-activity weighting is desired, we need to create a whole new set
    #of values, where the burn value of each pixel is the attribute value of the
    #polygon that it resides within. This means that we need the AOI raster, and
    #need to rebuild based on that, then move on from there. I'm abstracting
    #this to a different file for ease of reading. It will return a tuple of two
    #lists- the first will be the list of rasterized aoi/layers, and the second
    #will be a list of the original file names in the same order as the layers
    #so that the dictionaries with other weights can be cross referenced.
    if do_intra:
        weighted_raster_uris, weighted_raster_names = (
            make_indiv_weight_rasters(
                intermediate_dir, aoi_raster_uri, layers_dict, intra_name))

    #Need to get the X{max} now, so iterate through the features on a layer, and
    #make a dictionary that maps the name of the layer to the max potential
    #intra-activity weight
    if do_intra:
        max_intra_weights = {}
        for layer_uri in layers_dict:
            layer_name = os.path.splitext(os.path.basename(layer_uri))[0]
            datasource = ogr.Open(layer_uri)
            layer = datasource.GetLayer()
            for feature in layer:
                attribute = feature.items()[intra_name]
                try:
                    max_intra_weights[layer_name] = \
                        max(attribute, max_intra_weights[layer_name])
                except KeyError:
                    max_intra_weights[layer_name] = attribute

    #We also need to know the maximum of the inter-activity value weights, but
    #only if inter-activity weighting is desired at all. If it is not, we don't
    #need this value, so we can just set it to a None type.
    max_inter_weight = None
    if do_inter:
        max_inter_weight = max(inter_weights_dict.values())

    #Assuming that inter-activity valuation is desired, whereas intra-activity
    #is not, we should use the original rasterized layers as the pixels to
    #combine. If, on the other hand, inter is not wanted, then we should just
    #use 1 in our equation.

    def combine_weighted_pixels(*pixel_parameter_list):
        aoi_pixel_vector = pixel_parameter_list[0]
        curr_pix_sum_vector = numpy.zeros(aoi_pixel_vector.shape)
        #curr_pix_sum = 0
        aoi_nodata_mask = aoi_pixel_vector == aoi_nodata
        #if aoi_pixel == aoi_nodata:
        #    return aoi_nodata
        for i in range(1, n+1):
            #This will either be a 0 or 1, since the burn value for the
            #unweighted raster files was a 1.
            U_vector = pixel_parameter_list[i]
            #U = pixel_parameter_list[i]
            I = None
            if do_inter:
                layer_name = raster_names[i]
                Y = inter_weights_dict[layer_name]
                I = Y / max_inter_weight
            else:
                I = 1

            #This is coming from the documentation, refer to additional info in
            #the docstring. n gets cast to a float so that it can be used
            #in division.
            curr_pix_sum_vector += ((1/float(n)) * U_vector * I)
        return numpy.where(aoi_nodata_mask, aoi_nodata, curr_pix_sum_vector)

    def combine_weighted_pixels_intra(*pixel_parameter_list):
        aoi_pixel_vector = pixel_parameter_list[0]
        curr_pix_sum_vector = numpy.zeros(aoi_pixel_vector.shape)
        aoi_nodata_mask = aoi_pixel_vector == aoi_nodata
        #if aoi_pixel == aoi_nodata:
        #    return aoi_nodata
        for i in range(1, n+1):

            #Can assume that if we have gotten here, that intra-activity
            #weighting is desired. Compute U for that weighting, assuming the
            #raster pixels are the intra weights.
            layer_name = weighted_raster_names[i]
            X_vector = pixel_parameter_list[i]
            X_max = max_intra_weights[layer_name]

            U_vector = X_vector / X_max
            I = None

            if do_inter:
                layer_name = raster_names[i]
                Y = inter_weights_dict[layer_name]
                I = Y / max_inter_weight
            else:
                I = 1

            #This is coming from the documentation, refer to additional info in
            #the docstring.
            #n is getting cast to a float so that we can use non-integer
            #division in the calculations.
            curr_pix_sum_vector += ((1/float(n)) * U_vector * I)
        return numpy.where(aoi_nodata_mask, aoi_nodata, curr_pix_sum_vector)

    if do_intra:
        pygeoprocessing.geoprocessing.vectorize_datasets(
            weighted_raster_uris, combine_weighted_pixels_intra, outgoing_uri,
            gdal.GDT_Float32, aoi_nodata, pixel_size_out, "intersection",
            dataset_to_align_index=0, vectorize_op=False)
    else:
        pygeoprocessing.geoprocessing.vectorize_datasets(
            raster_uris, combine_weighted_pixels, outgoing_uri,
            gdal.GDT_Float32, aoi_nodata, pixel_size_out, "intersection",
            dataset_to_align_index=0, vectorize_op=False)

    #Now want to check if hu_impscore exists. If it does, use that as the
    #multiplier against the hubs raster. If not, use the hu_freq raster and
    #multiply against that.
    def combine_hubs_raster(*pixel_list):

        #We know that we are only ever multiplying these two, and that these
        #will be the only two in the list of pixels.
        hubs_layer = pixel_list[0]
        base_layer = pixel_list[1]

        return hubs_layer * base_layer

    if do_hubs:
        #This is where the weighted raster file exists (if do_inter or do_intra)
        if os.path.isfile(outgoing_uri):
            #Make a copy of the file so that we can use it to re-create the hub
            #weighted raster file.
            temp_uri = os.path.join(intermediate_dir, "temp_rast.tif")
            shutil.copyfile(outgoing_uri, temp_uri)

            base_raster_uri = temp_uri

        #Otherwise, if we don't have a weighted raster file, use the unweighted
        #frequency file.
        else:
            base_raster_uri = os.path.join(out_dir, "hu_freq.tif")
            temp_uri = None

        #h_rast_list = [hubs_raster, base_raster]
        h_rast_uri_list = [hubs_raster_uri, base_raster_uri]

        LOGGER.debug("this is the list %s" % h_rast_uri_list)
        pygeoprocessing.geoprocessing.vectorize_datasets(
            h_rast_uri_list, combine_hubs_raster, outgoing_uri,
            gdal.GDT_Float32, aoi_nodata, pixel_size_out, "intersection",
            vectorize_op=False)

        try:
            os.remove(temp_uri)
        except OSError as e:
            LOGGER.warn("in create_weighted_raster %s on file %s" % (
                e, temp_uri))


def make_indiv_weight_rasters(
        input_dir, aoi_raster_uri, layers_dict, intra_name):
    '''
    This is a helper function for create_weighted_raster, which abstracts
    some of the work for getting the intra-activity weights per pixel to a
    separate function. This function will take in a list of the activities
    layers, and using the aoi_raster as a base for the tranformation, will
    rasterize the shapefile layers into rasters where the burn value is based
    on a per-pixel intra-activity weight (specified in each polygon on the
    layer). This function will return a tuple of two lists- the first is a list
    of the rasterized shapefiles, starting with the aoi. The second is a list
    of the shapefile names (minus the extension) in the same order as they were
    added to the first list. This will be used to reference the dictionaries
    containing the rest of the weighting information for the final weighted
    raster calculation.

    Input:
        input_dir: The directory into which the weighted rasters should be
            placed.
        aoi_raster_uri: The uri to the rasterized version of the area of
            interest. This will be used as a basis for all following
            rasterizations.
        layers_dict: A dictionary of all shapefiles to be rasterized. The key
            is the name of the original file, minus the file extension. The
            value is an open shapefile datasource.
        intra_name: The string corresponding to the value we wish to pull out
            of the shapefile layer. This is an attribute of all polygons
            corresponding to the intra-activity weight of a given shape.

    Returns:
        weighted_raster_files: A list of raster versions of the original
            activity shapefiles. The first file will ALWAYS be the AOI,
            followed by the rasterized layers.
        weighted_names: A list of the filenames minus extensions, of the
            rasterized files in weighted_raster_files. These can be used to
            reference properties of the raster files that are located in other
            dictionaries.
    '''

    #aoi_raster has to be the first so that we can easily pull it out later when
    #we go to combine them. Will need the aoi_nodata for later as well.
    weighted_raster_uris = [aoi_raster_uri]
    #Inserting 'aoi' as a placeholder so that when I go through the list, I can
    #reference other indicies without having to convert for the missing first
    #element in names.
    weighted_names = ['aoi']
    LOGGER.debug('layers_dict %s', layers_dict)
    for layer_uri in layers_dict:
        basename = os.path.splitext(os.path.basename(layer_uri))[0]

        outgoing_uri = os.path.join(input_dir, basename + ".tif")

        #Setting nodata value to 0 so that the nodata pixels can be used
        #directly in calculations without messing up the weighted total
        #equations for the second output file.
        nodata = 0
        pygeoprocessing.geoprocessing.new_raster_from_base_uri(
            aoi_raster_uri,
            outgoing_uri,
            'GTiff',
            nodata,
            gdal.GDT_Float32,
            fill_value=nodata)

        pygeoprocessing.geoprocessing.rasterize_layer_uri(
            outgoing_uri,
            layer_uri,
            option_list=["ATTRIBUTE=%s" % intra_name])

        weighted_raster_uris.append(outgoing_uri)
        weighted_names.append(basename)

    return weighted_raster_uris, weighted_names


def make_indiv_rasters(out_dir, overlap_shape_uris, aoi_raster_uri):
    '''This will pluck each of the files out of the dictionary and create a new
    raster file out of them. The new file will be named the same as the
    original shapefile, but with a .tif extension, and will be placed in the
    intermediate directory that is being passed in as a parameter.

    Input:
        out_dir- This is the directory into which our completed raster files
            should be placed when completed.
        overlap_shape_uris- This is a dictionary containing all of the open
            shapefiles which need to be rasterized. The key for this dictionary
            is the name of the file itself, minus the .shp extension. This key
            maps to the open shapefile of that name.
        aoi_raster_uri- The dataset for our AOI. This will be the base map for
            all following datasets.

    Returns:
        raster_files- This is a list of the datasets that we want to sum. The
            first will ALWAYS be the AOI dataset, and the rest will be the
            variable number of other datasets that we want to sum.
        raster_names- This is a list of layer names that corresponds to the
            files in 'raster_files'. The first layer is guaranteed to be the
            AOI, but all names after that will be in the same order as the
            files so that it can be used for indexing later.
    '''
    #aoi_raster has to be the first so that we can use it as an easy "weed out"
    #for pixel summary later
    raster_uris = [aoi_raster_uri]
    raster_names = ['aoi']

    #Remember, this defaults to element being the keys of the dictionary
    for overlap_uri in overlap_shape_uris:
        element_name = os.path.splitext(
            os.path.basename(overlap_uri))[0]
        outgoing_uri = os.path.join(
            out_dir, element_name + ".tif")
        nodata = 0
        pygeoprocessing.geoprocessing.new_raster_from_base_uri(
            aoi_raster_uri,
            outgoing_uri,
            'GTiff',
            nodata,
            gdal.GDT_Int32,
            fill_value=nodata)

        LOGGER.debug('rasterizing %s to %s' % (overlap_uri, outgoing_uri))
        pygeoprocessing.geoprocessing.rasterize_layer_uri(
            outgoing_uri, overlap_uri, burn_values=[1],
            option_list=['ALL_TOUCHED=TRUE'])

        raster_uris.append(outgoing_uri)
        raster_names.append(element_name)

    LOGGER.debug("Just made the following URIs %s" % str(raster_uris))
    return raster_uris, raster_names
