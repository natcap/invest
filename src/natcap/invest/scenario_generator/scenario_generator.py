"""Scenario Generator Module."""

import os
import math
import struct
import logging
from decimal import Decimal

import numpy as np
import scipy as sp
from osgeo import gdal, ogr

import pygeoprocessing.geoprocessing as geoprocess
from .. import utils as invest_utils

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger(
    'natcap.invest.scenario_generator.scenario_generator')

shapeTypes = {0: "Null Shape", 1: "Point", 3: "PolyLine", 5: "Polygon",
              8: "MultiPoint", 11: "PointZ", 13: "PolyLineZ",
              15: "PolygonZ", 18: "MultiPointZ", 21: "PointM",
              23: "PolyLineM", 25: "PolygonM", 28: "MultiPointM",
              31: "MultiPatch"}

_OUTPUT = {
    'landcover_transition_uri': 'transitioned.tif',
    'override_dataset_uri': 'override.tif',
    'landcover_htm_uri': 'scenario-output-summary.html'
}

_INTERMEDIATE = {
    'scenario_uri': 'scenario.tif',
    'transition_name': "transition_%i.tif",
    'suitability_name': '%s_%s.tif',
    'normalized_name': '%s_%s_norm.tif',
    'combined_name': 'factors_%s.tif',
    'constraints_name': 'constraints.tif',
    'filter_name': 'filter_%i.tif',
    'factors_name': 'suitability_%s.tif',
    'cover_name': 'cover_%i.tif',
    'proximity_name': 'proximity_%s.tif',
    'normalized_proximity_name': 'proximity_norm_%s.tif',
    'adjusted_suitability_name': 'adjusted_suitability_%s.tif'
}


def calculate_weights(array, rounding=4):
    """Create list of priority weights by land-cover class.

    Args:
        array (np.array): input array
        rounding (int): number of decimal places to include

    Returns:
        weights_list (list): list of priority weights
    """
    LOGGER.info("Creating priority weights list...")

    decimal_places = Decimal(10) ** -(rounding)
    eigenvalues, eigenvectors = sp.linalg.eig(array)

    # get primary eigenvalue and vector
    max_eigenvalue = max(eigenvalues)
    max_eigenvalue_idx = eigenvalues.tolist().index(max_eigenvalue)
    eigenvector = eigenvectors.take((max_eigenvalue_idx,), axis=1)

    # priority vector = normalized primary eigenvector
    normalized_eigenvector = eigenvector / sum(eigenvector)

    # turn into list of real part values
    vector = [abs(e[0]) for e in normalized_eigenvector]

    # return nice rounded Decimal values with labels
    return [Decimal(str(v)).quantize(decimal_places) for v in vector]


def calculate_priority(priority_table_uri):
    """Create dictionary mapping each land-cover class to their priority weight.

    Args:
        priority_table_uri (str): path to priority csv table

    Returns:
        priority_dict (dict): land-cover and weights_matrix
    """
    LOGGER.info("Creating land-cover priority weights dictionary...")

    # parse priority table
    table = [line.strip().split(",") for line in open(
        priority_table_uri).readlines()]
    id_index = table[0].index("Id")

    cover_id_list = [row[id_index] for row in table]
    cover_id_list.pop(0)
    cover_id_index_list = \
        [table[0].index(cover_id) for cover_id in cover_id_list]
    cover_id_list = [int(cover_id) for cover_id in cover_id_list]

    # initialize priority matrix
    matrix = np.zeros((len(cover_id_list), len(cover_id_list)))
    for row in range(len(cover_id_list)):
        for col in range(row + 1):
            matrix[row][col] = float(table[row+1][cover_id_index_list[col]])
            matrix[col][row] = 1 / matrix[row][col]

    return dict(zip(cover_id_list, calculate_weights(matrix, 4)))


def calculate_distance_raster_uri(dataset_in_uri, dataset_out_uri):
    """Calculate distance to non-zero cell for all input zero-value cells.

    Args:
        dataset_in_uri (str): the input mask raster. Distances calculated from
            the non-zero cells in raster.
        dataset_out_uri (str): the output raster where all zero values are
            equal to the euclidean distance of the closest non-zero pixel.
    """
    LOGGER.info("Creating euclidean distance raster...")

    # Compute pixel distance
    geoprocess.distance_transform_edt(dataset_in_uri, dataset_out_uri)

    # Convert to meters
    def pixel_to_meters_op(x):
        x[x != nodata] *= cell_size
        return x

    cell_size = geoprocess.get_cell_size_from_uri(dataset_in_uri)
    nodata = geoprocess.get_nodata_from_uri(dataset_out_uri)
    tmp = geoprocess.temporary_filename()
    geoprocess.vectorize_datasets(
        [dataset_out_uri],
        pixel_to_meters_op,
        tmp,
        gdal.GDT_Float64,
        nodata,
        cell_size,
        'union',
        vectorize_op=False)

    def identity_op(x):
        return x

    geoprocess.vectorize_datasets(
        [tmp],
        identity_op,
        dataset_out_uri,
        gdal.GDT_Float64,
        nodata,
        cell_size,
        'union',
        vectorize_op=False)

    # Compute raster stats so the raster is viewable in QGIS and Arc
    geoprocess.calculate_raster_stats_uri(dataset_out_uri)


def get_geometry_type_from_uri(datasource_uri):
    """Get geometry type from a shapefile.

    Args:
        datasource_uri (str): path to shapefile

    Returns:
        shape_type (int): OGR geometry type
    """
    datasource = open(datasource_uri, 'r')
    datasource.seek(32)
    shape_type, = struct.unpack('<i', datasource.read(4))
    datasource.close()
    return shape_type


def get_transition_pairs_count_from_uri(dataset_uri_list):
    """Find transition summary statistics between lulc rasters.

    Args:
        dataset_uri_list (list): list of paths to rasters

    Returns:
        unique_raster_values_count (dict): cell type with each raster value
        transitions (dict): count of cells
    """
    LOGGER.info("Finding transition summary statistics between original"
                " and scenario land-cover rasters...")

    cell_size = geoprocess.get_cell_size_from_uri(dataset_uri_list[0])
    lulc_nodata = int(geoprocess.get_nodata_from_uri(dataset_uri_list[0]))
    nodata = 0

    # reclass rasters to compact bit space
    lulc_codes = set()
    unique_raster_values_count = {}

    for dataset_uri in dataset_uri_list:
        unique_raster_values_count[dataset_uri] = \
            geoprocess.unique_raster_values_count(dataset_uri)
        lulc_codes.update(unique_raster_values_count[dataset_uri].keys())

    lulc_codes = list(lulc_codes)
    lulc_codes.sort()

    if len(lulc_codes) < 2 ** 8:
        data_type = gdal.GDT_UInt16
        shift = 8
    elif len(lulc_codes) < 2 ** 16:
        data_type = gdal.GDT_UInt32
        shift = 16
    else:
        raise ValueError("Too many LULC codes.")

    # renumber and reclass rasters
    reclass_orig_dict = dict(zip(lulc_codes, range(1, len(lulc_codes) + 1)))

    reclass_dest_dict = {}
    for key in reclass_orig_dict:
        reclass_dest_dict[key] = reclass_orig_dict[key] << shift

    def add_op(orig, dest):
        return orig + dest

    counts = {}
    for i in range(len(dataset_uri_list) - 1):
        orig_uri = geoprocess.temporary_filename()
        dest_uri = geoprocess.temporary_filename()
        multi_uri = geoprocess.temporary_filename()

        # reclass orig values
        geoprocess.reclassify_dataset_uri(
            dataset_uri_list[i],
            reclass_orig_dict,
            orig_uri,
            data_type,
            nodata,
            exception_flag="values_required")

        # reclass dest values
        geoprocess.reclassify_dataset_uri(
            dataset_uri_list[i+1],
            reclass_dest_dict,
            dest_uri,
            data_type,
            nodata,
            exception_flag="values_required")

        # multiplex orig with dest
        geoprocess.vectorize_datasets(
            [orig_uri, dest_uri],
            add_op,
            multi_uri,
            data_type,
            nodata,
            cell_size,
            "union")

        # get unique counts
        counts[i] = geoprocess.unique_raster_values_count(multi_uri, False)

    restore_classes = {}
    for key in reclass_orig_dict:
        restore_classes[reclass_orig_dict[key]] = key
    restore_classes[nodata] = lulc_nodata

    LOGGER.debug("Decoding transition table.")
    transitions = {}
    for key in counts:
        transitions[key] = {}
        for k in counts[key]:
            try:
                orig = restore_classes[k % (2 ** shift)]
            except KeyError:
                orig = lulc_nodata
            try:
                dest = restore_classes[k >> shift]
            except KeyError:
                dest = lulc_nodata
            try:
                transitions[key][orig][dest] = counts[key][k]
            except KeyError:
                transitions[key][orig] = {dest: counts[key][k]}

    return unique_raster_values_count, transitions


def generate_chart_html(cover_dict, cover_names_dict, workspace_dir):
    """Create HTML page showing statistics about land-cover change.

        - Initial land-cover cell count
        - Scenario land-cover cell count
        - Land-cover percent change
        - Land-cover percent total: initial, final, change
        - Transition matrix
        - Unconverted pixels list

    Args:
        cover_dict (dict): land cover {'cover_id': [before, after]}
        cover_names_dict (dict): land cover names {'cover_id': 'cover_name'}
        workspace_dir (str): path to workspace directory

    Returns:
        chart_html (str): html chart
    """
    LOGGER.info("Creating HTML page...")

    html = "\n<table BORDER=1>"
    html += "\n<TR><td>Id</td><td>% Before</td><td>% After</td></TR>"
    cover_id_list = cover_dict.keys()
    cover_id_list.sort()

    cover_id_list_chart = cover_names_dict.keys()
    cover_id_list_chart.sort()

    pixcount = 0
    for cover_id in cover_id_list:
        pixcount += cover_dict[cover_id][0]
    pixcount = float(pixcount)

    for cover_id in cover_id_list:
        html += "\n<TR><td>%i</td><td>%i</td><td>%i</td></TR>" % (
             cover_id,
             (cover_dict[cover_id][0] / pixcount) * 100,
             (cover_dict[cover_id][1] / pixcount) * 100)
    html += "\n<table>"

    # create three charts for original, final and change
    thecharts = [
        ['Original', 0],
        ['Final', 1],
        ['Change', 2]]

    hainitial = ""
    hainitialnegative = ""
    hainitiallist = []
    hafinal = ""
    hafinalnegative = ""
    hafinallist = []
    hachange = ""
    hachangelist = []
    haall = []
    initialcover = []
    finalcover = []

    for cover_id in cover_id_list_chart:
        try:
            initialcover.append((cover_dict[cover_id][0] / pixcount) * 100)
        except KeyError:
            initialcover.append(0)
        try:
            finalcover.append((cover_dict[cover_id][1] / pixcount) * 100)
        except KeyError:
            finalcover.append(0)

    # return html
    html += "<style type='text/css'>"
    html += "body {font-family: Arial, Helvetica, "\
            "sans-serif; font-size: 0.9em;}"
    html += "table#results {margin: 20px auto}"
    html += "table#results th {text-align: left}"
    html += "</style>"
    html += "<script type='text/javascript'>\n"
    html += "var chart,\n"

    categories = []
    html += "categories = ["
    for cover_id in cover_id_list_chart:
        # pass
        categories.append("'"+cover_names_dict[cover_id]+"'")
    html += ",".join(categories)
    html += "]\n"

    html += "$(document).ready(function() {\n"

    for x in initialcover:
        hainitial = hainitial + str(x)+","
        hainitialnegative = hainitialnegative + "0,"
        hainitiallist.append(float(x))
    temp = []
    temp.append(hainitial)
    temp.append(hainitialnegative)
    haall.append(temp)

    thecharts[0].append(max(hainitiallist))
    thecharts[0].append(min(hainitiallist))

    for x in finalcover:
        hafinal = hafinal + str(x) + ","
        hafinalnegative = hafinalnegative + "0,"
        hafinallist.append(float(x))
    temp = []
    temp.append(hafinal)
    temp.append(hafinalnegative)
    haall.append(temp)

    thecharts[1].append(max(hafinallist))
    thecharts[1].append(min(hafinallist))

    for x in range(len(initialcover)):
        hachange = hachange + str(float(finalcover[x]) -
                                  float(initialcover[x])) + ","
        hachangelist.append(float(finalcover[x]) - float(initialcover[x]))

    # split the change values
    hachangelistnegative = ""
    hachangelistpositive = ""
    for item in hachangelist:
        if item < 0:
            hachangelistnegative = hachangelistnegative+str(item) + ","
            hachangelistpositive = hachangelistpositive + "0,"
        else:
            hachangelistpositive = hachangelistpositive + str(item) + ","
            hachangelistnegative = hachangelistnegative + "0,"

    temp = []

    temp.append(hachangelistpositive)
    temp.append(hachangelistnegative)
    haall.append(temp)

    thecharts[2].append(max(hachangelist))
    thecharts[2].append(min(hachangelist))

    if thecharts[0][2] > thecharts[1][2]:
        thecharts[1][2] = thecharts[0][2]
        thecharts[2][2] = thecharts[0][2]
    else:
        thecharts[0][2] = thecharts[1][2]
        thecharts[2][2] = thecharts[1][2]

    for x in thecharts:
        if x[0] == 'Change':
            themin = x[3]
        else:
            themin = 0
        html += "chart = new Highcharts.Chart({\n"
        html += "chart: {renderTo: '" + x[0] + \
                "container',defaultSeriesType: 'bar'},"
        html += "title: {text: '"+x[0]+" Landcover'},"
        html += "subtitle: {text: ''},"
        html += "xAxis: [{categories: categories,reversed: false}, {opposite:"\
                " true, reversed: false,categories: categories,linkedTo: 0}],"
        html += "yAxis: {title: {text: null},labels: {formatter: function(){"\
                "return Math.abs(this.value)}},min: " + str(themin) + ",max: "\
                + str(x[2]) + "},"
        html += "plotOptions: {series: { stacking: 'normal', showInLegend:"\
                " false } },"
        html += "tooltip: { formatter: function(){return '<b>'+"\
                " this.point.category +'</b><br/>'+'Area: '+ "\
                "Highcharts.numberFormat(Math.abs(this.point.y), 0)+'%';}},"
        html += "series: [{name: '',"
        html += "data: ["+haall[x[1]][0]+"]}, {"
        html += "name: '',"
        html += "data: ["+haall[x[1]][1]+"]}]});\n"
    html += "});\n"
    html += "</script>\n"

    for x in thecharts:
        html += "<div id='"+x[0]+"container' style='width: 800px; height: "\
                "400px; margin: 20px 0'></div>\n"
    return html


def filter_fragments(input_uri, size, output_uri):
    """Filter fragments.

    Args:
        input_uri (str): path to input raster
        size (float): patch (/fragments?) size threshold
        output_uri (str): path to output raster
    """
    LOGGER.debug(
        "Filtering patches smaller than %i from %s...", size, input_uri)

    # clump and sieve
    src_ds = gdal.Open(input_uri)
    src_band = src_ds.GetRasterBand(1)
    src_array = src_band.ReadAsArray()

    driver = gdal.GetDriverByName("GTiff")
    driver.CreateCopy(output_uri, src_ds, 0)

    dst_ds = gdal.Open(output_uri, 1)
    dst_band = dst_ds.GetRasterBand(1)
    dst_array = np.copy(src_array)

    suitability_values = np.unique(src_array)
    if suitability_values[0] == 0:
        suitability_values = suitability_values[1:]

    # 8 connectedness preferred, 4 connectedness allowed
    eight_connectedness = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    four_connectedness = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    suitability_values_count = suitability_values.size
    for v in range(1, suitability_values_count):
        LOGGER.debug('Processing suitability value ' + str(
            suitability_values.size - v))
        value = suitability_values[v]
        # Pixels of interest set to 1, 0 otherwise
        mask = src_array == value
        # Number of pixels to process
        ones_in_mask = np.sum(mask)
        # Label and count disconnected components (fragments)
        label_im, nb_labels = sp.ndimage.label(mask, four_connectedness)
        # Compute fragment sizes
        fragment_sizes = sp.ndimage.sum(mask, label_im, range(nb_labels + 1))
        # List fragments
        fragment_labels = np.array(range(nb_labels + 1))
        # Discard large fragments
        small_fragment_mask = np.where(fragment_sizes <= size)
        # Gather small fragment information
        small_fragment_sizes = fragment_sizes[small_fragment_mask]
        small_fragment_labels = fragment_labels[small_fragment_mask]
        combined_small_fragment_size = np.sum(small_fragment_sizes)
        # Find each fragment
        fragments_location = sp.ndimage.find_objects(label_im, nb_labels)
        removed_pixels = 0
        small_fragment_labels_count = small_fragment_labels.size
        for l in range(small_fragment_labels_count-1):
            label = small_fragment_labels[l+1]
            last_label = small_fragment_labels[l]
            size = small_fragment_sizes[l+1]
            source = label_im[fragments_location[last_label]]
            target = dst_array[fragments_location[last_label]]
            pixels_to_remove = np.where(source == label)
            target[pixels_to_remove] = 0

    dst_band.WriteArray(dst_array)


def execute(args):
    """Scenario Generator: Rule-Based.

    Model entry-point.

    Args:
        workspace_dir (str): path to workspace directory
        suffix (str): string to append to output files
        landcover (str): path to land-cover raster
        transition (str): path to land-cover attributes table

        calculate_priorities (bool): whether to calculate priorities
        priorities_csv_uri (str): path to priority csv table

        calculate_proximity (bool): whether to calculate proximity
        proximity_weight (float): weight given to proximity
        calculate_transition (bool): whether to specifiy transitions

        calculate_factors (bool): whether to use suitability factors
        suitability_folder (str): path to suitability folder
        suitability (str): path to suitability factors table
        weight (float): suitability factor weight
        factor_inclusion (int): the rasterization method -- all touched or
            center points
        factors_field_container (bool): whether to use suitability factor
            inputs

        calculate_constraints (bool): whether to use constraint inputs
        constraints (str): filepath to constraints shapefile layer
        constraints_field (str): shapefile field containing constraints field

        override_layer (bool): whether to use override layer
        override (str): path to override shapefile
        override_field (str): shapefile field containing override value
        override_inclusion (int): the rasterization method

    Example Args::

        args = {
            'workspace_dir': 'path/to/dir',
            'suffix': '',
            'landcover': 'path/to/raster',
            'transition': 'path/to/csv',
            'calculate_priorities': True,
            'priorities_csv_uri': 'path/to/csv',
            'calculate_proximity': True,
            'calculate_transition': True,
            'calculate_factors': True,
            'suitability_folder': 'path/to/dir',
            'suitability': 'path/to/csv',
            'weight': 0.5,
            'factor_inclusion': 0,
            'factors_field_container': True,
            'calculate_constraints': True,
            'constraints': 'path/to/shapefile',
            'constraints_field': '',
            'override_layer': True,
            'override': 'path/to/shapefile',
            'override_field': '',
            'override_inclusion': 0
        }

    Added Afterwards::

        d = {
            'proximity_weight': 0.3,
            'distance_field': '',
            'transition_id': 'ID',
            'percent_field': 'Percent Change',
            'area_field': 'Area Change',
            'priority_field': 'Priority',
            'proximity_field': 'Proximity',
            'suitability_id': '',
            'suitability_layer': '',
            'suitability_field': '',
        }

    """
    LOGGER.info("Starting Scenario Generator model run...")

    # SHOULD BE INPUT AND VALIDATION FUNCTIONS
    LOGGER.info("Fetching and validating inputs...")
    # Preliminary tests
    if ('transition' in args) and ('suitability' in args):
        assert args['transition'] != args['suitability'], \
            'Transition and suitability tables are the same: ' + \
            args['transition'] + '. The model expects different tables.'

    # Transition table fields
    args["transition_id"] = "id"
    args["percent_field"] = "percent change"
    args["area_field"] = "area change"
    args["priority_field"] = "priority"
    args["proximity_field"] = "proximity"
    args["proximity_weight"] = "0.3"
    args["patch_field"] = "patch ha"

    # Suitability factors table fields
    args["suitability_id"] = "id"
    args["suitability_layer"] = "layer"
    args["suitability_weight"] = "wt"
    args["suitability_field"] = "suitfield"
    args["distance_field"] = "dist"
    args["suitability_cover_id"] = "cover id"

    # Exercise fields
    args["returns_cover_id"] = "cover id"

    landcover_uri = args["landcover"]

    if len(args["suffix"]) > 0:
        suffix = "_" + args["suffix"].strip("_")
    else:
        suffix = ""

    # Get parameters, set outputs
    workspace = args["workspace_dir"]
    if not os.path.exists(workspace):
        os.makedirs(workspace)
    intermediate_dir = os.path.join(workspace, "intermediate")
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)

    file_registry = invest_utils.build_file_registry(
        [(_OUTPUT, workspace), (_INTERMEDIATE, intermediate_dir)],
        args['suffix'])

    proximity_weight = float(args["proximity_weight"])

    # it might be better to just check if factors being used
    try:
        physical_suitability_weight = float(args["weight"])
    except KeyError:
        LOGGER.info("Physical suitability weight not found in user arguments. "
                    "Setting weight to 0.5")
        physical_suitability_weight = 0.5

    # output file names
    landcover_transition_uri = file_registry['landcover_transition_uri']
    override_dataset_uri = file_registry['override_dataset_uri']
    landcover_htm_uri = file_registry['landcover_htm_uri']

    geoprocess.create_directories([workspace])

    # relative paths, or with patterned name
    transition_name = file_registry['transition_name']
    suitability_name = file_registry['suitability_name']
    normalized_name = file_registry['normalized_name']
    combined_name = file_registry['combined_name']
    constraints_name = file_registry['constraints_name']
    filter_name = file_registry['filter_name']
    factors_name = file_registry['factors_name']
    cover_name = file_registry['cover_name']
    proximity_name = file_registry['proximity_name']
    normalized_proximity_name = file_registry['normalized_proximity_name']
    adjusted_suitability_name = file_registry['adjusted_suitability_name']

    # Constants
    raster_format = "GTiff"
    transition_dtype = gdal.GDT_Int16
    transition_nodata = -1
    change_nodata = -9999
    suitability_dtype = gdal.GDT_Int16
    suitability_nodata = 0

    # Value to multiply transition matrix entries
    #    (ie covert 10 point scale to 100 point scale)
    transition_scale = 10
    distance_scale = 100

    ds_type = "GTiff"
    driver = gdal.GetDriverByName(ds_type)

    def suitability_op(trans, suit):
        if trans == 0:
            return 0
        return round(((1 - physical_suitability_weight) * trans) +
                     (physical_suitability_weight * suit))

    # Validate data
    if not any([args["calculate_transition"],
                args["calculate_factors"],
                args["override_layer"]]):
        msg = "You must select at least one of the following: specify "\
              "transitions, use factors, or override layer."
        LOGGER.error(msg)
        raise ValueError(msg)

    # Transition table validation
    if args["transition"] and not(args["calculate_transition"] or args[
            "calculate_factors"]):
        msg = "Transition table provided but not used."
        LOGGER.warn(msg)

    transition_dict = {}
    if args["calculate_transition"] or args["calculate_factors"]:
        transition_dict = geoprocess.get_lookup_from_table(
            args["transition"],
            args["transition_id"])
        landcover_count_dict = geoprocess.unique_raster_values_count(
            landcover_uri)
        missing_lulc = set(landcover_count_dict).difference(
            transition_dict.keys())
        if len(missing_lulc) > 0:
            missing_lulc = list(missing_lulc)
            missing_lulc.sort()
            mising_lulc = ", ".join([str(l) for l in missing_lulc])
            msg = "Missing suitability information for cover(s) %s."\
                  % missing_lulc
            LOGGER.error(msg)
            raise ValueError(msg)

        for cover_id in transition_dict:
            if (transition_dict[cover_id][args["percent_field"]] > 0) and not (
                    cover_id in landcover_count_dict):
                msg = "Cover %i does not exist in LULC and therefore cannot "\
                      "have a percent change." % cover_id
                raise ValueError(msg)

            # raise error if change by percent and area both specified
            if (transition_dict[cover_id][args["percent_field"]] > 0) and (
                    transition_dict[cover_id][args["area_field"]] > 0):
                msg = "Cover %i cannot have both an increase by percent and "\
                      "area." % cover_id
                raise ValueError(msg)

    # Factor parameters validation
    if args["calculate_factors"]:
        # error if overall physical weight not in [0, 1] range
        # factor table validation
        # if polygon no distance field allowed
        # if point or line, integer distance field only
        # error if same factor twice for same coverage
        pass

    if args["calculate_priorities"]:
        priorities_dict = calculate_priority(args["priorities_csv_uri"])

    # Check geographic extents, projections
    cell_size = geoprocess.get_cell_size_from_uri(landcover_uri)
    suitability_transition_dict = {}

    # THIS SHOULD BE OWN FUNCTION
    if args["calculate_transition"]:
        LOGGER.info("Calculating transition...")
        for next_lulc in transition_dict:
            transition_raster_fpath = os.path.join(
                workspace, transition_name % next_lulc)
            reclass_dict = {}
            all_zeros = True
            for this_lulc in transition_dict:
                value = int(transition_dict[this_lulc][str(int(next_lulc))])
                reclass_dict[this_lulc] = value * transition_scale
                all_zeros = all_zeros and (value == 0)

            if not all_zeros:
                geoprocess.reclassify_dataset_uri(
                    landcover_uri,
                    reclass_dict,
                    transition_raster_fpath,
                    transition_dtype,
                    suitability_nodata,
                    exception_flag="values_required")

                # Change nodata value so 0's no longer nodata
                dataset = gdal.Open(transition_raster_fpath, 1)
                band = dataset.GetRasterBand(1)
                nodata = band.SetNoDataValue(transition_nodata)
                dataset = None
                suitability_transition_dict[
                    next_lulc] = transition_raster_fpath

    # THIS SHOULD BE OWN FUNCTION
    suitability_factors_dict = {}
    if args["calculate_factors"]:
        LOGGER.info("Calculating suitability factors...")
        factor_dict = geoprocess.get_lookup_from_table(
            args["suitability"],
            args["suitability_id"])
        factor_uri_dict = {}
        factor_folder = args["suitability_folder"]

        if not args["factor_inclusion"]:
            option_list = ["ALL_TOUCHED=TRUE"]
        else:
            option_list = ["ALL_TOUCHED=FALSE"]

        for factor_id in factor_dict:
            factor = factor_dict[factor_id][args["suitability_layer"]]
            factor_stem, _ = os.path.splitext(factor)
            suitability_field_name = factor_dict[
                factor_id][args["suitability_field"]]
            distance = factor_dict[factor_id][args["distance_field"]]
            cover_id = int(factor_dict[factor_id][
                args["suitability_cover_id"]])
            weight = int(factor_dict[factor_id][args["suitability_weight"]])

            LOGGER.debug(
                "Found reference to factor (%s, %s, %s) for cover %i.",
                factor_stem, suitability_field_name, distance, cover_id)

            if not (factor_stem, suitability_field_name, distance) in \
                    factor_uri_dict:
                factor_uri = os.path.join(factor_folder, factor)
                if not os.path.exists(factor_uri):
                    msg = "Missing file %s." % factor_uri
                    LOGGER.error(msg)
                    raise ValueError(msg)

                shape_type = get_geometry_type_from_uri(factor_uri)
                LOGGER.debug("Processing %s.", shapeTypes[shape_type])

                if shape_type in [5, 15, 25, 31]:  # polygon
                    LOGGER.info("Rasterizing %s using suitability field %s.",
                                factor_stem, suitability_field_name)
                    ds_uri = os.path.join(workspace, suitability_name % (
                        factor_stem, suitability_field_name))

                    burn_value = [1]
                    suitability_field = [
                        "ATTRIBUTE=%s" % suitability_field_name]
                    gdal_format = gdal.GDT_Float64
                    geoprocess.new_raster_from_base_uri(
                        landcover_uri,
                        ds_uri,
                        raster_format,
                        transition_nodata,
                        gdal_format,
                        fill_value=0)
                    geoprocess.rasterize_layer_uri(
                        ds_uri,
                        factor_uri,
                        burn_value,
                        option_list=option_list+suitability_field)
                    factor_uri_dict[(
                        factor_stem, suitability_field_name, distance)] =\
                        ds_uri

                # Point or line
                elif shape_type in [1, 3, 8, 11, 13, 18, 21, 23, 28]:
                    # For features with no area, it's (almost) impossible to
                    # hit the center pixel, so we use ALL_TOUCHED=TRUE
                    option_list = ["ALL_TOUCHED=TRUE"]
                    distance = int(distance)

                    ds_uri = os.path.join(
                        workspace, suitability_name %
                        (factor_stem, str(distance) + '_raw_raster'))
                    distance_uri = os.path.join(
                        workspace, suitability_name %
                        (factor_stem, str(distance) + '_raw_distance'))
                    fdistance_uri = os.path.join(
                        workspace, suitability_name %
                        (factor_stem, distance))
                    normalized_uri = os.path.join(
                        workspace, normalized_name %
                        (factor_stem, distance))

                    burn_value = [1]
                    LOGGER.info(
                        "Buffering rasterization of %s to distance of %i.",
                        factor_stem, distance)
                    gdal_format = gdal.GDT_Byte
                    geoprocess.new_raster_from_base_uri(
                        landcover_uri, ds_uri, raster_format, -1, gdal_format)

                    landcover_nodata = geoprocess.get_nodata_from_uri(
                        landcover_uri)
                    ds_nodata = geoprocess.get_nodata_from_uri(ds_uri)

                    geoprocess.vectorize_datasets(
                        [landcover_uri],
                        lambda x: 0 if x != landcover_nodata else -1,
                        ds_uri,
                        geoprocess.get_datatype_from_uri(
                            ds_uri),
                        ds_nodata,
                        geoprocess.get_cell_size_from_uri(
                            ds_uri),
                        'intersection')

                    geoprocess.rasterize_layer_uri(
                        ds_uri, factor_uri, burn_value, option_list)

                    calculate_distance_raster_uri(ds_uri, distance_uri)

                    def threshold(value):
                        result = np.where(
                            value > distance, transition_nodata, value)
                        return np.where(
                            value == transition_nodata,
                            transition_nodata,
                            result)

                    geoprocess.vectorize_datasets(
                        [distance_uri],
                        threshold,
                        fdistance_uri,
                        geoprocess.get_datatype_from_uri(distance_uri),
                        transition_nodata,
                        cell_size,
                        "union",
                        vectorize_op=False)

                    geoprocess.calculate_raster_stats_uri(fdistance_uri)
                    minimum, maximum, _, _ =\
                        geoprocess.get_statistics_from_uri(fdistance_uri)

                    def normalize_op(value):
                        diff = float(maximum - minimum)

                        return np.where(
                            value == transition_nodata,
                            suitability_nodata,
                            ((distance_scale - 1) - (((value - minimum) /
                             diff) * (distance_scale - 1))) + 1)

                    geoprocess.vectorize_datasets(
                        [fdistance_uri],
                        normalize_op,
                        normalized_uri,
                        transition_dtype,
                        transition_nodata,
                        cell_size,
                        "union",
                        vectorize_op=False)

                    factor_uri_dict[(
                        factor_stem, suitability_field_name, distance)] =\
                        normalized_uri
                else:
                    raise ValueError("Invalid geometry type %i." % shape_type)

                # Apply nodata to the factors raster
                landcover_nodata = \
                    geoprocess.get_nodata_from_uri(landcover_uri)
                temp_uri = geoprocess.temporary_filename()

                def apply_nodata_op(landcover, value):
                    return np.where(landcover == landcover_uri, 0, value)

                geoprocess.vectorize_datasets(
                    [landcover_uri,
                     factor_uri_dict[(
                        factor_stem, suitability_field_name, distance)]],
                    apply_nodata_op,
                    temp_uri,
                    transition_dtype,
                    transition_nodata,
                    cell_size,
                    "union",
                    vectorize_op=False)

                def identity_op(x):
                    return x

                geoprocess.vectorize_datasets(
                    [temp_uri],
                    identity_op,
                    factor_uri_dict[(
                        factor_stem, suitability_field_name, distance)],
                    transition_dtype,
                    transition_nodata,
                    cell_size,
                    "union",
                    vectorize_op=False)

            else:
                LOGGER.debug("Skipping already processed suitability layer.")

            LOGGER.debug(
                "Adding factor (%s, %s, %s) to cover %i suitability list.",
                factor_stem, suitability_field_name, distance, cover_id)

            if cover_id in suitability_factors_dict:
                suitability_factors_dict[cover_id].append((factor_uri_dict[(
                    factor_stem, suitability_field_name, distance)], weight))
            else:
                suitability_factors_dict[cover_id] = [(
                    factor_uri_dict[(
                        factor_stem,
                        suitability_field_name,
                        distance)],
                    weight)]

        for cover_id in suitability_factors_dict:
            if len(suitability_factors_dict[cover_id]) > 1:
                LOGGER.info("Combining factors for cover type %i.", cover_id)
                ds_uri = os.path.join(workspace, combined_name % cover_id)

                uri_list, weights_list = apply(
                    zip, suitability_factors_dict[cover_id])

                total = float(sum(weights_list))
                weights_list = [weight / total for weight in weights_list]

                def weighted_op(*values):
                    result = (values[0] * weights_list[0]).astype(float)
                    for v, w in zip(values[1:], weights_list[1:]):
                        result += v * w
                    return result

                geoprocess.vectorize_datasets(
                    list(uri_list),
                    weighted_op,
                    ds_uri,
                    suitability_dtype,
                    transition_nodata,
                    cell_size,
                    "union",
                    vectorize_op=False)

                suitability_factors_dict[cover_id] = ds_uri

            else:
                suitability_factors_dict[cover_id] = suitability_factors_dict[
                    cover_id][0][0]

    suitability_dict = {}
    if args["calculate_transition"]:
        suitability_dict = suitability_transition_dict
        if args["calculate_factors"]:
            for cover_id in suitability_factors_dict:
                if cover_id in suitability_dict:
                    LOGGER.info(
                        "Combining suitability for cover %i.", cover_id)
                    ds_uri = os.path.join(workspace, factors_name % cover_id)

                    geoprocess.vectorize_datasets(
                        [suitability_transition_dict[cover_id],
                            suitability_factors_dict[cover_id]],
                        suitability_op,
                        ds_uri,
                        transition_dtype,
                        transition_nodata,
                        cell_size,
                        "union")
                    suitability_dict[cover_id] = ds_uri
                else:
                    suitability_dict[cover_id] = suitability_factors_dict[
                        cover_id]
    elif args["calculate_factors"]:
        suitability_dict = suitability_factors_dict

    # clump and sieve
    for cover_id in transition_dict:
        if (transition_dict[cover_id][args["patch_field"]] > 0) and (
                cover_id in suitability_dict):
            LOGGER.info("Filtering patches from %i.", cover_id)
            size = 10000 * int(math.ceil(
                transition_dict[cover_id][args["patch_field"]] /
                               (cell_size ** 2)))

            output_uri = os.path.join(workspace, filter_name % cover_id)
            filter_fragments(suitability_dict[cover_id], size, output_uri)
            suitability_dict[cover_id] = output_uri

    # SHOULD BE OWN FUNCTION
    # Contraints raster
    #   (reclass using permability values, filters on clump size)
    if args["calculate_constraints"]:
        LOGGER.info("Calculating constraints...")
        constraints_uri = args["constraints"]
        constraints_field_name = args["constraints_field"]
        constraints_ds_uri = os.path.join(workspace, constraints_name)
        option_list = ["ALL_TOUCHED=FALSE"]
        burn_value = [0]
        constraints_field = ["ATTRIBUTE=%s" % constraints_field_name]
        gdal_format = gdal.GDT_Float64
        geoprocess.new_raster_from_base_uri(
            landcover_uri,
            constraints_ds_uri,
            raster_format,
            transition_nodata,
            gdal_format,
            fill_value=1)
        geoprocess.rasterize_layer_uri(
            constraints_ds_uri,
            constraints_uri,
            burn_value,
            option_list=option_list+constraints_field)

        # Check that the values make sense
        raster = gdal.Open(constraints_ds_uri)
        band = raster.GetRasterBand(1)
        array = band.ReadAsArray()
        unique = np.unique(array)
        assert (unique[0] >= 0.0) and (unique[-1] <= 1.0), \
            'Invalid raster value in field ' + constraints_field_name + ' in '\
            + constraints_uri

    else:
        LOGGER.info("Constraints not included.")

    # SHOULD BE OWN FUNCTION
    proximity_dict = {}
    if args["calculate_proximity"]:
        LOGGER.info("Calculating proximity.")
        cover_types = transition_dict.keys()
        for cover_id in transition_dict:
            if transition_dict[cover_id][args["proximity_field"]] == u'':
                msg = "Proximity value not provided for lulc_code %i.  Check "\
                      "your land cover transition / attributes table."
                raise ValueError(msg % cover_id)
            if transition_dict[cover_id][args["proximity_field"]] > 0 and \
                    cover_id in suitability_dict:
                distance = int(
                    transition_dict[cover_id][args["proximity_field"]])
                LOGGER.info("Calculating proximity for %i.", cover_id)
                reclass_dict = dict(zip(cover_types, [1] * len(cover_types)))
                reclass_dict[cover_id] = 0

                ds_uri = os.path.join(
                    workspace, cover_name % cover_id)
                distance_uri = geoprocess.temporary_filename()
                fdistance_uri = os.path.join(
                    workspace, proximity_name % cover_id)
                normalized_uri = os.path.join(
                    workspace, normalized_proximity_name % cover_id)

                geoprocess.reclassify_dataset_uri(
                    landcover_uri,
                    reclass_dict,
                    ds_uri,
                    transition_dtype,
                    transition_nodata,
                    exception_flag="values_required")

                calculate_distance_raster_uri(ds_uri, distance_uri)

                def threshold(value):
                    if value > distance:
                        return transition_nodata
                    return value

                geoprocess.vectorize_datasets(
                    [distance_uri],
                    threshold,
                    fdistance_uri,
                    geoprocess.get_datatype_from_uri(distance_uri),
                    transition_nodata,
                    cell_size,
                    "union")

                minimum, maximum, _, _ = geoprocess.get_statistics_from_uri(
                    fdistance_uri)

                assert minimum < maximum, "Wrong distance (min, max) = (" + \
                    str(minimum) + ", " + str(maximum) + ") in " + \
                    fdistance_uri

                def normalize_op(value):
                    if value == transition_nodata:
                        return suitability_nodata
                    else:
                        return ((distance_scale - 1) -
                                (((value - minimum) /
                                 float(maximum - minimum)) *
                                (distance_scale - 1))) + 1

                geoprocess.vectorize_datasets(
                    [fdistance_uri],
                    normalize_op,
                    normalized_uri,
                    transition_dtype,
                    transition_nodata,
                    cell_size,
                    "union")

                proximity_dict[cover_id] = normalized_uri

    def es_change_op(final_es, initial_es):
        return final_es - initial_es

    def constraint_op(suit, cons):
        return suit * cons

    def proximity_op(suit, prox):
        v = suit + (prox * proximity_weight)
        if v > 100:
            return 100
        else:
            return v

    def constraint_proximity_op(suit, cons, prox):
        v = (cons * suit) + (prox * proximity_weight)
        if v > 100:
            return 100
        else:
            return v

    for cover_id in suitability_dict:
        suitability_uri = os.path.join(
            workspace, adjusted_suitability_name % cover_id)
        if args["calculate_constraints"]:
            if cover_id in proximity_dict:
                LOGGER.info("Combining suitability, proximity, and constraints"
                            " for %i.", cover_id)

                uri_list = [suitability_dict[cover_id],
                            constraints_ds_uri,
                            proximity_dict[cover_id]]
                LOGGER.info("Vectorizing: %s", ", ".join(uri_list))
                geoprocess.vectorize_datasets(
                    uri_list,
                    constraint_proximity_op,
                    suitability_uri,
                    transition_dtype,
                    transition_nodata,
                    cell_size,
                    "union")
                suitability_dict[cover_id] = suitability_uri

            else:
                LOGGER.info(
                    "Combining suitability and constraints for %i.", cover_id)

                uri_list = [suitability_dict[cover_id],
                            constraints_ds_uri]
                LOGGER.info("Vectorizing: %s", ", ".join(uri_list))
                geoprocess.vectorize_datasets(
                    uri_list,
                    constraint_op,
                    suitability_uri,
                    transition_dtype,
                    transition_nodata,
                    cell_size,
                    "union")
                suitability_dict[cover_id] = suitability_uri

        elif cover_id in proximity_dict:
            LOGGER.info(
                "Combining suitability and proximity for %i.", cover_id)
            uri_list = [suitability_dict[cover_id],
                        proximity_dict[cover_id]]
            LOGGER.info("Vectorizing: %s", ", ".join(uri_list))
            geoprocess.vectorize_datasets(
                uri_list,
                proximity_op,
                suitability_uri,
                transition_dtype,
                transition_nodata,
                cell_size,
                "union")
            suitability_dict[cover_id] = suitability_uri

    # normalize probabilities to be on a 10 point scale
    # probability raster (reclass using probability matrix)

    # proximity raster (gaussian for each landcover type, using max distance)
    # InVEST 2 uses 4-connectedness?

    # combine rasters for weighting into sutibility raster, multiply proximity
    # by 0.3
    # [suitability * (1-factor weight)] + (factors * factor weight) or only
    # single raster

    ###
    # reallocate pixels (disk heap sort, randomly reassign equal value pixels,
    # applied in order)
    ###

    # copy initial LULC
    scenario_uri = file_registry['scenario_uri']

    src_ds = gdal.Open(landcover_uri)
    n_cols = src_ds.RasterXSize
    n_rows = src_ds.RasterYSize

    dst_ds = driver.CreateCopy(scenario_uri, src_ds, 0)
    dst_ds = None
    src_ds = None

    # THIS SHOULD BE OWN FUNCTION -- also compress if-else into single func
    # identify LULC types undergoing change
    change_list = []
    if args["calculate_priorities"]:
        LOGGER.info("Calculating priorities using priority table...")
        for cover_id in transition_dict:
            percent_change = transition_dict[cover_id][args["percent_field"]]
            area_change = transition_dict[cover_id][args["area_field"]]
            if percent_change > 0:
                change_list.append((
                    priorities_dict[cover_id],
                    cover_id,
                    int((percent_change / 100.0) *
                        landcover_count_dict[cover_id])))
            elif area_change > 0:
                change_list.append((
                    priorities_dict[cover_id],
                    cover_id,
                    int(math.ceil(10000 * area_change / (cell_size**2)))))
            else:
                LOGGER.warn(
                    "Cover %i suitability specified, but no change indicated.",
                    cover_id)
    else:
        LOGGER.info("Calculating priorities with land attributes table...")
        for cover_id in transition_dict:
            percent_change = transition_dict[cover_id][args["percent_field"]]
            area_change = transition_dict[cover_id][args["area_field"]]
            if percent_change > 0:
                change_list.append((
                    transition_dict[cover_id][args["priority_field"]],
                    cover_id,
                    int((percent_change / 100.0) *
                        landcover_count_dict[cover_id])))
            elif area_change > 0:
                change_list.append((
                    transition_dict[cover_id][args["priority_field"]],
                    cover_id,
                    int(math.ceil(10000 * area_change / (cell_size**2)))))
            else:
                LOGGER.warn(
                    "Cover %i suitability specified, but no change indicated.",
                    cover_id)

    change_list.sort(reverse=True)

    # change pixels
    scenario_ds = gdal.Open(scenario_uri, 1)
    scenario_band = scenario_ds.GetRasterBand(1)
    scenario_array = scenario_band.ReadAsArray()

    unconverted_pixels = {}
    for index, (priority, cover_id, count) in enumerate(change_list):
        LOGGER.debug("Increasing cover %i by %i pixels.", cover_id, count)

        # open all lower priority suitability rasters and assign changed pixels
        # value of 0
        update_ds = {}
        update_bands = {}
        update_arrays = {}
        for _, update_id, _ in change_list[index+1:]:
            update_ds[update_id] = gdal.Open(suitability_dict[update_id], 1)
            update_bands[update_id] = update_ds[update_id].GetRasterBand(1)
            update_arrays[update_id] = update_bands[update_id].ReadAsArray()

        # select pixels
        # open suitability raster
        src_ds = gdal.Open(suitability_dict[cover_id], 1)
        src_band = src_ds.GetRasterBand(1)
        src_array = src_band.ReadAsArray()

        pixels_changed = 0
        suitability_values = list(np.unique(src_array))
        suitability_values.sort(reverse=True)
        if suitability_values[-1] == 0:
            suitability_values.pop(-1)
        for suitability_score in suitability_values:
            # Check if suitsbility is between 0 and 100 inclusive
            if abs(suitability_score - 50) > 50:
                print('suitability_values:', suitability_dict[cover_id])
                for v in suitability_values:
                    print v, ' ',

            assert abs(suitability_score - 50) <= 50, \
                'Invalid suitability score ' + str(suitability_score)
            if pixels_changed == count:
                LOGGER.debug("All necessay pixels converted.")
                break

            LOGGER.debug(
                "Checking pixels with suitability score of %i.",
                suitability_score)

            # mask out everything except the current suitability score
            mask = src_array == suitability_score

            # label patches
            label_im, nb_labels = sp.ndimage.label(mask)

            # get patch sizes
            patch_sizes = sp.ndimage.sum(
                mask, label_im, range(1, nb_labels + 1))
            patch_labels = np.array(range(1, nb_labels + 1))
            patch_locations = sp.ndimage.find_objects(label_im, nb_labels)

            # randomize patch order
            np.random.shuffle(patch_labels)

            # check patches for conversion
            patch_label_count = patch_labels.size
            for l in range(patch_label_count):
                label = patch_labels[l]
                source = label_im[patch_locations[label-1]]
                target = scenario_array[patch_locations[label-1]]
                pixels_to_change = np.where(source == label)
                assert pixels_to_change[0].size == patch_sizes[label-1]

                if patch_sizes[label-1] + pixels_changed > count:
                    # mask out everything except the current patch
                    # patch = np.where(label_im == label)
                    # patch_mask = np.zeros_like(scenario_array)
                    patch_mask = np.zeros_like(target)
                    # patch_mask[patch] = 1
                    patch_mask[pixels_to_change] = 1

                    # calculate the distance to exit the patch
                    # tmp_array = sp.ndimage.morphology.distance_transform_edt(
                    # patch_mask)
                    tmp_array = sp.ndimage.morphology.distance_transform_edt(
                        patch_mask)
                    # tmp_array = tmp_array[patch]
                    tmp_array = tmp_array[pixels_to_change]

                    # select the number of pixels that need to be converted
                    tmp_index = np.argsort(tmp_array)
                    tmp_index = tmp_index[:count - pixels_changed]

                    # convert the selected pixels into coordinates
                    # pixels_to_change = np.array(zip(patch[0], patch[1]))
                    pixels_to_change = np.array(zip(
                        pixels_to_change[0], pixels_to_change[1]))
                    pixels_to_change = pixels_to_change[tmp_index]
                    pixels_to_change = apply(zip, pixels_to_change)

                    # change the pixels in the scenario
                    # scenario_array[pixels_to_change] = cover_id
                    target[pixels_to_change] = cover_id

                    pixels_changed = count

                    # alter other suitability rasters to prevent double
                    # conversion
                    for _, update_id, _ in change_list[index+1:]:
                        # update_arrays[update_id][pixels_to_change] = 0
                        target = update_arrays[
                            update_id][patch_locations[label-1]]
                        target[pixels_to_change] = 0

                    break

                else:
                    # convert patch, increase count of changes
                    target[pixels_to_change] = cover_id
                    pixels_changed += patch_sizes[label-1]

                    # alter other suitability rasters to prevent double
                    # conversion
                    for _, update_id, _ in change_list[index+1:]:
                        target = update_arrays[
                            update_id][patch_locations[label-1]]
                        target[pixels_to_change] = 0

        # report and record unchanged pixels
        if pixels_changed < count:
            LOGGER.warn("Not all pixels converted.")
            unconverted_pixels[cover_id] = count - pixels_changed

        # write new suitability arrays
        for _, update_id, _ in change_list[index+1:]:
            update_bands[update_id].WriteArray(update_arrays[update_id])
            update_arrays[update_id] = None
            update_bands[update_id] = None
            update_ds[update_id] = None

    scenario_band.WriteArray(scenario_array)
    scenario_array = None
    scenario_band = None
    scenario_ds = None

    # THIS SHOULD BE OWN FUNCTION
    # apply override
    if args["override_layer"]:
        LOGGER.info(
            "Overriding pixels using values from field %s...",
            args["override_field"])

        datasource = ogr.Open(args["override"])
        layer = datasource.GetLayer()
        dataset = gdal.Open(scenario_uri, 1)

        if dataset is None:
            msg = "Could not open landcover transition raster."
            LOGGER.error(msg)
            raise IOError(msg)

        if datasource is None:
            msg = "Could not open override vector."
            LOGGER.error(msg)
            raise IOError(msg)

        if not bool(args["override_inclusion"]):
            LOGGER.debug("Overriding all touched pixels.")
            options = ["ALL_TOUCHED=TRUE", "ATTRIBUTE=%s" % args[
                "override_field"]]
        else:
            LOGGER.debug("Overriding only pixels with covered center points.")
            options = ["ATTRIBUTE=%s" % args["override_field"]]
        gdal.RasterizeLayer(dataset, [1], layer, options=options)
        dataset.FlushCache()
        datasource = None
        dataset = None

    # THIS SHOULD BE OWN FUNCTION
    LOGGER.info("Generating results...")
    # Tabulate coverages
    unique_raster_values_count, transitions = \
        get_transition_pairs_count_from_uri([landcover_uri, scenario_uri])

    htm = open(landcover_htm_uri, 'w')
    htm.write("<html><head><title>Scenario Generator Report</title>")

    htm.write("<style type='text/css'>")
    htm.write("table {border-collapse: collapse; font-size: 1em;}")
    htm.write("td {padding: 10px;}")
    htm.write('body {font-family: Arial, Helvetica, sans-serif; '
              'font-size: 1em;}')
    htm.write('h2 {background: #DDDDDD; padding: 10px;}')
    htm.write("</style>")

    jquery_uri = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "jquery-1.6.2.min.js")
    htm.write("<script>\n" + open(jquery_uri).read() + "\n</script>")
    highcharts_uri = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "highcharts.js")
    htm.write("<script>\n" + open(highcharts_uri).read() + "\n</script>")

    htm.write("</head><body>")
    htm.write("<div style=''>")
    htm.write("<h1>Scenario Output Summary</h1>")
    htm.write("<h2>Initial Landscape</h2>")
    htm.write("\n<table BORDER=1>")
    initial_cover_id_list = unique_raster_values_count[landcover_uri].keys()
    initial_cover_id_list.sort()

    htm.write("\n<tr><td>ID</td><td>")
    htm.write("</td><td>".join([
        str(cover_id) for cover_id in initial_cover_id_list]))
    htm.write("\n</td></tr>")

    htm.write("\n<tr><td>Count</td><td>")
    htm.write("</td><td>".join([str(
        unique_raster_values_count[landcover_uri][cover_id]) for
            cover_id in initial_cover_id_list]))
    htm.write("\n</td></tr>")

    htm.write("\n</table>")

    htm.write("<h2>Scenario Landscape</h2>")
    htm.write("\n<table BORDER=1>")
    scenario_cover_id_list = unique_raster_values_count[scenario_uri].keys()
    scenario_cover_id_list.sort()

    htm.write("\n<tr><td>ID</td><td>")
    htm.write("</td><td>".join([str(cover_id) for
                                cover_id in scenario_cover_id_list]))
    htm.write("\n</td></tr>")

    htm.write("\n<tr><td>Count</td><td>")
    htm.write("</td><td>".join([str(
        unique_raster_values_count[scenario_uri][cover_id]) for
        cover_id in scenario_cover_id_list]))
    htm.write("\n</td></tr>")

    htm.write("\n</table>")

    cover_dict = {}
    for cover_id in set(unique_raster_values_count[
            landcover_uri].keys()).union(set(
                unique_raster_values_count[scenario_uri].keys())):
        try:
            before = unique_raster_values_count[landcover_uri][cover_id]
        except KeyError:
            before = 0
        try:
            after = unique_raster_values_count[scenario_uri][cover_id]
        except KeyError:
            after = 0
        cover_dict[cover_id] = (before, after)

    htm.write("<h2>Change Table</h2>")

    cover_names_dict = {}

    transition_dict = geoprocess.get_lookup_from_table(
        args["transition"], args["transition_id"])
    cover_names_dict = {}
    for cover in transition_dict:
        cover_names_dict[cover] = transition_dict[cover]["name"]

    htm.write(generate_chart_html(cover_dict, cover_names_dict, workspace))

    htm.write("<h2>Transition Matrix</h2>")
    htm.write("\n<table BORDER=1>")
    htm.write("\n<tr><td>ID</td><td>")
    htm.write("</td><td>".join([str(cover_id) for cover_id
                                in scenario_cover_id_list]))
    htm.write("\n</td></tr>")

    for initial_cover_id in initial_cover_id_list:
        htm.write("\n<tr><td>%i</td>" % initial_cover_id)
        for scenario_cover_id in scenario_cover_id_list:
            try:
                htm.write("<td>%i</td>" % transitions[0][initial_cover_id][
                    scenario_cover_id])
            except KeyError:
                htm.write("<td><FONT COLOR=lightgray>%i</FONT></td>" % 0)

        htm.write("\n</tr>")

    htm.write("\n</table>")

    unconverted_cover_id_list = unconverted_pixels.keys()
    unconverted_cover_id_list.sort()
    if len(unconverted_cover_id_list) > 0:
        htm.write("<h2>Unconverted Pixels</h2>")
        htm.write("\n<table BORDER=1>")
        htm.write("<tr><td>ID</td><td>Count</td></tr>")
        for cover_id in unconverted_cover_id_list:
            htm.write("<tr><td>%i</td><td>%i</td></tr>" % (
                cover_id, unconverted_pixels[cover_id]))
        htm.write("\n</table>")
    else:
        htm.write("<p><i>All target pixels converted.</i></p>")
    htm.write("\n</html>")

    # Input CSVs
    input_csv_list = []

    if args["calculate_priorities"]:
        input_csv_list.append((args["priorities_csv_uri"], "Priorities Table"))

    if args["calculate_transition"] or args["calculate_factors"]:
        input_csv_list.append((args["transition"], "Transition Table"))

    if args["calculate_factors"]:
        input_csv_list.append((args["suitability"], "Factors Table"))

    htm.write("<h1>Input Tables</h1>")
    for csv_uri, name in input_csv_list:
        table = "\n<table BORDER=1><tr><td>" + \
            open(csv_uri).read().strip().replace(",", "</td><td>").replace(
                "\n", "</td></tr><tr><td>") + "</td></tr></table>"

        htm.write("<h2>%s</h2>" % name)
        htm.write(table)
    htm.write("\n</div>\n</body>\n</html>")
    htm.close()

    LOGGER.info("...model run complete.")
