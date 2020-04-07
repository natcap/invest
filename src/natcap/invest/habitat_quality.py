# coding=UTF-8
"""InVEST Habitat Quality model."""
import collections
import os
import logging
import pickle

import numpy
from osgeo import gdal
from osgeo import osr
import pygeoprocessing
import taskgraph

from . import utils
from . import validation

LOGGER = logging.getLogger(__name__)

ARGS_SPEC = {
    "model_name": "Habitat Quality",
    "module": __name__,
    "userguide_html": "habitat_quality.html",
    "args_with_spatial_overlap": {
        "spatial_keys": ["lulc_cur_path", "lulc_fut_path", "lulc_bas_path"],
    },
    "args": {
        "workspace_dir": validation.WORKSPACE_SPEC,
        "results_suffix": validation.SUFFIX_SPEC,
        "n_workers": validation.N_WORKERS_SPEC,
        "lulc_cur_path": {
            "type": "raster",
            "required": True,
            "validation_options": {
                "projected": True,
            },
            "about": (
                "A GDAL-supported raster file.  The current LULC must have "
                "its' own threat rasters, where each threat raster file path "
                "is defined in the <b>Threats Data</b> CSV.<br/><br/> "
                "Each cell should represent a LULC code as an Integer. "
                "The dataset should be in a projection where the units are "
                "in meters and the projection used should be defined.  The "
                "LULC codes must match the codes in the Sensitivity table."),
            "name": "Current Land Cover"
        },
        "lulc_fut_path": {
            "type": "raster",
            "required": False,
            "validation_options": {
                "projected": True,
            },
            "about": (
                "Optional.  A GDAL-supported raster file.  Inputting a "
                "future LULC will generate degradation, habitat quality, and "
                "habitat rarity (If baseline is input) outputs.  The future "
                "LULC must have it's own threat rasters, where each threat "
                "raster file path is defined in the <b>Threats Data</b> CSV. "
                "<br/><br/>Each cell should represent a LULC code as an "
                "Integer.  The dataset should be in a projection where the "
                "units are in meters and the projection used should be "
                "defined. The LULC codes must match the codes in the "
                "Sensitivity table."),
            "name": "Future Land Cover"
        },
        "lulc_bas_path": {
            "type": "raster",
            "required": False,
            "validation_options": {
                "projected": True,
            },
            "about": (
                "Optional.  A GDAL-supported raster file.  If the baseline "
                "LULC is provided, rarity outputs will be created for the "
                "current and future LULC. The baseline LULC can have it's "
                "own threat rasters (optional), where each threat raster "
                "file path is defined in the <b>Threats Data</b> CSV. "
                "If there are no threat rasters and the threat paths are "
                "left blank in the CSV column, degradation and habitat quality "
                "outputs will not be generated for the baseline LULC.<br/><br/> "
                "Each cell should represent a LULC code as an Integer.  The "
                "dataset should be in a projection where the units are in "
                "meters and the projection used should be defined. The LULC "
                "codes must match the codes in the Sensitivity table.  If "
                "possible the baseline map should refer to a time when "
                "intensive management of the landscape was relatively rare."),
            "name": "Baseline Land Cover"
        },
        "threats_table_path": {
            "validation_options": {
                "required_fields": [
                    "THREAT", "MAX_DIST", "WEIGHT", "DECAY", "BASE_PATH", 
                    "CUR_PATH", "FUT_PATH"],
            },
            "type": "csv",
            "required": True,
            "about": (
                "A CSV file of all the threats for the model to consider. "
                "Each row in the table is a degradation source. The columns "
                "(THREAT, MAX_DIST, WEIGHT, DECAY) are different attributes "
                "of each degradation source. The columns (BASE_PATH, CUR_PATH, "
                "FUT_PATH) specify the filepath name for the degradation "
                "source where the path is relative to the THREAT CSV. "
                "THREAT: "
                "The name of the threat source and this name must match "
                "exactly to the name of it's corresponding column in the "
                "sensitivity table. "
                "NOTE: The sensitivity column should have a "
                "prefix indicator (L_). The THREAT name in the threat table "
                "should not include 'L_' prefix. "
                "MAX_DIST: A number in kilometres (km) for the maximum "
                "distance a threat has an affect. WEIGHT: A "
                "floating point value between 0 and 1 for the threats "
                "weight relative to the other threats.  Depending on the "
                "type of habitat under review, certain threats may cause "
                "greater degradation than other threats. "
                "DECAY: A string value of either exponential or "
                "linear representing the type of decay over space for "
                "the threat. See the user's guide for valid values "
                "for these columns. "
                "BASE_PATH: optional. The THREAT raster filepath for "
                "the base scenario (if present) where the filepath is "
                "relative to the THREAT CSV input. CUR_PATH: required.  "
                "The THREAT raster filepath for the current scenario "
                "where the filepath is relative to the THREAT CSV input. "
                "FUT_PATH: optional. The THREAT raster filepath for the "
                "future scenario (if present) where the filepath is "
                "relative to the THREAT CSV input."
                ),
            "name": "Threats Data"
        },
        "access_vector_path": {
            "validation_options": {
                "required_fields": ["access"],
                "projected": True,
            },
            "type": "vector",
            "required": False,
            "about": (
                "A GDAL-supported vector file.  The input contains data on "
                "the relative protection that legal / institutional / social "
                "/ physical barriers provide against threats.  The vector "
                "file should contain polygons with a field ACCESS. "
                "The ACCESS values should range from 0 - 1, where 1 "
                "is fully accessible.  Any cells not covered by a polygon "
                "will be set to 1."),
            "name": "Accessibility to Threats (Vector) (Optional)"
        },
        "sensitivity_table_path": {
            "validation_options": {
                "required_fields": ["LULC", "NAME", "HABITAT"],
            },
            "type": "csv",
            "required": True,
            "about": (
                "A CSV file of LULC types, whether or not they are considered "
                "habitat, and, for LULC types that are habitat, their "
                "specific sensitivity to each threat. Each row is a LULC "
                "type with the following columns: LULC, HABITAT, "
                "L_THREAT1, L_THREAT2, ... LULC: Integer "
                "values that reflect each LULC code found in current, "
                "future, and baseline rasters. HABITAT: "
                "A value of 0 or 1 (presence / absence) or a value between 0 "
                "and 1 (continuum) depicting the suitability of "
                "habitat. L_THREATN: Each L_THREATN should "
                "match exactly with the threat names given in the threat "
                "CSV file, where the THREATN is the name that matches.  This "
                "is a floating point value between 0 and 1 that represents "
                "the sensitivity of a habitat to a threat."
                "Please see the users guide for more detailed information on "
                "proper column values and column names for each threat."),
            "name": "Sensitivity of Land Cover Types to Each Threat"
        },
        "half_saturation_constant": {
            "validation_options": {
                "expression": "value > 0",
            },
            "type": "number",
            "required": True,
            "about": (
                "A positive floating point value that is defaulted at 0.5. "
                "This is the value of the parameter k in equation (4). In "
                "general, set k to half of the highest grid cell degradation "
                "value on the landscape.  To perform this model calibration "
                "the model must be run once in order to find the highest "
                "degradation value and set k for the provided landscape.  "
                "Note that the choice of k only determines the spread and "
                "central tendency of habitat quality cores and does not "
                "affect the rank."),
            "name": "Half-Saturation Constant"
        },
    }
}

_OUT_NODATA = -1.0
_RARITY_NODATA = -64329.0
_SCALING_PARAM = 2.5
_THREAT_SCENARIO_MAP = {'_c':'CUR_PATH', '_f':'FUT_PATH','_b':'BASE_PATH'}


def execute(args):
    """Habitat Quality.

    This model calculates habitat degradation and quality for the current LULC
    as described in the InVEST user's guide. Optionally execute calculates
    habitat degradation and quality for a future LULC and habitat rarity for
    current and future LULC.

    Parameters:
        args['workspace_dir'] (string): a path to the directory that will write output
            and other temporary files (required)
        args['lulc_cur_path'] (string): a path to an input land use/land cover raster
            (required)
        args['lulc_fut_path'] (string): a path to an input land use/land cover raster
            (optional)
        args['lulc_bas_path'] (string): a path to an input land use/land cover raster
            (optional, but required for rarity calculations)
        args['threats_table_path'] (string): a path to an input CSV containing data
            of all the considered threats. Each row is a degradation source
            and each column a different attribute of the source with the
            following names: 'THREAT','MAX_DIST','WEIGHT', 'BASE_PATH', 
            'CUR_PATH', 'FUT_PATH' (required).
        args['access_vector_path'] (string): a path to an input polygon shapefile
            containing data on the relative protection against threats (optional)
        args['sensitivity_table_path'] (string): a path to an input CSV file of LULC
            types, whether they are considered habitat, and their sensitivity
            to each threat (required)
        args['half_saturation_constant'] (float): a python float that determines
            the spread and central tendency of habitat quality scores
            (required)
        args['results_suffix'] (string): a python string that will be inserted into all
            raster path paths just before the file extension.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process.

    Returns:
        None
    """

    # Append a _ to the suffix if it's not empty and doesn't already have one
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    # Check to see if each of the workspace folders exists.  If not, create the
    # folder in the filesystem.
    LOGGER.info("Creating workspace")
    output_dir = args['workspace_dir']
    intermediate_output_dir = os.path.join(args['workspace_dir'], 'intermediate')
    kernel_dir = os.path.join(intermediate_output_dir, 'kernels')
    utils.make_directories([intermediate_output_dir, output_dir, kernel_dir])

    work_token_dir = os.path.join(intermediate_output_dir, '_taskgraph_working_dir')
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous mode.
    graph = taskgraph.TaskGraph(work_token_dir, n_workers)

    LOGGER.info("Checking Threat and Sensitivity tables for compliance")
    # Get CSVs as dictionaries and ensure the key is a string for threats.
    threat_dict = {
        str(key): value for key, value in utils.build_lookup_from_csv(
            args['threats_table_path'], 'THREAT', to_lower=False).items()}
    sensitivity_dict = utils.build_lookup_from_csv(
        args['sensitivity_table_path'], 'LULC', to_lower=False)
    
    # check that the required headers exist in the threat table.
    # Raise exception if they don't.
    threat_header_list = list(list(threat_dict.values())[0].keys())
    required_threat_header_list = [
        'WEIGHT', 'MAX_DIST', 'DECAY', 'BASE_PATH', 'CUR_PATH', 'FUT_PATH']
    missing_threat_header_list = [
        h for h in required_threat_header_list if h not in threat_header_list]
    if missing_threat_header_list:
        raise ValueError(
            'Column(s) %s are missing in the threat table' %
            (', '.join(missing_threat_header_list)))    

    # check that the required headers exist in the sensitivity table.
    # Raise exception if they don't.
    sens_header_list = list(sensitivity_dict.values())[0]
    required_sens_header_list = ['LULC', 'NAME', 'HABITAT']
    missing_sens_header_list = [
        h for h in required_sens_header_list if h not in sens_header_list]
    if missing_sens_header_list:
        raise ValueError(
            'Column(s) %s are missing in the sensitivity table' %
            (', '.join(missing_sens_header_list)))    
    
    # check that the threat names in the threats table match with the threats
    # columns in the sensitivity table. 
    missing_threat_header_list = [] 
    for threat in threat_dict:
        if 'L_' + threat not in sens_header_list:
            missing_threat_header_list.append(threat) 

    if missing_threat_header_list:
        raise ValueError(
            'Threats "%s" does not match any column in the sensitivity '
            'table. Sensitivity columns: %s' % 
            (missing_threat_header_list, sens_header_list))
                
    # Get the directory path for the Threats CSV, used for locating threat 
    # rasters, which are relative to this path
    threat_csv_basepath = os.path.dirname(args['threats_table_path'])

    # get the half saturation constant
    try:
        half_saturation_constant = float(args['half_saturation_constant'])
    except ValueError:
        raise ValueError('Half-saturation constant is not a numeric number.'
                         'It is: %s' % args['half_saturation_constant'])

    # declare dictionaries to store the land cover and the threat rasters
    # pertaining to the different threats
    lulc_path_dict = {}
    threat_path_dict = {}
    # store land cover and threat rasters in a list for convenient access
    lulc_and_threat_raster_list = []
    # lookup for the unique lucode tasks
    unique_lucode_lookup = []
    unique_lucode_pickle_lookup = []
    LOGGER.info("Find threat rasters and collect unique LULC codes")
    # compile all the threat rasters associated with the land cover
    for lulc_key, lulc_args in (('_c', 'lulc_cur_path'),
                                ('_f', 'lulc_fut_path'),
                                ('_b', 'lulc_bas_path')):
        if lulc_args in args:
            lulc_path = args[lulc_args]
            lulc_path_dict[lulc_key] = lulc_path
            # save land cover paths in a list for alignment and resize
            lulc_and_threat_raster_list.append(lulc_path)

            unique_lucode_pickle_path = os.path.join(
                intermediate_output_dir, f'unique_lulc{lulc_key}.pickle')
            # save unique codes to check if it's missing in sensitivity table
            unique_lucode_task = graph.add_task(
                    _collect_unique_lucodes,
                    args=((lulc_path, 1), unique_lucode_pickle_path),
                    target_path_list=[unique_lucode_pickle_path],
                    task_name=f'unique_lucodes{lulc_key}')
            unique_lucode_lookup.append(unique_lucode_task)
            unique_lucode_pickle_lookup.append(unique_lucode_pickle_path)

            # add a key to the threat dictionary that associates all threat
            # rasters with this land cover
            threat_path_dict['threat' + lulc_key] = {}

            # for each threat given in the CSV file try opening the associated
            # raster which should be found relative to the Threat CSV 
            for threat in threat_dict:
                # Threat path from threat CSV is relative to CSV
                threat_path = os.path.join(
                        threat_csv_basepath, 
                        threat_dict[threat][_THREAT_SCENARIO_MAP[lulc_key]]) 

                # it's okay to have no threat raster for baseline scenario
                validated_threat_path = _resolve_threat_raster_path(threat_path)
                    
                if validated_threat_path is None:
                    if lulc_key != '_b':
                        raise ValueError(
                            'There was an Error locating a threat raster from '
                            'the path in CSV for column: '
                            f'{_THREAT_SCENARIO_MAP[lulc_key]} and threat: '
                            f'{threat}. The path in the CSV column should be '
                            'relative to the threat CSV table.')

                threat_path_dict['threat' + lulc_key][threat] = (
                        validated_threat_path)
                # save threat paths in a list for alignment and resize
                if validated_threat_path:
                    # check for duplicate absolute threat path names that 
                    # cause errors when trying to write aligned versions
                    if not validated_threat_path in lulc_and_threat_raster_list:
                        lulc_and_threat_raster_list.append(validated_threat_path)
                    else:
                        raise ValueError(
                            'Threat paths cannot be the same and must have '
                            'unique absolute filepaths. The threat path: '
                            f'{os.path.basename(validated_threat_path)} is a '
                            'duplicate.')

    LOGGER.info("Checking LULC codes against Sensitivity table")
    compare_lucodes_sens_task = graph.add_task(
            _compare_lucodes_sensitivity,
            args=(sensitivity_dict, unique_lucode_pickle_lookup),
            dependent_task_list=[*unique_lucode_lookup],
            task_name='lucode_sens_comparison')

    # don't continue if the compare_lucodes_sens_task throws error
    compare_lucodes_sens_task.join()

    LOGGER.info('Starting to align and resize land cover and threat rasters')
    lulc_raster_info = pygeoprocessing.get_raster_info(args['lulc_cur_path'])
    lulc_pixel_size = lulc_raster_info['pixel_size']
    lulc_bbox = lulc_raster_info['bounding_box']

    # create paths for aligned rasters checking for the case the raster path
    # is a folder
    aligned_raster_list = []
    for path in lulc_and_threat_raster_list:
        ext = os.path.splitext(path)[1]
        if not ext:
            threat_dir_name = os.path.basename(os.path.dirname(path))
            aligned_raster_list.append(
                os.path.join(intermediate_output_dir, 
                    f'{threat_dir_name}_aligned{file_suffix}.tif'))
        else:
            aligned_raster_list.append(
                os.path.join(intermediate_output_dir, 
                    os.path.basename(path).replace(
                        ext, f'_aligned{file_suffix}.tif')))

    LOGGER.debug(f"aligned raster list: {aligned_raster_list}")
    # Align and resize all the land cover and threat rasters,
    # and store them in the intermediate folder
    align_task = graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(
            lulc_and_threat_raster_list, aligned_raster_list,
            ['near']*len(lulc_and_threat_raster_list), lulc_pixel_size,
            lulc_bbox),
        target_path_list=aligned_raster_list,
        task_name='align_input_rasters')

    LOGGER.info('Finished aligning and resizing land cover and threat rasters')
    # list of tracking updated threat pixel rasters
    updated_threat_tasks = []
    threat_pixels_update_lookup = []
    LOGGER.debug("Updating dict raster paths to reflect aligned paths")
    # Modify paths in lulc_path_dict and threat_path_dict to be aligned rasters
    for lulc_key, lulc_path in lulc_path_dict.items():
        lulc_path_dict[lulc_key] = os.path.join(
            intermediate_output_dir, os.path.basename(lulc_path).replace(
                os.path.splitext(lulc_path)[1], 
                f'_aligned{file_suffix}.tif'))
        for threat in threat_dict:
            threat_path = threat_path_dict['threat' + lulc_key][threat]
            if threat_path in lulc_and_threat_raster_list:
                aligned_threat_path = os.path.join(
                    intermediate_output_dir, os.path.basename(threat_path).replace(
                        os.path.splitext(threat_path)[1], 
                        f'_aligned{file_suffix}.tif'))
                aligned_updated_threat_path = os.path.join(
                    intermediate_output_dir,
                    os.path.basename(aligned_threat_path).replace(
                        '.tif', '_updated.tif'))
                # Use these updated threat raster paths in future calculations
                threat_path_dict['threat' + lulc_key][threat] = (
                    aligned_updated_threat_path)

                # Update threat raster pixel values as needed so that:
                #  * Nodata values are replaced with 0
                #  * Anything other than 0 or nodata is replaced with 1
                update_threat_task = graph.add_task(
                    _update_threat_pixels,
                    args=(aligned_threat_path, aligned_updated_threat_path),
                    target_path_list=[aligned_updated_threat_path],
                    dependent_task_list=[align_task],
                    task_name=f'update_threat{lulc_key}_{threat}')
                updated_threat_tasks.append(update_threat_task)

    LOGGER.info('Starting habitat_quality biophysical calculations')

    # Rasterize access vector, if value is null set to 1 (fully accessible),
    # else set to the value according to the ACCESS attribute
    cur_lulc_path = lulc_path_dict['_c']
    fill_value = 1.0
    try:
        LOGGER.info('Handling Access Shape')
        access_raster_path = os.path.join(
            intermediate_output_dir, 'access_layer%s.tif' % file_suffix)

        # create a new raster based on the raster info of current land cover
        access_base_task = graph.add_task(
            pygeoprocessing.new_raster_from_base,
            args=(cur_lulc_path, access_raster_path, gdal.GDT_Float32,
                [_OUT_NODATA]),
            kwargs={
                'fill_value_list': [fill_value]
                },
            target_path_list=[access_raster_path],
            dependent_task_list=[align_task],
            task_name=f'access_raster')

        rasterize_access_task = graph.add_task(
            pygeoprocessing.rasterize,
            args=(args['access_vector_path'], access_raster_path),
            kwargs={
                'option_list': ['ATTRIBUTE=ACCESS'],
                'burn_values': None
                },
            target_path_list=[access_raster_path],
            dependent_task_list=[access_base_task],
            task_name=f'rasterize_access')
    except KeyError:
        LOGGER.info(
            'No Access Shape Provided, access raster filled with 1s.')

    # calculate the weight sum which is the sum of all the threats' weights
    weight_sum = 0.0
    for threat_data in threat_dict.values():
        # Sum weight of threats
        weight_sum = weight_sum + threat_data['WEIGHT']

    LOGGER.debug('lulc_path_dict : %s', lulc_path_dict)

    # for each land cover raster provided compute habitat quality
    for lulc_key, lulc_path in lulc_path_dict.items():
        LOGGER.info('Calculating habitat quality for landuse: %s', lulc_path)

        habitat_raster_lookup = []
        threat_convolve_lookup = []
        sensitivity_lookup = []

        # Create raster of habitat based on habitat field
        habitat_raster_path = os.path.join(
            intermediate_output_dir, 'habitat%s%s.tif' % (lulc_key, file_suffix))

        habitat_raster_task = graph.add_task(
                _map_raster_to_dict_values,
                args=(lulc_path, habitat_raster_path, sensitivity_dict,
                    'HABITAT', _OUT_NODATA),
                kwargs={
                    'values_required': False
                    },
                dependent_task_list=[align_task],
                task_name=f'habitat_raster{lulc_key}')
        habitat_raster_lookup.append(habitat_raster_task)

        # initialize a list that will store all the threat/threat rasters
        # after they have been adjusted for distance, weight, and access
        deg_raster_list = []

        # a list to keep track of the normalized weight for each threat
        weight_list = numpy.array([])

        # variable to indicate whether we should break out of calculations
        # for a land cover because a threat raster was not found
        exit_landcover = False

        # adjust each threat/threat raster for distance, weight, and access
        for threat, threat_data in threat_dict.items():
            LOGGER.debug('Calculating threat: %s.\nThreat data: %s' %
                        (threat, threat_data))

            # get the threat raster for the specific threat
            threat_raster_path = threat_path_dict['threat' + lulc_key][threat]
            LOGGER.debug('threat_raster_path %s', threat_raster_path)
            # if threat path is None then must be in Base scenario where 
            # threats are not required.
            if threat_raster_path is None:
                LOGGER.warning(
                    'The threat raster for %s could not be found for the land '
                    'cover %s. Skipping Habitat Quality calculation for this '
                    'land cover.' % (threat, lulc_key))
                exit_landcover = True
                break

            kernel_path = os.path.join(
                kernel_dir, 'kernel_%s%s%s.tif' % (threat, lulc_key, file_suffix))

            decay_type = threat_data['DECAY']

            decay_threat_task = graph.add_task(
                _decay_threat,
                args=(threat_raster_path, kernel_path, decay_type, 
                    threat_data['MAX_DIST']),
                target_path_list=[kernel_path],
                dependent_task_list=[*updated_threat_tasks],
                task_name=f'decay_kernel_{decay_type}{lulc_key}_{threat}')

            filtered_threat_raster_path = os.path.join(
                intermediate_output_dir, 
                'filtered_%s%s%s.tif' % (threat, lulc_key, file_suffix))

            convolve_task = graph.add_task(
                pygeoprocessing.convolve_2d,
                args=((threat_raster_path, 1), (kernel_path, 1),
                    filtered_threat_raster_path),
                kwargs={
                    'ignore_nodata': True
                    },
                target_path_list=[filtered_threat_raster_path],
                dependent_task_list=[*updated_threat_tasks, decay_threat_task],
                task_name=f'convolve_{decay_type}_{lulc_key}_{threat}')
            threat_convolve_lookup.append(convolve_task)

            # create sensitivity raster based on threat
            sens_raster_path = os.path.join(
                intermediate_output_dir, 
                'sens_%s%s%s.tif' % (threat, lulc_key, file_suffix))

            sens_threat_task = graph.add_task(
                _map_raster_to_dict_values,
                args=(lulc_path, sens_raster_path, sensitivity_dict,
                    'L_' + threat, _OUT_NODATA),
                kwargs={
                    'values_required': True
                    },
                target_path_list=[sens_raster_path],
                dependent_task_list=[align_task],
                task_name=f'sens_raster_{decay_type}_{lulc_key}_{threat}')
            sensitivity_lookup.append(sens_threat_task)

            # get the normalized weight for each threat
            weight_avg = threat_data['WEIGHT'] / weight_sum

            # add the threat raster adjusted by distance and the raster
            # representing sensitivity to the list to be past to
            # vectorized_rasters below
            deg_raster_list.append(filtered_threat_raster_path)
            deg_raster_list.append(sens_raster_path)

            # store the normalized weight for each threat in a list that
            # will be used below in total_degradation
            weight_list = numpy.append(weight_list, weight_avg)

        # check to see if we got here because a threat raster was missing
        # and if so then we want to skip to the next landcover
        if exit_landcover:
            continue

        # add the access_raster onto the end of the collected raster list. The
        # access_raster will be values from the shapefile if provided or a
        # raster filled with all 1's if not
        deg_raster_list.append(access_raster_path)

        deg_sum_raster_path = os.path.join(
            output_dir, f'deg_sum{lulc_key}{file_suffix}.tif')

        LOGGER.info('Starting raster calculation on total_degradation')

        total_degradation_task = graph.add_task(
            _calculate_total_degradation,
            args=(deg_raster_list, deg_sum_raster_path, weight_list),
            target_path_list=[deg_sum_raster_path],
            dependent_task_list=[*threat_convolve_lookup,
                *sensitivity_lookup, access_base_task],
            task_name=f'tot_degradation_{decay_type}{lulc_key}_{threat}')

        LOGGER.info('Finished raster calculation on total_degradation')

        # Compute habitat quality
        # ksq: a term used below to compute habitat quality
        ksq = half_saturation_constant**_SCALING_PARAM

        quality_path = os.path.join(
            output_dir, f'quality{lulc_key}{file_suffix}.tif')

        LOGGER.info('Starting raster calculation on quality_op')

        deg_hab_raster_list = [deg_sum_raster_path, habitat_raster_path]

        hq_task = graph.add_task(
            _calculate_habitat_quality,
            args=(deg_hab_raster_list, quality_path, ksq),
            target_path_list=[quality_path],
            dependent_task_list=[*habitat_raster_lookup,
                total_degradation_task],
            task_name=f'habitat_quality')

        LOGGER.info('Finished raster calculation on quality_op')

    # Compute Rarity if user supplied baseline raster
    if '_b' not in lulc_path_dict:
        LOGGER.info('Baseline not provided to compute Rarity')
    else:
        lulc_base_path = lulc_path_dict['_b']

        # compute rarity for current landscape and future (if provided)
        for lulc_key in ['_c', '_f']:
            if lulc_key not in lulc_path_dict:
                continue
            lulc_path = lulc_path_dict[lulc_key]
            lulc_time = 'current' if lulc_key == '_c' else 'future'

            new_cover_path = os.path.join(
                intermediate_output_dir, f'new_cover{lulc_key}{file_suffix}.tif')

            rarity_path = os.path.join(
                output_dir, f'rarity{lulc_key}{file_suffix}.tif')

            rarity_task = graph.add_task(
                _compute_rarity_operation,
                args=(lulc_base_path, lulc_path, new_cover_path, rarity_path),
                dependent_task_list=[align_task],
                task_name=f'rarity{lulc_time}')

        LOGGER.info('Finished habitat_quality biophysical calculations')

    graph.join()


def _calculate_habitat_quality(deg_hab_raster_list, quality_out_path, ksq):
    """Calculate habitat quality from degradation inputs.

    Parameters:
        deg_hab_raster_list (list): list of string paths for the degraded
            habitat rasters.
        quality_out_path (string): path to output the habitat quality raster.
        ksq (float): a number representing half-saturation**_SCALING_PARAM
    Returns:
        None
    """
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
                (degradation == _OUT_NODATA) | (habitat == _OUT_NODATA),
                _OUT_NODATA,
                (habitat * (1.0 - ((degredataion_clamped**_SCALING_PARAM) /
                 (degredataion_clamped**_SCALING_PARAM + ksq)))))

    deg_hab_raster_band_list = [
        (path, 1) for path in deg_hab_raster_list]

    pygeoprocessing.raster_calculator(deg_hab_raster_band_list, quality_op,
        quality_out_path, gdal.GDT_Float32, _OUT_NODATA)


def _calculate_total_degradation(deg_raster_list, deg_sum_raster_path, 
        weight_list):
    """Calculate habitat degradation.

    Parameters:
        deg_raster_list (list): list of string paths for the degraded
            threat rasters.
        deg_sum_raster_path (string): path to output the habitat quality 
            degradation raster.
        weight_list (list): normalized weight for each threat corresponding 
            to threats in ``deg_raster_list``.
    Returns:
        None
    """
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
        for index in range(len(raster) // 2):
            step = index * 2
            sum_degradation += (
                raster[step] * raster[step + 1] * weight_list[index])

        nodata_mask = numpy.empty(raster[0].shape, dtype=numpy.int8)
        nodata_mask[:] = 0
        for array in raster:
            nodata_mask = nodata_mask | (array == _OUT_NODATA)

        # the last element in raster is access
        return numpy.where(
                nodata_mask, _OUT_NODATA, sum_degradation * raster[-1])

    deg_raster_band_list = [(path, 1) for path in deg_raster_list]

    pygeoprocessing.raster_calculator(deg_raster_band_list, total_degradation,
            deg_sum_raster_path, gdal.GDT_Float32, _OUT_NODATA)

def _compute_rarity_operation(lulc_base_path, lulc_path, new_cover_path, 
        rarity_path):
    """Calculate habitat rarity.

    Parameters:
        lulc_base_path (string): path to input base LULC raster.
        lulc_path (string): path to LULC for current or future scenario.
        new_cover_path (string): path to intermediate raster file for trimming
            ``lulc_path`` to ``lulc_base_path``.
        rarity_path (string): path to output rarity raster.
    Returns:
        None
    """
    # get the area of a base pixel to use for computing rarity where the
    # pixel sizes are different between base and cur/fut rasters
    base_pixel_size = pygeoprocessing.get_raster_info(
        lulc_base_path)['pixel_size']
    base_area = float(abs(base_pixel_size[0]) * abs(base_pixel_size[1]))
    base_nodata = pygeoprocessing.get_raster_info(
        lulc_base_path)['nodata'][0]

    lulc_code_count_b = _raster_pixel_count(lulc_base_path)

    # get the area of a cur/fut pixel
    lulc_pixel_size = pygeoprocessing.get_raster_info(
        lulc_path)['pixel_size']
    lulc_area = float(abs(lulc_pixel_size[0]) * abs(lulc_pixel_size[1]))
    lulc_nodata = pygeoprocessing.get_raster_info(
        lulc_path)['nodata'][0]

    def trim_op(base, cover_x):
        """Trim cover_x to the mask of base.

        Parameters:
            base (numpy.ndarray): base raster from 'lulc_base'
            cover_x (numpy.ndarray): either future or current land
                cover raster from 'lulc_path' above

        Returns:
            _OUT_NODATA where either array has nodata, otherwise
            cover_x.
        """
        return numpy.where(
            (base == base_nodata) | (cover_x == lulc_nodata),
            base_nodata, cover_x)

    LOGGER.info('Starting masking %s land cover to base land cover.'
                % os.path.basename(lulc_path))

    pygeoprocessing.raster_calculator(
        [(lulc_base_path, 1), (lulc_path, 1)], trim_op, new_cover_path,
        gdal.GDT_Float32, _OUT_NODATA)

    LOGGER.info('Finished masking %s land cover to base land cover.'
                % os.path.basename(lulc_path))

    LOGGER.info('Starting rarity computation on %s land cover.'
                % os.path.basename(lulc_path))

    lulc_code_count_x = _raster_pixel_count(new_cover_path)

    # a dictionary to map LULC types to a number that depicts how
    # rare they are considered
    code_index = {}

    # compute rarity index for each lulc code
    # define 0.0 if an lulc code is found in the cur/fut landcover
    # but not the baseline
    for code in lulc_code_count_x:
        if code in lulc_code_count_b:
            numerator = lulc_code_count_x[code] * lulc_area
            denominator = lulc_code_count_b[code] * base_area
            ratio = 1.0 - (numerator / denominator)
            code_index[code] = ratio
        else:
            code_index[code] = 0.0

    pygeoprocessing.reclassify_raster(
        (new_cover_path, 1), code_index, rarity_path, gdal.GDT_Float32,
            _RARITY_NODATA)

    LOGGER.info('Finished rarity computation on %s land cover.'
                % os.path.basename(lulc_path))


def _decay_threat(threat_raster_path, kernel_path, decay_type, max_dist): 
    """Create a decay kernel as a raster.

    Parameters:
        threat_raster_path (string): path to threat raster to decay.
        kernel_path (string): path to output kernel raster.
        decay_type (string): type of decay kernel to create, either 
            'linear' | 'exponentional'.
        max_dist (float): max distance of threat in KM.
    Returns:
        None
    """
    # need the pixel size for the threat raster so we can create
    # an appropriate kernel for convolution
    threat_pixel_size = pygeoprocessing.get_raster_info(
        threat_raster_path)['pixel_size']
    # pixel size tuple could have negative value
    mean_threat_pixel_size = (
        abs(threat_pixel_size[0]) + abs(threat_pixel_size[1]))/2.0

    # convert max distance (given in KM) to meters
    max_dist_m = max_dist * 1000.0

    # convert max distance from meters to the number of pixels that
    # represents on the raster
    max_dist_pixel = max_dist_m / mean_threat_pixel_size
    LOGGER.debug('Max distance in pixels: %f', max_dist_pixel)

    # blur the threat raster based on the effect of the threat over
    # distance
    if decay_type == 'linear':
        decay_func = _make_linear_decay_kernel_path
    elif decay_type == 'exponential':
        decay_func = utils.exponential_decay_kernel_raster
    else:
        raise ValueError(
            "Unknown type of decay in biophysical table, should be "
            "either 'linear' or 'exponential'. Input was %s for threat"
            " %s." % (decay_type, 
                os.path.splitext(os.path.basename(threat_raster_path))[0]))

    decay_func(max_dist_pixel, kernel_path)

def _compare_lucodes_sensitivity(sensitivity_dict, unique_lucode_pickle_lookup):
    """Compare LULC unique codes to codes listed in dictionary.

    Parameters:
        sensitivity_dict (dict): dictionary representing sensitivity for LULC
            values with a column of 'codes'.
        unique_lucode_pickle_lookup (list): list of paths to pickle files that
            represent a python set of unique LULC values for the LULC rasters
            provided in the model.
    Returns:
        None
    """
    raster_unique_lucodes = set()
    for lucode_path in unique_lucode_pickle_lookup:
        with open(lucode_path, 'rb') as fh:
            lucode_dict = pickle.load(fh)
        raster_unique_lucodes.update(lucode_dict['codes'])

    # check if there's any lucode from the LULC rasters missing in the
    # sensitivity table
    table_unique_lucodes = set(sensitivity_dict.keys())
    missing_lucodes = raster_unique_lucodes.difference(table_unique_lucodes)
    if missing_lucodes:
        raise ValueError(
            'The following land cover codes were found in your landcover rasters '
            'but not in your sensitivity table. Check your sensitivity table '
            'to see if they are missing: %s. \n\n' %
            ', '.join([str(x) for x in sorted(missing_lucodes)]))


def _collect_unique_lucodes(raster_path_band, pickle_path):
    """Get unique pixel values from raster and pickle as a Python set.

    Parameters:
        raster_path_band (tuple): a 2 tuple of the form 
            (filepath to raster, band index).
        pickle_path (string): a path to output of a pickled Python set.
    Returns:
        None
    """
    lulc_path = raster_path_band[0]
    # declare a set to store unique codes from lulc rasters
    raster_unique_lucodes = set()

    for _, lulc_block in pygeoprocessing.iterblocks(raster_path_band):
        raster_unique_lucodes.update(numpy.unique(lulc_block))

    # Remove the nodata value from the set of landuser codes.
    nodata = pygeoprocessing.get_raster_info(lulc_path)['nodata'][0]
    try:
        raster_unique_lucodes.remove(nodata)
    except KeyError:
        # KeyError when the nodata value was not encountered in the
        # raster's pixel values.  Same result when nodata value is
        # None.
        pass

    data = {'codes': raster_unique_lucodes}
    with open(pickle_path, 'wb') as fh:
        pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)


def _resolve_threat_raster_path(path):
    """Determine if path is a valid gdal raster file.

    Parameters:
        path (string): file path.

    Return:
        ``path`` if valid raster, otherwise ``None``.
    """
    # Turning on exceptions so that if an error occurs when trying to open a
    # file path we can catch it and handle it properly

    def _error_handler(*_):
        """A dummy error handler that raises a ValueError."""
        raise ValueError()
    gdal.PushErrorHandler(_error_handler)

    # initialize dataset to None in the case that path does not exist
    dataset = None
    if os.path.exists(path):
        try:
            dataset = gdal.OpenEx(path, gdal.OF_RASTER | gdal.GA_ReadOnly)
        except ValueError:
            # If GDAL can't open the raster, our GDAL error handler will be
            # executed and ValueError raised.
            pass

    gdal.PopErrorHandler()

    # If a dataset comes back None, then it could not be found, set path 
    # to None and return
    if dataset is None:
        path = None

    dataset = None
    return path


def _raster_pixel_count(raster_path):
    """Count unique pixel values in raster.

    Parameters:
        raster_path (string): path to a raster

    Returns:
        dict of pixel values to frequency.
    """
    nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
    counts = collections.defaultdict(int)
    for _, raster_block in pygeoprocessing.iterblocks((raster_path, 1)):
        for value, count in zip(
                *numpy.unique(raster_block, return_counts=True)):
            if value == nodata:
                continue
            counts[value] += count
    return counts


def _map_raster_to_dict_values(
        key_raster_path, out_path, attr_dict, field, out_nodata, values_required):
    """Creates a new raster from 'key_raster' where the pixel values from
       'key_raster' are the keys to a dictionary 'attr_dict'. The values
       corresponding to those keys is what is written to the new raster. If a
       value from 'key_raster' does not appear as a key in 'attr_dict' then
       raise an Exception if 'raise_error' is True, otherwise return a
       'out_nodata'

       key_raster_path - a GDAL raster path dataset whose pixel values relate to
                     the keys in 'attr_dict'
       out_path - a string for the output path of the created raster
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

    LOGGER.info('Starting map_raster_to_dict_values')
    int_attr_dict = {}
    for key in attr_dict:
        int_attr_dict[int(key)] = float(attr_dict[key][field])

    pygeoprocessing.reclassify_raster(
        (key_raster_path, 1), int_attr_dict, out_path, gdal.GDT_Float32,
        out_nodata, values_required)


def _make_linear_decay_kernel_path(max_distance, kernel_path):
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

    col_index = numpy.array(range(kernel_size))
    integration = 0.0
    for row_index in range(kernel_size):
        distance_kernel_row = numpy.sqrt(
            (row_index - max_distance) ** 2 +
            (col_index - max_distance) ** 2).reshape(1, kernel_size)
        kernel = numpy.where(
            distance_kernel_row > max_distance, 0.0,
            (max_distance - distance_kernel_row) / max_distance)
        integration += numpy.sum(kernel)
        kernel_band.WriteArray(kernel, xoff=0, yoff=row_index)

    for row_index in range(kernel_size):
        kernel_row = kernel_band.ReadAsArray(
            xoff=0, yoff=row_index, win_xsize=kernel_size, win_ysize=1)
        kernel_row /= integration
        kernel_band.WriteArray(kernel_row, 0, row_index)


def _update_threat_pixels(aligned_threat_path, updated_pixel_threat_path):
    """Update raster's nodata values.

    Iterate though the threat raster and update pixel values as needed so that:
      * Nodata values are replaced with 0
      * Anything other than 0 or nodata is replaced with 1

    Parameters:
        aligned_threat_path (string): a path to the threat raster on disk.
        updated_pixel_threat_path (string): an output path for the updated
            raster.
    Returns:
        None
    """
    LOGGER.info('Preprocessing threat values for %s',
                aligned_threat_path)
    threat_nodata = pygeoprocessing.get_raster_info(
        aligned_threat_path)['nodata'][0]
    
    if threat_nodata is None:
        raise TypeError(
            'Raster NODATA value for Threat path'
            f' [ {os.path.basename(aligned_threat_path)} ] is UNDEFINED.'
            ' Please make sure each threat raster has a defined NODATA value.')
    
    threat_datatype = pygeoprocessing.get_raster_info(
        aligned_threat_path)['datatype']

    def _update_op(block):
        """ """
        # First check if we actually need to set anything.
        # No need to perform unnecessary writes!
        if set(numpy.unique(block)) == set([0, 1]):
            return block
        
        zero_threat = numpy.isclose(block, threat_nodata)
        block[zero_threat] = 0
        block[~numpy.isclose(block, 0)] = 1

        return block
    
    pygeoprocessing.raster_calculator(
        [(aligned_threat_path, 1)], _update_op, updated_pixel_threat_path,
        threat_datatype, threat_nodata)


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
    validation_warnings = validation.validate(
        args, ARGS_SPEC['args'], ARGS_SPEC['args_with_spatial_overlap'])

    invalid_keys = validation.get_invalid_keys(validation_warnings)
    
    if ("threats_table_path" not in invalid_keys and 
            "sensitivity_table_path" not in invalid_keys and
            "threat_raster_folder" not in invalid_keys):
        
        # Get CSVs as dictionaries and ensure the key is a string for threats.
        threat_dict = {
            str(key): value for key, value in utils.build_lookup_from_csv(
                args['threats_table_path'], 'THREAT', to_lower=False).items()}
        sensitivity_dict = utils.build_lookup_from_csv(
            args['sensitivity_table_path'], 'LULC', to_lower=False)

        # check that the threat names in the threats table match with the threats
        # columns in the sensitivity table. 
        sens_header_list = list(sensitivity_dict.values())[0]
        missing_sens_header_list = []
        for threat in threat_dict:
            if 'L_' + threat not in sens_header_list:
                missing_sens_header_list.append(threat)
        
        if missing_sens_header_list:
            validation_warnings.append((
                ['sensitivity_table_path'],
                (f'Threats "{missing_sens_header_list}" does not match any'
                  ' column in the sensitivity table. Sensitivity columns:'
                 f' {sens_header_list}')))
                
            invalid_keys.add('sensitivity_table_path')
   

        # Get the directory path for the Threats CSV, used for locating threat 
        # rasters, which are relative to this path
        threat_csv_basepath = os.path.dirname(args['threats_table_path'])
        
        # Validate threat raster paths and their nodata values
        bad_threat_paths = []
        bad_threat_nodatas = []
        duplicate_paths = []
        threat_paths = []
        for lulc_key, lulc_args in (('_c', 'lulc_cur_path'),
                                    ('_f', 'lulc_fut_path'),
                                    ('_b', 'lulc_bas_path')):
            if lulc_args in args:
                # for each threat given in the CSV file try opening the associated
                # raster which should be found in threat_raster_folder
                for threat in threat_dict:
                    # Threat path from threat CSV is relative to CSV
                    threat_path = os.path.join(
                            threat_csv_basepath, 
                            threat_dict[threat][_THREAT_SCENARIO_MAP[lulc_key]]) 

                    validated_threat_path = _resolve_threat_raster_path(
                            threat_path)
                    # it's okay to have no threat raster for baseline scenario
                    if lulc_key != '_b' and validated_threat_path is None:
                        bad_threat_paths.append(
                                (threat, _THREAT_SCENARIO_MAP[lulc_key]))
                        continue
                    
                    if validated_threat_path:
                        # check for duplicate absolute threat path names that 
                        # cause errors when trying to write aligned versions
                        if not validated_threat_path in threat_paths:
                            threat_paths.append(validated_threat_path)
                        else:
                            duplicate_paths.append(
                                os.path.basename(validated_threat_path))
                        
                        # Check NODATA value of the valid threat raster
                        threat_raster_info = pygeoprocessing.get_raster_info(
                                                validated_threat_path)
                        if threat_raster_info['nodata'][0] is None:
                            bad_threat_nodatas.append(threat)
        
        if bad_threat_paths:
            validation_warnings.append((
                ['threats_table_path'],
                (f'A threat raster for threats: {bad_threat_paths}'
                  ' was not found or it could not be opened by GDAL.')))
 
            invalid_keys.add('threats_table_path')

        if duplicate_paths:
            validation_warnings.append((
                ['threats_table_path'],
                ('Threat paths cannot be the same and must have unique '
                f'absolute filepaths. The threat paths: {duplicate_paths} were '
                'duplicates.')))

            if not 'threats_table_path' in invalid_keys:
                invalid_keys.add('threats_table_path')

        if bad_threat_nodatas:
            validation_warnings.append((
                ['threats_table_path'],
                (f'A threat raster for threats: {bad_threat_nodatas}'
                  ' has an undefined NODATA value. Please define' 
                  ' the NODATA value for all threat rasters.')))
            
            if not 'threats_table_path' in invalid_keys:
                invalid_keys.add('threats_table_path')
    
    return validation_warnings
