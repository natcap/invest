"""InVEST Nutrient Delivery Ratio (NDR) module."""
import logging
import os

from osgeo import gdal
from osgeo import ogr
import numpy

import pygeoprocessing
import pygeoprocessing.routing
import pygeoprocessing.routing.routing_core

from .. import utils
import ndr_core

LOGGER = logging.getLogger('natcap.invest.ndr.ndr')
logging.basicConfig(
    format='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
    level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

_OUTPUT_BASE_FILES = {
    'n_export_path': 'n_export.tif',
    'p_export_path': 'p_export.tif',
    'watershed_results_ndr_path': 'watershed_results_ndr.shp',
    }

_INTERMEDIATE_BASE_FILES = {
    'ic_factor_path': 'ic_factor.tif',
    'load_n_path': 'load_n.tif',
    'load_p_path': 'load_p.tif',
    'modified_load_n_path': 'modified_load_n.tif',
    'modified_load_p_path': 'modified_load_p.tif',
    'modified_sub_load_n_path': 'modified_sub_load_n.tif',
    'modified_sub_load_p_path': 'modified_sub_load_p.tif',
    'ndr_n_path': 'ndr_n.tif',
    'ndr_p_path': 'ndr_p.tif',
    'runoff_proxy_index_path': 'runoff_proxy_index.tif',
    's_accumulation_path': 's_accumulation.tif',
    's_bar_path': 's_bar.tif',
    's_factor_inverse_path': 's_factor_inverse.tif',
    'stream_path': 'stream.tif',
    'sub_crit_len_n_path': 'sub_crit_len_n.tif',
    'sub_crit_len_p_path': 'sub_crit_len_p.tif',
    'sub_eff_n_path': 'sub_eff_n.tif',
    'sub_eff_p_path': 'sub_eff_p.tif',
    'sub_effective_retention_n_path': 'sub_effective_retention_n.tif',
    'sub_effective_retention_p_path': 'sub_effective_retention_p.tif',
    'sub_load_n_path': 'sub_load_n.tif',
    'sub_load_p_path': 'sub_load_p.tif',
    'sub_ndr_n_path': 'sub_ndr_n.tif',
    'sub_ndr_p_path': 'sub_ndr_p.tif',
    'crit_len_n_path': 'crit_len_n.tif',
    'crit_len_p_path': 'crit_len_p.tif',
    'd_dn_path': 'd_dn.tif',
    'd_up_path': 'd_up.tif',
    'eff_n_path': 'eff_n.tif',
    'eff_p_path': 'eff_p.tif',
    'effective_retention_n_path': 'effective_retention_n.tif',
    'effective_retention_p_path': 'effective_retention_p.tif',
    'flow_accumulation_path': 'flow_accumulation.tif',
    'flow_direction_path': 'flow_direction.tif',
    'slope_path': 'slope.tif',
    'thresholded_slope_path': 'thresholded_slope.tif',
    'processed_cell_path': 'processed_cell.tif',
    }

_TMP_BASE_FILES = {
    'loss_path': 'loss.tif',
    'aligned_dem_path': 'aligned_dem.tif',
    'aligned_lulc_path': 'aligned_lulc.tif',
    'aligned_runoff_proxy_path': 'aligned_runoff_proxy.tif',
    'zero_absorption_source_path': 'zero_absorption_source.tif',
    }


def execute(args):
    """Nutrient Delivery Ratio.

    Parameters:
        args['workspace_dir'] (string):  path to current workspace
        args['dem_path'] (string): path to digital elevation map raster
        args['lulc_path'] (string): a path to landcover map raster
        args['runoff_proxy_path'] (string): a path to a runoff proxy raster
        args['watersheds_path'] (string): path to the watershed shapefile
        args['biophysical_table_path'] (string): path to csv table on disk
            containing nutrient retention values.

            For each nutrient type [t] in args['calc_[t]'] that is true, must
            contain the following headers:

            'load_[t]', 'eff_[t]', 'crit_len_[t]'

            If args['calc_n'] is True, must also contain the header
            'proportion_subsurface_n' field.

        args['calc_p'] (boolean): if True, phosphorous is modeled,
            additionally if True then biophysical table must have p fields in
            them
        args['calc_n'] (boolean): if True nitrogen will be modeled,
            additionally biophysical table must have n fields in them.
        args['results_suffix'] (string): (optional) a text field to append to
            all output files
        args['threshold_flow_accumulation']: a number representing the flow
            accumulation in terms of upstream pixels.
        args['_prepare']: (optional) The preprocessed set of data created by
            the ndr._prepare call.  This argument could be used in cases where
            the call to this function is scripted and can save a significant
            amount DEM processing runtime.

    Returns:
        None
    """
    def _validate_inputs(nutrients_to_process, lucode_to_parameters):
        """Validate common errors in inputs.

        Parameters:
            nutrients_to_process (list): list of 'n' and/or 'p'
            lucode_to_parameters (dictionary): biophysical input table mapping
                lucode to dictionary of table parameters.  Used to validate
                the correct columns are input

        Returns:
            None

        Raises:
            ValueError whenever a missing field in the parameter table is
            detected along with a message describing every missing field.
        """
        # Make sure all the nutrient inputs are good
        if len(nutrients_to_process) == 0:
            raise ValueError("Neither phosphorous nor nitrogen was selected"
                             " to be processed.  Choose at least one.")

        # Build up a list that'll let us iterate through all the input tables
        # and check for the required rows, and report errors if something
        # is missing.
        row_header_table_list = []

        lu_parameter_row = lucode_to_parameters.values()[0]
        row_header_table_list.append(
            (lu_parameter_row, ['load_', 'eff_', 'crit_len_'],
             args['biophysical_table_path']))

        missing_headers = []
        for row, header_prefixes, table_type in row_header_table_list:
            for nutrient_id in nutrients_to_process:
                for header_prefix in header_prefixes:
                    header = header_prefix + nutrient_id
                    if header not in row:
                        missing_headers.append(
                            "Missing header %s from %s" % (
                                header, table_type))

        # proportion_subsurface_n is a special case in which phosphorous does
        # not have an equivalent.
        if ('n' in nutrients_to_process and
                'proportion_subsurface_n' not in lu_parameter_row):
            missing_headers.append(
                "Missing header proportion_subsurface_n from " +
                args['biophysical_table_path'])

        if len(missing_headers) > 0:
            raise ValueError('\n'.join(missing_headers))

    # Load all the tables for preprocessing
    output_dir = os.path.join(args['workspace_dir'])
    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    output_dir = os.path.join(args['workspace_dir'])
    pygeoprocessing.create_directories(
        [output_dir, intermediate_output_dir])

    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    f_reg = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir),
         (_TMP_BASE_FILES, output_dir)], file_suffix)

    # Build up a list of nutrients to process based on what's checked on
    nutrients_to_process = []
    for nutrient_id in ['n', 'p']:
        if args['calc_' + nutrient_id]:
            nutrients_to_process.append(nutrient_id)
    lucode_to_parameters = pygeoprocessing.get_lookup_from_csv(
        args['biophysical_table_path'], 'lucode')

    _validate_inputs(nutrients_to_process, lucode_to_parameters)
    dem_pixel_size = pygeoprocessing.get_cell_size_from_uri(
        args['dem_path'])

    # Align all the input rasters
    pygeoprocessing.align_dataset_list(
        [args['dem_path']], [f_reg['aligned_dem_path']], ['nearest'],
        dem_pixel_size, 'intersection', dataset_to_align_index=0,
        aoi_uri=args['watersheds_path'])

    # Calculate flow accumulation
    LOGGER.info("calculating flow accumulation")
    pygeoprocessing.routing.flow_direction_d_inf(
        f_reg['aligned_dem_path'], f_reg['flow_direction_path'])
    pygeoprocessing.routing.flow_accumulation(
        f_reg['flow_direction_path'], f_reg['aligned_dem_path'],
        f_reg['flow_accumulation_path'])

    # Calculate slope
    LOGGER.info("Calculating slope")
    pygeoprocessing.calculate_slope(
        f_reg['aligned_dem_path'], f_reg['slope_path'])
    slope_nodata = pygeoprocessing.get_nodata_from_uri(
        f_reg['slope_path'])

    def threshold_slope(slope):
        """Threshold slope between 0.001 and 1.0."""
        valid_mask = slope != slope_nodata
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = slope_nodata
        slope_fraction = slope[valid_mask] / 100
        slope_fraction[slope_fraction < 0.005] = 0.005
        slope_fraction[slope_fraction > 1.0] = 1.0
        result[valid_mask] = slope_fraction
        return result

    pygeoprocessing.vectorize_datasets(
        [f_reg['slope_path']], threshold_slope,
        f_reg['thresholded_slope_path'], gdal.GDT_Float32, slope_nodata,
        dem_pixel_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)

    dem_pixel_size = pygeoprocessing.get_cell_size_from_uri(
        args['dem_path'])
    # pixel size is m, so square and divide by 10000 to get cell size in Ha
    cell_area_ha = dem_pixel_size ** 2 / 10000.0
    out_pixel_size = dem_pixel_size

    # align all the input rasters
    pygeoprocessing.align_dataset_list(
        [args['dem_path'], args['lulc_path'], args['runoff_proxy_path']],
        [f_reg['aligned_dem_path'], f_reg['aligned_lulc_path'],
         f_reg['aligned_runoff_proxy_path']], ['nearest'] * 3, out_pixel_size,
        'dataset', dataset_to_align_index=0, dataset_to_bound_index=0,
        aoi_uri=args['watersheds_path'])

    runoff_proxy_mean = pygeoprocessing.aggregate_raster_values_uri(
        f_reg['aligned_runoff_proxy_path'],
        args['watersheds_path']).pixel_mean[9999]
    runoff_proxy_nodata = pygeoprocessing.get_nodata_from_uri(
        f_reg['aligned_runoff_proxy_path'])

    def normalize_runoff_proxy_op(val):
        """Divide val by average runoff."""
        valid_mask = val != runoff_proxy_nodata
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = runoff_proxy_nodata
        result[valid_mask] = val[valid_mask] / runoff_proxy_mean
        return result

    pygeoprocessing.vectorize_datasets(
        [f_reg['aligned_runoff_proxy_path']], normalize_runoff_proxy_op,
        f_reg['runoff_proxy_index_path'], gdal.GDT_Float32,
        runoff_proxy_nodata, out_pixel_size, "intersection",
        vectorize_op=False)

    nodata_landuse = pygeoprocessing.get_nodata_from_uri(
        f_reg['aligned_lulc_path'])
    nodata_load = -1.0

    # classify streams from the flow accumulation raster
    LOGGER.info("Classifying streams from flow accumulation raster")
    pygeoprocessing.routing.stream_threshold(
        f_reg['flow_accumulation_path'],
        float(args['threshold_flow_accumulation']), f_reg['stream_path'])
    nodata_stream = pygeoprocessing.get_nodata_from_uri(
        f_reg['stream_path'])

    def map_load_function(load_type, subsurface_proportion_type=None):
        """Function generator to map arbitrary nutrient type to surface load.

        Parameters:
            load_type (string): either 'n' or 'p', used for indexing headers
            subsurface_proportion_type (string): if None no subsurface transfer
                is mapped.  Otherwise indexed from lucode_to_parameters

        Returns:
            map_load (function(lucode_array)): a function that can be passed to
                vectorize_datasets.
        """
        def map_load(lucode_array):
            """Convert unit load to total load & handle nodata."""
            result = numpy.empty(lucode_array.shape)
            result[:] = nodata_load
            for lucode in numpy.unique(lucode_array):
                if lucode != nodata_landuse:
                    if subsurface_proportion_type is not None:
                        result[lucode_array == lucode] = (
                            lucode_to_parameters[lucode][load_type] *
                            (1 - lucode_to_parameters[lucode]
                             [subsurface_proportion_type]) *
                            cell_area_ha)
                    else:
                        result[lucode_array == lucode] = (
                            lucode_to_parameters[lucode][load_type] *
                            cell_area_ha)
            return result
        return map_load

    def map_subsurface_load_function(
            load_type, subsurface_proportion_type=None):
        """Function generator to map arbitrary nutrient to subsurface load.

        Parameters:
            load_type (string): either 'n' or 'p', used for indexing headers
            subsurface_proportion_type (string): if None no subsurface transfer
                is mapped.  Otherwise indexed from lucode_to_parameters

        Returns:
            map_load (function(lucode_array)): a function that can be passed to
                vectorize_datasets to create subsurface load raster.
        """
        # If we don't have subsurface, just return 0.0.
        if subsurface_proportion_type is None:
            return lambda lucode_array: numpy.where(
                lucode_array != nodata_landuse, 0, nodata_load)

        keys = sorted(numpy.array(lucode_to_parameters.keys()))
        surface_values = numpy.array(
            [lucode_to_parameters[x][load_type] for x in keys])
        subsurface_values = numpy.array(
            [lucode_to_parameters[x][subsurface_proportion_type]
             for x in keys])

        def map_load(lucode_array):
            """Convert unit load to total load & handle nodata."""
            valid_mask = lucode_array != nodata_landuse
            result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
            result[:] = nodata_load
            index = numpy.digitize(
                lucode_array[valid_mask].ravel(), keys, right=True)
            result[valid_mask] = (
                surface_values[index] * subsurface_values[index] *
                cell_area_ha)
            return result
        return map_load

    def map_const_value(const_value, nodata):
        """Function generator to map arbitrary efficiency type."""
        def map_const(lucode_array):
            """Return constant value unless nodata."""
            return numpy.where(
                lucode_array == nodata_landuse, nodata, const_value)
        return map_const

    def map_eff_function(load_type):
        """Function generator to map arbitrary efficiency type."""
        keys = sorted(numpy.array(lucode_to_parameters.keys()))
        values = numpy.array(
            [lucode_to_parameters[x][load_type] for x in keys])

        def map_eff(lucode_array, stream_array):
            """Map efficiency from LULC and handle nodata/streams."""
            valid_mask = (
                (lucode_array != nodata_landuse) &
                (stream_array != nodata_stream))
            result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
            result[:] = nodata_load
            index = numpy.digitize(
                lucode_array[valid_mask].ravel(), keys, right=True)
            result[valid_mask] = (
                values[index] * (1 - stream_array[valid_mask]))
            return result
        return map_eff

    for nutrient in nutrients_to_process:
        load_path = f_reg['load_%s_path' % nutrient]
        modified_load_path = f_reg['modified_load_%s_path' % nutrient]
        # Perrine says that 'n' is the only case where we could consider a
        # prop subsurface component.  So there's a special case for that.
        if nutrient == 'n':
            subsurface_proportion_type = 'proportion_subsurface_n'
        else:
            subsurface_proportion_type = None
        pygeoprocessing.vectorize_datasets(
            [f_reg['aligned_lulc_path']], map_load_function(
                'load_%s' % nutrient, subsurface_proportion_type),
            load_path, gdal.GDT_Float32, nodata_load, out_pixel_size,
            "intersection", vectorize_op=False)

        def modified_load(load, runoff_proxy_index):
            """Multiply load by runoff proxy index to make modified load."""
            valid_mask = (
                (load != nodata_load) &
                (runoff_proxy_index != runoff_proxy_nodata))
            result = numpy.empty(valid_mask.shape)
            result[:] = nodata_load
            result[valid_mask] = (
                load[valid_mask] * runoff_proxy_index[valid_mask])
            return result

        pygeoprocessing.vectorize_datasets(
            [load_path, f_reg['runoff_proxy_index_path']], modified_load,
            modified_load_path, gdal.GDT_Float32, nodata_load,
            out_pixel_size, "intersection", vectorize_op=False)

        sub_load_path = f_reg['sub_load_%s_path' % nutrient]
        modified_sub_load_path = f_reg['modified_sub_load_%s_path' % nutrient]
        pygeoprocessing.vectorize_datasets(
            [f_reg['aligned_lulc_path']], map_subsurface_load_function(
                'load_%s' % nutrient, subsurface_proportion_type),
            sub_load_path, gdal.GDT_Float32, nodata_load,
            out_pixel_size, "intersection", vectorize_op=False)

        pygeoprocessing.vectorize_datasets(
            [sub_load_path, f_reg['runoff_proxy_index_path']], modified_load,
            modified_sub_load_path, gdal.GDT_Float32, nodata_load,
            out_pixel_size, "intersection", vectorize_op=False)

        sub_eff_path = f_reg['sub_eff_%s_path' % nutrient]
        pygeoprocessing.vectorize_datasets(
            [f_reg['aligned_lulc_path']], map_const_value(
                args['subsurface_eff_%s' % nutrient], nodata_load),
            sub_eff_path, gdal.GDT_Float32, nodata_load,
            out_pixel_size,
            "intersection", vectorize_op=False)

        sub_crit_len_path = f_reg['sub_crit_len_%s_path' % nutrient]
        pygeoprocessing.vectorize_datasets(
            [f_reg['aligned_lulc_path']], map_const_value(
                args['subsurface_critical_length_%s' % nutrient], nodata_load),
            sub_crit_len_path, gdal.GDT_Float32, nodata_load,
            out_pixel_size, "intersection", vectorize_op=False)

        eff_path = f_reg['eff_%s_path' % nutrient]
        pygeoprocessing.vectorize_datasets(
            [f_reg['aligned_lulc_path'], f_reg['stream_path']],
            map_eff_function('eff_%s' % nutrient), eff_path, gdal.GDT_Float32,
            nodata_load, out_pixel_size, "intersection", vectorize_op=False)

        crit_len_path = f_reg['crit_len_%s_path' % nutrient]
        pygeoprocessing.vectorize_datasets(
            [f_reg['aligned_lulc_path'], f_reg['stream_path']],
            map_eff_function('crit_len_%s' % nutrient), crit_len_path,
            gdal.GDT_Float32, nodata_load, out_pixel_size, "intersection",
            vectorize_op=False)

    watershed_output_datasource_uri = os.path.join(
        output_dir, 'watershed_results_ndr%s.shp' % file_suffix)
    # If there is already an existing shapefile with the same name and path,
    # delete it then copy the input shapefile into the designated output folder
    if os.path.isfile(watershed_output_datasource_uri):
        os.remove(watershed_output_datasource_uri)
    esri_driver = ogr.GetDriverByName('ESRI Shapefile')
    original_datasource = ogr.Open(args['watersheds_path'])
    output_datasource = esri_driver.CopyDataSource(
        original_datasource, watershed_output_datasource_uri)
    output_layer = output_datasource.GetLayer()

    # need this for low level route_flux function
    pygeoprocessing.make_constant_raster_from_base_uri(
        f_reg['aligned_dem_path'], 0.0, f_reg['zero_absorption_source_path'])

    flow_accumulation_nodata = (
        pygeoprocessing.get_nodata_from_uri(f_reg['flow_accumulation_path']))

    LOGGER.info("calculating %s", f_reg['s_accumulation_path'])
    pygeoprocessing.routing.route_flux(
        f_reg['flow_direction_path'], f_reg['aligned_dem_path'],
        f_reg['thresholded_slope_path'], f_reg['zero_absorption_source_path'],
        f_reg['loss_path'], f_reg['s_accumulation_path'], 'flux_only',
        aoi_uri=args['watersheds_path'])

    s_bar_nodata = pygeoprocessing.get_nodata_from_uri(
        f_reg['s_accumulation_path'])
    LOGGER.info("calculating %s", f_reg['s_bar_path'])

    def bar_op(base_accumulation, flow_accumulation):
        """Calculate bar operation."""
        valid_mask = (
            (base_accumulation != s_bar_nodata) &
            (flow_accumulation != flow_accumulation_nodata))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = s_bar_nodata
        result[valid_mask] = (
            base_accumulation[valid_mask] / flow_accumulation[valid_mask])
        return result

    pygeoprocessing.vectorize_datasets(
        [f_reg['s_accumulation_path'], f_reg['flow_accumulation_path']],
        bar_op, f_reg['s_bar_path'], gdal.GDT_Float32, s_bar_nodata,
        out_pixel_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)

    LOGGER.info('calculating d_up')
    cell_area = out_pixel_size ** 2
    d_up_nodata = -1.0

    def d_up(s_bar, flow_accumulation):
        """Calculate d_up index."""
        valid_mask = (
            (s_bar != s_bar_nodata) &
            (flow_accumulation != flow_accumulation_nodata))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = d_up_nodata
        result[valid_mask] = (
            s_bar[valid_mask] * numpy.sqrt(
                flow_accumulation[valid_mask] * cell_area))
        return result

    pygeoprocessing.vectorize_datasets(
        [f_reg['s_bar_path'], f_reg['flow_accumulation_path']], d_up,
        f_reg['d_up_path'], gdal.GDT_Float32, d_up_nodata, out_pixel_size,
        "intersection", dataset_to_align_index=0, vectorize_op=False)

    LOGGER.info('calculate inverse S factor')
    s_nodata = -1.0
    slope_nodata = pygeoprocessing.get_nodata_from_uri(
        f_reg['thresholded_slope_path'])

    def s_inverse_op(s_factor):
        """Calculate inverse of S factor."""
        valid_mask = s_factor != slope_nodata
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[valid_mask] = 1.0 / s_factor[valid_mask]
        return result

    pygeoprocessing.vectorize_datasets(
        [f_reg['thresholded_slope_path']], s_inverse_op,
        f_reg['s_factor_inverse_path'], gdal.GDT_Float32, s_nodata,
        out_pixel_size, "intersection", dataset_to_align_index=0,
        vectorize_op=False)

    LOGGER.info('calculating d_dn')
    pygeoprocessing.routing.distance_to_stream(
        f_reg['flow_direction_path'], f_reg['stream_path'], f_reg['d_dn_path'],
        factor_uri=f_reg['s_factor_inverse_path'])

    LOGGER.info('calculate ic')
    ic_nodata = -9999.0
    d_up_nodata = pygeoprocessing.get_nodata_from_uri(f_reg['d_up_path'])
    d_dn_nodata = pygeoprocessing.get_nodata_from_uri(f_reg['d_dn_path'])

    def ic_op(d_up, d_dn):
        """Calculate IC0."""
        valid_mask = (
            (d_up != d_up_nodata) & (d_dn != d_dn_nodata) & (d_up != 0) &
            (d_dn != 0))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = ic_nodata
        result[valid_mask] = numpy.log10(d_up[valid_mask] / d_dn[valid_mask])
        return result

    pygeoprocessing.vectorize_datasets(
        [f_reg['d_up_path'], f_reg['d_dn_path']], ic_op,
        f_reg['ic_factor_path'], gdal.GDT_Float32, ic_nodata, out_pixel_size,
        "intersection", dataset_to_align_index=0, vectorize_op=False)

    ic_min, ic_max, _, _ = (
        pygeoprocessing.get_statistics_from_uri(f_reg['ic_factor_path']))
    ic_0_param = (ic_min + ic_max) / 2.0
    k_param = float(args['k_param'])

    # define some variables outside the loop for closure
    effective_retention_nodata = None
    ndr_nodata = None
    sub_effective_retention_nodata = None
    load_nodata = None
    export_nodata = None
    field_header_order = []
    field_summaries = {}
    for nutrient in nutrients_to_process:
        effective_retention_path = (
            f_reg['effective_retention_%s_path' % nutrient])
        LOGGER.info('calculate effective retention')
        eff_path = f_reg['eff_%s_path' % nutrient]
        crit_len_path = f_reg['crit_len_%s_path' % nutrient]
        ndr_core.ndr_eff_calculation(
            f_reg['flow_direction_path'], f_reg['stream_path'], eff_path,
            crit_len_path, effective_retention_path)
        effective_retention_nodata = (
            pygeoprocessing.get_nodata_from_uri(
                effective_retention_path))
        LOGGER.info('calculate NDR')
        ndr_path = f_reg['ndr_%s_path' % nutrient]
        ndr_nodata = -1.0

        def calculate_ndr(effective_retention_array, ic_array):
            """Calculate NDR."""
            valid_mask = (
                (effective_retention_array != effective_retention_nodata) &
                (ic_array != ic_nodata))
            result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
            result[:] = ndr_nodata
            result[valid_mask] = (
                (1.0 - effective_retention_array[valid_mask]) /
                (1.0 + numpy.exp(
                    (ic_0_param - ic_array[valid_mask]) / k_param)))
            return result

        pygeoprocessing.vectorize_datasets(
            [effective_retention_path, f_reg['ic_factor_path']],
            calculate_ndr, ndr_path, gdal.GDT_Float32, ndr_nodata,
            out_pixel_size, 'intersection', vectorize_op=False)

        sub_effective_retention_path = (
            f_reg['sub_effective_retention_%s_path' % nutrient])
        LOGGER.info('calculate subsurface effective retention')
        ndr_core.ndr_eff_calculation(
            f_reg['flow_direction_path'], f_reg['stream_path'], sub_eff_path,
            sub_crit_len_path, sub_effective_retention_path)
        sub_effective_retention_nodata = (
            pygeoprocessing.get_nodata_from_uri(
                sub_effective_retention_path))
        LOGGER.info('calculate sub NDR')
        sub_ndr_path = f_reg['sub_ndr_%s_path' % nutrient]
        ndr_nodata = -1.0

        def calculate_sub_ndr(sub_eff_ret_array):
            """Calculate subsurface NDR."""
            valid_mask = (
                sub_eff_ret_array !=
                sub_effective_retention_nodata)
            result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
            result[:] = ndr_nodata
            result[valid_mask] = (1.0 - sub_eff_ret_array[valid_mask])
            return result

        pygeoprocessing.vectorize_datasets(
            [sub_effective_retention_path], calculate_sub_ndr, sub_ndr_path,
            gdal.GDT_Float32, ndr_nodata, out_pixel_size, 'intersection',
            vectorize_op=False)

        export_path = f_reg['%s_export_path' % nutrient]

        load_nodata = pygeoprocessing.get_nodata_from_uri(
            load_path)
        export_nodata = -1.0

        def calculate_export(
                modified_load_array, ndr_array, modified_sub_load_array,
                sub_ndr_array):
            """Combine NDR and subsurface NDR."""
            valid_mask = (
                (modified_load_array != load_nodata) &
                (ndr_array != ndr_nodata) &
                (modified_sub_load_array != load_nodata) &
                (sub_ndr_array != ndr_nodata))
            result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
            result[:] = export_nodata
            result[valid_mask] = (
                modified_load_array[valid_mask] * ndr_array[valid_mask] +
                modified_sub_load_array[valid_mask] *
                sub_ndr_array[valid_mask])
            return result

        modified_load_path = f_reg['modified_load_%s_path' % nutrient]
        modified_sub_load_path = f_reg['modified_sub_load_%s_path' % nutrient]
        pygeoprocessing.vectorize_datasets(
            [modified_load_path, ndr_path,
             modified_sub_load_path, sub_ndr_path], calculate_export,
            export_path, gdal.GDT_Float32, export_nodata,
            out_pixel_size, "intersection", vectorize_op=False)

        # summarize the results in terms of watershed:
        LOGGER.info("Summarizing the results of nutrient %s", nutrient)
        load_path = f_reg['load_%s_path' % nutrient]
        load_tot = pygeoprocessing.aggregate_raster_values_uri(
            load_path, args['watersheds_path'], 'ws_id').total
        export_tot = pygeoprocessing.aggregate_raster_values_uri(
            export_path, args['watersheds_path'], 'ws_id').total

        field_summaries['%s_load_tot' % nutrient] = load_tot
        field_summaries['%s_exp_tot' % nutrient] = export_tot
        field_header_order = (
            [x % nutrient for x in ['%s_load_tot', '%s_exp_tot']] +
            field_header_order)

    LOGGER.info('Writing summaries to output shapefile')
    _add_fields_to_shapefile(
        'ws_id', field_summaries, output_layer, field_header_order)

    LOGGER.info('cleaning up temp files')
    for tmp_filename_key in _TMP_BASE_FILES:
        try:
            os.remove(f_reg[tmp_filename_key])
        except OSError as os_error:
            LOGGER.warn(
                "Can't remove temporary file: %s\nOriginal Exception:\n%s",
                f_reg[tmp_filename_key], os_error)

    LOGGER.info(r'NDR complete!')
    LOGGER.info(r'  _   _    ____    ____     ')
    LOGGER.info(r' | \ |"|  |  _"\U |  _"\ u  ')
    LOGGER.info(r'<|  \| |>/| | | |\| |_) |/  ')
    LOGGER.info(r'U| |\  |uU| |_| |\|  _ <    ')
    LOGGER.info(r' |_| \_|  |____/ u|_| \_\   ')
    LOGGER.info(r' ||   \\,-.|||_   //   \\_  ')
    LOGGER.info(r' (_")  (_/(__)_) (__)  (__) ')


def _add_fields_to_shapefile(
        key_field, field_summaries, output_layer, field_header_order):
    """Add fields and values to an OGR layer open for writing.

    Parameters:
        key_field (string): name of the key field in the output_layer that
            uniquely identifies each polygon.
        field_summaries (dict): index for the desired field name to place in
            the polygon that indexes to another dictionary indexed by
            key_field value to map to that particular polygon.
            ex {'field_name_1': {key_val1: value,
            key_val2: value}, 'field_name_2': {key_val1: value, etc.
        output_layer (ogr.Layer): an open writable OGR layer
        field_header_order (list of string): a list of field headers in the
            order to appear in the output table.

    Returns:
        None.
    """
    for field_name in field_header_order:
        field_def = ogr.FieldDefn(field_name, ogr.OFTReal)
        output_layer.CreateField(field_def)

    # Initialize each feature field to 0.0
    for feature_id in xrange(output_layer.GetFeatureCount()):
        feature = output_layer.GetFeature(feature_id)
        for field_name in field_header_order:
            ws_id = feature.GetFieldAsInteger(key_field)
            feature.SetField(
                field_name, float(field_summaries[field_name][ws_id]))
        # Save back to datasource
        output_layer.SetFeature(feature)
