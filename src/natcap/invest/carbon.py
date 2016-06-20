"""Carbon Storage and Sequestration."""
import collections
import logging
import os
import time

from osgeo import gdal
import numpy
import pygeoprocessing

from . import utils

logging.basicConfig(
    format='%(asctime)s %(name)-18s %(levelname)-8s %(message)s',
    level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.carbon')

_OUTPUT_BASE_FILES = {
    'tot_c_cur': 'tot_c_cur.tif',
    'tot_c_fut': 'tot_c_fut.tif',
    'tot_c_redd': 'tot_c_redd.tif',
    'delta_cur_fut': 'delta_cur_fut.tif',
    'delta_cur_redd': 'delta_cur_redd.tif',
    'npv_fut': 'npv_fut.tif',
    'npv_redd': 'npv_redd.tif',
    'html_report': 'report.html',
    }

_INTERMEDIATE_BASE_FILES = {
    'c_above_cur': 'c_above_cur.tif',
    'c_below_cur': 'c_below_cur.tif',
    'c_soil_cur': 'c_soil_cur.tif',
    'c_dead_cur': 'c_dead_cur.tif',
    'c_above_fut': 'c_above_fut.tif',
    'c_below_fut': 'c_below_fut.tif',
    'c_soil_fut': 'c_soil_fut.tif',
    'c_dead_fut': 'c_dead_fut.tif',
    'c_above_redd': 'c_above_redd.tif',
    'c_below_redd': 'c_below_redd.tif',
    'c_soil_redd': 'c_soil_redd.tif',
    'c_dead_redd': 'c_dead_redd.tif',
    }

_TMP_BASE_FILES = {
    'aligned_lulc_cur_path': 'aligned_lulc_cur.tif',
    'aligned_lulc_fut_path': 'aligned_lulc_fut.tif',
    'aligned_lulc_redd_path': 'aligned_lulc_redd.tif',
    }

# -1.0 since carbon stocks are 0 or greater
_CARBON_NODATA = -1.0
# use min float32 which is unlikely value to see in a NPV raster
_VALUE_NODATA = numpy.finfo(numpy.float32).min


def execute(args):
    """InVEST Carbon Model.

    Calculate the amount of carbon stocks given a landscape, or the difference
    due to a future change, and/or the tradeoffs between that and a REDD
    scenario, and calculate economic valuation on those scenarios.

    The model can operate on a single scenario, a combined present and future
    scenario, as well as an additional REDD scenario.

    Parameters:
        args['workspace_dir'] (string): a path to the directory that will
            write output and other temporary files during calculation.
        args['results_suffix'] (string): appended to any output file name.
        args['lulc_cur_path'] (string): a path to a raster representing the
            current carbon stocks.
        args['lulc_fut_path'] (string): a path to a raster representing future
            landcover scenario.  Optional, but if present and well defined
            will trigger a sequestration calculation.
        args['lulc_redd_path'] (string): a path to a raster representing the
            alternative REDD scenario which is only possible if the
            args['lulc_fut_path'] is present and well defined.
        args['carbon_pools_path'] (string): path to CSV or that indexes carbon
            storage density to lulc codes. (required if 'do_uncertainty' is
            false)
        args['lulc_cur_year'] (int/string): an integer representing the year
            of `args['lulc_cur_path']` used in valuation required if
            `args['do_valuation']` is True.
        args['lulc_fut_year'](int/string): an integer representing the year
            of `args['lulc_fut_path']` used in valuation if it exists.
            Required if  `args['do_valuation']` is True and
            `args['lulc_fut_path']` is present and well defined.
        args['do_valuation'] (bool): if true then run the valuation model on
            available outputs.  At a minimum will run on carbon stocks, if
            sequestration with a future scenario is done and/or a REDD
            scenario calculate NPV for either and report in final HTML
            document.
        args['price_per_metric_ton_of_c'] (float): Is the present value of
            carbon per metric ton. Used if `args['do_valuation']` is present
            and True.
        args['discount_rate'] (float): Discount rate used if NPV calculations
            are required.  Used if `args['do_valuation']` is  present and
            True.
        args['rate_change'] (float): Annual rate of change in price of carbon
            as a percentage.  Used if `args['do_valuation']` is  present and
            True.
    Returns:
        None.
    """
    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    output_dir = args['workspace_dir']
    pygeoprocessing.create_directories([intermediate_output_dir, output_dir])

    LOGGER.info('Building file registry')
    file_registry = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir),
         (_TMP_BASE_FILES, output_dir)], file_suffix)

    carbon_pool_table = pygeoprocessing.get_lookup_from_table(
        args['carbon_pools_path'], 'lucode')

    cell_sizes = []
    valid_lulc_keys = []
    valid_scenarios = []
    for scenario_type in ['cur', 'fut', 'redd']:
        lulc_key = "lulc_%s_path" % (scenario_type)
        if lulc_key in args and len(args[lulc_key]) > 0:
            cell_sizes.append(
                pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                    args[lulc_key]))
            valid_lulc_keys.append(lulc_key)
            valid_scenarios.append(scenario_type)
    pixel_size_out = min(cell_sizes)

    # align the input datasets
    pygeoprocessing.align_dataset_list(
        [args[_] for _ in valid_lulc_keys],
        [file_registry['aligned_' + _] for _ in valid_lulc_keys],
        ['nearest'] * len(valid_lulc_keys),
        pixel_size_out, 'intersection', 0, assert_datasets_projected=True)

    LOGGER.info('Map all carbon pools to carbon storage rasters.')
    aligned_lulc_key = None
    pool_storage_path_lookup = collections.defaultdict(list)
    summary_stats = []  # use to aggregate storage, value, and more
    for pool_type in ['c_above', 'c_below', 'c_soil', 'c_dead']:
        carbon_pool_by_type = dict([
            (lucode, float(carbon_pool_table[lucode][pool_type]))
            for lucode in carbon_pool_table])
        for scenario_type in valid_scenarios:
            aligned_lulc_key = 'aligned_lulc_%s_path' % scenario_type
            storage_key = '%s_%s' % (pool_type, scenario_type)
            LOGGER.info(
                "Mapping carbon from '%s' to '%s' scenario.",
                aligned_lulc_key, storage_key)
            _generate_carbon_map(
                file_registry[aligned_lulc_key], carbon_pool_by_type,
                file_registry[storage_key])
            # store the pool storage path so they can be easily added later
            pool_storage_path_lookup[scenario_type].append(
                file_registry[storage_key])

    # Sum the individual carbon storage pool paths per scenario
    for scenario_type, storage_path_list in (
            pool_storage_path_lookup.iteritems()):
        output_key = 'tot_c_' + scenario_type
        LOGGER.info(
            "Calculate carbon storage for '%s'", output_key)
        _sum_rasters(storage_path_list, file_registry[output_key])

        # Tuple below is (sort_priority, description, value, unit, path)
        summary_stats.append((
            0, "Total %s" % scenario_type,
            _accumulate_totals(file_registry[output_key]), 'Mg of C',
            file_registry[output_key]))

    # calculate sequestration
    for fut_type in ['fut', 'redd']:
        if fut_type not in valid_scenarios:
            continue
        output_key = 'delta_cur_' + fut_type
        LOGGER.info("Calculate sequestration scenario '%s'", output_key)
        storage_path_list = [
            file_registry['tot_c_cur'], file_registry['tot_c_' + fut_type]]
        _diff_rasters(storage_path_list, file_registry[output_key])

        # Tuple below is (sort_priority, description, value, unit, path)
        summary_stats.append((
            1, "Change in C for %s" % fut_type,
            _accumulate_totals(file_registry[output_key]), 'Mg of C',
            file_registry[output_key]))

    if 'do_valuation' in args and args['do_valuation']:
        LOGGER.info('Constructing valuation formula.')
        valuation_constant = _calculate_valuation_constant(
            int(args['lulc_cur_year']), int(args['lulc_fut_year']),
            float(args['discount_rate']), float(args['rate_change']),
            float(args['price_per_metric_ton_of_c']))

        for scenario_type in ['fut', 'redd']:
            if scenario_type not in valid_scenarios:
                continue
            output_key = 'npv_%s' % scenario_type
            LOGGER.info("Calculating NPV for scenario '%s'", output_key)
            _calculate_npv(
                file_registry['delta_cur_%s' % scenario_type],
                valuation_constant, file_registry[output_key])

            # Tuple below is (sort_priority, description, value, unit, path)
            summary_stats.append((
                2, "Net present value from cur to %s" % scenario_type,
                _accumulate_totals(file_registry[output_key]),
                "currency units", file_registry[output_key]))

    _generate_report(summary_stats, args, file_registry['html_report'])

    for tmp_filename_key in _TMP_BASE_FILES:
        try:
            tmp_filename = file_registry[tmp_filename_key]
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)
        except OSError as os_error:
            LOGGER.warn(
                "Can't remove temporary file: %s\nOriginal Exception:\n%s",
                file_registry[tmp_filename_key], os_error)


def _accumulate_totals(raster_path):
    """Sum all non-nodata pixels in `raster_path` and return result."""
    nodata = pygeoprocessing.get_nodata_from_uri(raster_path)
    raster_sum = 0.0
    for _, block in pygeoprocessing.iterblocks(raster_path):
        raster_sum += numpy.sum(block[block != nodata])
    return raster_sum


def _generate_carbon_map(
        lulc_path, carbon_pool_by_type, out_carbon_stock_path):
    """Generate carbon stock raster by mapping LULC values to carbon pools.

    Parameters:
        lulc_path (string): landcover raster with integer pixels.
        out_carbon_stock_path (string): path to output raster that will have
            pixels with carbon storage values in them with units of Mg*C
        carbon_pool_by_type (dict): a dictionary that maps landcover values
            to carbon storage densities per area (Mg C/Ha).

    Returns:
        None.
    """
    nodata = pygeoprocessing.get_nodata_from_uri(lulc_path)
    pixel_size_out = pygeoprocessing.get_cell_size_from_uri(lulc_path)
    carbon_stock_by_type = dict([
        (lulcid, stock * pixel_size_out ** 2 / 10**4)
        for lulcid, stock in carbon_pool_by_type.iteritems()])

    carbon_stock_by_type[nodata] = _CARBON_NODATA

    pygeoprocessing.reclassify_dataset_uri(
        lulc_path, carbon_stock_by_type, out_carbon_stock_path,
        gdal.GDT_Float32, _CARBON_NODATA, exception_flag='values_required',
        assert_dataset_projected=True)


def _sum_rasters(storage_path_list, output_sum_path):
    """Sum all the rasters in `storage_path_list` to `output_sum_path`."""
    def _sum_op(*storage_arrays):
        """Sum all the arrays or nodata a pixel stack if one exists."""
        valid_mask = reduce(
            lambda x, y: x & y, [
                _ != _CARBON_NODATA for _ in storage_arrays])
        result = numpy.empty(storage_arrays[0].shape)
        result[:] = _CARBON_NODATA
        result[valid_mask] = numpy.sum([
            _[valid_mask] for _ in storage_arrays], axis=0)
        return result

    pixel_size_out = pygeoprocessing.get_cell_size_from_uri(
        storage_path_list[0])
    pygeoprocessing.vectorize_datasets(
        storage_path_list, _sum_op, output_sum_path,
        gdal.GDT_Float32, _CARBON_NODATA, pixel_size_out, "intersection",
        vectorize_op=False, datasets_are_pre_aligned=True)


def _diff_rasters(storage_path_list, output_diff_path):
    """Subtract rasters in `storage_path_list` to `output_sum_path`."""
    def _diff_op(base_array, future_array):
        """Subtract future_array from base_array and ignore nodata."""
        result = numpy.empty(base_array.shape, dtype=numpy.float32)
        result[:] = _CARBON_NODATA
        valid_mask = (
            (base_array != _CARBON_NODATA) &
            (future_array != _CARBON_NODATA))
        result[valid_mask] = (
            future_array[valid_mask] - base_array[valid_mask])
        return result

    pixel_size_out = pygeoprocessing.get_cell_size_from_uri(
        storage_path_list[0])
    pygeoprocessing.vectorize_datasets(
        storage_path_list, _diff_op, output_diff_path,
        gdal.GDT_Float32, _CARBON_NODATA, pixel_size_out, "intersection",
        vectorize_op=False, datasets_are_pre_aligned=True)


def _calculate_valuation_constant(
        lulc_cur_year, lulc_fut_year, discount_rate, rate_change,
        price_per_metric_ton_of_c):
    """Calculate a net present valuation constant to multiply carbon storage.

    Parameters:
        lulc_cur_year (int): calendar year in present
        lulc_fut_year (int): calendar year in future
        discount_rate (float): annual discount rate as a percentage
        rate_change (float): annual change in price of carbon as a percentage
        price_per_metric_ton_of_c (float): currency amount of Mg of carbon

    Returns:
        a floating point number that can be used to multiply a delta carbon
        storage value by to calculate NPV.
    """
    n_years = int(lulc_fut_year) - int(lulc_cur_year) - 1
    ratio = (
        1.0 / ((1 + float(discount_rate) / 100.0) *
               (1 + float(rate_change) / 100.0)))
    valuation_constant = (
        float(price_per_metric_ton_of_c) /
        (float(lulc_fut_year) - float(lulc_cur_year)) *
        (1.0 - ratio ** (n_years + 1)) / (1.0 - ratio))
    return valuation_constant


def _calculate_npv(delta_carbon_path, valuation_constant, npv_out_path):
    """Calculate net present value.

    Parameters:
        delta_carbon_path (string): path to change in carbon storage over
            time.
        valulation_constant (float): value to multiply each carbon storage
            value by to calculate NPV.
        npv_out_path (string): path to output net present value raster.
    Returns:
        None.
    """
    def _npv_value_op(carbon_array):
        """Calculate the NPV given carbon storage or loss values."""
        result = numpy.empty(carbon_array.shape, dtype=numpy.float32)
        result[:] = _VALUE_NODATA
        valid_mask = carbon_array != _CARBON_NODATA
        result[valid_mask] = carbon_array[valid_mask] * valuation_constant
        return result

    pixel_size_out = pygeoprocessing.get_cell_size_from_uri(delta_carbon_path)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [delta_carbon_path], _npv_value_op, npv_out_path, gdal.GDT_Float32,
        _VALUE_NODATA, pixel_size_out, "intersection",
        vectorize_op=False, datasets_are_pre_aligned=True)


def _generate_report(summary_stats, model_args, html_report_path):
    """Generate a human readable HTML report of summary stats of model run.

    Paramters:
        summary_stats (list of tuple): a list of tuples of the form
            (display_sort_priority, description, value, unit, file_path)
        model_args (dict): InVEST argument dictionary.
        html_report_path (string): path to output report file.
    Returns:
        None.
    """
    with open(html_report_path, 'w') as report_doc:
        # Boilerplate header that defines style and intro header.
        header = (
            '<!DOCTYPE html><html><head><title>Carbon Results</title><style t'
            'ype="text/css">body { background-color: #EFECCA; color: #002F2F '
            '} h1 { text-align: center } h1, h2, h3, h4, strong, th { color: '
            '#046380; } h2 { border-bottom: 1px solid #A7A37E; } table { bord'
            'er: 5px solid #A7A37E; margin-bottom: 50px; background-color: #E'
            '6E2AF; } td, th { margin-left: 0px; margin-right: 0px; padding-l'
            'eft: 8px; padding-right: 8px; padding-bottom: 2px; padding-top: '
            '2px; text-align:left; } td { border-top: 5px solid #EFECCA; } .n'
            'umber {text-align: right; font-family: monospace;} img { margin:'
            ' 20px; }</style></head><body><h1>InVEST Carbon Model Results</h1'
            '><p>This document summarizes the results from running the InVEST'
            ' carbon model with the following data.</p>')

        report_doc.write(header)
        report_doc.write('<p>Report generated at %s</p>' % (
            time.strftime("%Y-%m-%d %H:%M")))

        # Report input arguments
        report_doc.write('<table><tr><th>arg id</th><th>arg value</th></tr>')
        for key, value in model_args.iteritems():
            report_doc.write('<tr><td>%s</td><td>%s</td></tr>' % (key, value))
        report_doc.write('</table>')

        # Report aggregate results
        report_doc.write('<h3>Aggregate Results</h3>')
        report_doc.write(
            '<table><tr><th>Description</th><th>Value</th><th>Units</th><th>R'
            'aw File</th></tr>')
        for _, result_description, units, value, raw_file_path in sorted(
                summary_stats):
            report_doc.write(
                '<tr><td>%s</td><td class="number">%.2f</td><td>%s</td>'
                '<td>%s</td></tr>' % (
                    result_description, units, value, raw_file_path))
        report_doc.write('</body></html>')
