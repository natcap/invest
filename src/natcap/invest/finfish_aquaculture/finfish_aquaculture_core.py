"""
Implementation of the aquaculture calculations, and subsequent outputs.
This will pull from data passed in by finfish_aquaculture.
"""

import collections
import os
import math
import logging

from osgeo import ogr
from osgeo import gdal
import numpy as np
from scipy.stats import norm

from natcap.invest.reporting import html

LOGGER = logging.getLogger('natcap.invest.finfish_aquaculture.core')


def execute(args):
    """'
    Runs the biophysical and valuation parts of the finfish aquaculture model.
    This will output:
    1. a shape file showing farm locations w/ addition of # of harvest cycles,
        total processed weight at that farm, and if valuation is true, total
        discounted net revenue at each farm location.
    2. Three HTML tables summarizing all model I/O- summary of user-provided
        data, summary of each harvest cycle, and summary of the outputs/farm
    3. A .txt file that is named according to the date and time the model is
        run, which lists the values used during that run

    Data in args should include the following:
    --Biophysical Arguments--
    args: a python dictionary containing the following data:
    args['workspace_dir']- The directory in which to place all result files.
    args['ff_farm_file']- An open shape file containing the locations of
        individual fisheries
    args['farm_ID']- column heading used to describe individual farms. Used to
        link GIS location data to later inputs.
    args['g_param_a']- Growth parameter alpha, used in modeling fish growth,
        should be int or a float.
    args['g_param_b']- Growth parameter beta, used in modeling fish growth,
        should be int or a float.
    args['water_temp_dict']- A dictionary which links a specific date to the
        farm numbers, and their temperature values on that day. (Note: in this
        case, the outer keys 1 and 2 are calendar days out of 365, starting
        with January 1 (day 0), and the inner 1, 2, and 3 are farm numbers.)

            Format: {'0': '{'1': '8.447, '2': '8.447', '3':'8.947', ...}' ,
                     '1': '{'1': '8.406, '2': '8.406', '3':'8.906', ...}' ,
                    .                        .                    .
                    .                        .                    .
                    .                        .                    .       }
    args['farm_op_dict']- Dictionary which links a specific farm ID # to
        another dictionary containing operating parameters mapped to their
        value for that particular farm (Note: in this case, the 1 and 2
        are farm ID's, not dates out of 365.)

            Format: {'1': '{'Wt of Fish': '0.06', 'Tar Weight': '5.4', ...}',
                    '2': '{'Wt of Fish': '0.06', 'Tar Weight': '5.4', ...}',
                    .                        .                    .
                    .                        .                    .
                    .                        .                    .       }
    args['frac_post_process']- the fraction of edible fish left after
        processing is done to remove undesirable parts
    args['mort_rate_daily']- mortality rate among fish  in a year, divided by
        365
    args['duration']- duration of the simulation, in years
    args['outplant_buffer'] - This value will allow the outplant start day to
        be flexible plus or minus the number of days specified here.

    --Valuation arguments--
    args['do_valuation']- boolean indicating whether or not to run the
        valuation process
    args['p_per_kg']: Market price per kilogram of processed fish
    args['frac_p']: Fraction of market price that accounts for costs rather
        than profit
    args['discount']: Daily market discount rate

    returns nothing
    """
    LOGGER.debug("FROM CORE: %s" % args['farm_op_dict'])
    output_dir = os.path.join(args['workspace_dir'], 'output')

    cycle_history = calc_farm_cycles(
        args['outplant_buffer'], args['g_param_a'], args['g_param_b'],
        args['g_param_tau'], args['water_temp_dict'], args['farm_op_dict'],
        float(args['duration']))

    out_path = os.path.join(output_dir, 'Finfish_Harvest.shp')
    if os.path.isfile(out_path):
        # Remove so we can re-create.
        os.remove(out_path)

    curr_shp_file = gdal.OpenEx(args['ff_farm_file'], gdal.OF_VECTOR)
    driver = gdal.GetDriverByName('ESRI Shapefile')
    sf_copy = driver.CreateCopy(out_path, curr_shp_file)
    layer = sf_copy.GetLayer()

    #This adds the number of cycles completed by each farm to their shapefile
    #feature
    cycle_field = ogr.FieldDefn('Tot_Cycles', ogr.OFTReal)
    cycle_field.SetWidth(24)
    cycle_field.SetPrecision(11)
    layer.CreateField(cycle_field)

    for feature in layer:
        accessor = args['farm_ID']
        feature_ID = feature.items()[accessor]
        num_cycles = len(cycle_history[feature_ID])
        feature.SetField('Tot_Cycles', num_cycles)
        layer.SetFeature(feature)

    #Now want to add the total processed weight of each farm as a second
    #feature on the outgoing shapefile- abstracting the calculation of this to
    #a separate function, but it will return a dictionary with a int->float
    #mapping for farm_ID->processed weight
    sum_hrv_weight, hrv_weight = calc_hrv_weight(
        args['farm_op_dict'], args['frac_post_process'],
        args['mort_rate_daily'], cycle_history)

    #have to start at the beginning of the layer to access the attributes
    layer.ResetReading()

    #Now, add the total processed weight as a shapefile feature
    hrv_field = ogr.FieldDefn('Hrvwght_kg', ogr.OFTReal)
    hrv_field.SetWidth(24)
    hrv_field.SetPrecision(11)
    layer.CreateField(hrv_field)

    for feature in layer:
        accessor = args['farm_ID']
        feature_ID = feature.items()[accessor]
        feature.SetField('Hrvwght_kg', sum_hrv_weight[feature_ID])
        layer.SetFeature(feature)

    # Do valuation if requested.
    if args['do_valuation']:
        value_history, farms_npv = valuation(
            args['p_per_kg'], args['frac_p'], args['discount'],
            hrv_weight, cycle_history)

        #And add it into the shape file
        layer.ResetReading()
        npv_field = ogr.FieldDefn('NVP_USD_1k', ogr.OFTReal)
        npv_field.SetWidth(24)
        npv_field.SetPrecision(11)
        layer.CreateField(npv_field)

        for feature in layer:
            accessor = args['farm_ID']
            feature_ID = feature.items()[accessor]
            feature.SetField('NVP_USD_1k', farms_npv[feature_ID])
            layer.SetFeature(feature)
    else:
        value_history, farms_npv = None, None

    uncertainty_stats = None
    # Do uncertainty analysis if it's enabled.
    if 'g_param_a_sd' in args and 'g_param_b_sd' in args:
        uncertainty_stats = compute_uncertainty_data(
            args, output_dir)

    create_HTML_table(
        output_dir, args, cycle_history, sum_hrv_weight, hrv_weight, farms_npv,
        value_history, uncertainty_stats)


def calc_farm_cycles(outplant_buffer, a, b, tau, water_temp_dict,
                     farm_op_dict, dur):
    """
    Input:
        outplant_buffer: The number of days surrounding the outplant day during
            which the fish growth cycle can still be started.
        a: Growth parameter alpha. Float used as a scaler in the fish growth
            equation.
        b: Growth paramater beta. Float used as an exponential multiplier in
            the fish growth equation.
        water_temp_dict: 2D dictionary which contains temperature values for
            farms. The outer keys are calendar days as strings, and the inner
            are farm numbers as strings.
        farm_op_dict: 2D dictionary which contains individual operating
            parameters for each farm. The outer key is farm number as a string,
            and the inner is string descriptors of each parameter.
        dur: Float which describes the length for the growth simulation to run
            in years.

    Returns cycle_history where:

        cycle_history: Dictionary which contains mappings from farms to a
            history of growth for each cycle completed on that farm. These
            entries are formatted as follows...

            Farm->List of Type (day of outplanting,day of harvest, fish weight
                (grams))
    """

    cycle_history = {}

    for f in farm_op_dict.keys():

        #Are multiplying by 1000, because a and b are in grams, so need to do
        #the whole equation in grams. Have to explicit cast to get things in a
        #format that will be usable later.
        start_day = int(farm_op_dict[f]['start day for growing']) - 1
        fallow_per = int(farm_op_dict[f]['Length of Fallowing period'])
        start_weight = 1000 * float(
            farm_op_dict[f]['weight of fish at start (kg)'])
        tar_weight = 1000 * float(
            farm_op_dict[f]['target weight of fish at harvest (kg)'])

        fallow_days_left = start_day
        farm_history = []
        fish_weight = 0
        outplant_date = None
        #Have changed the water temp table to be accessed by keys 0 to 364, so
        #now can just grab straight from the table without having to deal with
        #change in day

        #However, it should be kept in mind that when doing calculations for a
        #given day, you are using YESTRDAY'S temperatures and weights to get
        #the value for today.

        #Are going 1 day beyond on the off-chance that you ended a harvest the
        #day before, and need to record today. This should not create any
        #false harvest records, because the fish would be reaching the end
        #growth weight today, not yesterday.
        for day in range(0, int((365*dur)) + 1):

            if fallow_days_left > 0:
                fallow_days_left -= 1

            elif fish_weight >= tar_weight:
                record = (outplant_date, day, fish_weight)
                farm_history.append(record)
                fallow_days_left = fallow_per
                fish_weight = 0

            elif fish_weight != 0:
                #Grow 'dem fishies!
                exponent = math.exp(
                    float(water_temp_dict[str((day-1) % 365)][f]) * tau)

                fish_weight = (a * (fish_weight ** b) * exponent) + \
                    fish_weight

                fish_weight = fish_weight

            #function that maps an incoming day to the same day % 365, then
            #creates a list to check against +/- buffer days from the start day

            elif (day % 365) in map(
                lambda x: x % 365, range(start_day - outplant_buffer,
                                         start_day + outplant_buffer + 1)):
                    fish_weight = start_weight
                    outplant_date = day + 1

        cycle_history[int(f)] = farm_history

    return cycle_history


def calc_hrv_weight(farm_op_dict, frac, mort, cycle_history):
    """
    Input:
        farm_op_dict: 2D dictionary which contains individual operating
            parameters for each farm. The outer key is farm number as a string,
            and the inner is string descriptors of each parameter.
        frac: A float representing the fraction of the fish that remains after
            processing.
        mort: A float referring to the daily mortality rate of fishes on an
            aquaculture farm.
        cycle_history: Farm->List of Type (day of outplanting,
            day of harvest, fish weight (grams))

    Returns a tuple (curr_cycle_totals,indiv_tpw_totals) where:
        curr_cycle_totals_: dictionary which will hold a mapping from every
            farm (as identified by farm_ID) to the total processed weight of
            each farm
        indiv_tpw_totals: dictionary which will hold a farm->list mapping,
            where the list holds the individual tpw for all cycles that the
            farm completed
    """

    curr_cycle_totals = {}
    indiv_tpw_totals = {}

    for f in farm_op_dict:

        #They keys from farm_op_dict are strings, since they came from a CSV.
        #So, have to cast to strings in order to make them usable for
        #refrencing everything else.
        f = int(f)

        #pre-load farm specific vars, have to cast some because they come out
        #of a CSV all as strings
        curr_cycle_totals[f] = 0
        f_num_fish = int(farm_op_dict[str(f)]['number of fish in farm'])
        cycles_comp = len(cycle_history[f])
        farm_history = cycle_history[f]
        mort = float(mort)
        indiv_tpw_totals[f] = []

        #We are starting this range at 0, and going to one less than the
        #number of cycles, since the list of cycles from the cycle calcs will
        #start at index 0
        for c in range(0, cycles_comp):

            #this will get the tuple referring to the current cycle
            #the information will be inside the tuple as:
            # (day of outplanting, day of harvest, harvest weight)
            current_cycle_info = farm_history[c]
            outplant_date, harvest_date, fish_weight = current_cycle_info

            #Now do the computation for each cycle individually, then add it
            #to the total within the dictionary
            #Note that we divide by 1000 to make sure the output in in kg
            cycle_length = harvest_date - outplant_date
            e_exponent = -mort * cycle_length

            #This equation comes from total weight of fish produced per farm
            #from the user's guide
            curr_cy_tpw = (fish_weight / 1000) * frac * f_num_fish * \
                math.exp(e_exponent)
            curr_cy_tpw = curr_cy_tpw

            indiv_tpw_totals[f].append(curr_cy_tpw)
            curr_cycle_totals[f] += curr_cy_tpw

    return (curr_cycle_totals, indiv_tpw_totals)


def valuation(price_per_kg, frac_mrkt_price, discount, hrv_weight,
              cycle_history):
    """
    This performs the valuation calculations, and returns tuple containing a
    dictionary with a farm-> float mapping, where each float is the net
    processed value of the fish processed on that farm, in $1000s of dollars,
    and a dictionary containing a farm-> list mapping, where each entry in the
    list is a tuple of (Net Revenue, Net Present Value) for every cycle on that
    farm.

    Inputs:
        price_per_kg: Float representing the price per kilogram of finfish for
                valuation purposes.
        frac_mrkt_price: Float that represents the fraction of market price
            that is attributable to costs.
        discount: Float that is the daily market discount rate.
        cycle_hisory: Farm->List of Type (day of outplanting,
                                          day of harvest, fish weight (grams))
        hrv_weight: Farm->List of TPW for each cycle (kilograms)



    Returns a tuple (val_history, valuations):
        val_history: dictionary which will hold a farm->list mapping, where the
            list holds tuples containing (Net Revenue, Net Present Value) for
            each cycle completed by that farm
        valuations: dictionary with a farm-> float mapping, where each float is
            the net processed value of the fish processed on that farm
    """
    val_history = {}
    valuations = {}

    for f in cycle_history:

        val_history[f] = []
        valuations[f] = 0.0

        #running from 0 to 1 less than the number of cycles that farm
        #completed, since the list that each farm ID is mapped to starts at
        #index 0
        for c in range(0, len(cycle_history[f])):

            tpw = hrv_weight[f][c]

            #the 1 refers to the placement of day of harvest in the tuple for
            #each cycle
            t = cycle_history[f][c][1]

            net_rev = tpw * (price_per_kg * (1 - frac_mrkt_price))
            npv = net_rev * (1 / (1 + discount) ** t)

            val_history[f].append((net_rev, npv))

            #divide by 1000, because the number we want to return is in
            #thousands of dollars
            valuations[f] += npv / 1000

    return val_history, valuations


def compute_uncertainty_data(args, output_dir):
    """Does uncertainty analysis via a Monte Carlo simulation.

    Returns:
    -a dict containining statistical results (mean and std deviation)
    The dict has farm IDs as outer keys, and result types (e.g. 'value',
    'weight', and 'cycles') as inner keys.
    """
    results = do_monte_carlo_simulation(args)

    LOGGER.info('Computing confidence statistics.')
    uncertainty_stats = collections.OrderedDict()
    for farm, farm_results in results.items():
        uncertainty_stats[farm] = {}
        for result_type, result_list in farm_results.items():
            uncertainty_stats[farm][result_type] = norm.fit(result_list)
        if not args['do_valuation']:
            uncertainty_stats[farm]['value'] = (
                '(no valuation)', '(no valuation)')
    uncertainty_stats['total']['cycles'] = ('n/a', 'n/a')

    LOGGER.info('Done with uncertainty analysis.')
    return uncertainty_stats


def do_monte_carlo_simulation(args):
    """
    Performs a Monte Carlo simulation and returns the results.
    """
    def sample_param(param):
        """Samples the normal distribution for the given growth parameter.

        Returns only positive values, discarding the rest."""
        while True:
            sample = np.random.normal(args['g_param_%s' % param],
                                      args['g_param_%s_sd' % param])
            if sample > 0:
                return sample

    # Set up a dict to contain the results of the simulation.
    farms = sorted(int(f) for f in args['farm_op_dict'])

    # Add a placeholder for aggregate results.
    farms.insert(0, 'total')

    fields = ['cycles', 'weight']
    if args['do_valuation']:
        fields.append('value')

    results = collections.OrderedDict()
    for farm in farms:
        results[farm] = {}
        for field in fields:
            # We don't log total cycles across all farms,
            # since it's not particularly meaningful.
            if not (farm == 'total' and field == 'cycles'):
                results[farm][field] = []

    LOGGER.info('Beginning Monte Carlo simulation. Doing %d runs.'
                % args['num_monte_carlo_runs'])

    for i in range(args['num_monte_carlo_runs']):
        if i > 0 and i % 100 == 0:
            LOGGER.info('Done with %d runs.' % i)

        # Compute the cycle history given samples for parameters a and b.
        cycle_history = calc_farm_cycles(
            args['outplant_buffer'], sample_param('a'), sample_param('b'),
            args['g_param_tau'], args['water_temp_dict'], args['farm_op_dict'],
            float(args['duration']))

        # Compute the total harvested weight.
        sum_hrv_weight, hrv_weight_per_cycle = calc_hrv_weight(
            args['farm_op_dict'], args['frac_post_process'],
            args['mort_rate_daily'], cycle_history)

        # Compute valuation data.
        if args['do_valuation']:
            _, farms_npv = valuation(
                args['p_per_kg'], args['frac_p'], args['discount'],
                hrv_weight_per_cycle, cycle_history)

        # Update our total results.
        results['total']['weight'].append(sum(sum_hrv_weight.values()))
        if args['do_valuation']:
            results['total']['value'].append(sum(farms_npv.values()))

        # Update our per-farm results.
        for farm, hrv_weight in sum_hrv_weight.items():
            results[farm]['weight'].append(hrv_weight)
            results[farm]['cycles'].append(len(cycle_history[farm]))
            if args['do_valuation']:
                results[farm]['value'].append(farms_npv[farm])

    LOGGER.info('Monte Carlo simulation complete.')
    return results


def create_HTML_table(output_dir, args, cycle_history, sum_hrv_weight,
                      hrv_weight, farms_npv, value_history,
                      uncertainty_stats):
    """
    Inputs:
        output_dir: The directory in which we will be creating our .html file
            output.
        cycle_history: dictionary mapping farm ID->list of tuples, each of
            which contains 3 things- (day of outplanting, day of harvest,
                harvest weight of a single fish in grams)
        sum_hrv_weight: dictionary which holds a mapping from farm ID->total
            processed weight of each farm
        hrv_weight: dictionary which holds a farm->list mapping, where the list
            holds the individual tpw for all cycles that the farm completed
        do_valuation: boolean variable that says whether or not valuation is
            desired
        farms_npv: dictionary with a farm-> float mapping, where each float is
            the net processed value of the fish processed on that farm, in
            $1000s of dollars.
        value_history: dictionary which holds a farm->list mapping, where the
            list holds tuples containing (Net Revenue, Net Present Value) for
            each cycle completed by that farm

    Output:
        HTML file: contains 3 tables that summarize inputs and outputs for the
            duration of the model.
            - Input Table: Farm Operations provided data, including Farm ID #,
                Cycle Number, weight of fish at start, weight of fish at
                harvest, number of fish in farm, start day for growing, and
                length of fallowing period
            - Output Table 1: Farm Harvesting data, including a summary table
                for each harvest cycle of each farm. Will show Farm ID, cycle
                number, days since outplanting date, harvested weight, net
                revenue, outplant day, and year.
            - Output Table 2: Model outputs for each farm, including Farm ID,
                net present value, number of completed harvest cycles, and
                total volume harvested.

        Returns nothing.
    """
    html_uri = os.path.join(output_dir,
                            ("Harvest_Results.html"))
    doc = html.HTMLDocument(html_uri, 'Marine InVEST',
                            'Aquaculture Model (Finfish Harvest)')

    doc.write_paragraph(
        'This page contains results from running the Marine InVEST Finfish '
        'Aquaculture model.')

    doc.insert_table_of_contents()

    doc.write_header('Farm Operations (input)')

    ops_table = doc.add(html.Table(id='farm_ops_table'))
    ops_table.add_row(['Farm ID Number',
                       'Weight of Fish at Start (kg)',
                       'Weight of Fish at Harvest (kg)',
                       'Number of Fish in Farm',
                       'Start Day for Growing (1-365)',
                       'Length of Fallowing Period (days)'
                       ],
                      is_header=True)

    for farm_id in cycle_history:
        farm_id = str(farm_id)
        cells = [farm_id]
        for column_key in ['weight of fish at start (kg)',
                           'target weight of fish at harvest (kg)',
                           'number of fish in farm',
                           'start day for growing',
                           'Length of Fallowing period']:
            cells.append(args['farm_op_dict'][farm_id][column_key])
        ops_table.add_row(cells, do_formatting=False)

    doc.write_header('Farm Harvesting (output)')
    harvest_table = doc.add(html.Table(id='harvest_table'))
    harvest_table.add_row([
        'Farm ID Number', 'Cycle Number',
        'Days Since Outplanting Date (Including Fallowing Period)',
        'Length of Given Cycle',
        'Harvested Weight After Processing (kg/cycle)',
        'Net Revenue (Thousands of $)',
        'Net Present Value (Thousands of $)',
        'Outplant Day (Julian Day)',
        'Outplant Year'],
        is_header=True)

    for farm_id in cycle_history:
        for cycle in range(0, len(cycle_history[farm_id])):
            cycle_num = cycle + 1

            curr_cycle = cycle_history[farm_id][cycle]
            outplant_date, harvest_date, harvest_weight = curr_cycle
            cycle_length = harvest_date - outplant_date

            # Want to get the processed weight on a farm for a given cycle. All
            # of the PW for all cycles should add to the third table's TPW.
            harvest_weight = hrv_weight[farm_id][cycle]

            out_day = outplant_date % 365
            out_year = outplant_date // 365 + 1

            if args['do_valuation']:
                # Revenue and NPV should be in thousands of dollars.
                indiv_rev, indiv_npv = value_history[farm_id][cycle]
                indiv_rev /= 1000.0
                indiv_npv /= 1000.0
            else:
                indiv_rev, indiv_npv = '(no valuation)', '(no valuation)'

            cells = [farm_id, cycle_num, harvest_date, cycle_length,
                     harvest_weight, indiv_rev, indiv_npv, out_day, out_year]
            harvest_table.add_row(cells)

    doc.write_header('Farm Result Totals (output)')

    doc.write_paragraph(
        'All values in the following table were also populated in the '
        'attribute table of the netpens feature class.')

    totals_table = doc.add(html.Table(id='totals_table'))
    totals_table.add_row([
        'Farm ID Number',
        'Net Present Value (Thousands of $) (For Duration of Model Run)',
        'Number of Completed Harvest Cycles',
        'Total Volume Harvested (kg)(After Processing Occurs)'],
        is_header=True)

    for farm_id in cycle_history:
        if args['do_valuation']:
            npv = round(farms_npv[farm_id], 4)
        else:
            npv = '(no valuation)'

        num_cy_complete = len(cycle_history[farm_id])
        total_harvested = round(sum_hrv_weight[farm_id], 4)

        cells = [farm_id, npv, num_cy_complete, total_harvested]
        totals_table.add_row(cells)

    if uncertainty_stats:
        doc.write_header('Uncertainty Analysis Results')

        doc.write_paragraph(
            'These results were obtained by running a Monte Carlo simulation. '
            'For each run of the simulation, each growth parameter was '
            'randomly sampled according to the provided normal distribution. '
            'The simulation involved %d runs of the model, each with '
            'different values for the growth '
            'parameters.' % args['num_monte_carlo_runs'])

        doc.write_paragraph(
            'Results labeled as <i>total results (all farms)</i> '
            'were calculated by, for each run, summing the harvested weights '
            'and the net present values of all farms for that run.')

        # Write a table with numerical results.
        doc.write_header('Numerical Results', level=3)
        doc.write_paragraph(
            'This table summarizes the mean and standard deviation for '
            'total harvested weight (after processing) and for total '
            'net present value. The mean and standard deviation were '
            'computed for results across all runs of the Monte Carlo '
            'simulation.')

        # Add a table with uncertainty stats (mean and standard deviation).
        uncertainty_table = doc.add(html.Table(id='uncertainty_table'))
        uncertainty_table.add_two_level_header(
            outer_headers=['Harvested weight after processing (kg)',
                           'Net present value (thousands of USD)',
                           'Number of cycles'],
            inner_headers=['Mean', 'Standard deviation'],
            row_id_header='Farm ID')

        for farm in uncertainty_stats:
            row = []
            if farm == 'total':
                row.append('Total (all farms)')
            else:
                row.append('Farm %s' % farm)
            for result_type in ['weight', 'value', 'cycles']:
                # Append the mean and the standard deviation to the row.
                row += uncertainty_stats[farm][result_type]
            uncertainty_table.add_row(row)

    doc.flush()
