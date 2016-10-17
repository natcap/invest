'''This is the core module for HRA functionality. This will perform all HRA
calcs, and return the appropriate outputs.
'''

import logging
import os
import collections
import math
import datetime
import matplotlib
matplotlib.use('AGG')  # Use the Anti-Grain Geometry backend (for PNG files)
from matplotlib import pyplot as plt
import re
import random
import numpy

from osgeo import gdal, ogr, osr
import pygeoprocessing.geoprocessing

LOGGER = logging.getLogger('natcap.invest.habitat_risk_assessment.hra_core')
logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s \
   %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')


def execute(args):
    '''
    This provides the main calculation functionaility of the HRA model. This
    will call all parts necessary for calculation of final outputs.

    Inputs:
        args- Dictionary containing everything that hra_core will need to
            complete the rest of the model run. It will contain the following.
        args['workspace_dir']- Directory in which all data resides. Output
            and intermediate folders will be subfolders of this one.
        args['h_s_c']- The same as intermediate/'h-s', but with the addition
            of a 3rd key 'DS' to the outer dictionary layer. This will map to
            a dataset URI that shows the potentially buffered overlap between
            the habitat and stressor. Additionally, any raster criteria will
            be placed in their criteria name subdictionary. The overall
            structure will be as pictured:

            {(Habitat A, Stressor 1):
                    {'Crit_Ratings':
                        {'CritName':
                            {'Rating': 2.0, 'DQ': 1.0, 'Weight': 1.0}
                        },
                    'Crit_Rasters':
                        {'CritName':
                            {
                                'DS': "CritName Raster URI",
                                'Weight': 1.0, 'DQ': 1.0
                            }
                        },
                    'DS':  "A-1 Dataset URI"
                    }
            }
        args['habitats']- Similar to the h-s dictionary, a multi-level
            dictionary containing all habitat-specific criteria ratings and
            rasters. In this case, however, the outermost key is by habitat
            name, and habitats['habitatName']['DS'] points to the rasterized
            habitat shapefile URI provided by the user.
        args['h_s_e']- Similar to the h_s_c dictionary, a multi-level
            dictionary containing habitat-stressor-specific criteria ratings
            and shapes. The same as intermediate/'h-s', but with the addition
            of a 3rd key 'DS' to the outer dictionary layer. This will map to
            a dataset URI that shows the potentially buffered overlap between
            the habitat and stressor. Additionally, any raster criteria will
            be placed in their criteria name subdictionary.
        args['risk_eq']- String which identifies the equation to be used
            for calculating risk.  The core module should check for
            possibilities, and send to a different function when deciding R
            dependent on this.
        args['max_risk']- The highest possible risk value for any given pairing
            of habitat and stressor.
        args['max_stress']- The largest number of stressors that the user
            believes will overlap. This will be used to get an accurate
            estimate of risk.
        args['aoi_tables']- May or may not exist within this model run, but if
            it does, the user desires to have the average risk values by
            stressor/habitat using E/C axes for each feature in the AOI layer
            specified by 'aoi_tables'. If the risk_eq is 'Euclidean', this will
            create risk plots, otherwise it will just create the standard HTML
            table for either 'Euclidean' or 'Multiplicative.'
        args['aoi_key']- The form of the word 'Name' that the aoi layer uses
            for this particular model run.
        args['warnings']- A dictionary containing items which need to be
            acted upon by hra_core. These will be split into two categories.
            'print' contains statements which will be printed using
            logger.warn() at the end of a run. 'unbuff' is for pairs which
            should use the unbuffered stressor file in lieu of the decayed
            rated raster.

            {'print': ['This is a warning to the user.', 'This is another.'],
              'unbuff': [(HabA, Stress1), (HabC, Stress2)]
            }
    Outputs:
        --Intermediate--
            These should be the temp risk and criteria files needed for the
            final output calcs.
        --Output--
            /output/maps/recov_potent_H[habitatname].tif- Raster layer
                depicting the recovery potential of each individual habitat.
            /output/maps/cum_risk_H[habitatname]- Raster layer depicting the
                cumulative risk for all stressors in a cell for the given
                habitat.
            /output/maps/ecosys_risk- Raster layer that depicts the sum of all
                cumulative risk scores of all habitats for that cell.
            /output/maps/[habitatname]_HIGH_RISK- A raster-shaped shapefile
                containing only the "high risk" areas of each habitat, defined
                as being above a certain risk threshold.

    Returns nothing.
    '''
    inter_dir = os.path.join(args['workspace_dir'], 'intermediate')
    output_dir = os.path.join(args['workspace_dir'], 'output')

    LOGGER.info('Applying CSV criteria to rasters.')
    crit_lists, denoms = pre_calc_denoms_and_criteria(
        inter_dir,
        args['h_s_c'],
        args['habitats'],
        args['h_s_e'])

    LOGGER.info('Calculating risk rasters for individual overlaps.')
    # Need to have the h_s_c dict in there so that we can use the H-S pair DS to
    # multiply against the E/C rasters in the case of decay.
    risk_dict = make_risk_rasters(
        args['h_s_c'],
        args['habitats'],
        inter_dir, crit_lists,
        denoms,
        args['risk_eq'],
        args['warnings'])

    # Know at this point that the non-core has re-created the ouput directory
    # So we can go ahead and make the maps directory without worrying that
    # it will throw an 'already exists.'
    maps_dir = os.path.join(output_dir, 'Maps')
    os.mkdir(maps_dir)

    LOGGER.info('Calculating habitat risk rasters.')
    # We will combine all of the h-s rasters of the same habitat into
    # cumulative habitat risk rastersma db return a list of the DS's of each,
    # so that it can be read into the ecosystem risk raster's vectorize.
    h_risk_dict, h_s_risk_dict = make_hab_risk_raster(maps_dir, risk_dict)

    LOGGER.info('Making risk shapefiles.')
    # Also want to output a polygonized version of high and low risk areas in
    # each habitat. Will polygonize everything that falls above a certain
    # percentage of the total raster risk, or below that threshold. These can
    # then be fed into different models.
    num_stress = make_risk_shapes(
        maps_dir,
        crit_lists,
        h_risk_dict,
        h_s_risk_dict,
        args['max_risk'],
        args['max_stress'])

    LOGGER.info('Calculating ecosystem risk rasters.')
    # Now, combine all of the habitat rasters unto one overall ecosystem
    # rasterusing the DS's from the previous function.
    make_ecosys_risk_raster(maps_dir, h_risk_dict)

    # Recovery potential will use the 'Recovery' subdictionary from the
    # crit_lists and denoms dictionaries
    make_recov_potent_raster(maps_dir, crit_lists, denoms)

    if 'aoi_tables' in args:
        LOGGER.info('Creating subregion maps and risk plots.')

        # Let's pre-calc stuff so we don't have to worry about it in the middle
        # of the file creation.
        avgs_dict, aoi_names = pre_calc_avgs(
            inter_dir,
            risk_dict,
            args['aoi_tables'],
            args['aoi_key'],
            args['risk_eq'],
            args['max_risk'])
        aoi_pairs = rewrite_avgs_dict(avgs_dict, aoi_names)

        tables_dir = os.path.join(output_dir, 'HTML_Plots')
        os.mkdir(tables_dir)

        make_aoi_tables(tables_dir, aoi_pairs)

        if args['risk_eq'] == 'Euclidean':
            make_risk_plots(
                tables_dir,
                aoi_pairs,
                args['max_risk'],
                args['max_stress'],
                num_stress,
                len(h_risk_dict))

    # Want to clean up the intermediate folder containing the added r/dq*w
    # rasters, since it serves no purpose for the users.
    # unecessary_folder = os.path.join(inter_dir, 'ReBurned_Crit_Rasters')
    # shutil.rmtree(unecessary_folder)

    # Want to remove that AOI copy that we used for ID number->name translation.
    #if 'aoi_tables' in args:
    #    unnecessary_file = os.path.join(inter_dir, 'temp_aoi_copy.shp')
    #    os.remove(unnecessary_file)

    # Want to print out our warnings as the last possible things in the
    # console window.
    for text in args['warnings']['print']:

        LOGGER.warn(text)


def make_risk_plots(out_dir, aoi_pairs, max_risk, max_stress, num_stress, num_habs):
    '''
    This function will produce risk plots when the risk equation is
    euclidean.

    Args:
        out_dir (string): The directory into which the completed risk plots should
            be placed.

        aoi_pairs (dictionary):

            {'AOIName':
                [(HName, SName, E, C, Risk), ...],
                ....
            }

        max_risk (float): Double representing the highest potential value for a
            single h-s raster. The amount of risk for a given Habitat raster
            would be SUM(s) for a given h.
        max_stress (float): The largest number of stressors that the user
            believes will overlap. This will be used to get an accurate
            estimate of risk.
        num_stress (dict): A dictionary that simply associates every habaitat
            with the number of stressors associated with it. This will help us
            determine the max E/C we should be expecting in our overarching
            ecosystem plot.

    Returns:
        None

    Outputs:
        A set of .png images containing the matplotlib plots for every H-S
        combination. Within that, each AOI will be displayed as plotted by
        (E,C) values.

        A single png that is the "ecosystem plot" where the E's for each AOI
        are the summed
    '''

    def plot_background_circle(max_value):
        circle_color_list = [(6, '#000000'),
                             (5, '#780000'),
                             (4.75, '#911206'),
                             (4.5, '#AB2C20'),
                             (4.25, '#C44539'),
                             (4, '#CF5B46'),
                             (3.75, '#D66E54'),
                             (3.5, '#E08865'),
                             (3.25, '#E89D74'),
                             (3, '#F0B686'),
                             (2.75, '#F5CC98'),
                             (2.5, '#FAE5AC'),
                             (2.25, '#FFFFBF'),
                             (2, '#EAEBC3'),
                             (1.75, '#CFD1C5'),
                             (1.5, '#B9BEC9'),
                             (1.25, '#9FA7C9'),
                             (1, '#8793CC'),
                             (0.75, '#6D83CF'),
                             (0.5, '#5372CF'),
                             (0.25, '#305FCF')]
        index = 0
        for radius, color in circle_color_list:
            index += 1
            linestyle = 'solid' if index % 2 == 0 else 'dashed'
            cir = plt.Circle(
                (0, 0),
                edgecolor='.25',
                linestyle=linestyle,
                radius=radius * max_value / 3.75,
                fc=color)
            plt.gca().add_patch(cir)

    def jigger(E, C):
        '''
        Want to return a fractionally offset set of coordinates so that
        each of the text related to strings is slightly offset.

        Range of x: E <= x <= E+.1
        Range of y: C-.1 <= y <= C+.1
        '''

        x = E + random.random() * .1
        y = C + ((random.random() * .4) - .2)

        return (x, y)

    # Create plots for each combination of AOI, Hab
    plot_index = 0

    for aoi_name, aoi_list in aoi_pairs.iteritems():

        LOGGER.debug("AOI list for %s: %s" % (aoi_name, aoi_list))

        fig = plt.figure(plot_index)
        plot_index += 1
        plt.suptitle(aoi_name)
        fig.text(0.5, 0.04, 'Exposure', ha='center', va='center')
        fig.text(0.06, 0.5, 'Consequence', ha='center', va='center',
                 rotation='vertical')

        hab_index = 1
        curr_hab_name = aoi_list[0][0]

        # Elements look like: (HabName, StressName, E, C, Risk)
        for element in aoi_list:
            if element == aoi_list[0]:

                # Want to have two across, and make sure there are enough
                # spaces going down for each of the subplots
                plt.subplot(int(math.ceil(num_habs / 2.0)),
                            2, hab_index)
                plot_background_circle(max_risk)
                plt.title(curr_hab_name)
                plt.xlim([-.5, max_risk])
                plt.ylim([-.5, max_risk])

            hab_name = element[0]
            if curr_hab_name == hab_name:

                plt.plot(
                    element[2], element[3], 'k^',
                    markerfacecolor='black', markersize=8)
                plt.annotate(
                    element[1], xy=(element[2], element[3]),
                    xytext=jigger(element[2], element[3]))
                continue

            # We get here once we get to the next habitat
            hab_index += 1
            plt.subplot(int(math.ceil(num_habs/2.0)),
                                      2, hab_index)
            plot_background_circle(max_risk)

            curr_hab_name = hab_name

            plt.title(curr_hab_name)
            plt.xlim([-.5, max_risk])
            plt.ylim([-.5, max_risk])

            # We still need to plot the element that gets us here.
            plt.plot(
                element[2],
                element[3],
                'k^',
                markerfacecolor='black',
                markersize=8)
            plt.annotate(
                element[1],
                xy=(element[2], element[3]),
                xytext=jigger(element[2], element[3]))

        out_uri = os.path.join(
            out_dir, 'risk_plot_' + 'AOI[' + aoi_name + '].png')

        plt.savefig(out_uri, format='png')

    # Create one ecosystem megaplot that plots the points as summed E,C from
    # a given habitat, AOI pairing. So each dot would be (HabitatName, AOI1)
    # for all habitats in the ecosystem.
    plot_index += 1
    max_tot_risk = max_risk * max_stress * num_habs

    plt.figure(plot_index)
    plt.suptitle("Ecosystem Risk")

    plot_background_circle(max_tot_risk)

    points_dict = {}

    for aoi_name, aoi_list in aoi_pairs.items():

        for element in aoi_list:

            if aoi_name in points_dict:
                points_dict[aoi_name]['E'] += element[2]
                points_dict[aoi_name]['C'] += element[3]
            else:
                points_dict[aoi_name] = {}
                points_dict[aoi_name]['E'] = 0
                points_dict[aoi_name]['C'] = 0

    for aoi_name, p_dict in points_dict.items():
        # Create the points which are summed AOI's across all Habitats.
        plt.plot(p_dict['E'], p_dict['C'], 'k^',
                               markerfacecolor='black', markersize=8)
        plt.annotate(
            aoi_name,
            xy=(p_dict['E'], p_dict['C']),
            xytext=(p_dict['E'], p_dict['C']+0.07))

    plt.xlim([0, max_tot_risk])
    plt.ylim([0, max_tot_risk])
    plt.xlabel("Exposure (Cumulative)")
    plt.ylabel("Consequence (Cumulative)")

    out_uri = os.path.join(out_dir, 'ecosystem_risk_plot.png')
    plt.savefig(out_uri, format='png')
    # Clearing the state of the axes / figures so we don't accumulate
    # duplicate information when creating plots in this function
    plt.cla()
    plt.clf()


def make_aoi_tables(out_dir, aoi_pairs):
    '''
    This function will take in an shapefile containing multiple AOIs, and
    output a table containing values averaged over those areas.

    Input:
        out_dir- The directory into which the completed HTML tables should be
            placed.
        aoi_pairs- Replacement for avgs_dict, holds all the averaged values on
            a H, S basis.

            {'AOIName':
                [(HName, SName, E, C, Risk), ...],
                ....
            }
    Output:
        A set of HTML tables which will contain averaged values of E, C, and
        risk for each H, S pair within each AOI. Additionally, the tables will
        contain a column for risk %, which is the averaged risk value in that
        area divided by the total potential risk for a given pixel in the map.

    Returns nothing.
    '''

    filename = os.path.join(
        out_dir,
        'Sub_Region_Averaged_Results_[%s].html'
        % datetime.datetime.now().strftime("%Y-%m-%d_%H_%M"))

    file = open(filename, "w")

    file.write("<html>")
    file.write("<title>" + "InVEST HRA" + "</title>")
    file.write("<CENTER><H1>" + "Habitat Risk Assessment Model" +
               "</H1></CENTER>")
    file.write("<br>")
    file.write("This page contains results from running the InVEST Habitat \
    Risk Assessment model." + "<p>" + "Each table displays values on a \
    per-habitat basis. For each overlapping stressor within the model, the \
    averages for the desired sub-regions are presented. C, E, and Risk values \
    are calculated as an average across a given subregion. Risk Percentage is \
    calculated as a function of total potential risk within that area.")
    file.write("<br><br>")
    file.write("<HR>")

    # Now, all of the actual calculations within the table. We want to make one
    # table for each AOi used on the subregions shapefile.
    for aoi_name, aoi_list in aoi_pairs.items():
        file.write("<H2>" + aoi_name + "</H2>")
        file.write('<table border="1", cellpadding="5">')

        # Headers row
        file.write(
            "<tr><b><td>Habitat Name</td><td>Stressor Name</td>" +
            "<td>E</td><td>C</td><td>Risk</td><td>Risk %</td></b></tr>")

        # Element looks like (HabName, StressName, E, C, Risk)
        for element in aoi_list:

            file.write("<tr>")
            file.write("<td>" + element[0] + "</td>")
            file.write("<td>" + element[1] + "</td>")
            file.write("<td>" + str(round(element[2], 2)) + "</td>")
            file.write("<td>" + str(round(element[3], 2)) + "</td>")
            file.write("<td>" + str(round(element[4], 2)) + "</td>")
            file.write("<td>" + str(round(element[5] * 100, 2)) + "</td>")
            file.write("</tr>")

        # End of the AOI-specific table
        file.write("</table>")

    # End of the page.
    file.write("</html>")
    file.close()


def rewrite_avgs_dict(avgs_dict, aoi_names):
    '''
    Aftermarket rejigger of the avgs_dict setup so that everything is AOI
    centric instead. Should produce something like the following:

    {'AOIName':
        [(HName, SName, E, C, Risk, R_Pct), ...],
        ....
    }
    '''
    pair_dict = {}

    for aoi_name in aoi_names:
        pair_dict[aoi_name] = []

        for h_name, h_dict in avgs_dict.items():
            for s_name, s_list in h_dict.items():
                for aoi_dict in s_list:
                    if aoi_dict['Name'] == aoi_name:
                        pair_dict[aoi_name].append((
                            h_name,
                            s_name,
                            aoi_dict['E'],
                            aoi_dict['C'],
                            aoi_dict['Risk'],
                            aoi_dict['R_Pct']))

    return pair_dict


def pre_calc_avgs(inter_dir, risk_dict, aoi_uri, aoi_key, risk_eq, max_risk):
    '''
    This funtion is a helper to make_aoi_tables, and will just handle
    pre-calculation of the average values for each aoi zone.

    Input:
        inter_dir- The directory which contains the individual E and C rasters.
            We can use these to get the avg. E and C values per area. Since we
            don't really have these in any sort of dictionary, will probably
            just need to explicitly call each individual file based on the
            names that we pull from the risk_dict keys.
        risk_dict- A simple dictionary that maps a tuple of
            (Habitat, Stressor) to the URI for the risk raster created when the
            various sub components (H/S/H_S) are combined.

            {('HabA', 'Stress1'): "A-1 Risk Raster URI",
            ('HabA', 'Stress2'): "A-2 Risk Raster URI",
            ...
            }
        aoi_uri- The location of the AOI zone files. Each feature within this
            file (identified by a 'name' attribute) will be used to average
            an area of E/C/Risk values.
        risk_eq- A string identifier, either 'Euclidean' or 'Multiplicative'
            that tells us which equation should be used for calculation of
            risk. This will be used to get the risk value for the average E
            and C.
        max_risk- The user reported highest risk score present in the CSVs.

    Returns:
        avgs_dict- A multi level dictionary to hold the average values that
            will be placed into the HTML table.

            {'HabitatName':
                {'StressorName':
                    [{'Name': AOIName, 'E': 4.6, 'C': 2.8, 'Risk': 4.2},
                        {...},
                    ...
                    ]
                },
                ....
            }
       aoi_names- Quick and dirty way of getting the AOI keys.
    '''
    # Since we know that the AOI will be consistent across all of the rasters,
    # want to create the new int field, and the name mapping dictionary upfront

    driver = ogr.GetDriverByName('ESRI Shapefile')
    aoi = ogr.Open(aoi_uri)
    cp_aoi_uri = os.path.join(inter_dir, 'temp_aoi_copy.shp')
    cp_aoi = driver.CopyDataSource(aoi, cp_aoi_uri)

    layer = cp_aoi.GetLayer()

    field_defn = ogr.FieldDefn('BURN_ID', ogr.OFTInteger)
    layer.CreateField(field_defn)

    name_map = {}
    count = 0
    ids = []

    for feature in layer:

        ids.append(count)
        name = feature.items()[aoi_key]
        feature.SetField('BURN_ID', count)
        name_map[count] = name
        count += 1

        layer.SetFeature(feature)

    layer.ResetReading()

    # Now we will loop through all of the various pairings to deal with all
    # their component parts across our AOI. Want to make sure to use our new
    # field as the index.
    avgs_dict = {}
    avgs_r_sum = {}

    # Set a temp filename for the AOI raster.
    aoi_rast_uri = pygeoprocessing.geoprocessing.temporary_filename()

    # Need an arbitrary element upon which to base the new raster.
    arb_raster_uri = next(risk_dict.itervalues())
    LOGGER.debug("arb_uri: %s" % arb_raster_uri)

    # Use the first overlap raster as the base for the AOI
    pygeoprocessing.geoprocessing.new_raster_from_base_uri(
        arb_raster_uri,
        aoi_rast_uri,
        'GTiff',
        -1,
        gdal.GDT_Float32)

    # This rasterize should burn a unique burn ID int to each. Need to have a
    # dictionary which associates each burn ID with the AOI 'name' attribute
    # that's required.
    pygeoprocessing.geoprocessing.rasterize_layer_uri(
        aoi_rast_uri,
        cp_aoi_uri,
        option_list=["ATTRIBUTE=BURN_ID", "ALL_TOUCHED=TRUE"])

    for pair in risk_dict:
        h, s = pair

        if h not in avgs_dict:
            avgs_dict[h] = {}
            avgs_r_sum[h] = {}
        if s not in avgs_dict[h]:
            avgs_dict[h][s] = []

        # Just going to have to pull explicitly. Too late to go back and
        # rejigger now.
        e_rast_uri = os.path.join(
            inter_dir, "H[" + h + ']_S[' + s +
            ']_E_Risk_Raster.tif')

        c_rast_uri = os.path.join(
            inter_dir, "H[" + h + ']_S[' + s +
            ']_C_Risk_Raster.tif')

        # Now, we are going to modify the e value by the spatial overlap value.
        # Get S.O value first.
        h_rast_uri = os.path.join(inter_dir, 'Habitat_Rasters', h + '.tif')
        hs_rast_uri = os.path.join(
            inter_dir, 'Overlap_Rasters', "H[" +
            h + ']_S[' + s + '].tif')

        LOGGER.debug("Entering new funct.")
        rast_uri_list = [e_rast_uri, c_rast_uri, h_rast_uri, hs_rast_uri]
        rast_labels = ['E', 'C', 'H', 'H_S']
        over_pix_sums = aggregate_multi_rasters_uri(
            aoi_rast_uri,
            rast_uri_list,
            rast_labels,
            [0])
        LOGGER.debug("%s,%s:%s" % (h, s, over_pix_sums))
        LOGGER.debug("Exiting new funct.")

        for burn_value in over_pix_sums:

            subregion_name = name_map[burn_value]

            # For a given layer under the AOI, first list item is #of pix,
            # second is pix sum
            if over_pix_sums[burn_value]['H'][0] == 0:
                frac_over = 0.
            else:
                # Casting to float because otherwise we end up with integer
                # division issues.
                frac_over = over_pix_sums[burn_value]['H_S'][0] / float(
                    over_pix_sums[burn_value]['H'][0])

            s_o_score = max_risk * frac_over + (1-frac_over)

            if frac_over == 0.:
                e_score = 0.
            # Know here that there is overlap. So now check whether we have
            # scoring from users. If no, just use spatial overlap.
            else:
                e_mean = (over_pix_sums[burn_value]['E'][1] /
                          over_pix_sums[burn_value]['E'][0])

                if e_mean == 0.:
                    e_score = s_o_score

                # If there is, want to average the spatial overlap into
                # everything else.
                else:
                    e_score = (e_mean + s_o_score) / 2

            # If there's no habitat, my E is 0 (indicating that there's no
            # spatial overlap), then my C and risk scores should also be 0.
            # Setting E to 0 should cascade to also make risk 0.
            if e_score == 0.:
                avgs_dict[h][s].append(
                    {'Name': subregion_name, 'E': 0.,
                     'C': 0.})
            else:
                c_mean = (over_pix_sums[burn_value]['C'][1] /
                          over_pix_sums[burn_value]['C'][0])
                avgs_dict[h][s].append(
                    {'Name': subregion_name, 'E': e_score,
                     'C': c_mean})

    for h, hab_dict in avgs_dict.iteritems():
        for s, sub_list in hab_dict.iteritems():
            for sub_dict in sub_list:

                # For the average risk, want to use the avg. E and C values
                # that we just got.
                if risk_eq == 'Euclidean':

                    c_val = 0 if sub_dict['C'] == 0. else sub_dict['C'] - 1
                    e_val = 0 if sub_dict['E'] == 0. else sub_dict['E'] - 1

                    r_val = math.sqrt((c_val)**2 + (e_val)**2)
                else:
                    r_val = sub_dict['C'] * sub_dict['E']

                sub_dict['Risk'] = r_val

                if sub_dict['Name'] in avgs_r_sum[h]:
                    avgs_r_sum[h][sub_dict['Name']] += r_val
                else:
                    avgs_r_sum[h][sub_dict['Name']] = r_val

    for h, hab_dict in avgs_dict.iteritems():
        for s, sub_list in hab_dict.iteritems():
            for sub_dict in sub_list:
                # Want to avoid div by 0 errors if there is none of a particular
                # habitat within a subregion. Thus, if the total for risk for a
                # habitat is 0, just return 0 as a percentage too.
                curr_total_risk = avgs_r_sum[h][sub_dict['Name']]

                if curr_total_risk == 0.:
                    sub_dict['R_Pct'] = 0.
                else:
                    sub_dict['R_Pct'] = sub_dict['Risk']/curr_total_risk

    return avgs_dict, name_map.values()


def aggregate_multi_rasters_uri(aoi_rast_uri, rast_uris, rast_labels, ignore_value_list=[]):
    '''Will take a stack of rasters and an AOI, and return a dictionary
    containing the number of overlap pixels, and the value of those pixels for
    each overlap of raster and AOI.

    Input:
        aoi_uri- The location of an AOI raster which MUST have individual ID
            numbers with the attribute name 'BURN_ID' for each feature on the
            map.
        rast_uris- List of locations of the rasters which should be overlapped
            with the AOI.
        rast_labels- Names for each raster layer that will be retrievable from
            the output dictionary.
        ignore_value_list- Optional argument that provides a list of values
            which should be ignored if they crop up for a pixel value of one
            of the layers.
    Returns:
        layer_overlap_info-
            {AOI Data Value 1:
                {rast_label: [#of pix, pix value],
                rast_label: [200, 2567.97], ...
            }
    '''

    cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(aoi_rast_uri)
    nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(aoi_rast_uri)

    rast_uris = [aoi_rast_uri] + rast_uris

    # Want to create a set of temporary filenames, just need to be sure to
    # clean them up at the end.
    temp_rast_uris = [
        pygeoprocessing.geoprocessing.temporary_filename() for _ in range(len(rast_uris))]

    pygeoprocessing.geoprocessing.align_dataset_list(
        rast_uris,
        temp_rast_uris,
        ['nearest'] * len(rast_uris),
        cell_size,
        "dataset",
        0,
        dataset_to_bound_index=0)

    rast_ds_list = [gdal.Open(uri) for uri in temp_rast_uris]
    rast_bands = [ds.GetRasterBand(1) for ds in rast_ds_list]

    # Get the AOI to use for line by line, then cell by cell iterration.
    aoi_band = rast_bands[0]

    n_cols = aoi_band.XSize
    n_rows = aoi_band.YSize

    # Set up numpy arrays that currently hold only zeros, but will be used for
    # each row read.
    aoi_row = numpy.zeros((1, n_cols), numpy.float64, 'C')

    rows_dict = {}
    for layer_name in rast_labels:
        rows_dict[layer_name] = numpy.zeros((1, n_cols), numpy.float64, 'C')

    # Now iterate through every cell of the aOI, and concat everything that's
    # undr it and store that.

    # this defaults a dictionary so we can initalize layer_overlap
    # info[aoi_pix][layer_name] = [0,0.]
    layer_overlap_info = collections.defaultdict(
        lambda: collections.defaultdict(lambda: list([0, 0.])))
    for row_index in range(n_rows):
        aoi_band.ReadAsArray(
            yoff=row_index,
            win_xsize=n_cols,
            win_ysize=1,
            buf_obj=aoi_row)

        for idx, layer_name in enumerate(rast_labels):
            rast_bands[idx+1].ReadAsArray(
                yoff=row_index,
                win_xsize=n_cols,
                win_ysize=1,
                buf_obj=rows_dict[layer_name])

        for aoi_pix_value in numpy.unique(aoi_row):
            if aoi_pix_value == nodata:
                continue

            aoi_mask = (aoi_row == aoi_pix_value)

            for layer_name in rast_labels:
                valid_rows_dict_mask = (
                    rows_dict[layer_name] != nodata) & aoi_mask
                for ignore_value in ignore_value_list:
                    valid_rows_dict_mask = valid_rows_dict_mask & (
                        rows_dict[layer_name] != ignore_value)

                layer_sum = numpy.sum(
                    rows_dict[layer_name][valid_rows_dict_mask])
                layer_count = numpy.count_nonzero(valid_rows_dict_mask)

                layer_overlap_info[aoi_pix_value][layer_name][0] += layer_count
                layer_overlap_info[aoi_pix_value][layer_name][1] += layer_sum
    return layer_overlap_info


def make_recov_potent_raster(dir, crit_lists, denoms):
    '''
    This will do the same h-s calculation as used for the individual E/C
    calculations, but instead will use r/dq as the equation for each criteria.
    The full equation will be:

        SUM HAB CRITS( r/dq )
        ---------------------
        SUM HAB CRITS( 1/dq )

    Input:
        dir- Directory in which the completed raster files should be placed.
        crit_lists- A dictionary containing pre-burned criteria which can be
            combined to get the E/C for that H-S pairing.

            {'Risk': {
                'h_s_c': {
                    (hab1, stressA):
                        ["indiv num raster URI",
                                    "raster 1 URI", ...],
                                 (hab1, stressB): ...
                               },
                        'h':   {
                            hab1: ["indiv num raster URI", "raster 1 URI"],
                                ...
                               },
                        'h_s_e': { (hab1, stressA): ["indiv num raster URI"]
                               }
                     }
             'Recovery': { hab1: ["indiv num raster URI", ...],
                           hab2: ...
                         }
            }
        denoms- Dictionary containing the combined denominator for a given
            H-S overlap. Once all of the rasters are combined, each H-S raster
            can be divided by this.

            {'Risk': {
                'h_s_c': {
                    (hab1, stressA): {
                        'CritName': 2.0, ...},
                    (hab1, stressB): {'CritName': 1.3, ...}
                               },
                        'h':   { hab1: {'CritName': 1.3, ...},
                                ...
                               },
                        'h_s_e': { (hab1, stressA): {'CritName': 1.3, ...}
                               }
                     }
             'Recovery': { hab1: {'critname': 1.6, ...}
                           hab2: ...
                         }
            }
    Output:
        A raster file for each of the habitats included in the model displaying
            the recovery potential within each potential grid cell.

    Returns nothing.
    '''
    # Want all of the unique habitat names
    habitats = denoms['Recovery'].keys()

    # First, going to try doing everything all at once. For every habitat,
    # concat the lists of criteria rasters.
    for h in habitats:

        curr_list = crit_lists['Recovery'][h]
        curr_crit_names = map(lambda uri: re.match(
            '.*\]_([^_]*)',
            os.path.splitext(os.path.basename(uri))[0]).group(1), curr_list)
        curr_denoms = denoms['Recovery'][h]

        def add_recov_pix(*pixels):
            '''We will have burned numerator values for the recovery potential
            equation. Want to add all of the numerators (r/dq), then divide by
            the denoms added together (1/dq).'''

            value = numpy.zeros(pixels[0].shape)
            denom_val = numpy.zeros(pixels[0].shape)
            all_nodata = numpy.zeros(pixels[0].shape, dtype=numpy.bool)
            all_nodata[:] = True

            for i in range(len(pixels)):
                valid_mask = pixels[i] != -1
                value = numpy.where(valid_mask, pixels[i] + value, value)
                denom_val = numpy.where(
                    valid_mask,
                    curr_denoms[curr_crit_names[i]] + denom_val,
                    denom_val)

                # Bitwise and- if both are true, will still return True
                all_nodata = ~valid_mask & all_nodata

            # turn off dividie by zero warning because we probably will divide
            # by zero
            olderr = numpy.seterr(divide='ignore')
            result = numpy.where(denom_val != 0, value / denom_val, 0.0)
            # return numpy error state to old value
            numpy.seterr(**olderr)

            # mask out nodata stacks
            return numpy.where(all_nodata, -1, result)

            '''
            all_nodata = True
            for p in pixels:
                if p not in [-1., -1]:
                    all_nodata = False
            if all_nodata:
                return -1.

            value = 0.
            denom_val = 0.

            for i in range(0, len(pixels)):

                p = pixels[i]

                if p not in [-1., -1]:
                    value += p
                    denom_val += curr_denoms[curr_crit_names[i]]

            if value in [0, 0.]:
                return 0
            else:

                value = value / denom_val
                return value'''

        # Need to get the arbitrary first element in order to have a pixel size
        # to use in vectorize_datasets. One hopes that we have at least 1 thing
        # in here.
        pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(curr_list[0])

        out_uri = os.path.join(dir, 'recov_potent_H[' + h + '].tif')

        pygeoprocessing.geoprocessing.vectorize_datasets(
            curr_list,
            add_recov_pix,
            out_uri,
            gdal.GDT_Float32,
            -1.,
            pixel_size,
            "union",
            resample_method_list=None,
            dataset_to_align_index=0,
            aoi_uri=None,
            vectorize_op=False)


def make_ecosys_risk_raster(dir, h_dict):
    '''
    This will make the compiled raster for all habitats within the ecosystem.
    The ecosystem raster will be a direct sum of each of the included habitat
    rasters.

    Input:
        dir- The directory in which all completed should be placed.
        h_dict- A dictionary of raster dataset URIs which can be combined to
            create an overall ecosystem raster. The key is the habitat name,
            and the value is the dataset URI.

            {'Habitat A': "Overall Habitat A Risk Map URI",
            'Habitat B': "Overall Habitat B Risk URI"
             ...
            }
    Output:
        ecosys_risk.tif- An overall risk raster for the ecosystem. It will
            be placed in the dir folder.

    Returns nothing.
    '''
    # Need a straight list of the values from h_dict
    h_list = h_dict.values()
    pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(h_list[0])

    nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(h_list[0])

    out_uri = os.path.join(dir, 'ecosys_risk.tif')

    def add_e_pixels(*pixels):
        '''
        Sum all risk pixels to make a single habitat raster out of all the
        h-s overlap rasters.
        '''

        value = numpy.zeros(pixels[0].shape)
        all_nodata = numpy.zeros(pixels[0].shape, dtype=numpy.bool)
        all_nodata[:] = True

        for i in range(len(pixels)):
            valid_mask = pixels[i] != -1

            value = numpy.where(valid_mask, pixels[i] + value, value)

            all_nodata = ~valid_mask & all_nodata

        return numpy.where(all_nodata, -1, value)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        h_list,
        add_e_pixels,
        out_uri,
        gdal.GDT_Float32,
        -1.,
        pixel_size,
        "union",
        resample_method_list=None,
        dataset_to_align_index=0,
        aoi_uri=None,
        vectorize_op=False)


def make_risk_shapes(dir, crit_lists, h_dict, h_s_dict, max_risk, max_stress):
    '''
    This function will take in the current rasterized risk files for each
    habitat, and output a shapefile where the areas that are "HIGH RISK" (high
    percentage of risk over potential risk) are the only existing polygonized
    areas.

    Additonally, we also want to create a shapefile which is only the
    "low risk" areas- actually, those that are just not high risk (it's the
    combination of low risk areas and medium risk areas).

    Since the pygeoprocessing.geoprocessing function can only take in ints, want to predetermine

    what areas are or are not going to be shapefile, and pass in a raster that
    is only 1 or nodata.

    Input:
        dir- Directory in which the completed shapefiles should be placed.
        crit_lists- A dictionary containing pre-burned criteria which can be
            combined to get the E/C for that H-S pairing.

            {'Risk': {
                'h_s_c': { (hab1, stressA): ["indiv num raster URI",
                                    "raster 1 URI", ...],
                                 (hab1, stressB): ...
                               },
                        'h':   {
                            hab1: ["indiv num raster URI", "raster 1 URI"],
                                ...
                               },
                        'h_s_e': {(hab1,  stressA): ["indiv num raster URI"]
                               }
                     }
             'Recovery': { hab1: ["indiv num raster URI", ...],
                           hab2: ...
                         }
            }
        h_dict- A dictionary that contains raster dataset URIs corresponding
            to each of the habitats in the model. The key in this dictionary is
            the name of the habiat, and it maps to the open dataset.
        h_s_dict- A dictionary that maps a habitat name to the risk rasters
            for each of the applicable stressors.

            {'HabA': ["A-1 Risk Raster URI", "A-2 Risk Raster URI", ...],
             'HabB': ["B-1 Risk Raster URI", "B-2 Risk Raster URI", ...], ...
            }
        max_risk- Double representing the highest potential value for a single
            h-s raster. The amount of risk for a given Habitat raster would be
            SUM(s) for a given h.
        max_stress- The largest number of stressors that the user believes will
            overlap. This will be used to get an accurate estimate of risk.

     Output:
        Returns two shapefiles for every habitat, one which shows features only
        for the areas that are "high risk" within that habitat, and one which
        shows features only for the combined low + medium risk areas.

     Return:
        num_stress- A dictionary containing the number of stressors being
            associated with each habitat. The key is the string name of the
            habitat, and it maps to an int counter of number of stressors.
     '''
    # For each h, want  to know how many stressors are associated with it. This
    # allows us to not have to think about whether or not a h-s pair was zero'd
    # out by weighting or DQ.
    num_stress = collections.Counter()
    for pair in crit_lists['Risk']['h_s_c']:
        h, _ = pair

        if h in num_stress:
            num_stress[h] += 1
        else:
            num_stress[h] = 1

    # This is the user definied threshold overlap of stressors, multipled by the
    # maximum potential risk for any given overlap between habitat and stressor
    # This yields a user defined threshold for risk.
    user_max_risk = max_stress * max_risk

    def high_risk_raster(*pixels):

        # H_Raster is first in the stack.
        high_h_mask = numpy.where(
            pixels[0] != -1,
            pixels[0] / float(user_max_risk) >= .666,
            False)

        high_hs = numpy.zeros(pixels[0].shape, dtype=numpy.bool)

        for i in range(1, len(pixels)):
            high_hs = high_hs | (pixels[i] / float(max_risk) >= .666)

        return numpy.where(high_hs | high_h_mask, 3, -1)

        '''#We know that the overarching habitat pixel is the first in the list
        h_pixel = pixels[0]
        h_percent = float(h_pixel)/ user_max_risk

        #high risk is classified as the top third of risk
        if h_percent >= .666:
            return 1
        #If we aren't getting high risk from just the habitat pixel,
        #want to secondarily check each of the h_s pixels.
        for p in pixels[1::]:

            p_percent = float(p) / max_risk
            if p_percent >= .666:
                return 1

        #If we get here, neither the habitat raster nor the h_s_raster are
        #considered high risk. Can return nodata.
        return -1.'''

    def med_risk_raster(*pixels):

        med_h_mask = numpy.where(
            pixels[0] != -1,
            (pixels[0] / float(user_max_risk) < .666) &
            (pixels[0] / float(user_max_risk) >= .333),
            False)

        med_hs = numpy.zeros(pixels[0].shape, dtype=numpy.bool)

        for i in range(1, len(pixels)):
            med_hs = med_hs | \
                ((pixels[i] / float(max_risk) < .666) &
                    (pixels[i] / float(max_risk) >= .333))

        return numpy.where(med_hs | med_h_mask, 2, -1)
        '''#We know that the overarching habitat pixel is the first in the list
        h_pixel = pixels[0]
        h_percent = float(h_pixel)/ user_max_risk

        #medium risk is classified as the middle third of risk
        if .333 <= h_percent < .666:
            return 1
        #If we aren't getting medium risk from just the habitat pixel,
        #want to secondarily check each of the h_s pixels.
        for p in pixels[1::]:

            p_percent = float(p) / max_risk
            if .333 <= p_percent < .666:
                return 1

        #If we get here, neither the habitat raster nor the h_s_raster are
        #considered med risk. Can return nodata.
        return -1.'''

    def low_risk_raster(*pixels):
        low_h_mask = numpy.where(
            pixels[0] != -1,
            (pixels[0] / float(user_max_risk) < .333) &
            (pixels[0] / float(user_max_risk) >= 0),
            False)

        low_hs = numpy.zeros(pixels[0].shape, dtype=numpy.bool)

        for i in range(1, len(pixels)):
            low_hs = (low_hs |
                      ((pixels[i] / float(user_max_risk) < .333) &
                       (pixels[i] / float(user_max_risk) >= 0)))

        return numpy.where(low_hs | low_h_mask, 1, -1)

        '''#We know that the overarching habitat pixel is the first in the list
        h_pixel = pixels[0]
        h_percent = float(h_pixel)/ user_max_risk

        #low risk is classified as the lowest third of risk
        if 0. <= h_percent < .333:
            return 1
        #If we aren't getting low risk from just the habitat pixel,
        #want to secondarily check each of the h_s pixels.
        for p in pixels[1::]:

            p_percent = float(p) / max_risk
            if 0. <= p_percent < .333:
                return 1

        #If we get here, neither the habitat raster nor the h_s_raster are
        #considered low risk. Can return nodata.
        return -1.'''

    def combo_risk_raster(*pixels):
        # We actually know that there will be a l_pix, m_pix, and h_pix
        # But it's easier to just loop through all of them.

        combo_risk = numpy.zeros(pixels[0].shape)
        combo_risk[:] = -1

        for layer in pixels:
            combo_risk = numpy.where(layer != -1, layer, combo_risk)

        return combo_risk

        '''if h_pix != -1.:
            return 3
        elif m_pix != -1.:
            return 2
        elif l_pix != -1.:
            return 1
        else:
            return -1.'''

    for h in h_dict:
        # Want to know the number of stressors for the current habitat
        # curr_top_risk = num_stress[h] * max_risk
        curr_top_risk = 3 * max_risk
        # Make list on the fly for rasters which could be high risk. Want to
        # make sure that we're passing in the h risk raster first so that we
        # know it from the rest.
        old_ds_uri = h_dict[h]
        risk_raster_list = [old_ds_uri] + h_s_dict[h]

        grid_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(old_ds_uri)

        h_out_uri_r = os.path.join(dir, '[' + h + ']_HIGH_RISK.tif')

        pygeoprocessing.geoprocessing.vectorize_datasets(
            risk_raster_list,
            high_risk_raster,
            h_out_uri_r,
            gdal.GDT_Float32,
            -1.,
            grid_size,
            "union",
            resample_method_list=None,
            dataset_to_align_index=0,
            aoi_uri=None,
            vectorize_op=False)

        # Medium area would be here.
        m_out_uri_r = os.path.join(dir, '[' + h + ']_MED_RISK.tif')

        pygeoprocessing.geoprocessing.vectorize_datasets(
            risk_raster_list,
            med_risk_raster,
            m_out_uri_r,
            gdal.GDT_Float32,
            -1.,
            grid_size,
            "union",
            resample_method_list=None,
            dataset_to_align_index=0,
            aoi_uri=None,
            vectorize_op=False)

        # Now, want to do the low area.
        l_out_uri_r = os.path.join(dir, '[' + h + ']_LOW_RISK.tif')

        pygeoprocessing.geoprocessing.vectorize_datasets(
            risk_raster_list,
            low_risk_raster,
            l_out_uri_r,
            gdal.GDT_Float32,
            -1.,
            grid_size,
            "union",
            resample_method_list=None,
            dataset_to_align_index=0,
            aoi_uri=None,
            vectorize_op=False)

        # Want to do another vectorize in order to create a single shapefile
        # with high, medium, low values.
        single_raster_uri_r = os.path.join(dir, '[' + h + ']_ALL_RISK.tif')
        single_raster_uri = os.path.join(dir, '[' + h + ']_RISK.shp')

        pygeoprocessing.geoprocessing.vectorize_datasets(
            [l_out_uri_r, m_out_uri_r, h_out_uri_r],
            combo_risk_raster,
            single_raster_uri_r,
            gdal.GDT_Float32,
            -1.,
            grid_size,
            "union",
            resample_method_list=None,
            dataset_to_align_index=0,
            aoi_uri=None,
            vectorize_op=False)

        raster_to_polygon(
            single_raster_uri_r,
            single_raster_uri,
            h,
            'VALUE')

        # Now, want to delete all the other rasters that we don't need for risk.
        for file_uri in [h_out_uri_r,
                         m_out_uri_r,
                         l_out_uri_r,
                         single_raster_uri_r]:
            os.remove(file_uri)

    return num_stress


def raster_to_polygon(raster_uri, out_uri, layer_name, field_name):
    '''
    This will take in a raster file, and output a shapefile of the same
    area and shape.

    Input:
        raster_uri- The raster that needs to be turned into a shapefile. This
            is only the URI to the raster, we will need to get the band.
        out_uri- The desired URI for the new shapefile.
        layer_name- The name of the layer going into the new shapefile.
        field-name- The name of the field that will contain the raster pixel
            value.

    Output:
        This will be a shapefile in the shape of the raster. The raster being
        passed in will be solely "high risk" areas that conatin data, and
        nodata values for everything else.

    Returns nothing.
    '''
    raster = gdal.Open(raster_uri)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(out_uri)

    spat_ref = osr.SpatialReference()
    proj = raster.GetProjectionRef()
    spat_ref.ImportFromWkt(proj)

    layer_name = layer_name.encode('utf-8')
    layer = ds.CreateLayer(layer_name, spat_ref, ogr.wkbPolygon)

    field_defn = ogr.FieldDefn(field_name, ogr.OFTReal)
    layer.CreateField(field_defn)

    band = raster.GetRasterBand(1)
    mask = band.GetMaskBand()

    gdal.Polygonize(band, mask, layer, 0)

    # Now, want to loop through the polygons that we just created, and add a new
    # field with a string description, depending on what the 3/2/1 number is.
    field_defn = ogr.FieldDefn('CLASSIFY', ogr.OFTString)
    layer.CreateField(field_defn)

    for feature in layer:

        class_number = feature.items()['VALUE']

        if class_number == 3:
            feature.SetField('CLASSIFY', 'HIGH')
        elif class_number == 2:
            feature.SetField('CLASSIFY', 'MED')
        elif class_number == 1:
            feature.SetField('CLASSIFY', 'LOW')

        layer.SetFeature(feature)

    layer = None
    ds.SyncToDisk()


def make_hab_risk_raster(dir, risk_dict):
    '''
    This will create a combined raster for all habitat-stressor pairings
    within one habitat. It should return a list of open rasters that correspond
    to all habitats within the model.

    Input:
        dir- The directory in which all completed habitat rasters should be
            placed.
        risk_dict- A dictionary containing the risk rasters for each pairing of
            habitat and stressor. The key is the tuple of (habitat, stressor),
            and the value is the raster dataset URI corresponding to that
            combination.

            {('HabA', 'Stress1'): "A-1 Risk Raster URI",
            ('HabA', 'Stress2'): "A-2 Risk Raster URI",
            ...
            }
    Output:
        A cumulative risk raster for every habitat included within the model.

    Returns:
        h_rasters- A dictionary containing habitat names mapped to the dataset
            URI of the overarching habitat risk map for this model run.

            {'Habitat A': "Overall Habitat A Risk Map URI",
            'Habitat B': "Overall Habitat B Risk URI"
             ...
            }
        h_s_rasters- A dictionary that maps a habitat name to the risk rasters
            for each of the applicable stressors.

            {'HabA': ["A-1 Risk Raster URI", "A-2 Risk Raster URI", ...],
             'HabB': ["B-1 Risk Raster URI", "B-2 Risk Raster URI", ...], ...
            }
    '''

    #Use arbitrary element to get the nodata for habs
    nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(risk_dict.values()[0])

    def add_risk_pixels(*pixels):
        '''Sum all risk pixels to make a single habitat raster out of all the
        h-s overlap rasters.'''
        #Pulling the first one in teh list to use for masking purposes.
        value = numpy.zeros(pixels[0].shape)
        all_nodata = numpy.zeros(pixels[0].shape, dtype=numpy.bool)
        all_nodata[:] = True

        for i in range(len(pixels)):
            valid_mask = pixels[i] != -1

            value = numpy.where(
                valid_mask,
                pixels[i] + value,
                value)

            all_nodata = ~valid_mask & all_nodata

        return numpy.where(all_nodata, -1, value)
        '''all_nodata = True
        for p in pixels:
            if p != nodata:
                all_nodata = False
        if all_nodata:
            return nodata

        pixel_sum = 0.0

        for p in pixels:

            if p != nodata:

                pixel_sum += p

        return pixel_sum'''

    #This will give us two lists where we have only the unique habs and
    #stress for the system. List(set(list)) cast allows us to only get the
    #unique names within each.
    habitats, stressors = zip(*risk_dict.keys())
    habitats = list(set(habitats))
    stressors = list(set(stressors))

    #Want to get an arbitrary element in order to have a pixel size.
    pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
        risk_dict[(habitats[0], stressors[0])])

    #List to store the completed h rasters in. Will be passed on to the
    #ecosystem raster function to be used in vectorize_dataset.
    h_rasters = {}

    #Also need to store which h_s rasters apply to each habitat
    h_s_rasters = {}

    #Run through all potential pairings, and make lists for the ones that
    #share the same habitat.
    for h in habitats:

        ds_list = []
        for s in stressors:
            pair = (h, s)

            ds_list.append(risk_dict[pair])

        #Once we have the complete list, we can pass it to vectorize.
        out_uri = os.path.join(dir, 'cum_risk_[' + h + '].tif')

        pygeoprocessing.geoprocessing.vectorize_datasets(
            ds_list,
            add_risk_pixels,
            out_uri,
            gdal.GDT_Float32,
            -1.,
            pixel_size,
            "union",
            resample_method_list=None,
            dataset_to_align_index=0,
            aoi_uri=None,
            vectorize_op=False)

        h_rasters[h] = out_uri
        h_s_rasters[h] = ds_list

    return h_rasters, h_s_rasters


def make_risk_rasters(h_s_c, habs, inter_dir, crit_lists, denoms, risk_eq, warnings):
    '''
    This will combine all of the intermediate criteria rasters that we
    pre-processed with their r/dq*w. At this juncture, we should be able to
    straight add the E/C within themselves. The way in which the E/C rasters
    are combined depends on the risk equation desired.

    Input:
        h_s_c- Args dictionary containing much of the H-S overlap data in
            addition to the H-S base rasters. (In this function, we are only
            using it for the base h-s raster information.)
        habs- Args dictionary containing habitat criteria information in
            addition to the habitat base rasters. (In this function, we are
            only using it for the base raster information.)
        inter_dir- Intermediate directory in which the H_S risk-burned rasters
            can be placed.
        crit_lists- A dictionary containing pre-burned criteria which can be
            combined to get the E/C for that H-S pairing.

            {'Risk': {
                'h_s_c': {
                    (hab1, stressA): ["indiv num raster URI",
                                    "raster 1 URI", ...],
                                 (hab1, stressB): ...
                               },
                        'h':   {
                            hab1: ["indiv num raster URI",
                                   "raster 1 URI", ...],
                                ...
                               },
                        'h_s_e': { (hab1, stressA): ["indiv num raster URI",
                                                     ...]
                               }
                     }
             'Recovery': { hab1: ["indiv num raster URI", ...],
                           hab2: ...
                         }
            }
        denoms- Dictionary containing the denomincator scores for each overlap
            for each criteria. These can be combined to get the final denom by
            which the rasters should be divided.

            {'Risk': {  'h_s_c': { (hab1, stressA): {'CritName': 2.0,...},
                                 (hab1, stressB): {CritName': 1.3, ...}
                               },
                        'h':   { hab1: {'CritName': 2.5, ...},
                                ...
                               },
                        'h_s_e': { (hab1, stressA): {'CritName': 2.3},
                               }
                     }
             'Recovery': { hab1: {'CritName': 3.4},
                           hab2: ...
                         }
            }
        risk_eq- A string description of the desired equation to use when
            preforming risk calculation.
        warnings- A dictionary containing items which need to be acted upon by
            hra_core. These will be split into two categories. 'print' contains
            statements which will be printed using logger.warn() at the end of
            a run. 'unbuff' is for pairs which should use the unbuffered
            stressor file in lieu of the decayed rated raster.

            {'print': ['This is a warning to the user.', 'This is another.'],
              'unbuff': [(HabA, Stress1), (HabC, Stress2)]
            }
    Output:
        A new raster file for each overlapping of habitat and stressor. This
        file will be the overall risk for that pairing from all H/S/H-S
        subdictionaries.
    Returns:
        risk_rasters- A simple dictionary that maps a tuple of
            (Habitat, Stressor) to the URI for the risk raster created when the
            various sub components (H/S/H_S) are combined.

            {('HabA', 'Stress1'): "A-1 Risk Raster URI",
            ('HabA', 'Stress2'): "A-2 Risk Raster URI",
            ...
            }
    '''

    #Create dictionary that we can pass back to execute to be passed along to
    #make_habitat_rasters
    risk_rasters = {}

    #We will use the h-s pairs as the way of iterrating through everything
    #else.
    for pair in crit_lists['Risk']['h_s_c']:

        h, s = pair

        #Want to create an E and a C raster from the applicable
        #pre-calc'd rasters. We should be able to use vec_ds to straight add
        #the pixels and divide by the saved denoms total. These are the URIs to
        #which these parts of the risk equation will be burned.
        c_out_uri = os.path.join(
            inter_dir,
            "H[" + h + ']_S[' + s + ']_C_Risk_Raster.tif')
        e_out_uri = os.path.join(
            inter_dir,
            "H[" + h + ']_S[' + s + ']_E_Risk_Raster.tif')

        #Each of the E/C calculations should take in all of the relevant
        #subdictionary data, and return a raster to be used in risk
        #calculation. If, however, the pair contained no e criteria data, we
        #are using spatial overlap to substitute for the criteria burned
        #raster.
        if pair in warnings['unbuff']:

            unbuff_stress_uri = os.path.join(
                inter_dir,
                'Stressor_Rasters', s + '.tif')
            copy_raster(unbuff_stress_uri, e_out_uri)

        else:
            #Going to add in the h_s_c dict to have the URI for the
            #overlap raster
            calc_E_raster(
                e_out_uri,
                crit_lists['Risk']['h_s_e'][pair],
                denoms['Risk']['h_s_e'][pair],
                h_s_c[pair]['DS'],
                habs[h]['DS'])

        calc_C_raster(
            c_out_uri,
            crit_lists['Risk']['h_s_c'][pair],
            denoms['Risk']['h_s_c'][pair],
            crit_lists['Risk']['h'][h],
            denoms['Risk']['h'][h],
            habs[h]['DS'],
            h_s_c[pair]['DS'])

        #Function that we call now will depend on what the risk calculation
        #equation desired is.
        risk_uri = os.path.join(
            inter_dir,
            'H[' + h + ']_S[' + s + ']_Risk.tif')

        #Want to get the relevant ds for this H-S pair.
        base_ds_uri = h_s_c[pair]['DS']

        if risk_eq == 'Multiplicative':

            make_risk_mult(base_ds_uri, e_out_uri, c_out_uri, risk_uri)

        elif risk_eq == 'Euclidean':

            make_risk_euc(base_ds_uri, e_out_uri, c_out_uri, risk_uri)

        risk_rasters[pair] = risk_uri

    return risk_rasters


def make_risk_mult(base_uri, e_uri, c_uri, risk_uri):
    '''
    Combines the E and C rasters according to the multiplicative combination
    equation.

    Input:
        base- The h-s overlap raster, including potentially decayed values from
            the stressor layer.
        e_rast- The r/dq*w burned raster for all stressor-specific criteria
            in this model run.
        c_rast- The r/dq*w burned raster for all habitat-specific and
            habitat-stressor-specific criteria in this model run.
        risk_uri- The file path to which we should be burning our new raster.

    Returns the URI for a raster representing the multiplied E raster,
        C raster, and the base raster.
    '''
    grid_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(base_uri)

    # Rules should be similar to euclidean risk in that nothing happens
    # without there being c_pixels there.
    def combine_risk_mult(b_pix, e_pix, c_pix):

        risk_map = numpy.zeros(b_pix.shape)
        risk_map[:] = -1

        risk_map = numpy.where((b_pix == -1) & (c_pix != -1), 0, risk_map)
        risk_map = numpy.where(
            (b_pix != -1) & (c_pix != -1),
            e_pix * c_pix, risk_map)

        return risk_map

        '''if c_pix == c_nodata:
            return base_nodata

        #Here, we know that c_pix is not nodata, but want to return 0 if
        #there is habitat without overlap.
        elif b_pix == base_nodata:
            return 0

        #Here, we know that c_pix isn't nodata, and that overlap exists, so
        #can just straight multiply.
        else:
            return e_pix * c_pix'''

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [base_uri, e_uri, c_uri],
        combine_risk_mult,
        risk_uri,
        gdal.GDT_Float32,
        -1.,
        grid_size,
        "union",
        resample_method_list=None,
        dataset_to_align_index=0,
        aoi_uri=None,
        vectorize_op=False)


def make_risk_euc(base_uri, e_uri, c_uri, risk_uri):
    '''
    Combines the E and C rasters according to the euclidean combination
    equation.

    Input:
        base- The h-s overlap raster, including potentially decayed values from
            the stressor layer.
        e_rast- The r/dq*w burned raster for all stressor-specific criteria
            in this model run.
        c_rast- The r/dq*w burned raster for all habitat-specific and
            habitat-stressor-specific criteria in this model run.
        risk_uri- The file path to which we should be burning our new raster.

    Returns a raster representing the euclidean calculated E raster, C raster,
    and the base raster. The equation will be sqrt((C-1)^2 + (E-1)^2)
    '''
    # Already have base open for nodata values, just using pixel_size
    # version of the function.
    grid_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(base_uri)

    # we need to know very explicitly which rasters are being passed in which
    # order. However, since it's all within the make_risk_euc function, should
    # be safe.
    def combine_risk_euc(b_pix, e_pix, c_pix):

        b_mask = b_pix != -1
        c_mask = c_pix != -1

        e_vals = e_pix - 1
        c_vals = c_pix - 1

        e_vals = e_vals ** 2
        c_vals = c_vals ** 2

        # Per email from kwyatt and karkema, the h/(bufferedstressor) overlap
        # layer should be applied after the sqrt is taken, not multiplied by E
        # before the sqrt.  See https://bitbucket.org/natcap/invest/issues/3564
        risk_map = numpy.sqrt(e_vals + c_vals) * b_pix

        risk_map = numpy.where(c_mask, risk_map, -1)
        risk_map = numpy.where(c_mask & ~b_mask, 0, risk_map)

        return risk_map

        '''#If there is no C data (no habitat/overlap), we will always
        #be returning nodata.
        if c_pix == c_nodata:
            return base_nodata

        #Already know here that c_pix (hab/hab-overlap) exists.
        #If habitat exists without stressor, want to return 0 as the overall
        #risk, so that it will show up as "no risk" but still show up.
        elif b_pix == base_nodata:
            #If there's no spatial overlap, want the outcome of risk to just be
            #be 0.
            return 0

        #At this point, we know that there is data in c_pix, and we know that
        #there is overlap. So now can do the euc. equation.
        else:

            #Want to make sure that the decay is applied to E first, then that
            #product is what is used as the new E
            e_val = b_pix * e_pix

            c_val = c_pix - 1
            e_val -= 1

            #Now square both.
            c_val = c_val ** 2
            e_val = e_val ** 2

            #Combine, and take the sqrt
            value = math.sqrt(e_val + c_val)

            return value'''

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [base_uri, e_uri, c_uri],
        combine_risk_euc,
        risk_uri,
        gdal.GDT_Float32,
        -1.,
        grid_size,
        "union",
        resample_method_list=None,
        dataset_to_align_index=0,
        aoi_uri=None,
        vectorize_op=False)


def calc_E_raster(out_uri, h_s_list, denom_dict, h_s_base_uri, h_base_uri):
    '''
    Should return a raster burned with an 'E' raster that is a combination
    of all the rasters passed in within the list, divided by the denominator.

    Input:
        out_uri- The location to which the E raster should be burned.
        h_s_list- A list of rasters burned with the equation r/dq*w for every
            criteria applicable for that h, s pair.
        denom_dict- A double representing the sum total of all applicable
            criteria using the equation 1/dq*w.
            criteria applicable for that s.

    Returns nothing.
    '''
    grid_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(h_s_base_uri)

    # Using regex to pull out the criteria name after the last ]_. Will do this
    # for all full URI's.
    # See notebook notes from 8/22/13 for explanation for that regex.
    crit_name_list = map(
        lambda uri: re.match(
            '.*\]_([^_]*)',
            os.path.splitext(os.path.basename(uri))[0]).group(1), h_s_list)

    def add_e_pix(*pixels):

        h_s_pixels = pixels[2::]

        value = numpy.zeros(pixels[0].shape)
        denom_val = numpy.zeros(pixels[0].shape)

        for i in range(len(h_s_pixels)):
            valid_mask = h_s_pixels[i] != -1
            value = numpy.where(valid_mask, h_s_pixels[i] + value, value)
            denom_val = numpy.where(
                valid_mask,
                denom_dict[crit_name_list[i]] + denom_val,
                denom_val)

        # turn off dividie by zero warning because we probably will divide by
        # zero
        olderr = numpy.seterr(divide='ignore')
        result = numpy.where(denom_val != 0, value / denom_val, 0)
        # return numpy error state to old value
        numpy.seterr(**olderr)

        result[pixels[0] == -1] = -1

        return result

        '''h_base_pix = pixels[0]
        h_s_base_pix = pixels[1]
        h_s_pixels = pixels[2::]

        all_nodata = True
        for p in pixels:
            if p != nodata:
                all_nodata = False
        if all_nodata:
            return nodata

        #Know here that at least some pixels exist. h_s_pixels and h_s_base_pix
        #should cover the same area since they're all burned to h_s overlap.
        #Need to check if the one that exists is only the h pixel. If not
        #catching here, can assume that there are h_s values, and continue with
        #equation.
        if h_s_base_pix == nodata and h_base_pix != nodata:
            return 0

        #If we're here, want to go ahead and calculate out the values, since
        #we know there is overlap.
        value = 0.
        denom_val = 0.

        for i in range(len(h_s_pixels)):

            p = h_s_pixels[i]

            if p != nodata:
                value += p
                denom_val += denom_dict[crit_name_list[i]]

        return value / denom_val'''

    uri_list = [h_base_uri, h_s_base_uri] + h_s_list

    pygeoprocessing.geoprocessing.vectorize_datasets(
        uri_list,
        add_e_pix,
        out_uri,
        gdal.GDT_Float32,
        -1.,
        grid_size,
        "union",
        resample_method_list=None,
        dataset_to_align_index=0,
        aoi_uri=None,
        vectorize_op=False)


def calc_C_raster(out_uri, h_s_list, h_s_denom_dict, h_list, h_denom_dict, h_uri, h_s_uri):
    '''
    Should return a raster burned with a 'C' raster that is a combination
    of all the rasters passed in within the list, divided by the denominator.

    Input:
        out_uri- The location to which the calculated C raster should be
            bGurned.
        h_s_list- A list of rasters burned with the equation r/dq*w for every
            criteria applicable for that h, s pair.
        h_s_denom_dict- A dictionary containing criteria names applicable to
            this particular h,s pair. Each criteria string name maps to a
            double representing the denominator for that raster, using the
            equation 1/dq*w.
        h_list- A list of rasters burned with the equation r/dq*w for every
            criteria applicable for that s.
        h_denom_dict- A dictionary containing criteria names applicable to this
            particular habitat. Each criteria string name maps to a double
            representing the denominator for that raster, using the equation
            1/dq*w.

    Returns nothing.
    '''
    tot_crit_list = [h_uri, h_s_uri] + h_list + h_s_list

    h_s_names = map(
        lambda uri: re.match(
            '.*\]_([^_]*)',
            os.path.splitext(os.path.basename(uri))[0]).group(1), h_s_list)
    h_names = map(
        lambda uri: re.match(
            '.*\]_([^_]*)',
            os.path.splitext(os.path.basename(uri))[0]).group(1), h_list)

    grid_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(tot_crit_list[0])
    nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(h_s_list[0])

    # The first two spots are habitat raster and h_s raster
    h_count = len(h_list)

    def add_c_pix(*pixels):
        h_pixels = pixels[2:h_count+2]
        h_s_pixels = pixels[2+h_count::]

        value = numpy.zeros(pixels[0].shape)
        denom_val = numpy.zeros(pixels[0].shape)

        for i in range(len(h_pixels)):
            valid_mask = h_pixels[i] != -1
            value = numpy.where(valid_mask, h_pixels[i] + value, value)
            denom_val = numpy.where(
                valid_mask,
                h_denom_dict[h_names[i]] + denom_val,
                denom_val)

        # The h will need to get put into the h_s, so might as well have the
        # h_s loop start with the average returned from h.
        # This will essentiall treat all resilience criteria (h) as a single
        # entry alongside the h_s criteria.
        value = value / h_count
        denom_val = denom_val / h_count

        for i in range(len(h_s_pixels)):
            valid_mask = h_s_pixels[i] != -1
            value = numpy.where(valid_mask, h_s_pixels[i] + value, value)
            denom_val = numpy.where(
                valid_mask,
                h_s_denom_dict[h_s_names[i]] + denom_val,
                denom_val)

        # turn off dividie by zero warning because we probably will divide by
        # zero
        olderr = numpy.seterr(divide='ignore')
        result = numpy.where(denom_val != 0, value / denom_val, 0)
        # return numpy error state to old value
        numpy.seterr(**olderr)

        # Where there's just habitat but nothing else, we want 0, but evrything
        # outside that habitat should be nodata.
        result[pixels[0] == -1] = -1

        return result

        '''h_base_pix = pixels[0]
        h_s_base_pix = pixels[1]
        h_pixels = pixels[2:h_count+2]
        h_s_pixels = pixels[2+h_count::]

        all_nodata = True
        for p in pixels:
            if p != nodata:
                all_nodata = False
        if all_nodata:
            return nodata

        h_value = 0.
        h_denom_value = 0.

        #If we have habitat without overlap, want to return 0.
        if h_s_base_pix == nodata and h_base_pix != nodata:
            return 0.

        for i in range(len(h_pixels)):

            if p != nodata:
                h_value += h_pixels[i]
                h_denom_value += h_denom_dict[h_names[i]]

        #The h will need to get put into the h_s, so might as well have the
        #h_s loop start with the average returned from h.
        #This will essentiall treat all resilience criteria (h) as a single
        #entry alongside the h_s criteria.
        h_s_value = h_value / h_count
        h_s_denom_value = h_denom_value / h_count

        for i in range(len(h_s_pixels)):

            if p != nodata:
                h_s_value += h_s_pixels[i]
                h_s_denom_value += h_s_denom_dict[h_s_names[i]]

        return h_s_value / h_s_denom_value'''

    pygeoprocessing.geoprocessing.vectorize_datasets(
        tot_crit_list,
        add_c_pix,
        out_uri,
        gdal.GDT_Float32,
        -1.,
        grid_size,
        "union",
        resample_method_list=None,
        dataset_to_align_index=0,
        aoi_uri=None,
        vectorize_op=False)


def copy_raster(in_uri, out_uri):
    '''
    Quick function that will copy the raster in in_raster, and put it
    into out_raster.
    '''

    raster = gdal.Open(in_uri)
    drv = gdal.GetDriverByName('GTiff')
    drv.CreateCopy(out_uri, raster)


def pre_calc_denoms_and_criteria(dir, h_s_c, hab, h_s_e):
    '''
    Want to return two dictionaries in the format of the following:
    (Note: the individual num raster comes from the crit_ratings
    subdictionary and should be pre-summed together to get the numerator
    for that particular raster. )

    Input:
        dir- Directory into which the rasterized criteria can be placed. This
            will need to have a subfolder added to it specifically to hold the
            rasterized criteria for now.
        h_s_c- A multi-level structure which holds all criteria ratings,
            both numerical and raster that apply to habitat and stressor
            overlaps. The structure, whose keys are tuples of
            (Habitat, Stressor) names and map to an inner dictionary will have
            3 outer keys containing numeric-only criteria, raster-based
            criteria, and a dataset that shows the potentially buffered overlap
            between the habitat and stressor. The overall structure will be as
            pictured:

            {(Habitat A, Stressor 1):
                    {'Crit_Ratings':
                        {'CritName':
                            {'Rating': 2.0, 'DQ': 1.0, 'Weight': 1.0}
                        },
                    'Crit_Rasters':
                        {'CritName':
                            {
                                'DS': "CritName Raster URI",
                                    'Weight': 1.0, 'DQ': 1.0}
                        },
                    'DS':  "A-1 Raster URI"
                    }
            }
        hab- Similar to the h-s dictionary, a multi-level
            dictionary containing all habitat-specific criteria ratings and
            rasters. In this case, however, the outermost key is by habitat
            name, and habitats['habitatName']['DS'] points to the rasterized
            habitat shapefile URI provided by the user.
        h_s_e- Similar to the h_s_c dictionary, a multi-level
            dictionary containing habitat-stressor-specific criteria ratings
            and rasters. The outermost key is by (habitat, stressor) pair, but
            the criteria will be applied to the exposure portion of the risk
            calcs.

    Output:
        Creates a version of every criteria for every h-s paring that is
        burned with both a r/dq*w value for risk calculation, as well as a
        r/dq burned raster for recovery potential calculations.

    Returns:
        crit_lists- A dictionary containing pre-burned criteria URI which can
            be combined to get the E/C for that H-S pairing.

            {'Risk': {
                'h_s_c':
                    { (hab1, stressA): ["indiv num raster", "raster 1", ...],
                                   (hab1, stressB): ...
                                 },
                        'h': {
                            hab1: ["indiv num raster URI",
                                   "raster 1 URI", ...],
                                ...
                               },
                        'h_s_e': {
                                    (hab1, stressA):
                                        ["indiv num raster URI", ...]
                                 }
                     }
             'Recovery': { hab1: ["indiv num raster URI", ...],
                           hab2: ...
                         }
            }
        denoms- Dictionary containing the combined denominator for a given
            H-S overlap. Once all of the rasters are combined, each H-S raster
            can be divided by this.

            {'Risk': {
                'h_s_c': {
                    (hab1, stressA): {'CritName': 2.0, ...},
                                 (hab1, stressB): {'CritName': 1.3, ...}
                               },
                        'h':   { hab1: {'CritName': 1.3, ...},
                                ...
                               },
                        'h_s_e': { (hab1, stressA): {'CritName': 1.3, ...}
                               }
                     }
             'Recovery': { hab1: 1.6,
                           hab2: ...
                         }
            }
    '''

    pre_raster_dir = os.path.join(dir, 'ReBurned_Crit_Rasters')

    os.mkdir(pre_raster_dir)

    crit_lists = {
        'Risk': {'h_s_c': {}, 'h': {}, 'h_s_e': {}}, 'Recovery': {}}
    denoms = {'Risk': {'h_s_c': {}, 'h': {}, 'h_s_e': {}}, 'Recovery': {}}

    # Now will iterrate through the dictionaries one at a time, since each has
    # to be placed uniquely.

    # For Hab-Stress pairs that will be applied to the consequence portion
    # of risk.
    for pair in h_s_c:
        h, s = pair

        crit_lists['Risk']['h_s_c'][pair] = []
        denoms['Risk']['h_s_c'][pair] = {}

        # The base dataset for all h_s overlap criteria. Will need to load bases
        # for each of the h/s crits too.
        base_ds_uri = h_s_c[pair]['DS']
        base_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(base_ds_uri)
        base_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(base_ds_uri)

        # First, want to make a raster of added individual numerator criteria.
        # We will pre-sum all r / (dq*w), and then vectorize that with the
        # spatially explicit criteria later. Should be okay, as long as we keep
        # the denoms separate until after all raster crits are added.

        '''The following handle the cases for each dictionary for rasterizing
        the individual numerical criteria, and then the raster criteria.'''

        crit_rate_numerator = 0

        # H-S-C dictionary, Numerical Criteria: should output a
        # single raster that equals to the sum of r/dq*w for all single number
        # criteria in H-S

        # For the summed individual ratings, want the denominator to be
        # concatonated only with the other individual scores. Will make a single
        # entry that will correspond to the file name being output.
        denoms['Risk']['h_s_c'][pair]['Indiv'] = 0.

        for crit_dict in (h_s_c[pair]['Crit_Ratings']).values():
            r = crit_dict['Rating']
            dq = crit_dict['DQ']
            w = crit_dict['Weight']

            #Explicitly want a float output so as not to lose precision.
            crit_rate_numerator += r / float(dq*w)
            denoms['Risk']['h_s_c'][pair]['Indiv'] += 1 / float(dq*w)

        # This will not be spatially explicit, since we need to add the
        # others in first before multiplying against the decayed raster.
        # Instead, want to only have the crit_rate_numerator where data
        # exists, but don't want to multiply it.

        single_crit_C_uri = os.path.join(
            pre_raster_dir,
            'H[' + h + ']_S[' + s + ']' + '_Indiv_C_Raster.tif')
        # To save memory, want to use vectorize rasters instead of casting to an
        # array. Anywhere that we have nodata, leave alone. Otherwise, use
        # crit_rate_numerator as the burn value.

        def burn_numerator_single_hs(pixel):
            return numpy.where(pixel == -1, -1, crit_rate_numerator)

            '''if pixel == base_nodata:
                return base_nodata
            else:
                return crit_rate_numerator'''

        pygeoprocessing.geoprocessing.vectorize_datasets(
            [base_ds_uri],
            burn_numerator_single_hs,
            single_crit_C_uri,
            gdal.GDT_Float32,
            -1.,
            base_pixel_size,
            "union",
            resample_method_list=None,
            dataset_to_align_index=0,
            aoi_uri=None,
            vectorize_op=False)

        # Add the burned ds URI containing only the numerator burned ratings to
        # the list in which all rasters will reside
        crit_lists['Risk']['h_s_c'][pair].append(single_crit_C_uri)

        # H-S-C dictionary, Raster Criteria: should output multiple rasters,
        # each of which is reburned with the pixel value r, as r/dq*w.

        #.iteritems creates a key, value pair for each one.
        for crit_name, crit_dict in h_s_c[pair]['Crit_Rasters'].iteritems():

            crit_ds_uri = crit_dict['DS']
            crit_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(crit_ds_uri)

            dq = crit_dict['DQ']
            w = crit_dict['Weight']
            denoms['Risk']['h_s_c'][pair][crit_name] = 1 / float(dq * w)

            crit_C_uri = os.path.join(
                pre_raster_dir,
                'H[' + h + ']_S[' + s + ']_' + crit_name + '_' +
                'C_Raster.tif')

            def burn_numerator_hs(pixel):

                return numpy.where(pixel == -1, -1, pixel / (dq * w))
                '''if pixel == crit_nodata:
                    return crit_nodata

                else:
                    burn_rating = float(pixel) / (dq * w)
                    return burn_rating'''

            pygeoprocessing.geoprocessing.vectorize_datasets(
                [crit_ds_uri],
                burn_numerator_hs,
                crit_C_uri,
                gdal.GDT_Float32,
                -1.,
                base_pixel_size,
                "union",
                resample_method_list=None,
                dataset_to_align_index=0,
                aoi_uri=None,
                vectorize_op=False)

            crit_lists['Risk']['h_s_c'][pair].append(crit_C_uri)

    # Habitats are a special case, since each raster needs to be burned twice-
    # once for risk (r/dq*w), and once for recovery potential (r/dq).
    for h in hab:

        crit_lists['Risk']['h'][h] = []
        crit_lists['Recovery'][h] = []
        denoms['Risk']['h'][h] = {}
        denoms['Recovery'][h] = {}

        # The base dataset for all h_s overlap criteria. Will need to load bases
        # for each of the h/s crits too.
        base_ds_uri = hab[h]['DS']
        base_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(base_ds_uri)
        base_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(base_ds_uri)

        rec_crit_rate_numerator = 0
        risk_crit_rate_numerator = 0

        denoms['Risk']['h'][h]['Indiv'] = 0.
        denoms['Recovery'][h]['Indiv'] = 0.

        for crit_dict in hab[h]['Crit_Ratings'].values():

            r = crit_dict['Rating']
            dq = crit_dict['DQ']
            w = crit_dict['Weight']

            # Explicitly want a float output so as not to lose precision.
            risk_crit_rate_numerator += r / float(dq*w)
            rec_crit_rate_numerator += r / float(dq)
            denoms['Risk']['h'][h]['Indiv'] += 1 / float(dq*w)
            denoms['Recovery'][h]['Indiv'] += 1 / float(dq)

        # First, burn the crit raster for risk
        single_crit_C_uri = os.path.join(
            pre_raster_dir,
            'H[' + h + ']' + '_Indiv_C_Raster.tif')

        def burn_numerator_risk_single(pixel):

            return numpy.where(
                pixel == base_nodata,
                base_nodata,
                risk_crit_rate_numerator)
            '''if pixel == base_nodata:
                return base_nodata

            else:
                return risk_crit_rate_numerator
            '''
        pygeoprocessing.geoprocessing.vectorize_datasets(
            [base_ds_uri],
            burn_numerator_risk_single,
            single_crit_C_uri,
            gdal.GDT_Float32,
            -1.,
            base_pixel_size,
            "union",
            resample_method_list=None,
            dataset_to_align_index=0,
            aoi_uri=None,
            vectorize_op=False)

        crit_lists['Risk']['h'][h].append(single_crit_C_uri)

        # Now, burn the recovery potential raster, and add that.
        single_crit_rec_uri = os.path.join(
            pre_raster_dir,
            'H[' + h + ']' + '_Indiv_Recov_Raster.tif')

        def burn_numerator_rec_single(pixel):
            return numpy.where(pixel == -1, -1, rec_crit_rate_numerator)

            '''if pixel == base_nodata:
                return base_nodata

            else:
                return rec_crit_rate_numerator'''

        pygeoprocessing.geoprocessing.vectorize_datasets(
            [base_ds_uri],
            burn_numerator_rec_single,
            single_crit_rec_uri,
            gdal.GDT_Float32,
            -1.,
            base_pixel_size,
            "union",
            resample_method_list=None,
            dataset_to_align_index=0,
            aoi_uri=None,
            vectorize_op=False)

        crit_lists['Recovery'][h].append(single_crit_rec_uri)

        # Raster Criteria: should output multiple rasters, each
        # of which is reburned with the old pixel value r as r/dq*w, or r/dq.
        for crit_name, crit_dict in hab[h]['Crit_Rasters'].iteritems():
            dq = crit_dict['DQ']
            w = crit_dict['Weight']

            crit_ds_uri = crit_dict['DS']
            crit_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(crit_ds_uri)

            denoms['Risk']['h'][h][crit_name] = 1 / float(dq * w)
            denoms['Recovery'][h][crit_name] = 1 / float(dq)

            # First the risk rasters
            crit_C_uri = os.path.join(
                pre_raster_dir,
                'H[' + h + ']' + '_' + crit_name + '_' + 'C_Raster.tif')

            def burn_numerator_risk(pixel):
                return numpy.where(pixel == -1, -1, pixel / (w*dq))

                '''if pixel == crit_nodata:
                    return -1.

                else:
                    burn_rating = float(pixel) / (w*dq)
                    return burn_rating'''

            pygeoprocessing.geoprocessing.vectorize_datasets(
                [crit_ds_uri],
                burn_numerator_risk,
                crit_C_uri,
                gdal.GDT_Float32,
                -1.,
                base_pixel_size,
                "union",
                resample_method_list=None,
                dataset_to_align_index=0,
                aoi_uri=None,
                vectorize_op=False)

            crit_lists['Risk']['h'][h].append(crit_C_uri)

            # Then the recovery rasters
            crit_recov_uri = os.path.join(
                pre_raster_dir,
                'H[' + h + ']_' + crit_name + '_' + 'Recov_Raster.tif')

            def burn_numerator_rec(pixel):
                return numpy.where(pixel == -1, -1, pixel / (w*dq))
                '''if pixel == crit_nodata:
                    return 0.

                else:
                    burn_rating = float(pixel) / dq
                    return burn_rating'''

            pygeoprocessing.geoprocessing.vectorize_datasets(
                [crit_ds_uri],
                burn_numerator_rec,
                crit_recov_uri,
                gdal.GDT_Float32,
                -1.,
                base_pixel_size,
                "union",
                resample_method_list=None,
                dataset_to_align_index=0,
                aoi_uri=None,
                vectorize_op=False)

            crit_lists['Recovery'][h].append(crit_recov_uri)

    # Hab-Stress for Exposure
    for pair in h_s_e:
        h, s = pair

        crit_lists['Risk']['h_s_e'][pair] = []
        denoms['Risk']['h_s_e'][pair] = {}

        # The base dataset for all h_s overlap criteria. Will need to load bases
        # for each of the h/s crits too.
        base_ds_uri = h_s_e[pair]['DS']
        base_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(base_ds_uri)
        base_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(base_ds_uri)

        # First, want to make a raster of added individual numerator criteria.
        # We will pre-sum all r / (dq*w), and then vectorize that with the
        # spatially explicit criteria later. Should be okay, as long as we keep
        # the denoms separate until after all raster crits are added.

        '''The following handle the cases for each dictionary for rasterizing
        the individual numerical criteria, and then the raster criteria.'''

        crit_rate_numerator = 0
        denoms['Risk']['h_s_e'][pair]['Indiv'] = 0.

        # H-S-E dictionary, Numerical Criteria: should output a
        # single raster that equals to the sum of r/dq*w for all single number
        # criteria in H-S

        for crit_dict in (h_s_e[pair]['Crit_Ratings']).values():
            r = crit_dict['Rating']
            dq = crit_dict['DQ']
            w = crit_dict['Weight']

            #Explicitly want a float output so as not to lose precision.
            crit_rate_numerator += r / float(dq*w)
            denoms['Risk']['h_s_e'][pair]['Indiv'] += 1 / float(dq*w)

        # This will not be spatially explicit, since we need to add the
        # others in first before multiplying against the decayed raster.
        # Instead, want to only have the crit_rate_numerator where data
        # exists, but don't want to multiply it.

        single_crit_E_uri = os.path.join(
            pre_raster_dir,
            'H[' + h + ']_S[' + s + ']' + '_Indiv_E_Raster.tif')
        # To save memory, want to use vectorize rasters instead of casting to an
        # array. Anywhere that we have nodata, leave alone. Otherwise, use
        # crit_rate_numerator as the burn value.

        def burn_numerator_single_hs(pixel):
            return numpy.where(pixel == -1, -1, crit_rate_numerator)
            '''if pixel == base_nodata:
                return base_nodata
            else:
                return crit_rate_numerator'''

        pygeoprocessing.geoprocessing.vectorize_datasets(
            [base_ds_uri],
            burn_numerator_single_hs,
            single_crit_E_uri,
            gdal.GDT_Float32,
            -1.,
            base_pixel_size,
            "union",
            resample_method_list=None,
            dataset_to_align_index=0,
            aoi_uri=None,
            vectorize_op=False)

        # Add the burned ds URI containing only the numerator burned ratings to
        # the list in which all rasters will reside
        crit_lists['Risk']['h_s_e'][pair].append(single_crit_E_uri)

        # H-S-E dictionary, Raster Criteria: should output multiple rasters,
        # each of which is reburned with the pixel value r, as r/dq*w.

        # .iteritems creates a key, value pair for each one.
        for crit_name, crit_dict in h_s_e[pair]['Crit_Rasters'].iteritems():

            crit_ds_uri = crit_dict['DS']
            crit_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(crit_ds_uri)

            dq = crit_dict['DQ']
            w = crit_dict['Weight']
            denoms['Risk']['h_s_e'][pair][crit_name] = 1 / float(dq * w)

            crit_E_uri = os.path.join(
                pre_raster_dir,
                'H[' + h + ']_S[' + s + ']_' + crit_name + '_' +
                'E_Raster.tif')

            def burn_numerator_hs(pixel):
                return numpy.where(pixel == -1, -1, pixel / (w*dq))
                '''if pixel == crit_nodata:
                    return crit_nodata

                else:
                    burn_rating = float(pixel) / (dq * w)
                    return burn_rating'''

            pygeoprocessing.geoprocessing.vectorize_datasets(
                [crit_ds_uri],
                burn_numerator_hs,
                crit_E_uri,
                gdal.GDT_Float32,
                -1.,
                base_pixel_size,
                "union",
                resample_method_list=None,
                dataset_to_align_index=0,
                aoi_uri=None,
                vectorize_op=False)

            crit_lists['Risk']['h_s_e'][pair].append(crit_E_uri)

    # This might help.
    return (crit_lists, denoms)
