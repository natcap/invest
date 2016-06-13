"""Entry point for the Habitat Risk Assessment module"""

import csv
import os
import logging
import json
import fnmatch
import shutil

logging.basicConfig(format='%(asctime)s %(name)-18s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')
LOGGER = logging.getLogger('natcap.invest.habitat_risk.preprocessor')


class MissingHabitatsOrSpecies(Exception):
    '''An exception to pass if the hra_preprocessor args dictionary being
    passed is missing a habitats directory or a species directory.'''
    pass


class NotEnoughCriteria(Exception):
    '''An exception for hra_preprocessor which can be passed if the number of
    criteria in the resilience, exposure, and sensitivity categories all sums
    to less than 4.'''
    pass


class ImproperCriteriaSpread(Exception):
    '''An exception for hra_preprocessor which can be passed if there are not
    one or more criteria in each of the 3 criteria categories: resilience,
    exposure, and sensitivity.'''
    pass


class ZeroDQWeightValue(Exception):
    '''An exception specifically for the parsing of the preprocessor tables in
    which the model should break loudly if a user tries to enter a zero value
    for either a data quality or a weight. However, we should confirm that it
    will only break if the rating is not also zero. If they're removing the
    criteria entirely from that H-S overlap, it should be allowed.'''
    pass


class UnexpectedString(Exception):
    '''An exception for hra_preprocessor that should catch any strings that are
    left over in the CSVs. Since everything from the CSV's are being cast to
    floats, this will be a hook off of python's ValueError, which will re-raise
    our exception with a more accurate message. '''
    pass


class ImproperECSelection(Exception):
    '''An exception for hra_preprocessor that should catch selections for
    exposure vs consequence scoring that are not either E or C. The user must
    decide in this column which the criteria applies to, and my only designate
    this with an 'E' or 'C'. '''
    pass


class MissingSensOrResilException(Exception):
    '''An exception for hra_preprocessor that catches h-s pairings who are
    missing either Sensitivity or Resilience or C criteria, though not both.
    The user must either zero all criteria for that pair, or make sure that
    both E and C are represented.
    '''
    pass

class NA_RatingsError(Exception):
    """
    An exception that is raised on an invalid 'NA' input.

    When one or more Ratings value is set to "NA" for a habitat - stressor
    pair, but not ALL are set to "NA". If ALL Rating values for a
    habitat - stressor pair are "NA", then the habitat - stressor pair is
    considered to have NO interaction.
    """

    pass

def execute(args):
    """Habitat Risk Assessment Preprocessor.

    Want to read in multiple hab/stressors directories, in addition to named
    criteria, and make an appropriate csv file.

    Parameters:
        args['workspace_dir'] (string): The directory to dump the output CSV
            files to. (required)
        args['habitats_dir'] (string): A directory of shapefiles that are habitats.
            This is not required, and may not exist if there is a species layer
            directory. (optional)
        args['species_dir'] (string): Directory which holds all species
            shapefiles, but may or may not exist if there is a habitats
            layer directory. (optional)
        args['stressors_dir'] (string): A directory of ArcGIS shapefiles
            that are stressors. (required)
        args['exposure_crits'] (list): list containing string names of exposure
            criteria (hab-stress) which should be applied to the exposure
            score. (optional)
        args['sensitivity-crits'] (list): List containing string names of
            sensitivity (habitat-stressor overlap specific) criteria which
            should be applied to the consequence score. (optional)
        args['resilience_crits'] (list): List containing string names of
            resilience (habitat or species-specific) criteria which should be
            applied to the consequence score. (optional)
        args['criteria_dir'] (string): Directory which holds the criteria
            shapefiles. May not exist if the user does not desire criteria
            shapefiles. This needs to be in a VERY specific format, which
            shall be described in the user's guide. (optional)

    Returns:
        ``None``

    This function creates a series of CSVs within ``args['workspace_dir']``.
    There will be one CSV for every habitat/species. These files will contain information
    relevant to each habitat or species, including all criteria. The criteria
    will be broken up into those which apply to only the habitat, and those
    which apply to the overlap of that habitat, and each stressor.

    JSON file containing vars that need to be passed on to hra non-core
    when that gets run. Should live inside the preprocessor folder which
    will be created in ``args['workspace_dir']``. It will contain habitats_dir,
    species_dir, stressors_dir, and criteria_dir.
    """
    #Create two booleans to indicate which of the layers we should be using in
    #this model run.
    do_habs = 'habitats_dir' in args
    do_species = 'species_dir' in args

    #First, want to raise two exceptions if things are wrong.
    #1. Shouldn't be able to run with no species or habitats.
    if not do_species and not do_habs:

        raise MissingHabitatsOrSpecies("This model requires you to provide \
                either habitat or species information for comparison against \
                potential stressors.")

    #2. There should be > 4 criteria total.
    total_crits = (len(args['exposure_crits']) +
                   len(args['resilience_crits']) +
                   len(args['sensitivity_crits']))

    if total_crits < 4:
        raise NotEnoughCriteria("This model requires you to use at least 4 \
                criteria in order to display an accurate picture of habitat \
                risk.")

    #Now we can run the meat of the model.
    #Make the workspace directory if it doesn't exist
    output_dir = os.path.join(
        args['workspace_dir'],
        'habitat_stressor_ratings')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    #Make the dictionary first, then write the JSON file with the directory
    #pathnames if they exist in args
    json_uri = os.path.join(output_dir, 'dir_names.txt')

    json_dict = {'stressors_dir': args['stressors_dir']}
    for var in ('criteria_dir', 'habitats_dir', 'species_dir'):
        if var in args:
            json_dict[var] = args[var]

    with open(json_uri, 'w') as outfile:

        json.dump(json_dict, outfile)

    #Get the names of all potential habs
    hab_list = []
    for ele in ('habitats_dir', 'species_dir'):
        if ele in args:
            names = listdir(args[ele])
            hab_list = fnmatch.filter(names, '*.shp')
            hab_list = \
                map(lambda uri: os.path.splitext(
                    os.path.basename(uri))[0],
                    hab_list)

    #And all potential stressors
    names = listdir(args['stressors_dir'])
    stress_list = fnmatch.filter(names, '*.shp')
    stress_list = map(lambda uri: os.path.splitext(
        os.path.basename(uri))[0],
        stress_list)

    #Now that we know the stressor names, let's create the simple CSV file to
    #track the stressor buffers for each stressor.
    s_buff_uri = os.path.join(output_dir, 'stressor_buffers.csv')

    with open(s_buff_uri, 'wb') as s_file:
        s_writer = csv.writer(s_file)

        s_writer.writerow(['STRESSOR NAME', 'STRESSOR BUFFER (meters)'])
        s_writer.writerow([])

        for s_name in stress_list:

            s_writer.writerow([s_name, '<enter a buffer region in meters>'])

    #Clean up the incoming criteria name strings coming in from the IUI
    exposure_crits = map(lambda name:
                         name.replace('_', ' ').lower(),
                         args['exposure_crits'])
    resilience_crits = map(lambda name:
                           name.replace('_', ' ').lower(),
                           args['resilience_crits'])
    sensitivity_crits = map(lambda name:
                            name.replace('_', ' ').lower(),
                            args['sensitivity_crits'])

    '''If shapefile criteria are desired, want to pull the shapefile criteria
    from the folder structure specified. This function will return a dictionary
    with the following form:
        {'h_s_e':
            {('HabA', 'Stress1'):
                {'CritName': "Shapefile URI", ...}
            },
         'h_s_c':
            {'Stress1':
                {'CritName': "Shapefile URI", ...}
            },
         'h':
            {'HabA':
                {'CritName': "Shapefile URI", ...}
        }
    '''
    if 'criteria_dir' in args:
        crit_shapes = make_crit_shape_dict(args['criteria_dir'])

    crit_descriptions = {
        'change in area rating': '<enter (3) 50-100% loss, ' +
        '(2) 20-50% loss, (1) 0-20% loss, (0) no score>',
        'change in structure rating': '<enter (3) 50-100% loss, ' +
        '(2) 20-50% loss, (1) 0-20% loss, (0) no score>',
        'temporal overlap rating': '<enter (3) co-occur 8-12 mo/year, ' +
        '(2) 4-8 mo/yr, (1) 0-4 mo/yr, (0) no score>',
        'frequency of disturbance': '<enter (3) Annually or less often, ' +
        '(2) Several times per year, (1) Weekly or more often, ' +
        '(0) no score>',
        'intensity rating': '<enter (3) high, (2) medium, ' +
        '(1) low, (0) no score>',
        'management effectiveness': '<enter (3) not effective, ' +
        '(2) somewhat effective, (1) very effective, (0) no score>',
        'natural mortality rate': '<enter (3) 0-20%, (2) 20-50%, ' +
        '(1) >80% mortality, or (0) no score>',
        'recruitment rate': '<enter (3) every 2+ yrs, (2) every 1-2 yrs, ' +
        '(1) every <1 yrs, or (0) no score>',
        'recovery time': '<enter (3) >10 yrs, (2) 1-10 yrs, ' +
        '(1) <1 yr, or (0) no score>',
        'connectivity rate': '<enter (3) <10km, (2) 10-100km, ' +
        '(1) >100km, or (0) no score>'
        }

    default_dq_message = '<enter (3) best, (2) adequate, (1) limited>'
    default_weight_message = '<enter (3) more important, ' + \
        '(2) equal importance, (1) less important>'
    default_table_headers = ['', 'Rating', 'DQ', 'Weight', 'E/C']
    default_row = [default_dq_message, default_weight_message]
    default_rating = ['<enter (3) high, (2) medium, (1) low, (0) no score>']

    #Create habitat-specific CSV's
    for habitat_name in hab_list:

        csv_filename = os.path.join(output_dir,
                                    habitat_name + '_ratings.csv')

        with open(csv_filename, 'wb') as habitat_csv_file:
            habitat_csv_writer = csv.writer(habitat_csv_file)
            #Write the habitat name
            habitat_csv_writer.writerow(['HABITAT NAME', habitat_name])
            habitat_csv_writer.writerow([])
            habitat_csv_writer.writerow(['HABITAT ONLY PROPERTIES'])

            habitat_csv_writer.writerow(default_table_headers)

            ##### HERE WILL BE WHERE USER INPUT HABITAT-SPECIFIC (Resilience)
            #####CRITERIA GO.####
            for c_name in resilience_crits:

                curr_row = default_row

                #Need to first check to make sure that crit_shapes
                #was instantiated when
                if 'crit_shapes' in locals() and \
                        (habitat_name in crit_shapes['h'] and
                            c_name in crit_shapes['h'][habitat_name]):
                    curr_row = [c_name] + ['SHAPE'] + curr_row + ['C']
                elif c_name in crit_descriptions:
                    curr_row = [c_name] + [crit_descriptions[
                        c_name]] + curr_row + ['C']
                else:
                    curr_row = [c_name] + default_rating + curr_row + ['C']

                habitat_csv_writer.writerow(curr_row)

            ##### HERE WILL BE WHERE ALL THE H-S INPUT CRITERIA GO.####
            ##### THIS WILL ENCOMPASS BOTH THE SENSITIVITY AND EXPOSURE
            #     CRITS ###
            habitat_csv_writer.writerow([])
            habitat_csv_writer.writerow(
                ['HABITAT STRESSOR OVERLAP PROPERTIES'])

            for stressor_name in stress_list:

                habitat_csv_writer.writerow([])
                habitat_csv_writer.writerow([habitat_name + '/' +
                                            stressor_name + ' OVERLAP'])
                habitat_csv_writer.writerow(default_table_headers)

                ##SENSITIVITY##
                for c_name in sensitivity_crits:
                    curr_row = default_row

                    if 'crit_shapes' in locals() and \
                        ((habitat_name, stressor_name) in crit_shapes[
                            'h_s_c'] and c_name in crit_shapes['h_s_c'][
                            (habitat_name, stressor_name)]):

                        curr_row = [c_name] + ['SHAPE'] + curr_row + ['C']
                    elif c_name in crit_descriptions:

                        curr_row = [c_name] + [crit_descriptions[
                            c_name]] + curr_row + ['C']
                    else:
                        curr_row = [c_name] + default_rating + curr_row + ['C']

                    habitat_csv_writer.writerow(curr_row)

                ##EXPOSURE ###

                for c_name in exposure_crits:
                    curr_row = default_row

                    if 'crit_shapes' in locals() and \
                        ((habitat_name, stressor_name) in crit_shapes[
                            'h_s_e'] and c_name in crit_shapes['h_s_e'][
                            (habitat_name, stressor_name)]):

                        curr_row = [c_name] + ['SHAPE'] + curr_row + ['E']
                    elif c_name in crit_descriptions:

                        curr_row = [c_name] + [crit_descriptions[c_name]] + \
                            curr_row + ['E']
                    else:
                        curr_row = [c_name] + default_rating + curr_row + ['E']

                    habitat_csv_writer.writerow(curr_row)


def listdir(path):
    '''
    A replacement for the standar os.listdir which, instead of returning
    only the filename, will include the entire path. This will use os as a
    base, then just lambda transform the whole list.

    Input:
        path- The location container from which we want to gather all files.

    Returns:
        A list of full URIs contained within 'path'.
    '''
    file_names = os.listdir(path)
    uris = map(lambda x: os.path.join(path, x), file_names)

    return uris


def make_crit_shape_dict(crit_uri):
    '''
    This will take in the location of the file structure, and will return
    a dictionary containing all the shapefiles that we find. Hypothetically, we
    should be able to parse easily through the files, since it should be
    EXACTLY of the specs that we laid out.

    Input:
        crit_uri- Location of the file structure containing all of the
            shapefile criteria.


    Returns:
        A dictionary containing shapefile URI's, indexed by their criteria
        name, in addition to which dictionaries and h-s pairs they apply to.
        The structure will be as follows:

        {'h':
            {'HabA':
                {'CriteriaName: "Shapefile Datasource URI"...}, ...
            },
         'h_s_c':
            {('HabA', 'Stress1'):
                {'CriteriaName: "Shapefile Datasource URI", ...}, ...
            },
         'h_s_e'
            {('HabA', 'Stress1'):
                {'CriteriaName: "Shapefile Datasource URI", ...}, ...
            }
        }
    '''
    c_shape_dict = {'h_s_e': {}, 'h': {}, 'h_s_c': {}}

    res_dir = os.path.join(crit_uri, 'Resilience')
    exps_dir = os.path.join(crit_uri, 'Exposure')
    sens_dir = os.path.join(crit_uri, 'Sensitivity')

    for folder in [res_dir, exps_dir, sens_dir]:
        if not os.path.isdir(folder):
            raise IOError("Using spatically explicit critiera requires you to \
                    have subfolders named \"Resilience\", \"Exposure\", and \
                    \"Sensitivity\". Check that all these folders exist, and \
                    that your criteria are placed properly.")

    #First, want to get the things that are either habitat specific or
    #species specific. These should all be in the 'Resilience subfolder
    #of raster_criteria.
    res_names = listdir(os.path.join(crit_uri, 'Resilience'))
    res_shps = fnmatch.filter(res_names, '*.shp')

    #Now we have a list of all habitat specific shapefile criteria. Now we need
    #to parse them out.
    for path in res_shps:

        #The return of os.path.split is a tuple where everything after the
        #final slash is returned as the 'tail' in the second element of the
        #tuple path.splitext returns a tuple such that the first element is
        #what comes before the file extension, and the second is the extension
        #itself
        filename = os.path.splitext(os.path.split(path)[1])[0]

        #want the second part to all be one piece
        parts = filename.split('_', 1)
        hab_name = parts[0]
        crit_name = parts[1].replace('_', ' ').lower()

        if hab_name not in c_shape_dict['h']:
            c_shape_dict['h'][hab_name] = {}

        c_shape_dict['h'][hab_name][crit_name] = path

    #Next, get all of our pair-centric, exposure applicable criteria.
    exps_names = listdir(os.path.join(crit_uri, 'Exposure'))
    exps_shps = fnmatch.filter(exps_names, '*.shp')

    #Now we have a list of all pair specific shapefile criteria.
    #Now we need to parse them out.
    for path in exps_shps:
        #The return of os.path.split is a tuple where everything after the
        #final slash is returned as the 'tail' in the second element of the
        #tuple path.splitext returns a tuple such that the first element is
        #what comes before the file extension, and the second is the
        #extension itself
        filename = os.path.splitext(os.path.split(path)[1])[0]

        #want the first and second part to be separate, since they are the
        #habitatName and the stressorName, but want the criteria name to be
        #self contained.
        parts = filename.split('_', 2)

        hab_name = parts[0]
        stress_name = parts[1]
        crit_name = parts[2].replace('_', ' ').lower()

        if (hab_name, stress_name) not in c_shape_dict['h_s_e']:
            c_shape_dict['h_s_e'][(hab_name, stress_name)] = {}

        c_shape_dict['h_s_e'][(hab_name, stress_name)][crit_name] = path

    #Finally, want to get all of our pair-centric, but consequence
    #applicable criteria.
    sens_names = listdir(os.path.join(crit_uri, 'Sensitivity'))
    sens_shps = fnmatch.filter(sens_names, '*.shp')

    #Now we have a list of all pair specific shapefile criteria.
    #Now we need to parse them out.
    for path in sens_shps:

        #The return of os.path.split is a tuple where everything after the
        #final slash is returned as the 'tail' in the second element of the
        #tuple path.splitext returns a tuple such that the first element is
        #what comes before the file extension, and the second is the extension
        #itself
        filename = os.path.splitext(os.path.split(path)[1])[0]

        #want the first and second part to be separate, since they are the
        #habitatName and the stressorName, but want the criteria name to be
        #self contained.
        parts = filename.split('_', 2)
        hab_name = parts[0]
        stress_name = parts[1]
        crit_name = parts[2].replace('_', ' ').lower()

        if (hab_name, stress_name) not in c_shape_dict['h_s_c']:
            c_shape_dict['h_s_c'][(hab_name, stress_name)] = {}

        c_shape_dict['h_s_c'][(hab_name, stress_name)][crit_name] = path

    #Et, voila! C'est parfait.
    return c_shape_dict


def parse_hra_tables(folder_uri):
    '''
    This takes in the directory containing the criteria rating csv's,
    and returns a coherent set of dictionaries that can be used to do
    EVERYTHING in non-core and core.

    It will return a massive dictionary containing all of the subdictionaries
    needed by non core, as well as directory URI's. It will be of the following
    form:

    {'habitats_dir': 'Habitat Directory URI',
    'species_dir': 'Species Directory URI',
    'stressors_dir': 'Stressors Directory URI',
    'criteria_dir': 'Criteria Directory URI',
    'buffer_dict':
        {'Stressor 1': 50,
        'Stressor 2': ...,
        },
    'h_s_c':
        {(Habitat A, Stressor 1):
            {'Crit_Ratings':
                {'CritName':
                    {'Rating': 2.0, 'DQ': 1.0, 'Weight': 1.0}
                },
            'Crit_Rasters':
                {'CritName':
                    {'Weight': 1.0, 'DQ': 1.0}
                },
            }
        },
    'h_s_c':
        {(Habitat A, Stressor 1):
            {'Crit_Ratings':
                {'CritName':
                    {'Rating': 2.0, 'DQ': 1.0, 'Weight': 1.0}
                },
            'Crit_Rasters':
                {'CritName':
                    {'Weight': 1.0, 'DQ': 1.0}
                },
            }
        },
     'habitats':
        {Habitat A:
            {'Crit_Ratings':
                {'CritName':
                    {'Rating': 2.0, 'DQ': 1.0, 'Weight': 1.0}
                },
            'Crit_Rasters':
                {'CritName':
                    {'Weight': 1.0, 'DQ': 1.0}
                },
            }
        }
     'warnings':
        {'print':
            ['This is a warning to the user.', 'This is another.'],
          'unbuff':
            [(HabA, Stress1), (HabC, Stress2)]
        }
    }
    '''
    #Create the dictionary in which everything will be stored.
    parse_dictionary = {}

    #Get the arguments out of the json file.
    json_uri = os.path.join(folder_uri, 'dir_names.txt')

    with open(json_uri, 'rb') as infile:
        parse_dictionary = json.load(infile)

    #This is the file name in which we will store all buffer information. This
    #file will be explicitly created when preprocessor is run. Want to parse
    #and pull into it's own dictionary do that it can be placed in
    #mega-dictionary.

    s_buff_uri = os.path.join(folder_uri, 'stressor_buffers.csv')
    stress_dict = parse_stress_buffer(s_buff_uri)

    #Now we can compile the information from habitat csv's into other
    #dictionaries
    file_names = listdir(folder_uri)
    csv_uris = fnmatch.filter(file_names, '*_ratings.csv')

    #Initialize the three dictionaries that we will use to store criteria info
    habitat_dict = {}
    h_s_e_dict = {}
    h_s_c_dict = {}

    for habitat_uri in csv_uris:
        #Instead of having to know what came from where, let's just have it
        #update the global dictionaries while the function is running.
        parse_overlaps(habitat_uri, habitat_dict, h_s_e_dict, h_s_c_dict)

    #Creating a dictionary structure to hold warnings that will get passed back
    #to the core and used there.
    warnings = zero_check(h_s_c_dict, h_s_e_dict, habitat_dict)

    #Add everything to the parse dictionary
    parse_dictionary['buffer_dict'] = stress_dict
    parse_dictionary['habitats'] = habitat_dict
    parse_dictionary['h_s_e'] = h_s_e_dict
    parse_dictionary['h_s_c'] = h_s_c_dict
    parse_dictionary['warnings'] = warnings

    return parse_dictionary


def zero_check(h_s_c, h_s_e, habs):
    '''
    Any criteria that have a rating of 0 mean that they are not a desired input
    to the assessment. We should delete the criteria's entire subdictionary
    out of the dictionary.

    Input:
        habs- A dictionary which contains all resilience specific criteria
            info. The key for these will be the habitat name. It will map to a
            subdictionary containing criteria information. The whole dictionary
            will look like the following:

            {Habitat A:
                {'Crit_Ratings':
                    {'CritName':
                        {'Rating': 2.0, 'DQ': 1.0, 'Weight': 1.0}
                    },
                'Crit_Rasters':
                    {'CritName':
                        {'Weight': 1.0, 'DQ': 1.0}
                    },
                }
            }

        h_s_e- A dictionary containing all information applicable to exposure
            criteria. The dictionary will look identical to the 'habs'
            dictionary, but each key will be a tuple of two strings -
            (HabName, StressName).
        h_s_c- A dictionary containing all information applicable to
            sensitivity criteria. The dictionary will look identical to the
            'habs' dictionary, but each key will be a tuple of two strings -
            (HabName, StressName).

    Output:
        Will update each of the three dictionaries by deleting any criteria
        where the rating aspect is 0.

    Returns:
        warnings- A dictionary containing items which need to be acted upon by
            hra_core. These will be split into two categories. 'print' contains
            statements which will be printed using logger.warn() at the end of
            a run. 'unbuff' is for pairs which should use the unbuffered
            stressor file in lieu of the decayed rated raster.

            {'print': ['This is a warning to the user.', 'This is another.'],
              'unbuff': [(HabA, Stress1), (HabC, Stress2)]
            }
    '''

    #Want to do zero checks for each of the criteria dictionaries.
    for dictionary in [h_s_c, h_s_e, habs]:

        #These are the subdictionaries mapped to the habitat, h-s tuple key.
        for key_1, subdict_lvl_1 in dictionary.items():

            #These are the subdictionaries mapped to the keys of 'Crit_Ratings'
            #'Crit_Rasters'.
            for key_2, subdict_lvl_2 in subdict_lvl_1.items():

                #Only want to check for 0 ratings if they're giving a single
                #explicit rating. If they gave a shapefile, they're on their
                #own.
                if key_2 == 'Crit_Ratings':

                    #These are key, value pairs of crit_name, dict containing
                    #rating/dq/weight info.
                    for key_3, subdict_lvl_3 in subdict_lvl_2.items():

                        #Want to make sure that the Rating key isn't 0.
                        if subdict_lvl_3['Rating'] == 0.0:

                            del subdict_lvl_2[key_3]
            '''
            #Now that we have removed that, check the dictionary isn't
            #now empty
            if len(subdict_lvl_1['Crit_Ratings']) == 0 and \
                        len(subdict_lvl_1['Crit_Rasters']) == 0:

                del dictionary[key_1]
            '''

    warnings = {'print': [], 'unbuff': []}

    #At this point, the pair keys for h_s_c and h_s_e should be identical.
    #Going to use h_s_e as the iterator.
    for pair, overlap_dict in h_s_e.items():

        h, s = pair

        e_crit_count = len(overlap_dict['Crit_Rasters']) + \
            len(overlap_dict['Crit_Ratings'])
        c_crit_count = len(h_s_c[pair]['Crit_Rasters']) + \
            len(h_s_c[pair]['Crit_Ratings'])
        h_crit_count = len(habs[h]['Crit_Rasters']) + \
            len(habs[h]['Crit_Ratings'])

        #If there are no sens. and no resilience, ERROR.
        if h_crit_count == 0 and c_crit_count == 0:
            raise MissingSensOrResilException("For a given habitat-stressor \
                    pairing, there must be at least one relevant resilience \
                    or sensitivity criteria. Currently, the (%s, %s) pairing \
                    is missing one or the other. This can be corrected in \
                    the %s ratings csv file." % (h, s, h))

        #The next two are not mutually exclusive
        if e_crit_count == 0:
            #Will store this for later to be checked against when we go to do
            #the risk calculation.
            warnings['unbuff'].append(pair)

        if e_crit_count == 0 and (c_crit_count == 0 or h_crit_count == 0):
            #Any strings put in here will be printed out using a logger.warn()
            #at the end of the core function.
            warnings['print'].append("Please note that the (%s, %s) is being \
                run with insufficient data. We recommend entering criteria \
                scores for both exposure and consequence." % (h, s))

    return warnings


def parse_overlaps(uri, habs, h_s_e, h_s_c):
    '''
    This function will take in a location, and update the dictionaries being
    passed with the new Hab/Stress subdictionary info that we're getting from
    the CSV at URI.

    Input:
        uri- The location of the CSV that we want to get ratings info from.
            This will contain information for a given habitat's individual
            criteria ratings, as well as criteria ratings for the overlap of
            every stressor.
        habs- A dictionary which contains all resilience specific criteria
            info. The key for these will be the habitat name. It will map to a
            subdictionary containing criteria information. The whole dictionary
            will look like the following:

            {Habitat A:
                {'Crit_Ratings':
                    {'CritName':
                        {'Rating': 2.0, 'DQ': 1.0, 'Weight': 1.0}
                    },
                'Crit_Rasters':
                    {'CritName':
                        {'Weight': 1.0, 'DQ': 1.0}
                    },
                }
            }

        h_s_e- A dictionary containing all information applicable to exposure
            criteria. The dictionary will look identical to the 'habs'
            dictionary, but each key will be a tuple of two strings -
            (HabName, StressName).
        h_s_c- A dictionary containing all information applicable to
            sensitivity criteria. The dictionary will look identical to the
            'habs' dictionary, but each key will be a tuple of two strings -
            (HabName, StressName).
    '''

    with open(uri, 'rU') as hab_file:

        dialect = csv.Sniffer().sniff(hab_file.read(1024))
        hab_file.seek(0)
        csv_reader = csv.reader(hab_file, dialect)

        hab_name = csv_reader.next()[1]

        #Can at least initialized the habs dictionary part.
        habs[hab_name] = {'Crit_Rasters': {}, 'Crit_Ratings': {}}

        #Drain the next two lines
        for _ in range(2):
            csv_reader.next()

        #Get the headers
        headers = csv_reader.next()[1:]
        line = csv_reader.next()

        #Drain the habitat-specific dictionary
        #Since line is an array, try to join to an empty list. Either an empty
        #array or an array of '' will return ''. Anything else will have stuff
        #in the string.
        while ''.join(line) != '':

            if line[0] != '':
                key = line[0]

                #If we are dealing with a shapefile criteria, we only want  to
                #add the DQ and the W, and we will add a rasterized version of
                #the shapefile later.
                if line[1] == 'SHAPE':
                    try:
                        habs[hab_name]['Crit_Rasters'][key] = dict(
                            zip(headers[1:3], [float(x) for x in line[2:4]]))
                    except ValueError:
                        raise UnexpectedString("Entries in CSV table may not \
                            be strings, and may not be left blank. Check \
                            your %s CSV for any leftover strings or spaces \
                            within Rating, Data Quality or Weight \
                            columns." % hab_name)
                #Should catch any leftovers from the autopopulation of the
                #helptext
                else:
                    try:
                        habs[hab_name]['Crit_Ratings'][key] = dict(
                            zip(headers, [float(x) for x in line[1:4]]))
                    except ValueError:
                        raise UnexpectedString("Entries in CSV table may not \
                            be strings, and may not be left blank. Check \
                            your %s CSV for any leftover strings or spaces \
                            within Rating, Data Quality or Weight \
                            columns." % hab_name)

            line = csv_reader.next()

        #We will have just loaded in a null line from under the hab-specific
        #criteria, now drainthe next two, since they're just headers for users.
        #Drain the next two lines, and load in the third.
        for _ in range(2):
            a = csv_reader.next()

        #Now we will pick up all the E/C habitat-stressor information for this
        #specific habitat.

        counter = 0

        #Want to do this for all pairings of h-s  within the habitat.
        while True:

            try:
                line = csv_reader.next()
            except StopIteration:
                break

            if len(line) == 0:
                counter += 1
            #If there's something in here, we have another set of h-s to go.
            else:
                counter = 0

            #If our current line is blank, we're at the end of the file.
            if counter > 1:
                break

            stress_name = (line[0].split(hab_name+'/')[1]).split(' ')[0]
            headers = csv_reader.next()[1:]

            #Drain the overlap table
            line = csv_reader.next()

            #Create empty entries for this overlap in both the _e and _c
            #dictionaries.
            h_s_e[(hab_name, stress_name)] = {'Crit_Ratings': {},
                                              'Crit_Rasters': {}}
            h_s_c[(hab_name, stress_name)] = {'Crit_Ratings': {},
                                              'Crit_Rasters': {}}

            # Create a new entry in the habitat / stressor exposure
            # and consequence dictionaries. This list is to determine
            # if this pair has a valid interaction and therefore to
            # compute overlap. Used in hra.make_add_overlap_rasters
            h_s_e[(hab_name, stress_name)]['overlap_list'] = []
            h_s_c[(hab_name, stress_name)]['overlap_list'] = []

            # Draining the ratings scores.
            while ''.join(line) != '':

                # Just abstract all of the erroring out, so that we know if
                # we're below here, it should all work perfectly. LOL
                error_check(line, hab_name, stress_name)

                for criteria_type, h_s_x, in zip(['E', 'C'], [h_s_e, h_s_c]):
                    # Exposure/Consequence criteria.
                    if line[4] == criteria_type:
                        # Add a default True value for each criteria to the
                        # overlap list, assuming overlap should occur.
                        h_s_x[(hab_name, stress_name)]['overlap_list'].append(True)
                        # If criteria rasters are desired for that criteria.
                        if line[1] == 'SHAPE':
                            h_s_x[(hab_name, stress_name)]['Crit_Rasters'][
                                line[0]] = dict(
                                    zip(headers[1:3],
                                        [float(x) for x in line[2:4]]))
                        # Have already error checked, so this must be a float.
                        else:
                            # Check to see if the user has set a rating to "NA"
                            if line[1].lower() == 'na':
                                # Since the user has entered a Ratings value of
                                # "NA", they indicate there should be no
                                # interaction between stressor and habitat

                                # For consistency, set all other headers but
                                # Ratings to specified value
                                h_s_x[(hab_name, stress_name)]['Crit_Ratings'][
                                    line[0]] = dict(
                                        zip(headers[1:3],
                                            [float(x) for x in line[2:4]]))
                                # Set the Rating to 0.0, this is so that later
                                # in the function "zero_check", the dictionary
                                # line is removed and not used in future
                                # calculations
                                h_s_x[(hab_name, stress_name)]['Crit_Ratings'][
                                    line[0]][headers[0]] = 0.0
                                # Set the habitat / stressor overlap indicator
                                # to False
                                h_s_x[(hab_name, stress_name)]['overlap_list'][-1] = False
                            else:
                                h_s_x[(hab_name, stress_name)]['Crit_Ratings'][
                                    line[0]] = dict(
                                        zip(headers,
                                            [float(x) for x in line[1:4]]))
                    else:
                        continue

                try:
                    line = csv_reader.next()
                except StopIteration:
                    break

            # Combine consequence and exposure overlap lists
            overlap_list = (h_s_e[(hab_name, stress_name)]['overlap_list'] +
                h_s_c[(hab_name, stress_name)]['overlap_list'])
            # Error if at least one, but not all Ratings for the
            # habitat - stressor pair are set to 'NA'
            if any(overlap_list) and not all(overlap_list):
                raise NA_RatingsError("There were 'NA' Rating values found "
                    "for habitat - stressor pair : %s - %s , however, not "
                    "all Rating values were set to 'NA'. ALL or NONE Rating "
                    "values for the habitat - stressor pair must be set to "
                    "'NA'. " % (hab_name, stress_name))

            #Assume if we've gotten here, we're at the end of a block. Iterate
            #counter by one, since we know it's A whitespace line. If there is
            #a second after, the while loop will jump to here.
            counter += 1


def error_check(line, hab_name, stress_name):
    '''
    Throwing together a simple error checking function for all of the inputs
    coming from the CSV file. Want to do checks for strings vs floats, as well
    as some explicit string checking for 'E'/'C'.

    Input:
        line- An array containing a line of H-S overlap data. The format of a
            line would look like the following:

            ['CritName', 'Rating', 'Weight', 'DataQuality', 'Exp/Cons']

            The following restrictions should be placed on the data:

                CritName- This will be propogated by default by
                    HRA_Preprocessor. Since it's coming in as a string, we
                    shouldn't need to check anything.
                Rating- Can either be the explicit string 'SHAPE', which would
                    be placed automatically by HRA_Preprocessor, or a float.
                    ERROR: if string that isn't 'SHAPE'.
                Weight- Must be a float (or an int), but cannot be 0.
                    ERROR: if string, or anything not castable to float, or 0.
                DataQuality- Most be a float (or an int), but cannot be 0.
                    ERROR: if string, or anything not castable to float, or 0.
                Exp/Cons- Most be the string 'E' or 'C'.
                    ERROR: if string that isn't one of the acceptable ones,
                    or ANYTHING else.

    Returns nothing, should raise exception if there's an issue.
    '''
    # Check that, if the Ratings value is not SHAPE or NA that it is a float.
    #Rating
    if line[1] != 'SHAPE' and line[1].lower() != 'na':
        try:
            float(line[1])
        except ValueError:
            raise UnexpectedString("Entries in CSV table may not be strings, \
                and may not be left blank. Check your %s CSV in %s section \
                for any leftover strings or spaces within Rating, Data \
                Quality or Weight columns." % (hab_name, stress_name))

    #Weight and DQ

    #They may not be 0 unless Rating is 0.
    if (line[2] in ['0', '0.0'] or line[3] in ['0', '0.0']) and line[1] \
            not in ['0', '0.0']:
        raise ZeroDQWeightValue("Individual criteria data qualities and \
            weights may not be 0. Check your %s CSV table in the %s section \
            to correct this." % (hab_name, stress_name))

    #Assuming neither is 0, they also must be floats.
    try:
        float(line[2])
        float(line[3])
    except ValueError:
        raise UnexpectedString("Entries in CSV table may not be strings, \
            and may not be left blank. Check your %s CSV in %s section for \
            any leftover strings or spaces within Rating, Data Quality or \
            Weight columns." % (hab_name, stress_name))

    #Exposure vs Consequence
    if line[4] != 'E' and line[4] != 'C':

        raise ImproperECSelection("Entries in the E/C column of a CSV table \
            may only be \"E\" or \"C\". Please select one of those options \
            for the criteria in the %s section of the %s CSV \
            table." % (stress_name, hab_name))


def parse_stress_buffer(uri):
    '''
    This will take the stressor buffer CSV and parse it into a dictionary
    where the stressor name maps to a float of the about by which it should be
    buffered.

    Input:
        uri- The location of the CSV file from which we should pull the buffer
            amounts.

    Returns:
        A dictionary containing stressor names mapped to their corresponding
        buffer amounts. The float may be 0, but may not be a string. The form
        will be the following:

            {'Stress 1': 2000, 'Stress 2': 1500, 'Stress 3': 0, ...}
    '''

    buff_dict = {}

    with open(uri, 'rU') as buff_file:

        dialect = csv.Sniffer().sniff(buff_file.read(1024))
        buff_file.seek(0)
        csv_reader = csv.reader(buff_file, dialect)

        #Drain the first two lines, since just headers and blank
        for _ in range(2):
            csv_reader.next()

        #We know that the rest of the table will just be stressor names and
        #their mappings, so we are clear to just drain the table.
        for row in csv_reader:
            LOGGER.debug(row)
            s_name = row[0]

            try:
                #Make sure that what they're passing in as a buffer is a
                #number, not the leftover help string.
                buff_dict[s_name] = float(row[1])

            except ValueError:
                raise UnexpectedString("Entries in CSV table may not be \
                    strings, and may not be left blank. Check your Stressor \
                    Buffer CSV for any leftover strings or spaces within the \
                    buffer amount. Entries must be a number, and may not be \
                    left blank.")

    return buff_dict
