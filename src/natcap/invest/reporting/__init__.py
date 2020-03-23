"""natcap.invest.reporting package."""

import os
import logging
import codecs
import re
import copy

from ... import invest
from .. import utils
from osgeo import gdal
from . import table_generator


LOGGER = logging.getLogger('natcap.invest.reporting')
REPORTING_DATA = os.path.join(invest.local_dir(__file__), 'reporting_data/')
JQUERY_URI = os.path.join(REPORTING_DATA, 'jquery-1.10.2.min.js')
SORTTABLE_URI = os.path.join(REPORTING_DATA, 'sorttable.js')
TOTALS_URI = os.path.join(REPORTING_DATA, 'total_functions.js')


def generate_report(args):
    """Generate an html page from the arguments given in 'reporting_args'

        reporting_args[title] - a string for the title of the html page
            (required)

        reporting_args[sortable] - a boolean value indicating whether
            the sorttable.js library should be added for table sorting
            functionality (optional)

        reporting_args[totals] - a boolean value indicating whether
            the totals_function.js script should be added for table totals
            functionality (optional)

        reporting_args[out_uri] - a URI to the output destination for the html
            page (required)

        reporting_args[elements] - a list of dictionaries that represent html
            elements to be added to the html page. (required) If no elements
            are provided (list is empty) a blank html page will be generated.
            The 3 main element types are 'table', 'head', and 'text'.
            All elements share the following arguments:
                'type' - a string that depicts the type of element being add.
                    Currently 'table', 'head', and 'text' are defined
                    (required)

                'section' - a string that depicts whether the element belongs
                    in the body or head of the html page.
                    Values: 'body' | 'head' (required)

            Table element dictionary has at least the following additional
            arguments:
                'attributes' - a dictionary of html table attributes. The
                    attribute name is the key which gets set to the value
                    of the key. (optional)
                    Example: {'class': 'sorttable', 'id': 'parcel_table'}

                'sortable' - a boolean value for whether the tables columns
                    should be sortable (required)

                'checkbox' - a boolean value for whether there should be a
                    checkbox column. If True a 'selected total' row will be
                    added to the bottom of the table that will show the
                    total of the columns selected (optional)

                'checkbox_pos' - an integer value for in which column
                    position the the checkbox column should appear
                    (optional)

                'data_type' - one of the following string values:
                    'shapefile'|'hg csv'|'dictionary'. Depicts the type of data
                    structure to build the table from (required)

                'data' - either a list of dictionaries if 'data_type' is
                    'dictionary' or a URI to a CSV table or shapefile if
                    'data_type' is 'shapefile' or 'csv' (required). If a
                    list of dictionaries, each dictionary should have
                    keys that represent the columns, where each dictionary
                    is a row (list could be empty)
                    How the rows are ordered are defined by their
                    index in the list. Formatted example:
                    [{col_name_1: value, col_name_2: value, ...},
                     {col_name_1: value, col_name_2: value, ...},
                     ...]

                'key' - a string that defines which column or field should be
                    used as the keys for extracting data from a shapefile or
                    csv table 'key_field'.
                    (required for 'data_type' = 'shapefile' | 'csv')

                'columns'- a list of dictionaries that defines the column
                    structure for the table (required). The order of the
                    columns from left to right is depicted by the index
                    of the column dictionary in the list. Each dictionary
                    in the list has the following keys and values:
                        'name' - a string for the column name (required)
                        'total' - a boolean for whether the column should be
                            totaled (required)
                        'attr' - a dictionary that has key value pairs for
                            optional tag attributes (optional). Ex:
                            'attr': {'class': 'offsets'}
                        'td_class' - a String to assign as a class name to
                            the table data tags under the column. Each
                            table data tag under the column will have a class
                            attribute assigned to 'td_class' value (optional)

                'total'- a boolean value for whether there should be a constant
                    total row at the bottom of the table that sums the column
                    values (optional)

            Head element dictionary has at least the following additional
            arguments:
                'format' - a string representing the type of head element being
                    added. Currently 'script' (javascript) and 'style' (css
                    style) accepted (required)

                'data_src'- a URI to the location of the external file for
                    either the 'script' or the 'style' OR a String representing
                    the html script or style (DO NOT include the tags)
                    (required)

                'input_type' -  a String, 'File' or 'Text' that refers to how
                    'data_src' is being passed in (URI vs String) (required).

                'attributes' - a dictionary that has key value pairs for
                    optional tag attributes (optional). Ex:
                    'attributes': {'id': 'muni_data'}

            Text element dictionary has at least the following additional
            arguments:
                'text'- a string to add as a paragraph element in the html page
                    (required)

        returns - nothing"""

    LOGGER.info('Creating HTML Report')
    # Since the dictionary being is mutated, make a copy to mutate on
    # while keeping the integrity of the original
    reporting_args = copy.deepcopy(args)
    # Get the title for the html page and place it in a string with html
    # title tags
    html_title = '<title>%s</title>' % reporting_args['title']

    # Initiate the html dictionary which will store all the head and body
    # elements. The 'head' and 'body' keys points to a tuple of two lists. The
    # first list holds the string representations of the html elements and the
    # second list is the corresponding 'position' of those elements. This allows
    # for proper ordering later in 'write_html'.
    # Initialize head's first element to be the title where the -1 position
    # ensures it will be the first element
    html_obj = {'head':[html_title], 'body':[]}

    # A dictionary of 'types' that point to corresponding functions. When an
    # 'element' is passed in the 'type' will be one of the defined types below
    # and will execute a function that properly handles that element
    report = {
            'table': build_table,
            'text' : add_text_element,
            'head': add_head_element
            }

    LOGGER.debug('Adding default JavaScript libs')
    # Add Jquery file to the elements list any time a html page is generated
    jquery_dict = {
            'type': 'head', 'section': 'head', 'format': 'script',
            'data_src': JQUERY_URI, 'input_type':'File'}
    reporting_args['elements'].insert(0, jquery_dict)

    # A list of tuples of possible default js libraries / scripts to add
    jsc_lib_list = [('totals', TOTALS_URI), ('sortable', SORTTABLE_URI)]
    # Used to have control of how the js libraries / scripts get added
    index = 1
    for lib_tup in jsc_lib_list:
        if (lib_tup[0] in reporting_args) and reporting_args[lib_tup[0]]:
            # Build up the dictionary for the script head element
            lib_dict = {
                'type': 'head', 'section': 'head', 'format': 'script',
                'data_src': lib_tup[1], 'input_type':'File'}
            # Add dictionary to elements list
            reporting_args['elements'].insert(index, lib_dict)
            index = index + 1

    # Iterate over the elements to be added to the html page
    for element in reporting_args['elements']:
        # There are 2 general purpose arguments that each element will have,
        # 'type' and 'section'. Get and remove these from the
        # elements dictionary (they should not be added weight passed to the
        # individual element functions)
        fun_type = element.pop('type')
        section = element.pop('section')

        # Process the element by calling it's specific function handler which
        # will return a string. Append this to html dictionary to be written
        # in write_html
        html_obj[section].append(report[fun_type](element))

    # Write the html page to 'out_uri'
    write_html(html_obj, reporting_args['out_uri'])


def write_html(html_obj, out_uri):
    """Write an html file to 'out_uri' from html element represented as strings
        in 'html_obj'

        html_obj - a dictionary with two keys, 'head' and 'body', that point to
            lists. The list for each key is a list of the htmls elements as
            strings (required)
            example: {'head':['elem_1', 'elem_2',...],
                      'body':['elem_1', 'elem_2',...]}

        out_uri - a URI for the output html file

        returns - nothing"""

    LOGGER.debug('Writing HTML page')

    # Start the string that will be written as the html file
    html_str = '<html>'

    for section in ['head', 'body']:
        # Ensure the browser interprets the html file as utf-8
        if section == 'head':
            html_str += '<meta charset="UTF-8">'

        # Write the tag for the section
        html_str += '<%s>' % section
        # Get the list of html string elements for this section
        sect_elements = html_obj[section]

        for element in sect_elements:
            # Add each element to the html string
            if type(element) is str:
                element = element
            html_str += element

        # Add the closing tag for the section
        html_str += '</%s>' % section

    # Finish the html tag
    html_str += '</html>'

    #LOGGER.debug('HTML Complete String : %s', html_str)

    # If the URI for the html output file exists remove it
    if os.path.isfile(out_uri):
        os.remove(out_uri)

    # Open the file, write the string and close the file
    html_file = codecs.open(out_uri, 'wb', 'utf-8')
    html_file.write(html_str)
    html_file.close()


def build_table(param_args):
    """Generates a string representing a table in html format.

        param_args - a dictionary that has the parameters for building up the
            html table. The dictionary includes the following:

            'attributes' - a dictionary of html table attributes. The attribute
                    name is the key which gets set to the value of the key.
                    (optional)
                    Example: {'class': 'sorttable', 'id': 'parcel_table'}

            param_args['sortable'] - a boolean value that determines whether the
                table should be sortable (required)

            param_args['data_type'] - a string depicting the type of input to
                build the table from. Either 'shapefile', 'csv', or 'dictionary'
                (required)

            param_args['data'] - a URI to a csv or shapefile OR a list of
                dictionaries. If a list of dictionaries the data should be
                represented in the following format: (required)
                    [{col_name_1: value, col_name_2: value, ...},
                     {col_name_1: value, col_name_2: value, ...},
                     ...]

            param_args['key'] - a string that depicts which column (csv) or
                field (shapefile) will be the unique key to use in extracting
                the data into a dictionary. (required for 'data_type'
                'shapefile' and 'csv')

            param_args['columns'] - a list of dictionaries that defines the
                    column structure for the table (required). The order of
                    the columns from left to right is depicted by the index
                    of the column dictionary in the list. Each dictionary
                    in the list has the following keys and values:
                        'name' - a string for the column name (required)
                        'total' - a boolean for whether the column should be
                            totaled (required)
                        'attr' - a dictionary that has key value pairs for
                            optional tag attributes (optional). Ex:
                            'attr': {'class': 'offsets'}
                        'td_class' - a String to assign as a class name to
                            the table data tags under the column. Each
                            table data tag under the column will have a class
                            attribute assigned to 'td_class' value (optional)

            param_args['total'] - a boolean value where if True a constant
                total row will be placed at the bottom of the table that sums
                the columns (required)

        returns - a string that represents an html table
    """
    LOGGER.debug('Building Table Structure')
    # Initialize an intermediate dictionary which will hold the physical data
    # elements of the table
    data_dict = {}

    # Initialize the final dictionary which will have the data of the table as
    # well as parameters needed to build up the html table
    table_dict = {}

    # Get the data type of the input being passed in so that it can properly be
    # pre-processed
    data_type = param_args['data_type']

    # Get a handle on the input data being passed in, whether it a URI to a
    # shapefile / csv file or a list of dictionaries
    input_data = param_args['data']

    # Depending on the type of input being passed in, pre-process it
    # accordingly
    if data_type == 'shapefile':
        key = param_args['key']
        data_dict = extract_datasource_table_by_key(
            input_data, key)
        # Convert the data_dict to a list of dictionaries where each dictionary
        # in the list represents a row of the table
        data_list = data_dict_to_list(data_dict)
    elif data_type == 'csv':
        key = param_args['key']
        data_dict = utils.build_lookup_from_csv(input_data, key)
        # Convert the data_dict to a list of dictionaries where each dictionary
        # in the list represents a row of the table
        data_list = data_dict_to_list(data_dict)
    else:
        data_list = input_data

    #LOGGER.debug('Data Collected from Input Source: %s', data_list)
    LOGGER.debug('Data Collected from Input Source')

    # Add the columns data to the final dictionary that is to be passed
    # off to the table generator
    table_dict['cols'] = param_args['columns']

    # Add the properly formatted row data to the final dictionary that is
    # to be passed to the table generator
    table_dict['rows'] = data_list

    # If a totals row is present, add it to the final dictionary
    if 'total' in param_args:
        table_dict['total'] = param_args['total']

    # If table attributes were passed in check to see if the 'sortable' class
    # needs to be added to that list
    if 'attributes' in param_args:
        table_dict['attributes'] = param_args['attributes']
        if param_args['sortable']:
            try:
                class_list = table_dict['attributes']['class'] + ' sortable'
                table_dict['attributes']['class'] = class_list
            except KeyError:
                table_dict['attributes']['class'] = 'sortable'
    else:
        # Attributes were not passed in, however if sortable is True
        # create attributes key and dictionary to pass in to table
        # handler
        if param_args['sortable']:
            table_dict['attributes'] = {'class': 'sortable'}

    # If a checkbox column is wanted pass in the table dictionary
    if 'checkbox' in param_args and param_args['checkbox']:
        table_dict['checkbox'] = True
        if 'checkbox_pos' in param_args:
            table_dict['checkbox_pos'] = param_args['checkbox_pos']

    LOGGER.debug('Calling table_generator')
    # Call generate table passing in the final dictionary and attribute
    # dictionary. Return the generate string
    return table_generator.generate_table(table_dict)


def extract_datasource_table_by_key(datasource_uri, key_field):
    """Return vector attribute table of first layer as dictionary.

    Create a dictionary lookup table of the features in the attribute table
    of the datasource referenced by datasource_uri.

    Args:
        datasource_uri (string): a uri to an OGR datasource
        key_field: a field in datasource_uri that refers to a key value
            for each row such as a polygon id.

    Returns:
        attribute_dictionary (dict): returns a dictionary of the
            form {key_field_0: {field_0: value0, field_1: value1}...}
    """
    # Pull apart the datasource
    datasource = gdal.OpenEx(datasource_uri)
    layer = datasource.GetLayer()
    layer_def = layer.GetLayerDefn()

    # Build up a list of field names for the datasource table
    field_names = []
    for field_id in range(layer_def.GetFieldCount()):
        field_def = layer_def.GetFieldDefn(field_id)
        field_names.append(field_def.GetName())

    # Loop through each feature and build up the dictionary representing the
    # attribute table
    attribute_dictionary = {}
    for feature in layer:
        feature_fields = {}
        for field_name in field_names:
            feature_fields[field_name] = feature.GetField(field_name)
        key_value = feature.GetField(key_field)
        attribute_dictionary[key_value] = feature_fields

    # Explictly clean up the layers so the files close
    layer = None
    datasource = None
    return attribute_dictionary


def data_dict_to_list(data_dict):
    """Abstract out inner dictionaries from data_dict into a list, where
        the inner dictionaries are added to the list in the order of
        their sorted keys

        data_dict - a dictionary with unique keys pointing to dictionaries.
            Could be empty (required)

        returns - a list of dictionaries, or empty list if data_dict is empty"""

    data_list = []
    data_keys = list(data_dict)
    data_keys.sort()
    for key in data_keys:
        data = data_dict[key]
        data_list.append(data)

    return data_list


def add_text_element(param_args):
    """Generates a string that represents a html text block. The input string
        should be wrapped in proper html tags

        param_args - a dictionary with the following arguments:

            param_args['text'] - a string

        returns - a string
    """

    return param_args['text']


def add_head_element(param_args):
    """Generates a string that represents a valid element in the head section
        of an html file. Currently handles 'style' and 'script' elements,
        where both the script and style are locally embedded

        param_args - a dictionary that holds the following arguments:

            param_args['format'] - a string representing the type of element to
                be added. Currently : 'script', 'style' (required)

            param_args['data_src'] - a string URI path for the external source
                of the element OR a String representing the html
                (DO NOT include html tags, tags are automatically generated).
                If a URI the file is read in as a String. (required)

            param_args['input_type'] - 'Text' or 'File'. Determines how the
                input from 'data_src' is handled (required)

            'attributes' - a dictionary that has key value pairs for
                optional tag attributes (optional). Ex:
                'attributes': {'class': 'offsets'}

        returns - a string representation of the html head element"""

    LOGGER.info('Preparing to generate head element as String')

    # Get the type of element to add
    form = param_args['format']
    # Get a handle on the data whether it be a String or URI
    src = param_args['data_src']
    # Get the input type of the data, 'File' or 'Text'
    input_type = param_args['input_type']
    if input_type == 'File':
        # Read in file and save as string. Using latin1 to decode, seems to
        # work on the current javascript / css files
        head_file = codecs.open(src, 'rb', 'latin1')
        file_str = head_file.read()
    else:
        file_str = src

    attr = ''
    if 'attributes' in param_args:
        for key, val in param_args['attributes'].items():
            attr += '%s="%s" ' % (key, val)

    # List of regular expression strings to search against
    reg_list = [r'<script', r'/script>', r'<style', r'/style>']

    # Iterate over the String object to make sure there are no conflicting html
    # tags
    for exp in reg_list:
        if re.search(exp, file_str) != None:
            raise Exception('The following html tag was found in header'
                    ' string : %s. Please do not place any html tags in'
                    ' the header elements' % exp)

    if form == 'style':
        html_str = """<style type=text/css %s> %s </style>""" % (attr, file_str)
    elif form == 'script':
        html_str = """<script type=text/javascript %s> %s </script>""" % (attr, file_str)
    elif form == 'json':
        html_str = """<script type=application/json %s> %s </script>""" % (attr, file_str)
    else:
        raise Exception('Currently this type of head element is not supported'
                ' : %s' % form)

    return html_str
