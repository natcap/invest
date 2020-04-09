"""A helper module for generating html tables that are represented as Strings"""
import logging

LOGGER = logging.getLogger('natcap.invest.reporting.table_generator')

def generate_table(table_dict, attributes=None):
    """Takes in a dictionary representation of a table and generates a String of
        the the table in the form of hmtl

        table_dict - a dictionary with the following arguments:
            'cols'- a list of dictionaries that defines the column
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

            'rows' - a list of dictionaries that represent the rows. Each
                dictionaries keys should match the column names found in
                'cols' (possibly empty list) (required) Example:
                [{col_name_1: value, col_name_2: value, ...},
                 {col_name_1: value, col_name_2: value, ...},
                 ...]

            'checkbox' - a boolean value for whether there should be a
                checkbox column. If True a 'selected total' row will be added
                to the bottom of the table that will show the total of the
                columns selected (optional)

            'checkbox_pos' - an integer value for in which column
                position the the checkbox column should appear
                (optional)

            'total'- a boolean value for whether there should be a constant
                total row at the bottom of the table that sums the column
                values (optional)

            'attributes' - a dictionary of html table attributes. The attribute
                    name is the key which gets set to the value of the key.
                    (optional)
                    Example: {'class': 'sorttable', 'id': 'parcel_table'}

            returns - a string representing an html table
    """

    LOGGER.info('Generating HTML Table String')

    # Initialize the string that will store the html representation of the table
    table_string = ''

    if 'attributes' in table_dict:
        table_string += '<table'
        for attr_key, attr_value in table_dict['attributes'].items():
            table_string += ' %s="%s"' % (attr_key, attr_value)

        table_string += '>'
    else:
        table_string += '<table>'

    # If checkbox column is wanted set it up
    if ('checkbox' in table_dict) and (table_dict['checkbox']):
        # Set default checkbox column position to 1
        checkbox_pos = 1
        if 'checkbox_pos' in table_dict:
            # If user specified checkbox position, update here
            checkbox_pos = table_dict['checkbox_pos']

        # Get a copy of the column and row lists of dictionaries
        # to pass into checkbox function
        cols_copy = list(table_dict['cols'])
        rows_copy = list(table_dict['rows'])
        # Get the updated column and row data after adding a
        # checkbox column
        table_cols, table_rows = add_checkbox_column(
            cols_copy, rows_copy, checkbox_pos)
        add_checkbox_total = True
    else:
        # The column and row lists of dictionaries need to update,
        # so get the originals
        table_cols = table_dict['cols']
        table_rows = table_dict['rows']
        add_checkbox_total = False

    # Get the column headers
    col_headers = get_dictionary_values_ordered(table_cols, 'name')
    # Get a list of booleans indicating whether the above column
    # headers should be allowed to be totaled
    total_cols = get_dictionary_values_ordered(table_cols, 'total')

    def construct_td_classes(table_columns):
        """Creates a list of tuples based on whether a column has
            'td_class' attribute present. If present the first index
            in the tuple is set to True and the second is set to a
            String value for the value of the 'td_class' attribute.
            If not present, the first index is set to False and None
            is set for the second.

            table_columns - a Python list of dictionaries representing
                the columns of the table

            returns - a list of tuples, the first index in the tuple of type
                boolean and the second of type String
        """
        # List to hold the tuples
        table_data_list = []
        for col_dict in table_columns:
            if 'td_class' in col_dict:
                # If 'td_class' is a key in 'col_dict', then set the
                # tuple to True with the it's String value
                table_data_list.append((True, col_dict['td_class']))
            else:
                # If 'td_class' is not a key in 'col_dict', then
                # set the tuple to False with a value of None
                table_data_list.append((False, None))

        return table_data_list

    # Get a list of tuples representing if the key 'td_class'
    # is found in 'table_cols' and if so set the value
    LOGGER.debug('Construct table data classes')
    tdata_tuples = construct_td_classes(table_cols)

    def attr_to_string(attr_dict):
        """Concatenates a string from key value pairs in 'attr_dict'.
            The string is generated in such a way where each key is
            set equal to it's value and each key/value are separated
            by a space. Example: 'key1=value1 key2=value2...'

            attr_dict - a dictionary of key value pairs

            returns - a string
        """
        attr_str = ''
        for key, value in attr_dict.items():
            attr_str += ' %s= "%s"' % (key, str(value))

        return attr_str

    # Write table header tag followed by table row tag
    table_string = table_string + '<thead><tr>'
    for col_dict in table_cols:
        # Add each column header to the html string
        try:
            col_attr = attr_to_string(col_dict['attr'])
            table_string += '<th%s>%s</th>' % (col_attr, col_dict['name'])
        except KeyError:
            table_string += '<th>%s</th>' % col_dict['name']

    # Add the closing tag for the table header
    table_string += '</tr></thead>'

    # Get the row data as 2D list
    row_data = get_row_data(table_rows, col_headers)

    footer_string = ''

    if add_checkbox_total:
        footer_string += add_totals_row(
                col_headers, total_cols, 'Selected Total', True, tdata_tuples)

    # Add any total rows as 'tfoot' elements in the table
    if 'total' in table_dict and table_dict['total']:
        footer_string += add_totals_row(
                col_headers, total_cols, 'Total', False, tdata_tuples)

    if not footer_string == '':
        table_string += '<tfoot>%s</tfoot>' % footer_string

    # Add the start tag for the table body
    table_string += '<tbody>'

    LOGGER.debug('Construct html string for table body')
    # For each data row add a row in the html table and fill in the data
    for row in row_data:
        table_string += '<tr>'
        # Iterate over each row data, where the index indicates the column
        # index as well
        for row_index in range(len(row)):
            if total_cols[row_index]:
                class_str = 'rowDataSd '
                if tdata_tuples[row_index][0]:
                    class_str += tdata_tuples[row_index][1]
                # Add row data
                table_string += ('<td class="%s">%s</td>' %
                                    (class_str, row[row_index]))
            else:
                if tdata_tuples[row_index][0]:
                    class_str = tdata_tuples[row_index][1]
                    table_string += ('<td class="%s">%s</td>' %
                                        (class_str, row[row_index]))
                else:
                    table_string += '<td>%s</td>' % row[row_index]

        table_string += '</tr>'

    # Add the closing tag for the table body and table
    table_string += '</tbody></table>'

    return table_string

def add_totals_row(col_headers, total_list, total_name, checkbox_total,
                    tdata_tuples):
    """Construct a totals row as an html string. Creates one row element with
        data where the row gets a class name and the data get a class name if
        the corresponding column is a totalable column

        col_headers - a list of the column headers in order (required)

        total_list - a list of booleans that corresponds to 'col_headers' and
            indicates whether a column should be totaled (required)

        total_name - a string for the name of the total row, ex: 'Total', 'Sum'
            (required)

        checkbox_total - a boolean value that distinguishes whether a checkbox
            total row is being added or a regular total row. Checkbox total row
            is True. This will determine the row class name and row data class
            name (required)

        tdata_tuples - a list of tuples where the first index in the tuple is a
            boolean which indicates if a table data element has a attribute
            class. The second index is the String value of that class or None
            (required)

        return - a string representing the html contents of a row which should
            later be used in a 'tfoot' element"""

    LOGGER.debug('Generating a String for a Totals row')
    # Check to determine whether a checkbox total row is being added or a
    # regular totals row
    if checkbox_total:
        row_class = 'checkTotal'
        data_class = 'checkTot'
    else:
        row_class = 'totalColumn'
        data_class = 'totalCol'

    # Begin constructing the html string for the new totals row
    # Give the row a class name and have the first data element be the name or
    # header for that row. Also assign table data a class if tdata_tuples
    # indicates to do so
    if tdata_tuples[0][0]:
        class_name = tdata_tuples[0][1]
        html_str = ('<tr class="%s"><td class="%s">%s</td>' %
                        (row_class, class_name, total_name))
    else:
        html_str = '<tr class="%s"><td>%s</td>' % (row_class, total_name)

    # Iterate over the number of columns and add proper row data value,
    # starting from the second column as the first columns row data value was
    # defined above
    for col_index in range(1, len(col_headers)):
        # Check to see if this columns values should be totaled
        if total_list[col_index]:
            # Set temp variable that may need to be adjusted
            comp_class = data_class
            # Add table data class name if true
            if tdata_tuples[col_index][0]:
                comp_class = comp_class + ' ' + tdata_tuples[col_index][1]
            # If column should be totaled then add a class name
            html_str += '<td class="%s">--</td>' % comp_class
        else:
            # Add table data class name if true
            if tdata_tuples[col_index][0]:
                html_str += '<td class="%s">--</td>' % tdata_tuples[col_index][1]
            else:
                # If the column should not be totaled leave off the class name
                html_str += '<td>--</td>'

    html_str += '</tr>'

    return html_str

def get_dictionary_values_ordered(dict_list, key_name):
    """Generate a list, with values from 'key_name' found in each dictionary
        in the list of dictionaries 'dict_list'. The order of the values in the
        returned list match the order they are retrieved from 'dict_list'

        dict_list - a list of dictionaries where each dictionary has the same
            keys. Each dictionary should have at least one key:value pair
            with the key being 'key_name' (required)

        key_name - a String or Int for the key name of interest in the
            dictionaries (required)

        return - a list of values from 'key_name' in ascending order based
            on 'dict_list' keys"""

    # Initiate an empty list to store values
    ordered_value_list = []

    # Iterate over the list and extract the wanted value from each dictionaries
    # 'key_name'. Append the value to the new list
    for item in dict_list:
        ordered_value_list.append(item[key_name])

    return ordered_value_list

def add_checkbox_column(col_list, row_list, checkbox_pos=1):
    """Insert a new column into the list of column dictionaries so that it
        is the second column dictionary found in the list. Also add the
        checkbox column header to the list of row dictionaries and
        subsequent checkbox value

        'col_list'- a list of dictionaries that defines the column
            structure for the table (required). The order of the
            columns from left to right is depicted by the index
            of the column dictionary in the list. Each dictionary
            in the list has the following keys and values:
                'name' - a string for the column name (required)
                'total' - a boolean for whether the column should be
                    totaled (required)

        'row_list' - a list of dictionaries that represent the rows. Each
            dictionaries keys should match the column names found in
            'col_list' (required) Example:
            [{col_name_1: value, col_name_2: value, ...},
             {col_name_1: value, col_name_2: value, ...},
             ...]

        checkbox_pos - an integer for the position of the checkbox
            column. Defaulted at 1 (optional)

        returns - a tuple of the updated column and rows list of dictionaries
            in that order"""

    LOGGER.debug('Adding a checkbox column to the column structure')
    # Insert a new column dictionary in the list in the second spot
    col_list.insert(checkbox_pos, {'name':'Select', 'total':False,
                        'attr':{'class':'checkbox'}, 'td_class':'checkbox'})

    # For each dictionary in the row list add a 'Select' key which
    # refers to the new column and set the value as a checkbox
    for val in row_list:
        val['Select'] = '<input type=checkbox name=cb value=1>'

    # Return a tuple of the updated / modified column and row list of
    # dictionaries
    return (col_list, row_list)

def get_row_data(row_list, col_headers):
    """Construct the rows in a 2D List from the list of dictionaries,
        using col_headers to properly order the row data.

        'row_list' - a list of dictionaries that represent the rows. Each
            dictionaries keys should match the column names found in
            'col_headers'. The rows will be ordered the same as they are found
            in the dictionary list (required) Example:
            [{'col_name_1':'9/13', 'col_name_3':'expensive',
                'col_name_2':'chips'},
             {'col_name_1':'3/13', 'col_name_2':'cheap',
                'col_name_3':'peanuts'},
             {'col_name_1':'5/12', 'col_name_2':'moderate',
                'col_name_3':'mints'}]

        col_headers - a List of the names of the column headers in order
            example : [col_name_1, col_name_2, col_name_3...]

        return - a 2D list with each inner list representing a row"""
    LOGGER.debug('Compile and return row data as a 2D list')
    # Initialize a list to hold output rows represented as lists
    row_data = []

    # Iterate over each dictionary in the row_list and append the values to a
    # list in the order the keys are found in 'col_headers'. Add each generated
    # list to 'row_data'
    for row_dict in row_list:
        row = []
        for col in col_headers:
            row.append(row_dict[col])
        row_data.append(row)

    return row_data
