"""Usage: python get_args.py <json_filename>"""

import json
import sys


#: This is the basic template for a google-style docstring.  We're assuming
#: that the execute() call is at the module level.
LINE_TEMPLATE = '        {args_id} ({type}): {description}'


def main(filename):
    """Extract a Google-style docstring from an IUI json file.

    This script assumes that the function being called conforms to the InVEST
    API: the function has a single parameter, a dictionary of string
    argument names mapping to some value.  Based on the IUI json dictionary
    provided, this script extracts the following information:

        * ``args_id``: If an element doesn't have this
          attribute, it isn't considered an input to the model and won't be
          documented by this script.
        * ``helpText``: If an element doesn't have a
          defined helpText string, we try to use the label.  If no label is
          defined, a default string is assumed and will need to be filled in.
        * ``required``: If this attribute isn't defined by an element, we
          assume that th element is not required.
        * ``type``: The type of the element (and in some cases, the
          ``dataType`` as well) indicates what the return type of the element
          should be.

    The returned string is formatted such that each line does not exceed 79
    characters in length.

    Note:
        Type analysis here is imperfect.  Be sure to double-check the output!

    Parameters:
        filename (string): The path to an IUI json file on disk.

    Returns:
        ``None``, but the formatted docstring is printed to stdout.
    """
    iui_dict = json.load(open(filename))

    def recurse(args_dict):
        """Recurse through the input dict and print the docstring if needed.

        Parameters:
            args_dict (dict): A dict defining a whole form or a portion of the
                form.

        Returns:
            ``None``
        """
        try:
            for element_config in args_dict['elements']:
                recurse(element_config)
        except KeyError:
            pass

        try:
            args_id = args_dict['args_id']
        except KeyError:
            return

        try:
            helptext = args_dict['helpText']
        except KeyError:
            try:
                helptext = args_dict['label']
            except KeyError:
                helptext = "See the User's Guide for more information."

        try:
            required = bool(args_dict['required'])
        except KeyError:
            required = False

        return_type = ''
        if args_dict['type'] in ['file', 'folder']:
            return_type = 'string'
        else:
            try:
                if args_dict['dataType'] in ['string']:
                    return_type = 'string'
            except KeyError:
                pass

        helptext += ' (%s)' % 'required' if required else 'optional'

        # 8 for initial indent, 10 for type, 1 for colon, 8 more for good
        # measure
        param_name_len = len(args_id) + 8 + 10 + 1 + 8
        max_firstline_len = 79 - param_name_len

        formatted_helptext = ''
        first_line_satisfied = False
        current_line_counter = 12  # indent for new lines.
        for word in helptext.split(' '):
            if not first_line_satisfied:
                if len(formatted_helptext) > max_firstline_len:
                    formatted_helptext += '\n            ' + word
                    first_line_satisfied = True
                else:
                    formatted_helptext += (' ' + word)
            else:
                if current_line_counter + len(word) > 79:
                    formatted_helptext += ('\n' + '            ' + word)
                    current_line_counter = 12 + len(word)
                else:
                    formatted_helptext += (' ' + word)
                    current_line_counter += len(word) + 1

        formatted_template = LINE_TEMPLATE.format(
            args_id=args_id, description=formatted_helptext, type=return_type)
        print formatted_template

    recurse(iui_dict)


if __name__ == '__main__':
    main(sys.argv[1])
