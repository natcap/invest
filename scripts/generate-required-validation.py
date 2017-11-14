import argparse
import json
import sys


IS_ARG_COMPLETE = ("    if context.is_arg_complete('{args_key}', "
                   "require={required}):")


def main(args=None):
    if not args:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('iui_config', help='The path to the iui json file')
    args = parser.parse_args(args)

    config_params = json.load(open(args.iui_config))
    model_location = config_params['targetScript'].replace('natcap.invest.', '')
    relative_import_dots = '.' * len(model_location.split('.'))

    print 'from __future__ import absolute_import'
    print "from {dots} import validation".format(dots=relative_import_dots)
    print ""
    print "@validation.validator"
    print "def validate(args, limit_to=None):"
    print "    context = validation.ValidationContext(args, limit_to)"

    def _recurse(element_config):
        if 'elements' in element_config:
            for sub_element_config in element_config['elements']:
                _recurse(sub_element_config)
        else:
            if 'args_id' not in element_config:
                return

            if 'workspace' in element_config['args_id']:
                return
            if 'suffix' in element_config['args_id']:
                return

            try:
                required = element_config['required']
            except KeyError:
                if 'requiredIf' in element_config:
                    print '    # element has requiredIf logic.  See config.'
                required = False
            args_key = element_config['args_id']

            print IS_ARG_COMPLETE.format(args_key=args_key, required=required)
            print '        # Implement validation for %s here' % args_key
            print '        pass'
            print ''

    _recurse(config_params)

    print '    if limit_to is None:'
    print '        # Implement any validation that uses multiple inputs here.'
    print '        # Report multi-input warnings with:'
    print '        # context.warn(<warning>, keys=<keys_iterable>)'
    print '        pass'
    print ''
    print '    return context.warnings'


if __name__ == '__main__':
    main()
