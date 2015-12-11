"""InVEST specific code utils"""


def make_suffix_string(args, suffix_key):
    """Make an InVEST appropriate suffix string given the args dictionary and
    suffix key.  In general, prepends an '_' when necessary and generates an
    empty string when necessary.

    Parameters:
        args (dict): the classic InVEST model parameter dictionary that is
            passed to `execute`.
        suffix_key (string): the key used to index the base suffix.

    Returns:
        If `suffix_key` is not in `args`, or `args['suffix_key']` is ""
            return "",
        If `args['suffix_key']` starts with '_' return `args['suffix_key']`
            else return '_'+`args['suffix_key']`
        """

    try:
        file_suffix = args[suffix_key]
        if file_suffix != "" and not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''

    return file_suffix
