def require(key, args, warnings_):
    """Append an error to warnings_ when a required key is missing.

    A warning is recorded about a required key when either of these conditions
    are met:

        * The key is missing from ``args``.
        * The key's value is one of ``''`` or ``None``.

    Parameters:
        key (string): The string key to index into args.
        args (dict): The full args dict.
        warnings_ (list): the warnings list.

    Returns:
        ``None``
    """
    try:
        if args[key] in ('', None):
            raise KeyError
    except KeyError:
        warnings_.append((key, 'Key is required'))
