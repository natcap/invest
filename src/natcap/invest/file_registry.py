import collections
import os.path
import re


class FileRegistry:
    """
    The FileRegistry creates and tracks absolute paths that correspond to
    model outputs defined in a ModelSpec. Instantiate a FileRegistry from a
    list of spec.Outputs, a target directory, and a file suffix:

    ``file_registry = FileRegistry(MODEL_SPEC.outputs, workspace_dir, file_suffix)``

    The file registry is responsible for creating the output filepaths defined
    in the ModelSpec, in their real locations within the workspace directory,
    and adding the file suffix if one is provided.

    The primary way of interacting with a FileRegistry is by indexing. You can
    index the FileRegistry by output IDs. For output IDs that represent a
    pattern (that use square brackets to represent a section of the path that
    is dynamically substituted), pass in the value(s) to substitute to the
    index operator. For example,

    ``file_registry['aligned_dem']`` returns the equivalent of
    ``os.path.join(workspace_dir, 'aligned_dem_{file_suffix}.tif')``

    ``file_registry['[CROP]_[PERCENTILE]_coarse_yield', 'corn', '25th']``
    returns the equivalent of
    ``os.path.join(workspace_dir, f'intermediate_outputs/corn_25th_coarse_yield_{file_suffix}.tif')``

    The FileRegistry tracks which paths have been accessed in its ``registry``
    attribute, which is a nested dictionary mapping keys to absolute paths.
    For example, after performing the indexing examples above,
    ``file_registry.registry`` would be:

    .. code-block::

        {
            'aligned_dem': '/.../workspace_dir/aligned_dem_suffix.tif',
            '[CROP]_[PERCENTILE]_coarse_yield': {
                'corn': {
                    'yield_25th': '/.../workspace_dir/intermediate_outputs/corn_yield_25th_coarse_yield_suffix.tif''
                }
            }
        }

    """

    def __init__(self, outputs, workspace_dir, file_suffix=None):
        self.registry = {}
        self._keys_to_paths = {}
        self._pattern_fields = {}

        for output in outputs:
            path, extension = os.path.splitext(output.path)
            # Distinguish between paths that are/aren't patterns
            if re.match(r'(.*)\[(\w+)\](.*)', path):
                self._pattern_fields[output.id] = [
                    field.lower() for field in re.findall(r'\[(\w+)\]', output.id)]

            full_path = os.path.abspath(os.path.join(
                workspace_dir, path + (file_suffix or '') + extension))
            # Check for duplicate keys or paths
            if full_path in self._keys_to_paths.values():
                raise ValueError(f'Duplicate path: {full_path}')
            elif output.id in self._keys_to_paths:
                raise ValueError(f'Duplicate id: {output.id}')

            self._keys_to_paths[output.id] = full_path

    def __getitem__(self, keys):
        """Return the result of indexing the FileRegistry.

        For outputs that represent a single file (not a pattern), index by
        the output ID string. For outputs that represent a file pattern (those
        that contain one or more variables in square brackets), index by the
        output ID string followed by the value for each of the variables. For
        instance, an output with ID '[FOO]_result_[BAR]' should be accessed like
        ``file_registry['[FOO]_result_[BAR]', 'a', 'b']``. This will return the
        absolute path for '[FOO]_result_[BAR]', where '[FOO]' is replaced with 'a'
        and '[BAR]' is replaced with 'b'.

        Args:
            keys (str | tuple(obj)): key(s) to index the file registry by. Must
                be castable to string.

        Returns:
            absolute path (string) for the given key(s)

        """
        if isinstance(keys, str):
            keys = (keys,)
        key, *field_values = keys
        field_values = [str(value) for value in field_values]
        if key not in self._keys_to_paths:
            raise KeyError(f'Key not found: {key}')

        path = self._keys_to_paths[key]
        if key in self._pattern_fields:
            fields = self._pattern_fields[key]
            if len(field_values) != len(fields):
                raise KeyError(
                    f'Expected exactly {len(fields)} field values but received {len(field_values)}')

            for field, val in zip(fields, field_values):
                path = path.replace(f'[{field.upper()}]', val)

            if key not in self.registry:
                self.registry[key] = {}

            # Build nested entry. Last field_value will point to path.
            # (If only one field_value, it maps directly to path.)
            entry = path
            for i in range(len(field_values) - 1, -1, -1):
                entry = {field_values[i]: entry}

            self.registry[key].update(entry)

        else:
            if field_values:
                raise KeyError('Received field values for a key that has no fields')
            self.registry[key] = path
        return path
