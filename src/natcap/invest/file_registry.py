import collections
import os.path
import re


class FileRegistry:
    """
    The FileRegistry creates and tracks absolute paths that correspond to
    model outputs defined in a ModelSpec. Instantiate a FileRegistry from a
    ModelSpec, a target directory, and a file suffix:

    ``file_registry = FileRegistry(MODEL_SPEC, workspace_dir, file_suffix)``

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

    ```
    {
        'aligned_dem': '/.../workspace_dir/aligned_dem_suffix.tif',
        '[CROP]_[PERCENTILE]_coarse_yield': {
            ('corn', '25th'): '/.../workspace_dir/intermediate_outputs/corn_25th_coarse_yield_suffix.tif''
        }
    }
    ```
    """

    def __init__(self, model_spec, workspace_dir, file_suffix):
        self.model_spec = model_spec
        self.file_suffix = file_suffix

        self.registry = collections.defaultdict(dict)
        self.keys_to_paths = {}
        self.pattern_fields = {}

        for output in model_spec.outputs:

            path, extension = os.path.splitext(output.path)

            # Distinguish between paths that are/aren't patterns
            if re.match(r'(.*)\[(\w+)\](.*)', path):
                self.pattern_fields[output.id] = [
                    field.lower() for field in re.findall(r'\[(\w+)\]', path)]

            full_path = os.path.join(
                workspace_dir, path + file_suffix + extension)

            # Check for duplicate keys or paths
            if full_path in self.keys_to_paths.values():
                raise ValueError(f'Duplicate path: {full_path}')
            elif output.id in self.keys_to_paths:
                raise ValueError(f'Duplicate id: {output.id}')

            self.keys_to_paths[output.id] = full_path


    def __getitem__(self, keys):
        """Return the result of indexing the FileRegistry.

        Args:
            keys (str | tuple(str)): key(s) to index the file registry by.

        Returns:
            absolute path (string) for the given key(s)


        """
        key = keys[0] if isinstance(keys, tuple) else keys
        if key not in self.keys_to_paths:
            raise ValueError(f'Key not found: {key}')

        path = self.keys_to_paths[key]
        if key in self.pattern_fields:
            field_values = keys[1:]
            fields = self.pattern_fields[key]
            if len(field_values) != len(fields):
                raise ValueError(
                    f'Expected exactly {len(fields)} field values but received {len(field_values)}')

            for field, val in zip(fields, field_values):
                path = path.replace(f'[{field.upper()}]', str(val))

            sub_key = tuple(field_values) if len(field_values) > 1 else field_values[0]
            self.registry[key][sub_key] = path
        else:
            self.registry[key] = path
        return path
