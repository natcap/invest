import os.path
import re

class FileRegistryItem:

    def __init__(self, key, path):
        self.key = key
        self.path = path

class FileRegistryGroup:

    def __init__(self, key, path_pattern):

        match = re.match(r'(.*)\[([A-Z]+)\](.*)', path_pattern)
        if not match:
            raise ValueError('Path pattern does not match the expected format')

        self.field = match.group(2).lower()
        self.fstring = match.group(1) + '{' + self.field + '}' + match.group(3)

        self.items = {}

    def create_item(self, **kwargs):

        if set(kwargs) != {self.field}:
            raise ValueError(f'Expected exactly one kwarg: {self.field}')
        field_value = kwargs[self.field]
        item = FileRegistryItem(
            field_value, self.fstring.format(**kwargs))
        self.items[field_value] = item
        return item

    @property
    def path_list(self):
        return [item.path for item in self.items.values()]


class FileRegistry:

    def __init__(self, model_spec, workspace_dir, file_suffix):
        self.model_spec = model_spec
        self.file_suffix = file_suffix

        all_paths = set()
        self.registry = {}

        for output in model_spec.outputs:

            path, extension = os.path.splitext(output.path)
            full_path = os.path.join(
                workspace_dir, path + file_suffix + extension)

            # Check for duplicate keys or paths
            if full_path in all_paths:
                raise ValueError(f'Duplicate path: {full_path}')
            elif output.id in self.registry:
                raise ValueError(f'Duplicate id: {output.id}')

            all_paths.add(full_path)


            match = re.match(r'(.*)\[([A-Z]+)\](.*)', full_path)
            if match:
                self.registry[output.id] = FileRegistryGroup(output.id, full_path)
            else:
                self.registry[output.id] = FileRegistryItem(output.id, full_path)

    def get(self, key, **kwargs):
        if key not in self.registry:
            raise ValueError(f'Key not found in registry: {key}')

        if isinstance(self.registry[key], FileRegistryGroup):
            sub_key = list(kwargs.values)[0]
            return self.registry[key].items[sub_key]
        else:
            return self.registry[key]

    def get_path(self, key, **kwargs):
        return self.get(key, **kwargs).path

    def create_item(self, key, **kwargs):
        return self.registry[key].create_item(**kwargs)

    def get_group_path_list(self, key):
        return self.registry[key].path_list

    def as_dict(self):
        d = {}
        for entry in self.registry.values():
            if isinstance(entry, FileRegistryGroup):
                d[entry.key] = {}
                for item in entry.items:
                    d[entry.key][item.key] = item.path
            else:
                d[entry.key] = entry.path
        return d


