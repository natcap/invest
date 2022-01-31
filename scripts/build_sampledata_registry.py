import json
import os
import pprint
import sys

REGISTRY_PATH = os.path.join(
    os.path.dirname(__file__),
    '../src/renderer/sampledata_registry.json')

if __name__ == '__main__':
    storage_url = sys.argv[1]
    zip_dir = sys.argv[2]
    print(zip_dir)

    with open(REGISTRY_PATH, 'r') as json_file:
        registry = json.load(json_file)
        for model, data in registry.items():
            zipname = data['filename']
            registry[model]['filesize'] = os.stat(
                os.path.join(zip_dir, zipname)).st_size
            registry[model]['url'] = f'{storage_url}/{zipname}'

    with open(REGISTRY_PATH, 'w') as json_file:
        json_file.write(json.dumps(registry, indent=2))
