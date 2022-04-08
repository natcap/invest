import json
import os
import sys

if __name__ == '__main__':
    zip_dir = sys.argv[1]
    target_registry_path = os.path.join(zip_dir, 'registry.json')

    registry = {}
    for zipname in os.listdir(zip_dir):
        if zipname.endswith('.zip'):
            filesize = os.stat(os.path.join(zip_dir, zipname)).st_size
            registry[zipname] = filesize

    with open(target_registry_path, 'w') as json_file:
        json_file.write(json.dumps(registry, indent=2))
