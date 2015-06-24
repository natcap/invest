This file contains referential information about the various resources used
by IUI.

To override any of these settings, create your own resources folder with this
structure:
  resources_dir/
    resources.json
    -- any files and folders referred to in resources.json --

Then, define your overrides in the resources.json file.  All paths in
resources.json are required to be relative to the location of resources.json,
so it is most convenient to store all your resources in the resources
directory.
