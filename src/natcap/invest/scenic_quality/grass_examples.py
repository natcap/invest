"""
GRASS Python script examples.
"""
import sys
import os

import shutil

import random, string

def random_string(length):
   return ''.join(random.choice(string.lowercase) for i in range(length))

#add the path to GRASS
sys.path.append('/usr/lib/grass64/etc/python')

import grass.script
import grass.script.setup

class grasswrapper():
    def __init__(self,
                 dbBase='',
                 location='',
                 mapset=''):

        self.gisbase = os.environ['GISBASE']
        self.gisdb = dbBase
        self.loc = location
        self.mapset = mapset
        grass.script.setup.init(self.gisbase,
                                self.gisdb,
                                self.loc,
                                self.mapset)

if __name__ == '__main__':
    #get raster path from first parameter
    dataset_uri = sys.argv[1]

    #get coordinate from second and third parameter
    coordinate = [sys.argv[2], sys.argv[3]]

    #get user's home directory
    home = os.path.expanduser("~")

    random_input_name = random_string(6)
    random_output_name = random_string(6)

    #determine GRASS database path
    temp_uri = grass.script.parse_command('g.tempfile',
                                          pid = 1).keys()[0]

    database_uri = temp_uri.split('.tmp')[0]
    location_uri = os.path.join(database_uri, random_input_name)

    #create location with raster's reference system
    print 'Creating location %s with raster\'s reference system' % location_uri
    grass.script.run_command('g.proj',
                             'c',
                             georef = dataset_uri,
                             location = random_input_name)

    #import raster into GRASS
    print 'Importing raster from ', dataset_uri
    grass.script.run_command('r.in.gdal',
                             input = dataset_uri,
                             output = random_input_name)

    #calculate viewshed
    print 'Calculating viewshed'
    grass.script.run_command('r.viewshed',
                             'c',
                             'b',
                             input = random_input_name,
                             output = random_output_name,
                             coordinate = coordinate)

    #export raster from GRASS
    dataset_out_uri = os.path.join(home,
                                   "%s.tif" % random_output_name)
    print 'Saving raster to ', dataset_out_uri
    grass.script.run_command('r.out.tiff',
                             input = 'dem',
                             output = dataset_out_uri)

    #remove imported raster from GRASS
    print 'Removing imported raster.'
    grass.script.run_command('g.remove',
                             rast = random_input_name)

    #remove viewshed
    print 'Removing viewshed.'
    grass.script.run_command('g.remove',
                             rast = random_output_name)

    #remove the location from disk
    print 'Removing location %s' % location_uri
    shutil.rmtree(location_uri)
