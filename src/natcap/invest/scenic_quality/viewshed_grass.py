import sys, os
from osgeo import ogr

sys.path.append("/usr/lib/grass64/etc/python")

import grass.script
import grass.script.setup

import logging

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.scenic_quality.viewshed_grass')

class grasswrapper():
    def __init__(self,
                 dbBase="",
                 location="/home/mlacayo/workspace/newLocation",
                 mapset="PERMANENT"):
        '''
        Wrapper of "python.setup.init" defined in GRASS python.
        Initialize system variables to run scripts without starting GRASS explicitly.

        @param dbBase: path to GRASS database (default: '').
        @param location: location name (default: '').
        @param mapset: mapset within given location (default: 'PERMANENT')

        @return: Path to gisrc file.
        '''
        self.gisbase = os.environ['GISBASE']
        self.gisdb = dbBase
        self.loc = location
        self.mapset = mapset
        grass.script.setup.init(self.gisbase, self.gisdb, self.loc, self.mapset)

def execute(args):
    os.putenv('GIS_LOCK', 'default')

    project_setup(args["in_raster"])

    #preprocess data

    #calculate viewshed
    viewshed(args["in_raster"],
             args["in_observer_features"],
             os.path.join(args["workspace_dir"],"viewshed.tif"))

    project_cleanup()

def project_setup(dataset_uri):
    #this might not be necessary depending on how the InVEST installer is configured
    LOGGER.debug("Creating location.")
    grass.script.run_command('g.proj',
                             'c',
                             georef = dataset_uri,
                             location = 'invest')

    LOGGER.debug("Changing location.")
    grass.script.run_command('g.mapset',
                             mapset = 'PERMANENT',
                             location = 'invest')

##    LOGGER.debug("Adding mapset.")
##    grass.script.run_command('g.mapsets',
##                             addmapset = 'invest')

def project_cleanup():
    LOGGER.debug("Removing raster mapsets.")
    #this might delete source files
    grass.script.run_command('g.mremove',
                             'f',
                             rast = '*')

def viewshed(dataset_uri, feature_set_uri, dataset_out_uri):
    LOGGER.debug("Registering raster with GRASS.")
    grass.script.run_command('r.in.gdal',
                             'o', #overide projection
                             input=dataset_uri,
                             output='dem')

    LOGGER.debug("g.list: %s",
                 str(grass.script.parse_command("g.list", _type="vect")))

    LOGGER.debug("Computing viewshed for each point.")
    shapefile = ogr.Open(feature_set_uri)
    layer = shapefile.GetLayer()
    feature_count = layer.GetFeatureCount()
    for feature_id in xrange(feature_count):
        feature = layer.GetFeature(feature_id)
        geom = feature.GetGeometryRef()
        x, y, _ = geom.GetPoint()
        coordinate = [x, y]

        LOGGER.debug("Creating viewshed for feature %i.", feature_id)

        grass.script.run_command('r.viewshed',
                                 'c', #earth curvature
                                 'b', #boolean viewshed
                                 input='dem',
                                 output='feature_%i' % feature_id,
                                 coordinate=coordinate,
##                                 obs_elev=1.75,
##                                 tgt_elev=0.0,
##                                 memory=4098,
##                                 overwrite=True,
                                 quiet=True)

    shapefile = None

    LOGGER.debug("Summing viewsheds.")
    grass.script.run_command('r.mapcalc',
                             'viewshed = ' + ' + '.join(
                                 ['feature_%i' % i for i in xrange(
                                     feature_count)]))

    LOGGER.debug("Exporting viewshed.")
    grass.script.run_command('r.out.tiff',
                             input='viewshed',
                             output=dataset_out_uri)

if __name__ == "__main__":

    execute()
