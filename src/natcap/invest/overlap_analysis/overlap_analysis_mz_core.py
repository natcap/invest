'''
This is the core module for the management zone analysis portion of the
Overlap Analysis model.
'''

import os
import logging

from osgeo import ogr

LOGGER = logging.getLogger('natcap.invest.overlap_analysis.mz_core')
logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')


def execute(args):
    '''This is the core module for the management zone model, which was
    extracted from the overlap analysis model. This particular one will take
    in a shapefile conatining a series of AOI's, and a folder containing
    activity layers, and will return a modified shapefile of AOI's, each of
    which will have an attribute stating how many activities take place within
    that polygon.

    Input:
        args['workspace_dir']- The folder location into which we can write an
            Output or Intermediate folder as necessary, and where the final
            shapefile will be placed.
        args['zone_layer_file']- An open shapefile which contains our
            management zone polygons. It should be noted that this should not
            be edited directly but instead, should have a copy made in order
            to add the attribute field.
        args['over_layer_dict'] - A dictionary which maps the name of the
            shapefile (excluding the .shp extension) to the open datasource
            itself. These files are each an activity layer that will be counted
            within the totals per management zone.

    Output:
        A file named [workspace_dir]/Ouput/mz_frequency.shp which is a copy of
        args['zone_layer_file'] with the added attribute "ACTIV_CNT" that will
        total the number of activities taking place in each polygon.

     Returns nothing.'''

    output_dir = os.path.join(args['workspace_dir'], 'output')

    #Want to run through all polygons in the AOI, and see if any intersect or
    #contain all shapefiles from all other layers. Little bit gnarly in terms
    #of runtime, but at least doable.

    driver = ogr.GetDriverByName('ESRI Shapefile')
    zone_shape_old = args['zone_layer_file']
    layers_dict = args['over_layer_dict']

    path = os.path.join(output_dir, 'mz_frequency.shp')
    if os.path.isfile(path):
        os.remove(path)

    #This creates a new shapefile that is a copy of the old one, but at the path
    #location. That way we can edit without worrying about changing the Input
    #file.
    mz_freq_shape = driver.CopyDataSource(zone_shape_old, path)

    mz_freq_layer = mz_freq_shape.GetLayer()
    LOGGER.debug(mz_freq_layer)

    #Creating a definition for our new activity count field.
    field_defn = ogr.FieldDefn('ACTIV_CNT', ogr.OFTReal)
    mz_freq_layer.CreateField(field_defn)

    #This will loop through all management zone polygons, as defined by the MZ
    #input file. For each of those polygons, it will look through the list of
    #activity layers, and check against each shape on every one of those layers.
    for mz_polygon in mz_freq_layer:

        zone_geom = mz_polygon.GetGeometryRef()
        activity_count = 0

        for activ in layers_dict:

            shape_file = layers_dict[activ]
            activ_layer = shape_file.GetLayer()

            for feature in activ_layer:
                #If it contains or overlaps
                activ_geom = feature.GetGeometryRef()

                if zone_geom.Contains(activ_geom) or zone_geom.Overlaps(activ_geom):
                    activity_count += 1
                    break

            activ_layer.ResetReading()

        mz_polygon.SetField('ACTIV_CNT', activity_count)

        mz_freq_layer.SetFeature(mz_polygon)
