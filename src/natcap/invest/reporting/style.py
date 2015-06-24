"""A raster styling/visualizing module"""
import os
import numpy as np
# commenting out so we dont have a PIL dependency
# from PIL import Image
import logging
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from shapely.wkb import loads
from matplotlib import pyplot

import pygeoprocessing.geoprocessing

logging.basicConfig(format='%(asctime)s %(name)-18s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.reporting.style')

def grayscale_raster(raster_in_uri, raster_out_uri):
    """Create a grayscale image from 'raster_in_uri' by using linear
        interpolation to transform the float values to byte values
        between 0 and 256.

        raster_in_uri - a URI to a gdal raster

        raster_out_uri - a URI to a location on disk to save the output
            gdal raster

        returns - nothing
    """

    # If the output URI already exists, remove it
    if os.path.isfile(raster_out_uri):
        os.remove(raster_out_uri)

    # Max and Min value for the grayscaling
    gray_min = 0
    gray_max = 254

    # Make sure raster stats have been calculated
    pygeoprocessing.geoprocessing.calculate_raster_stats_uri(raster_in_uri)
    # Get the raster statistics, looking for Min and Max specifcally
    stats = pygeoprocessing.geoprocessing.get_statistics_from_uri(raster_in_uri)
    # Get Min, Max values from raster
    raster_min = stats[0]
    raster_max = stats[1]

    LOGGER.debug('Min:Max : %s:%s', raster_min, raster_max)

    # Set the x ranges to interpolate from
    x_range = [raster_min, raster_max]
    # Set the y ranges to interpolate to
    y_range = [gray_min, gray_max]

    # Get the pixel size of the input raster to use as the output
    # cell size
    pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(raster_in_uri)
    nodata_in = pygeoprocessing.geoprocessing.get_nodata_from_uri(raster_in_uri)
    out_nodata = 255

    def to_gray(pix):
        """Vectorize function that does a 1d interpolation
            from floating point values to grayscale values"""
        if pix == nodata_in:
            return out_nodata
        else:
            return int(np.interp(pix, x_range, y_range))

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [raster_in_uri], to_gray, raster_out_uri, gdal.GDT_Byte, 255,
        pixel_size, 'intersection')

def tif_to_png(tiff_uri, png_uri):
    """Save a tif of type BYTE as a png file

       raster_in_uri - a URI to a gdal raster of type BYTE

       raster_out_uri - a URI to a location on disk to save the output
           png image

       returns - nothing
    """

    raise NotImplementedError
    # If the output URI already exists, remove it
    if os.path.isfile(png_uri):
        os.remove(png_uri)

    #commenting this out so we don't have a dependancy on PIL
    #img = Image.open(tiff_uri)
    img.save(png_uri, 'PNG')

def create_thumbnail(image_in_uri, thumbnail_out_uri, size):
    """Generates a thumbnail image as a PNG file given of size 'size'

        image_in_uri - a URI to an image file

        thumbnail_out_uri - a URI to a location on disk for the output
            thumbnail image (png image)

        size - a tuple of integers with the dimensions of the thumbnail

        returns - nothing"""

    raise NotImplementedError
    # If the output URI already exists, remove it
    if os.path.isfile(thumbnail_out_uri):
        os.remove(thumbnail_out_uri)

    #commenting this out so we don't have a dependancy on PIL
    #img = Image.open(image_in_uri)
    img.thumbnail(size)
    img.save(thumbnail_out_uri, 'PNG')

#def shape_to_svg(shape_in_uri, image_out_uri, css_uri, args):
#   """Create a svg file from and OGR shapefile

#       shape_in_uri - a URI to the OGR shapefile to convert to an svg
#       image_out_uri - a URI path to save SVG to disk
#       css_uri - a URI to a CSS file for styling
#       args - a Dictionary with the following parameters used in creating the svg configuration:
#           size - a Tuple for width, height in pixels
#           field_id - a String for an attribute in 'source_uri' for the
#               unique field for the shapefile
#           key_id - the unique field for the shapefile
#           proj_type - a String for how the image projection should be interpreted

#       returns - Nothing
#   """

#   if os.path.isfile(image_out_uri):
#       os.remove(image_out_uri)
#   base_dir = os.path.dirname(image_out_uri)

#   # Copy the datasource to make some preprocessing adjustments
#   shape_copy_uri = os.path.join(base_dir, 'tmp_shp_copy.shp')

#   def remove_shapefile(shape_uri):
#       drv = ogr.GetDriverByName("ESRI Shapefile")
#       drv.DeleteDataSource(shape_uri)
#       drv = None

#   def convert_ogr_fields_to_strings(orig_shape_uri, shape_copy_uri):
#       """Converts an OGR Shapefile's fields to String values by
#           creating a new OGR Shapefile, building it from the
#           originals definitions

#           orig_shape_uri -

#           shape_copy_uri -

#           returns - nothing"""

#       orig_shape = ogr.Open(orig_shape_uri)
#       orig_layer = orig_shape.GetLayer()

#       if os.path.isfile(shape_copy_uri):
#           remove_shapefile(shape_copy_uri)

#       out_driver = ogr.GetDriverByName('ESRI Shapefile')
#       out_ds = out_driver.CreateDataSource(shape_copy_uri)
#       orig_layer_dfn = orig_layer.GetLayerDefn()
#       out_layer = out_ds.CreateLayer(
#           orig_layer_dfn.GetName(), orig_layer.GetSpatialRef(),
#           orig_layer_dfn.GetGeomType())

#       orig_field_count = orig_layer_dfn.GetFieldCount()

#       for fld_index in range(orig_field_count):
#           orig_field = orig_layer_dfn.GetFieldDefn(fld_index)
#           out_field = ogr.FieldDefn(orig_field.GetName(), ogr.OFTString)
#           out_layer.CreateField(out_field)

#       for orig_feat in orig_layer:
#           out_feat = ogr.Feature(feature_def = out_layer.GetLayerDefn())
#           geom = orig_feat.GetGeometryRef()
#           out_feat.SetGeometry(geom)

#           for fld_index in range(orig_field_count):
#               field = orig_feat.GetField(fld_index)
#               out_feat.SetField(fld_index, str(field))

#           out_layer.CreateFeature(out_feat)
#           out_feat = None

#   convert_ogr_fields_to_strings(shape_in_uri, shape_copy_uri)

#   aoi_sr = pygeoprocessing.geoprocessing.get_spatial_ref_uri(shape_copy_uri)
#   aoi_wkt = aoi_sr.ExportToWkt()

#   wkt_file = open('../test/invest-data/test/data/style_data/wkt_file.txt', 'wb')
#   wkt_file.write(aoi_wkt)
#   wkt_file.close()

#   # Get the Well Known Text of the shapefile
#   wgs84_sr = osr.SpatialReference()
#   wgs84_sr.SetWellKnownGeogCS("WGS84")
#   wgs84_wkt = wgs84_sr.ExportToWkt()

#   # NOTE: I think that kartograph is supposed to do the projection
#   # adjustment on the fly but it does not seem to be working for
#   # me.

#   # Reproject the AOI to the spatial reference of the shapefile so that the
#   # AOI can be used to clip the shapefile properly
#   tmp_uri = os.path.join(base_dir, 'tmp_shp_proj.shp')
#   pygeoprocessing.geoprocessing.reproject_datasource_uri(
#           shape_copy_uri, wgs84_wkt, tmp_uri)

#   css = open(css_uri).read()

#   kart = Kartograph()

#   config = {"layers":
#               {"mylayer":
#                   {"src":tmp_uri,
#                    "simplify": 1,
#                    "labeling": {"key": args['field_id']},
#                    "attributes":[args['key_id']]}
#                },
#             "proj":{"id": args['proj_type']},
#             "export":{"width":args['size'][0], "height": args['size'][1]}
#             }

#   kart.generate(config, outfile=image_out_uri, stylesheet=css)

#   remove_shapefile(shape_copy_uri)
#   remove_shapefile(tmp_uri)


