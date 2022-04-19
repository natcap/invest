from osgeo import gdal
from pygeoprocessing import get_raster_info, DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS

dem_path = '/Users/emily/invest-test-data/sdr/input/dem.tif'
lulc_path = '/Users/emily/invest-test-data/sdr/input/lulc.tif'
aligned_dem_path = '/Users/emily/Documents/aligned_dem.tif'
aligned_lulc_path = '/Users/emily/Documents/aligned_lulc.tif'

dem_raster_info = pygeoprocessing.get_raster_info(dem_path)
lulc_raster_info = get_raster_info(lulc_path)
min_pixel_size = numpy.min(numpy.abs(dem_raster_info['pixel_size']))

pygeoprocessing.align_and_resize_raster_stack(
    [dem_path, lulc_path],
    [aligned_dem_path, aligned_lulc_path],
    [bilinear, mode],
    (min_pixel_size, -min_pixel_size))

target_bounding_box = merge_bounding_box_list(
    [dem_raster_info['bounding_box'], lulc_raster_info['bounding_box']],
    'intersection')

# bounding box needs alignment
align_bounding_box = dem_raster_info['bounding_box']
align_pixel_size = dem_raster_info['pixel_size']

# adjust bounding box so lower left corner aligns with a pixel in
# raster[0]
for index in [0, 1]:
    n_pixels = int(
        (target_bounding_box[index] - align_bounding_box[index]) /
        float(align_pixel_size[index]))
    target_bounding_box[index] = (
        n_pixels * align_pixel_size[index] +
        align_bounding_box[index])

# ensure it's a sequence so we can modify it
working_bb = list(target_bounding_box)

# determine the raster size that bounds the input bounding box and then
# adjust the bounding box to be that size
target_x_size = int(abs(
    float(working_bb[2] - working_bb[0]) / target_pixel_size[0]))
target_y_size = int(abs(
    float(working_bb[3] - working_bb[1]) / target_pixel_size[1]))

# sometimes bounding boxes are numerically perfect, this checks for that
x_residual = (
    abs(target_x_size * target_pixel_size[0]) -
    (working_bb[2] - working_bb[0]))
if not numpy.isclose(x_residual, 0.0):
    target_x_size += 1
y_residual = (
    abs(target_y_size * target_pixel_size[1]) -
    (working_bb[3] - working_bb[1]))
if not numpy.isclose(y_residual, 0.0):
    target_y_size += 1

if target_x_size == 0:
    target_x_size = 1
if target_y_size == 0:
    target_y_size = 1

# this ensures the bounding boxes perfectly fit a multiple of the target
# pixel size
working_bb[2] = working_bb[0] + abs(target_pixel_size[0] * target_x_size)
working_bb[3] = working_bb[1] + abs(target_pixel_size[1] * target_y_size)

warped_raster_path = target_lulc_path
base_raster = gdal.OpenEx(lulc_path, gdal.OF_RASTER)

raster_creation_options = list(DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS[1])
if (lulc_raster_info['numpy_type'] == numpy.int8 and
        'PIXELTYPE' not in ' '.join(raster_creation_options)):
    raster_creation_options.append('PIXELTYPE=SIGNEDBYTE')

gdal.Warp(
    warped_raster_path, base_raster,
    format=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS[0],
    outputBounds=working_bb,
    xRes=abs(target_pixel_size[0]),
    yRes=abs(target_pixel_size[1]),
    resampleAlg='mode',
    outputBoundsSRS=dem_raster_info['projection_wkt'],
    srcSRS=None,
    dstSRS=dem_raster_info['projection_wkt'],
    multithread=False,
    warpOptions=[],
    creationOptions=raster_creation_options,
    callback=reproject_callback,
    callback_data=[target_lulc_path])

