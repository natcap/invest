import numpy
import os
import pygeoprocessing
from scipy import ndimage
import time

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

def identify_pour_points(flow_direction_raster_path, target_vector_path):
    """
    Create a pour point vector from D8 flow direction raster.

    A pour point is the center point of a pixel which:
        - flows off of the raster, or
        - flows into a nodata pixel

    Args:
        flow_direction_raster_path (string): path to the flow direction 
            raster to use. Values are defined as pointing to one of the 
                321
                4x0
                567
        target_vector_path (string): path to save pour point vector to.

    Returns:
        None
    """

    raster_info = pygeoprocessing.get_raster_info(flow_direction_raster_path)
    width, height = raster_info['raster_size']
    nodata = raster_info['nodata'][0]


    flow_direction_array = pygeoprocessing.raster_to_numpy_array(
                                           flow_direction_raster_path)

    # make a 1-pixel-wide nodata border around the flow direction data
    # this will make it easier to identify points flowing off the raster

    # t10 = time.perf_counter()
    # padded_flow_direction_array = numpy.full((height + 2, width + 2), nodata)
    # padded_flow_direction_array[1:-1, 1:-1] = flow_direction_array
    # t11 = time.perf_counter()

    # alternate

    padded_flow_direction_array = numpy.pad(
        flow_direction_array,
        1,
        constant_values=nodata
    )


    # flow direction: (delta rows, delta columns)
    # for a pixel at (i, j) with flow direction x,
    # the pixel it flows into is located at:
    # (i + directions[x][0], j + directions[x][1])
    directions = {
        0: (0, 1),
        1: (-1, 1),
        2: (-1, 0),
        3: (-1, -1),
        4: (0, -1),
        5: (1, -1),
        6: (1, 0),
        7: (1, 1)

    }

    def option_1(padded_flow_direction_array):
        # boolean array where 1 = pixel is a pour point
        # this will be converted to a point vector
        pour_point_array = numpy.empty((height, width))

        # iterate over each pixel
        for row in range(height):
            for col in range(width):
                flow_dir = padded_flow_direction_array[row + 1, col + 1]

                if flow_dir == nodata:
                    pour_point_array[row, col] = nodata

                else:
                    delta_x, delta_y = directions[flow_dir]
                    sink_value = padded_flow_direction_array[row + 1 + delta_x, col + 1 + delta_y]

                    if sink_value == nodata:
                        pour_point_array[row, col] = 1
                    else:
                        pour_point_array[row, col] = 0

        return pour_point_array




    def option_2(flow_direction_array):
        pour_point_array = numpy.empty((height, width))

        # iterate over each pixel
        for row in range(height):
            for col in range(width):
                # get the flow direction [0-7] for this pixel
                flow_dir = flow_direction_array[row, col]

                if flow_dir != nodata:
                    # get the location of the pixel it flows into
                    delta_x, delta_y = directions[flow_dir]
                    sink_row, sink_col = row + delta_x, col + delta_y

                    # if either index is <0, it's flowing off the edge of the raster
                    # if the sink pixel value is nodata, it's flowing into nodata
                    # either case means this is a pour point
                    if (sink_row < 0 or 
                        sink_col < 0 or 
                        flow_direction_array[sink_row, sink_col] == nodata):
                        pour_point_array[row, col] = 1
                    else:
                        pour_point_array[row, col] = 0

        return pour_point_array



    def option_3(flow_direction_array):

        def func(x):
            # 3 2 1       0 1 2 
            # 4 x 0  -->  3 4 5 
            # 5 6 7       6 7 8 
            convert = {
                0: 5,
                1: 2,
                2: 1,
                3: 0,
                4: 3,
                5: 6,
                6: 7,
                7: 8
            }
            flow_dir = x[4]
            if flow_dir == nodata:
                return nodata
            else:
                return x[convert[flow_dir]] == nodata

       
        footprint = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]

        pour_point_array = ndimage.generic_filter(
                                flow_direction_array, 
                                func, 
                                footprint=footprint, 
                                mode='constant', 
                                cval=nodata)

        return pour_point_array



    # opt_1_start = time.perf_counter()
    # option_1(padded_flow_direction_array)
    # opt_1_end = time.perf_counter()

    # opt_2_start = time.perf_counter()
    # option_2(flow_direction_array)
    # opt_2_end = time.perf_counter()

    opt_3_start = time.perf_counter()
    out_array = option_3(flow_direction_array)
    opt_3_end = time.perf_counter()

    # print('Option 1:', opt_1_end - opt_1_start)
    # print('Option 2:', opt_2_end - opt_2_start)
    # print('Option 3:', opt_3_end - opt_3_start)

    print(out_array)
    print(numpy.min(out_array), numpy.max(out_array))
    print(numpy.where(out_array == 1))
    pygeoprocessing.numpy_array_to_raster(
        out_array, 
        nodata, 
        raster_info['pixel_size'],
        (raster_info['geotransform'][0], raster_info['geotransform'][3]),
        raster_info['projection_wkt'],
        target_vector_path)



    # Save points to vector file

    # use same spatial reference as the input
    aoi_spatial_reference = osr.SpatialReference()
    aoi_spatial_reference.ImportFromWkt(raster_info['projection_wkt'])

    gpkg_driver = ogr.GetDriverByName("GPKG")
    target_vector = gpkg_driver.CreateDataSource(target_vector_path)
    layer_name = os.path.splitext(
        os.path.basename(target_vector_path))[0]
    target_layer = target_vector.CreateLayer(
        layer_name, aoi_spatial_reference, ogr.wkbPoint)
    target_defn = target_layer.GetLayerDefn()

    # It's important to have a user-facing unique ID field for post-processing
    # (e.g. table-joins) that is not the FID. FIDs are not stable across file
    # conversions that users might do. FIDs still used internally in this module.
    target_layer.CreateField(
        ogr.FieldDefn('point_id', ogr.OFTInteger64))

    ys, xs = numpy.where(out_array == 1)
    point_list = zip(ys, xs)

    target_layer.StartTransaction()
    for idx, (y, x) in enumerate(point_list):
        print(x, y)
        geometry = ogr.Geometry(ogr.wkbPoint)
        geometry.AddPoint(int(x), int(y))
        feature = ogr.Feature(target_defn)
        feature.SetGeometry(geometry)
        feature.SetField('point_id', idx)
        target_layer.CreateFeature(feature)
    target_layer.CommitTransaction()

    target_layer = None
    target_vector = None

if __name__ == '__main__':
    identify_pour_points('/Users/emily/invest/test_flow_direction.tif', '/Users/emily/invest/test_pour_points_output.gpkg')


            




