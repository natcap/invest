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

    # opt_3_start = time.perf_counter()
    # out_array = option_3(flow_direction_array)
    # opt_3_end = time.perf_counter()

    # print('Option 1:', opt_1_end - opt_1_start)
    # print('Option 2:', opt_2_end - opt_2_start)
    # print('Option 3:', opt_3_end - opt_3_start)

    # print(out_array)
    # print(numpy.min(out_array), numpy.max(out_array))
    # print(numpy.where(out_array == 1))
    # pygeoprocessing.numpy_array_to_raster(
    #     out_array, 
    #     nodata, 
    #     raster_info['pixel_size'],
    #     (raster_info['geotransform'][0], raster_info['geotransform'][3]),
    #     raster_info['projection_wkt'],
    #     target_vector_path)


    def write_to_point_vector(pour_point_set):
        """Save a list of points to a point vector.
        
        Args:
            pour_points_set (set): set of (x, y) tuples representing point
                coordinates, where the upper-left pixel of the raster is at
                (0, 0), and each pixel is 1x1.

        Returns:
            None

        """
        # use same spatial reference as the input
        aoi_spatial_reference = osr.SpatialReference()
        aoi_spatial_reference.ImportFromWkt(raster_info['projection_wkt'])

        # Will need origin and pixel size to convert each point to the
        # same spatial reference
        x_origin, y_origin = raster_info['geotransform'][0], raster_info['geotransform'][3]
        x_pixel_size, y_pixel_size = raster_info['pixel_size']

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

        target_layer.StartTransaction()
        for idx, (x, y) in enumerate(pour_point_set):
            # print(x, y)
            
            # Convert x (which is spatially referenced to the raster as a 
            # numpy array) to the correct location in the output spatial 
            # reference. (x + 0.5) * x_pixel_size so that the point is 
            # centered on the pixel.
            x_coord = x_origin + (x + 0.5) * x_pixel_size
            y_coord = y_origin + (y + 0.5) * y_pixel_size

            geometry = ogr.Geometry(ogr.wkbPoint)
            geometry.AddPoint(int(x), int(y))
            feature = ogr.Feature(target_defn)
            feature.SetGeometry(geometry)
            feature.SetField('point_id', idx)
            target_layer.CreateFeature(feature)
        target_layer.CommitTransaction()

        target_layer = None
        target_vector = None


    def find_pour_points_by_block(flow_direction_raster_path):

        pour_point_array = numpy.empty((height, width))
        pour_point_set = set()

        for block in pygeoprocessing.iterblocks((flow_direction_raster_path, 1),
                                                offset_only=True):
            print(block)
            # Expand each block towards the upper left by a one-pixel-wide strip,
            # if possible. This way the blocks will overlap so the watershed 
            # calculation will be continuous.
            if block['xoff'] > 0:
                block['xoff'] -= 1
                block['win_xsize'] += 1
            if block['yoff'] > 0:
                block['yoff'] -= 1
                block['win_ysize'] += 1

            # Read in the block frome the flow direction raster
            in_raster = gdal.OpenEx(flow_direction_raster_path, gdal.OF_RASTER)
            if in_raster is None:
                raise ValueError(
                    "Raster at %s could not be opened." % flow_direction_raster_path)
            in_band = in_raster.GetRasterBand(1)

            # Calculate pour point
            flow_dir_block = in_band.ReadAsArray(**block)
            pour_point_block = option_3(flow_dir_block)

            in_raster = None
            in_band = None

            # Add any pour points found in this block as (x, y) pairs
            # Use a set so that any duplicates in the overlap areas
            # won't be double-counted
            ys, xs = numpy.where(pour_point_block == 1)
            # Add the offsets so that all points reference the same coordinate
            # system (such that the upper-left corner of the raster is (0, 0),
            # and each pixel is 1x1).
            ys += block['yoff']
            xs += block['xoff']
            print(set(zip(xs, ys)))
            pour_point_set = pour_point_set.union(set(zip(xs, ys)))

        return pour_point_set


    write_to_point_vector(find_pour_points_by_block(flow_direction_raster_path))






if __name__ == '__main__':
    identify_pour_points('/Users/emily/invest/test_flow_direction.tif', '/Users/emily/invest/test_pour_points_output.gpkg')


            




