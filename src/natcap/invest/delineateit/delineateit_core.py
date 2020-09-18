import numpy
import os
import pygeoprocessing
from scipy import ndimage
import time

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

@profile
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

    # for this algorithm I'm using two different nodata values
    # because there are two, replace the original nodata value with TMP_NODATA
    # to be sure that the values are different.
    # the flow direction array gets padded with a border of TMP_NODATA, so that
    # 
    TMP_NODATA = 100
    FILL_VALUE = 101

    flow_direction_array = pygeoprocessing.raster_to_numpy_array(
                                           flow_direction_raster_path)


    @profile
    def calculate_pour_point_array(flow_direction_array):
        """
        Return a binary array indicating which pixels are pour points.

        Args:
            flow_direction_array (numpy.ndarray): a 2D array of D8 flow
                direction values (0 - 7) and possiby the nodata value.

        Returns:
            numpy.ndarray of the same shape as ``flow_direction_array``.
            Each element is 1 if that pixel is a pour point, 0 if not,
            nodata if it cannot be determined.
        """
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

        @profile
        def is_pour_point(kernel):
            """
            Determine if pixel is a pour point given its neighborhood.

            Args:
                kernel (list): list of 9 integers representing a 3x3 
                    kernel from a flow direction array:
                    0 1 2
                    3 4 5  -->  [0,1,2,3,4,5,6,7,8]
                    6 7 8
                    Used to determine if the center element, at index 4, is a
                    pour point based on the surrounding elements.

            Returns:
                integer: 1 if kernel[4] is a pour point, 0 if not, 
                    nodata if it cannot be determined.
            """

            # 3 2 1       0 1 2 
            # 4 x 0  -->  3 4 5 
            # 5 6 7       6 7 8 

            if FILL_VALUE in kernel:
                return nodata
            if kernel[4] == TMP_NODATA:
                return nodata
            else:
                return kernel[convert[kernel[4]]] == TMP_NODATA

        footprint = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]

        # 
        flow_direction_array[flow_direction_array == nodata] = TMP_NODATA

        pour_point_array = ndimage.generic_filter(
                                flow_direction_array, 
                                is_pour_point, 
                                footprint=footprint, 
                                mode='constant', 
                                cval=FILL_VALUE)

        return pour_point_array

    @profile
    def find_pour_points_by_block(flow_direction_raster_path):
        """
        Memory-safe pour point calculation from a flow direction raster.
        
        Args:
            flow_direction_raster_path (string): path to flow direction raster

        Returns:
            set of (x, y) coordinate tuples 
        """

        # Read in the block from the flow direction raster
        raster = gdal.OpenEx(flow_direction_raster_path, gdal.OF_RASTER)
        if raster is None:
            raise ValueError(
                "Raster at %s could not be opened." % flow_direction_raster_path)
        band = raster.GetRasterBand(1)

        pour_point_set = set()
        # horizontal_overlap = None
        # vertical_overlap = numpy.empty((2, 0))
        # for block, flow_dir_block in pygeoprocessing.iterblocks((flow_direction_raster_path, 1)):
        for block in pygeoprocessing.iterblocks((flow_direction_raster_path, 1),
                                                offset_only=True):
            # Expand each block by a one-pixel-wide margin, if possible. 
            # This way the blocks will overlap so the watershed 
            # calculation will be continuous.
            if block['xoff'] > 0:
                block['xoff'] -= 1
                block['win_xsize'] += 1
            if block['yoff'] > 0:
                block['yoff'] -= 1
                block['win_ysize'] += 1
            if block['xoff'] + block['win_xsize'] < width:
                block['win_xsize'] += 1
            if block['yoff'] + block['win_ysize'] < height:
                block['win_ysize'] += 1

            flow_dir_block = band.ReadAsArray(**block)

            # # this relies on the blocks going by rows, left -> right
            # if horizontal_overlap and block['xoff'] > 0:
            #     flow_dir_block = numpy.hstack(horizontal_overlap, flow_dir_block)
            # horizontal_overlap = flow_dir_block[:, -2:]

            # if vertical_overlap and block['yoff'] > 0:
            #     xmin = block['xoff']
            #     xmax = block['xoff'] + block['win_xsize']
            #     flow_dir_block = numpy.vstack(
            #         vertical_overlap[xmin:xmax + 1],
            #         flow_dir_block)
            # # keep track of 
            # vertical_overlap = numpy.hstack(vertical_overlap, flow_dir_block[-2:])


            # Add 1-pixel-wide nodata border relative to the whole raster
            if block['xoff'] == 0:
                border = numpy.full(flow_dir_block.shape[0], TMP_NODATA)
                flow_dir_block = numpy.column_stack([border, flow_dir_block])
            if block['yoff'] == 0:
                border = numpy.full(flow_dir_block.shape[1], TMP_NODATA)
                flow_dir_block = numpy.vstack([border, flow_dir_block])
            if block['xoff'] + block['win_xsize'] == width:
                border = numpy.full(flow_dir_block.shape[0], TMP_NODATA)
                flow_dir_block = numpy.column_stack([flow_dir_block, border])
            if block['yoff'] + block['win_ysize'] == height:
                border = numpy.full(flow_dir_block.shape[1], TMP_NODATA)
                flow_dir_block = numpy.vstack([flow_dir_block, border])

            # Calculate pour points
            pour_point_block = calculate_pour_point_array(flow_dir_block)

            # Add any pour points found in this block as (x, y) pairs
            # Use a set so that any duplicates in the overlap areas
            # won't be double-counted
            ys, xs = numpy.where(pour_point_block == 1)
            # Add the offsets so that all points reference the same coordinate
            # system (such that the upper-left corner of the raster is (0, 0),
            # and each pixel is 1x1).
            ys += block['yoff']
            xs += block['xoff']
            pour_point_set = pour_point_set.union(set(zip(xs, ys)))

        raster = None
        band = None

        print('pour points:', pour_point_set)
        # Return coordinates in the same coordinate system as the input raster
        origin = (raster_info['geotransform'][0], raster_info['geotransform'][3])
        return _convert_numpy_coords_to_geotransform(
            pour_point_set, origin, raster_info['pixel_size'])

    @profile
    def _convert_numpy_coords_to_geotransform(coords, origin, pixel_size):
        """
        Convert array index coordinates to a geotransform.

        Args:
            coords (set): set of (x, y) integer coordinate tuples. These are
                referenced to the entire raster as a numpy array: (0, 0) is the
                upper-left pixel, (raster_width, raster_height) is at
                the bottom-right.
            origin (tuple): (x, y) coordinate tuple of the location of the 
                upper-left corner of the raster in the desired geotransform.
            pixel_size (tuple): (x_size, y_size) tuple giving the dimensions of
                a pixel in the desired geotransform.

        Returns:
            set of (x, y) coordinate tuples that have been converted to the
                given geotransform's coordinate system.
        """
             
        transformed_coords = set()

        for x, y in coords:
            # +0.5 so that the point is centered in the pixel
            transformed_x = origin[0] + (x + 0.5) * pixel_size[0]
            transformed_y = origin[1] + (y + 0.5) * pixel_size[1]
            transformed_coords.add((transformed_x, transformed_y))
        return transformed_coords

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

    @profile
    def write_to_point_vector(pour_point_set):
        """Save a list of points to a point vector.
        
        Args:
            pour_point_set (set): set of (x, y) tuples representing
                point coordinates in the same coordinate system as 
                the flow direction raster.

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
        # conversions that users might do.
        target_layer.CreateField(
            ogr.FieldDefn('point_id', ogr.OFTInteger64))

        # Add a feature to the layer for each point
        target_layer.StartTransaction()
        for idx, (x, y) in enumerate(pour_point_set):
            geometry = ogr.Geometry(ogr.wkbPoint)
            geometry.AddPoint(x, y)
            feature = ogr.Feature(target_defn)
            feature.SetGeometry(geometry)
            feature.SetField('point_id', idx)
            target_layer.CreateFeature(feature)
        target_layer.CommitTransaction()

        target_layer = None
        target_vector = None

    pour_points = find_pour_points_by_block(flow_direction_raster_path)
    write_to_point_vector(pour_points)


if __name__ == '__main__':
    identify_pour_points('/Users/emily/invest/test_flow_direction.tif', '/Users/emily/invest/test_pour_points_output.gpkg')


            
# padded_flow_direction_array = numpy.pad(
#         flow_direction_array,
#         1,
#         constant_values=nodata
#     )


#     # flow direction: (delta rows, delta columns)
#     # for a pixel at (i, j) with flow direction x,
#     # the pixel it flows into is located at:
#     # (i + directions[x][0], j + directions[x][1])
#     directions = {
#         0: (0, 1),
#         1: (-1, 1),
#         2: (-1, 0),
#         3: (-1, -1),
#         4: (0, -1),
#         5: (1, -1),
#         6: (1, 0),
#         7: (1, 1)

#     }

#     def option_1(padded_flow_direction_array):
#         # boolean array where 1 = pixel is a pour point
#         # this will be converted to a point vector
#         pour_point_array = numpy.empty((height, width))

#         # iterate over each pixel
#         for row in range(height):
#             for col in range(width):
#                 flow_dir = padded_flow_direction_array[row + 1, col + 1]

#                 if flow_dir == nodata:
#                     pour_point_array[row, col] = nodata

#                 else:
#                     delta_x, delta_y = directions[flow_dir]
#                     sink_value = padded_flow_direction_array[row + 1 + delta_x, col + 1 + delta_y]

#                     if sink_value == nodata:
#                         pour_point_array[row, col] = 1
#                     else:
#                         pour_point_array[row, col] = 0

#         return pour_point_array


#     def option_2(flow_direction_array):
#         pour_point_array = numpy.empty((height, width))

#         # iterate over each pixel
#         for row in range(height):
#             for col in range(width):
#                 # get the flow direction [0-7] for this pixel
#                 flow_dir = flow_direction_array[row, col]

#                 if flow_dir != nodata:
#                     # get the location of the pixel it flows into
#                     delta_x, delta_y = directions[flow_dir]
#                     sink_row, sink_col = row + delta_x, col + delta_y

#                     # if either index is <0, it's flowing off the edge of the raster
#                     # if the sink pixel value is nodata, it's flowing into nodata
#                     # either case means this is a pour point
#                     if (sink_row < 0 or 
#                         sink_col < 0 or 
#                         flow_direction_array[sink_row, sink_col] == nodata):
#                         pour_point_array[row, col] = 1
#                     else:
#                         pour_point_array[row, col] = 0

#         return pour_point_array



