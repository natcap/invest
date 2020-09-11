import numpy
from scipy import ndimage
import pygeoprocessing

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
    height, width = raster_info['raster_size']
    nodata = raster_info['nodata'][0]


    flow_direction_array = pygeoprocessing.raster_to_numpy_array(
                                           flow_direction_raster_path)

    # make a 1-pixel-wide nodata border around the flow direction data
    # this will make it easier to identify points flowing off the raster
    padded_flow_direction_array = numpy.full((height + 2, width + 2), nodata)
    padded_flow_direction_array[1:-1, 1:-1] = flow_direction_array

    # alternate
    padded_flow_direction_array = numpy.pad(
        flow_direction_array,
        1,
        constant_values=-1
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
        pour_point_array = numpy.empty(shape)

        # iterate over each pixel
        for row in range(height):
            for col in range(width):
                flow_dir = flow_direction_array[row + 1, col + 1]

                deltaX, deltaY = directions[flow_dir]
                sink_value = flow_direction_array[row + 1 + deltaX, col + 1 + deltaY]

                if sink_value == nodata:
                    pour_point_array[row, col] = 1
                else:
                    pour_point_array[row, col] = 0




    def option_2(flow_direction_array):

        # iterate over each pixel
        for row in range(height):
            for col in range(width):
                # get the flow direction [0-7] for this pixel
                flow_dir = flow_direction_array[row, col]

                # get the location of the pixel it flows into
                delta_x, delta_y = directions[flow_dir]
                sink_row, sink_col = row + delta_x, col + delta_y

                # if either index is <0, it's flowing off the edge of the raster
                # if the sink pixel value is nodata, it's flowing into nodata
                # either case means this is a pour point
                if (sink_row < 0 or 
                    sink_col < 0 or 
                    flow_direction_array[sink_row, sink_col] == nodata):
                    pour_point_array[row - 1, col - 1] = 1
                else:
                    pour_point_array[row - 1, col - 1] = 0


    # alternate 3

    flow_direction_array = pygeoprocessing.raster_to_numpy_array(
                                               flow_direction_raster_path)


    def option_3(flow_direction_array):

        def func(x, nodata):
            print(x) # array of 8 surrounding pixels from top left to bottom right
            # 3 2 1       0 1 2 
            # 4 x 0  -->  3 4 5 
            # 5 6 7       6 7 8 
            convert_ = {
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
            return x[convert[flow_dir]] == nodata

       
        footprint = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]

        pour_points_array = ndimage.generic_filter(
                                flow_direction_array, 
                                func, 
                                footprint=footprint, 
                                mode='constant', 
                                cval=nodata)



    option_1(padded_flow_direction_array)
    option_2(flow_direction_array)
    option_3(flow_direction_array)



            




