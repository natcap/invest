import numpy
import pygeoprocessing
cimport numpy
cimport cython
from libcpp.map import map



cpdef _calculate_pour_point_array2(map flow_dir_array, map edges, int nodata):
    """
    Return a binary array indicating which pixels are pour points.

    Args:
        flow_dir_array (numpy.ndarray): a 2D array of D8 flow direction 
            values (0 - 7) and possiby the nodata value.
        edges (dict): has boolean keys 'top', 'left', 'bottom', 'right'
            indicating whether or not each edge is an edge of the raster.
the 
    Returns:
        numpy.ndarray of the same shape as ``flow_direction_array``.
        Each element is 1 if that pixel is a pour point, 0 if not,
        nodata if it cannot be determined.
    """   

    print(flow_dir_array)
    # flow direction: (delta rows, delta columns)
    # for a pixel at (i, j) with flow direction x,
    # the pixel it flows into is located at:
    # (i + directions[x][0], j + directions[x][1])
    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]

    cdef int height, width
    cdef int row, col
    cdef int flow_dir
    cdef int delta_x, delta_y
    cdef int sink_row, sink_col

    height, width = flow_dir_array.shape
    pour_point_array = numpy.empty((height, width))
    # The array passed into this function should already have a nodata border 
    # if applicable marking which edges are edges of the original raster. 
    # To avoid doing the calculation on these nodata borders, start from
    # index 1 on the sides that have the border.


    # iterate over each pixel
    for row in range(height):
        for col in range(width):

            # get the flow direction [0-7] for this pixel
            flow_dir = flow_dir_array[row, col]
            print((row, col), flow_dir)
            if flow_dir == nodata:
                pour_point_array[row, col] = nodata
            else:
                # get the location of the pixel it flows into
                delta_x, delta_y = directions[flow_dir]
                sink_row = row + delta_x
                sink_col = col + delta_y
                print('flows into', (sink_row, sink_col))
                # if the sink pixel value is nodata, it's either flowing
                # into a nodata area, or off the edge of the whole raster
                # either case means this is a pour point
                print('height, width:', height, width)
                print('out of bounds:', -1, -1, height, width)
                edge_conditions = [
                    ('top', sink_row == -1),
                    ('left', sink_col == -1),
                    ('bottom', sink_row == height),
                    ('right', sink_col == width)
                ]
                for edge, condition in edge_conditions:
                    if condition:
                        if edges[edge]:
                            print('1')
                            pour_point_array[row, col] = 1
                        else:
                            print('nodata')
                            pour_point_array[row, col] = TMP_NODATA
                        break
                else:
                    if flow_dir_array[sink_row, sink_col] == TMP_NODATA:
                        pour_point_array[row, col] = 1
                    else:
                        pour_point_array[row, col] = 0


    return pour_point_array
