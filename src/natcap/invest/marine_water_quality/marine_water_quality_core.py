import logging
import time

import scipy.sparse
from scipy.sparse.linalg import spsolve
import numpy as np
import pyamg

def diffusion_advection_solver(source_point_data, kps, in_water_array,
                               tide_e_array, adv_u_array,
                               adv_v_array, nodata, cell_size, layer_depth):
    """2D Water quality model to track a pollutant in the ocean.  Three input
       arrays must be of the same shape.  Returns the solution in an array of
       the same shape.

    source_point_data - dictionary of the form:
        { source_point_id_0: {'point': [row_point, col_point] (in gridspace),
                            'WPS': float (loading?),
                            'point': ...},
          source_point_id_1: ...}
    kps - absorption rate for the source point pollutants
    in_water_array - 2D numpy array of booleans where False is a land pixel and
        True is a water pixel.
    tide_e_array - 2D numpy array with tidal E values or nodata values, must
        be same shape as in_water_array (m^2/sec)
    adv_u_array, adv_v_array - the u and v components of advection, must be
       same shape as in_water_array (units?)
    nodata - the value in the input arrays that indicate a nodata value.
    cell_size - the length of the side of a cell in meters
    layer_depth - float indicating the depth of the grid cells in
            meters.
    """

    n_rows = in_water_array.shape[0]
    n_cols = in_water_array.shape[1]

    #Flatten arrays for use in the matrix building step
    in_water = in_water_array.flatten()
    e_array_flat = tide_e_array.flatten()
    adv_u_flat = adv_u_array.flatten()
    adv_v_flat = adv_v_array.flatten()

    LOGGER = logging.getLogger('natcap.invest.marine_water_quality.core')
    LOGGER.info('Calculating advection diffusion')
    t0 = time.clock()

    def calc_index(i, j):
        """used to abstract the 2D to 1D index calculation below"""
        if i >= 0 and i < n_rows and j >= 0 and j < n_cols:
            return i * n_cols + j
        else:
            return -1

    #set up variables to hold the sparse system of equations
    #upper bound  n*m*5 elements
    b_vector = np.zeros(n_rows * n_cols)

    #holds the rows for diagonal sparse matrix creation later, row 4 is
    #the diagonal
    a_matrix = np.zeros((9, n_rows * n_cols))
    diags = np.array([-2 * n_cols, -n_cols, -2, -1, 0, 1, 2, n_cols, 2 * n_cols])

    #Set up a data structure so we can index point source data based on 1D
    #indexes
    source_points = {}
    for source_id, source_data in source_point_data.iteritems():
        source_index = calc_index(*source_data['point'])
        source_points[source_index] = source_data

    #Build up an array of valid indexes.  These are locations where there is
    #water and well defined E and ADV points.
    LOGGER.info('Building valid index lookup table.')
    valid_indexes = in_water
    valid_indexes *= e_array_flat != nodata
    valid_indexes *= adv_u_flat != nodata
    valid_indexes *= adv_v_flat != nodata

    LOGGER.info('Building diagonals for linear advection diffusion system.')
    for i in range(n_rows):
        for j in range(n_cols):
            #diagonal element i,j always in bounds, calculate directly
            a_diagonal_index = calc_index(i, j)
            a_up_index = calc_index(i - 1, j)
            a_down_index = calc_index(i + 1, j)
            a_left_index = calc_index(i, j - 1)
            a_right_index = calc_index(i, j + 1)

            #if land then s = 0 and quit
            if not valid_indexes[a_diagonal_index]:
                a_matrix[4, a_diagonal_index] = 1
                b_vector[a_diagonal_index] = nodata
                continue

            if  a_diagonal_index in source_points:
                #Set wps to be daily loading the concentration, convert to / sec
                #loading
                wps = source_points[a_diagonal_index]['WPS'] / (cell_size ** 2 * \
                    layer_depth)
                b_vector[a_diagonal_index] = -wps

            E = e_array_flat[a_diagonal_index]
            adv_u = adv_u_flat[a_diagonal_index]
            adv_v = adv_v_flat[a_diagonal_index]

            #Build up terms
            #Ey
            if a_up_index > 0 and a_down_index > 0 and \
                valid_indexes[a_up_index] and valid_indexes[a_down_index]:
                #Ey
                a_matrix[4, a_diagonal_index] += -2.0 * E / cell_size ** 2
                a_matrix[7, a_down_index] += E / cell_size ** 2
                a_matrix[1, a_up_index] += E / cell_size ** 2

                #Uy
                a_matrix[7, a_down_index] += adv_v / (2.0 * cell_size)
                a_matrix[1, a_up_index] += -adv_v / (2.0 * cell_size)
            if a_up_index < 0 and valid_indexes[a_down_index]:
                #we're at the top boundary, forward expansion down
                #Ey
                a_matrix[4, a_diagonal_index] += -E / cell_size ** 2
                a_matrix[7, a_down_index] += E / cell_size ** 2

                #Uy
                a_matrix[7, a_down_index] += adv_v / (2.0 * cell_size)
                a_matrix[4, a_diagonal_index] += -adv_v / (2.0 * cell_size)
            if a_down_index < 0 and valid_indexes[a_up_index]:
                #we're at the bottom boundary, forward expansion up
                #Ey
                a_matrix[4, a_diagonal_index] += -E / cell_size ** 2
                a_matrix[1, a_up_index] += E / cell_size ** 2

                #Uy
                a_matrix[1, a_up_index] += adv_v / (2.0 * cell_size)
                a_matrix[4, a_diagonal_index] += -adv_v / (2.0 * cell_size)
            if not valid_indexes[a_up_index]:
                #Ey
                a_matrix[4, a_diagonal_index] += -2.0 * E / cell_size ** 2
                a_matrix[7, a_down_index] += E / cell_size ** 2

                #Uy
                a_matrix[7, a_down_index] += adv_v / (2.0 * cell_size)
            if not valid_indexes[a_down_index]:
                #Ey
                a_matrix[4, a_diagonal_index] += -2.0 * E / cell_size ** 2
                a_matrix[1, a_up_index] += E / cell_size ** 2

                #Uy
                a_matrix[1, a_up_index] += -adv_v / (2.0 * cell_size)

            if a_left_index > 0 and a_right_index > 0 and \
                valid_indexes[a_left_index] and valid_indexes[a_right_index]:
                #Ex
                a_matrix[4, a_diagonal_index] += -2.0 * E / cell_size ** 2
                a_matrix[5, a_right_index] += E / cell_size ** 2
                a_matrix[3, a_left_index] += E / cell_size ** 2

                #Ux
                a_matrix[5, a_right_index] += adv_u / (2.0 * cell_size)
                a_matrix[3, a_left_index] += -adv_u / (2.0 * cell_size)
            if a_left_index < 0 and valid_indexes[a_right_index]:
                #we're on left boundary, expand right
                #Ex
                a_matrix[4, a_diagonal_index] += -E / cell_size ** 2
                a_matrix[5, a_right_index] += E / cell_size ** 2

                a_matrix[5, a_right_index] += adv_u / (2.0 * cell_size)
                a_matrix[4, a_diagonal_index] += -adv_u / (2.0 * cell_size)
                #Ux
            if a_right_index < 0 and valid_indexes[a_left_index]:
                #we're on right boundary, expand left
                #Ex
                a_matrix[4, a_diagonal_index] += -E / cell_size ** 2
                a_matrix[3, a_left_index] += E / cell_size ** 2

                #Ux
                a_matrix[3, a_left_index] += adv_u / (2.0 * cell_size)
                a_matrix[4, a_diagonal_index] += -adv_u / (2.0 * cell_size)

            if not valid_indexes[a_right_index]:
                #Ex
                a_matrix[4, a_diagonal_index] += -2.0 * E / cell_size ** 2
                a_matrix[3, a_left_index] += E / cell_size ** 2

                #Ux
                a_matrix[3, a_left_index] += -adv_u / (2.0 * cell_size)

            if not valid_indexes[a_left_index]:
                #Ex
                a_matrix[4, a_diagonal_index] += -2.0 * E / cell_size ** 2
                a_matrix[5, a_right_index] += E / cell_size ** 2

                #Ux
                a_matrix[5, a_right_index] += adv_u / (2.0 * cell_size)

            #K
            a_matrix[4, a_diagonal_index] += -kps

            if not valid_indexes[a_up_index]:
                a_matrix[1, a_up_index] = 0
            if not valid_indexes[a_down_index]:
                a_matrix[7, a_down_index] = 0
            if not valid_indexes[a_left_index]:
                a_matrix[3, a_left_index] = 0
            if not valid_indexes[a_right_index]:
                a_matrix[5, a_right_index] = 0

    if n_cols <= 2 or n_rows <= 2:
        raise ValueError(
            'The number of inferred columns and rows in the output raster'
            'are less than 2, probably because the Output Pixel size in the UI '
            'is set too low for the projection of the AOI. '
            'Try a smaller value. Current n_cols, n_rows (%d, %d)',
            n_cols, n_rows)

    matrix = scipy.sparse.spdiags(
        a_matrix, [-2 * n_cols, -n_cols, -2, -1, 0, 1, 2, n_cols, 2 * n_cols],
        n_rows * n_cols, n_rows * n_cols, "csc")

    LOGGER.info('generating preconditioner')
    ml = pyamg.smoothed_aggregation_solver(matrix)
    M = ml.aspreconditioner()

    LOGGER.info('Solving via gmres iteration')
    result = scipy.sparse.linalg.lgmres(matrix, b_vector, tol=1e-5, M=M)[0]
    LOGGER.info('(' + str(time.clock() - t0) + 's elapsed)')

    #Result is a 1D array of all values, put it back to 2D
    result.resize(n_rows,n_cols)
    return result
