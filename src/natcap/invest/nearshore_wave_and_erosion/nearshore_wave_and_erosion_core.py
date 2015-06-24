"""Module that contains the core computational components for
    the coastal protection model"""

import math
import sys
import os
import logging
import csv

import numpy as np
import scipy as sp
from scipy import interpolate
from scipy import ndimage
from scipy import sparse
import h5py

from osgeo import ogr
from osgeo import gdal

import cProfile, pstats

import pygeoprocessing
import nearshore_wave_and_erosion_core as core

from NearshoreWaveFunctions_3p0 import*
import CPf_SignalSmooth as SignalSmooth

logging.getLogger("pygeoprocessing").setLevel(logging.WARNING)
logging.getLogger("raster_cython_utils").setLevel(logging.WARNING)
LOGGER = logging.getLogger('natcap.invest.nearshore_wave_and_erosion.core')
logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

def execute(args):
    """Executes the coastal protection model

        args - is a dictionary with at least the following entries:

        returns nothing"""
    logging.info('executing coastal_protection_core')

    # User-specified transect data URI
    if 'cross_shore_profile_uri' in args:
        args['transect_data_uri'] = args['cross_shore_profile_uri']
    # Default transect data URI.
    else:
        args['transect_data_uri'] = \
            os.path.join(args['output_dir'], 'transect_data.h5')

    # Run the profile generator
    if 'Profile generator' in args['modules_to_run']:
        compute_transects(args)
        export_transect_coordinates_to_CSV(args['transect_data_uri'])

    # Run the nearshore wave and erosion model
    if 'wave & erosion' in args['modules_to_run']:
        compute_nearshore_and_wave_erosion(args)

        # Reconstruct 2D shore maps from biophysical data
        reconstruct_2D_shore_map(args)

    # Debug purposes
#    p = cProfile.Profile()
#    transects_uri = p.runctx('compute_transects(args)', globals(), locals())
#    transects_uri = p.runctx('compute_transects(args)', globals(), locals())
#    s = pstats.Stats(p)
#    s.sort_stats('time').print_stats(20)


# Compute the shore transects
def compute_transects(args):
    LOGGER.debug('Computing transects...')

    # Store shore and transect information
    shore_nodata = -20000.0
    args['transects_uri'] = os.path.join(args['output_dir'], 'transects.tif')
    pygeoprocessing.geoprocessing.new_raster_from_base_uri(args['landmass_raster_uri'], \
        args['transects_uri'], 'GTIFF', shore_nodata, gdal.GDT_Float64)
    transect_raster = gdal.Open(args['transects_uri'], gdal.GA_Update)
    transect_band = transect_raster.GetRasterBand(1)
    block_size = transect_band.GetBlockSize()
    transects = \
        sp.sparse.lil_matrix((transect_band.YSize, transect_band.XSize))

    # Store transect profiles to reconstruct shore profile
    args['shore_profile_uri'] = os.path.join( \
        os.path.split(args['landmass_raster_uri'])[0], 'shore_profile.tif')
    pygeoprocessing.geoprocessing.new_raster_from_base_uri(args['landmass_raster_uri'], \
        args['shore_profile_uri'], 'GTIFF', shore_nodata, gdal.GDT_Float64)
    shore_raster = gdal.Open(args['shore_profile_uri'], gdal.GA_Update)
    shore_band = shore_raster.GetRasterBand(1)
    shore = \
        sp.sparse.lil_matrix((shore_band.XSize, shore_band.YSize))

    # Landmass
    landmass = pygeoprocessing.geoprocessing.load_memory_mapped_array( \
        args['landmass_raster_uri'], pygeoprocessing.geoprocessing.temporary_filename())

    landmass_raster = gdal.Open(args['landmass_raster_uri'])

    if landmass_raster is None:
        raise IOError, 'Cannot open file ' + args['landmass_raster_uri']

    fine_geotransform = landmass_raster.GetGeoTransform()
    landmass_band = landmass_raster.GetRasterBand(1)

    row_count = landmass_band.YSize
    col_count = landmass_band.XSize

    # AOI
    aoi_raster = gdal.Open(args['aoi_raster_uri'])

    if aoi_raster is None:
        raise IOError, 'Cannot open file ' + args['aoi_raster_uri']

    aoi_band = aoi_raster.GetRasterBand(1)

    # Bathymetry
    bathymetry = pygeoprocessing.geoprocessing.load_memory_mapped_array( \
        args['bathymetry_raster_uri'], pygeoprocessing.geoprocessing.temporary_filename())

   # Get the fine and coarse raster cell sizes, ensure consistent signs
    i_side_fine = int( \
        math.sqrt(fine_geotransform[4]**2 + fine_geotransform[5]**2))
    j_side_fine = int( \
        math.sqrt(fine_geotransform[1]**2 + fine_geotransform[2]**2))

#    print('fine_geotransform', fine_geotransform)

    i_side_coarse = int(math.copysign(args['transect_spacing'], i_side_fine))
    j_side_coarse = int(math.copysign(args['transect_spacing'], j_side_fine))

    args['i_side_coarse'] = i_side_coarse
    args['i_side_fine'] = i_side_fine

    # Start and stop coordinates in meters
    i_start = int(round(fine_geotransform[3]))
    j_start = int(round(fine_geotransform[0]))
    i_end = int(round(i_start + i_side_fine * row_count))
    j_end = int(round(j_start + j_side_fine * col_count))
    i_max_fine = (i_end - i_start) / i_side_fine
    j_max_fine = (j_end - j_start) / j_side_fine

    # Size of a tile. The + 6 at the end ensure the tiles overlap,
    # leaving no gap in the shoreline, and allows a buffer big enough for
    # computing shore orientation
    i_offset = i_side_coarse/i_side_fine + 6
    j_offset = j_side_coarse/j_side_fine + 6
#    print('(i_side_coarse, i_side_fine)', (i_side_coarse, i_side_fine))
#    print('(i_offset, j_offset)', (i_offset, j_offset))
    mask = np.ones((i_offset, j_offset))
    tile_size = np.sum(mask)


    tiles = 0 # Number of valid tiles

    # Caching transect information for faster raster information processing:
    # Each entry is a dictionary with the following information:
    #   'raw_positions': numpy array
    #   'clip_limits': tuple (first, last)
    transect_info = []
    max_transect_length = 0

    # Going through the bathymetry raster tile-by-tile.
    LOGGER.debug('Detecting shore')
    progress_step = max((i_end - i_start) / i_side_coarse / 50, 1)
#    for i in range(i_start, i_start+10*i_side_coarse, i_side_coarse):
    for i in range(i_start, i_end, i_side_coarse):
        if (i / i_side_coarse)  % progress_step == 0:
            print '.',

        # Top of the current tile
        i_base = max((i - i_start) / i_side_fine - 3, 0)

        # Adjust offset so it doesn't get outside raster bounds
        # TODO: Allow to use fractional tile size
        if (i_base + i_offset) >= row_count:
            continue

        for j in range(j_start, j_end, j_side_coarse):

            # Left coordinate of the current tile
            j_base = max((j - j_start) / j_side_fine - 3, 0)

            # Adjust offset so it doesn't get outside raster bounds
            # TODO: Allow to use fractional tile size
            if (j_base + j_offset) >= col_count:
                continue

            # Data under the tile
            #data = aoi[i_base:i_base+i_offset, j_base:j_base+j_offset]

            # Look for landmass cover on tile
            tile = landmass_band.ReadAsArray(j_base, i_base, j_offset, i_offset)

            land = np.sum(tile)

            # If land and sea, we have a shore: detect it and store
            if land and land < tile_size:

#                if tiles > 237:
#                    continue

                i_first = i_base+3
                j_first = j_base+3
                i_last = i_base+i_offset-3
                j_last = j_base+j_offset-3

#                print('tile', tiles)
#                print('tile limits', (i_first, j_first, i_last, j_last))
                #Draw the tile
#                for row in range(i_first, i_last+1):
#                    for col in range(j_first, j_last+1):
#                        transects[(col, row)] = tiles #-12
#                for left in range(i_first, i_last+1):
#                    transects[(left, j_first)] = -10
#                for right in range(i_first, i_last+1):
#                    transects[(left, j_last)] = -2
#                for up in range(j_first, j_last+1):
#                    transects[(i_first, up)] = -10
#                for down in range(j_first, j_last+1):
#                    transects[(i_last, down)] = -2

                shore_patch = detect_shore(tile, mask, 0, 3, connectedness = 4)
                shore_pts = np.where(shore_patch == 1)


                if shore_pts[0].size:
                    # Store shore position
                    for p in zip(shore_pts[0], shore_pts[1]):
                        if p[0] < 3:
                            continue
                        elif p[0] > i_offset-3:
                            continue
                        if p[1] < 3:
                            continue
                        elif p[1] > j_offset-3:
                            continue

                        transects[(p[0] + i_base, p[1] + j_base)] = tiles # -1

                    # Estimate shore orientation
                    shore_orientations = \
                        compute_shore_orientations(shore_patch, \
                            shore_pts, i_base, j_base)

                    # Skip if no shore orientation
                    if not shore_orientations:
#                        print("No valid shore orientations")
                        continue
#                    for p in shore_orientations:
#                        transects[(p[0], p[1])] = -2

                    # Pick transect position among valid shore points
                    assert len(shore_pts) == 2, \
                        "Can't process shore segment " + str((i, j)) + \
                        ' ' + str(shore_pts)
                    transect_positions = \
                        select_transect(shore_orientations.keys(), \
                            i_base+3, j_base+3, \
                            i_base+i_offset-3, j_base+j_offset-3)

                    # Skip tile if no valid shore points
                    if len(transect_positions) == 0:
#                        print("No well defined position")
                        continue
#                    for p in transect_positions:
#                        transects[(p[0], p[1])] = -3

                    # Compute transect orientation
                    transect_orientations = \
                        compute_transect_orientations( \
                            transect_positions, shore_orientations,landmass)
#                    print('orientations')
#                    for p in transect_orientations:
#                        print('position', p, 'orientation', transect_orientations[p])
                    # Skip tile if can't compute valid orientation
                    if len(transect_orientations) == 0:
#                        print("  No valid transect orientation for", transect_positions)
                        continue
#                    for p in transect_orientations:
#                       o = transect_orientations[p]
#                       transects[(int(round(p[0]+o[0]*1)), int(round(p[1]+o[1]*1)))] = tiles
#                       transects[(int(round(p[0]+o[0]*1)), int(round(p[1]+o[1]*1)))] = -4
#                       transects[(int(round(p[0]+o[0]*2)), int(round(p[1]+o[1]*2)))] = -5
#                       transects[(int(round(p[0]+o[0]*3)), int(round(p[1]+o[1]*3)))] = -6

#                    tiles += 1

#                    continue

                    # Look for the first valid transect
                    for transect_position in transect_orientations:
#                        print('transect_position', transect_position)
                        transect_orientation = transect_orientations[transect_position]
                        # Compute raw transect depths
                        raw_depths, raw_positions = \
                            compute_raw_transect_depths(transect_position, \
                            transect_orientation, bathymetry, \
                            args['bathymetry_raster_uri'], \
                            landmass, args['landmass_raster_uri'], \
                            i_side_fine, \
                            args['max_land_profile_len'], \
                            args['max_land_profile_height'], \
                            args['max_profile_length'])
                        # The index positions might be outside of valid bathymetry
                        if raw_depths is None:
#                            print('  Could not sample transect depth')
                            continue
#                        transects[(transect_position[0], transect_position[1])] = -7

#                    tiles += 1

#                    continue


                        # Interpolate transect to the model resolution
                        interpolated_depths = \
                            raw_depths if raw_depths.size > 5 else None

                        # Not enough values for interpolation
                        if interpolated_depths is None:
                            continue
#                        transects[(transect_position[0], transect_position[1])] = -6

                        # Smooth transect
                        smoothed_depths = \
                            smooth_transect(interpolated_depths, \
                                args['smoothing_percentage'])

                        # Apply additional constraints on the smoothed transect
                        (clipped_transect, (start, shore, end)) = \
                            clip_transect(smoothed_depths, \
                                args['max_profile_depth'])

                        # Transect could be invalid, skip it
                        if clipped_transect is None:
                            continue


                        # Perform additional QAQC:

                        # Enforce total profile length
                        profile_length = end - start

                        profile_length_m = profile_length * i_side_fine

                        if profile_length_m < args['min_profile_length']:
#                            print('Profile too short (' + str(profile_length_m) + \
#                                ' < ' + str(args['min_profile_length']))
                            continue


                        # Enforce minimum offshore profile length
                        offshore_length = end - shore

                        offshore_length_m = offshore_length * i_side_fine

                        if offshore_length_m < args['min_offshore_profile_length']:
#                            print('Offshore portion too short (' + \
#                                str(offshore_length_m) + \
#                                ' < ' + str(args['min_offshore_profile_length']))
                            continue


                        # Enforce minimum profile depth
                        averge_depth = \
                            np.sum(clipped_transect[shore:end]) / offshore_length

                        if averge_depth > args['min_profile_depth']:
#                            print('Average transect depth too shallow (' + \
#                                str(averge_depth) + \
#                                ' < ' + str(args['min_profile_depth']))
                            continue


                        # At this point, the transect is valid:
                        else:
                            # Store important information about it
                            transect_info.append( \
                                {'raw_positions': \
                                    (raw_positions[0][start:end], \
                                    raw_positions[1][start:end]), \
                                'depths':smoothed_depths[start:end], \
                                'clip_limits':(0, shore-start, end-start)})

                            # Update the longest transect length if necessary
                            if (end - start) > max_transect_length:
                                max_transect_length = end - start

                            tiles += 1

                            # Found valid transect, break out of the loop
                            break
    print('')

    # Cleanup
    bathymetry = None
    landmass = None
    aoi_band = None
    aoi_raster = None
    bathymetry_band = None
    bathymetry_raster = None
    landmass_band = None
    landmass_raster = None

    LOGGER.debug('found %i tiles.' % tiles)

    tiles = len(transect_info)

    transect_count = tiles

    args['tiles'] = tiles
    args['max_transect_length'] = max_transect_length

    habitat_nodata = -99999


    # Create a numpy array to store the habitat type
    habitat_field_count = args['habitat_field_count']
    soil_field_count = args['soil_field_count']
    climatic_forcing_field_count = args['climatic_forcing_field_count']
    tidal_forcing_field_count = args['tidal_forcing_field_count']

    # Creating HDF5 file that will store the transect data
    LOGGER.debug('Creating HDF5 file %s.' % args['transect_data_uri'])

    transect_data_file = h5py.File(args['transect_data_uri'], 'w')


    bathymetry_dataset = \
        transect_data_file.create_dataset('bathymetry', \
            (tiles, max_transect_length), \
            compression = 'gzip', fillvalue = habitat_nodata)

    positions_dataset = \
        transect_data_file.create_dataset('ij_positions', \
            (tiles, 2, max_transect_length), \
            compression = 'gzip', fillvalue = habitat_nodata, \
            dtype = 'i4')

    xy_positions_dataset = \
        transect_data_file.create_dataset('xy_positions', \
            (tiles, 2, max_transect_length), \
            compression = 'gzip', fillvalue = habitat_nodata)

    shore_dataset = \
        transect_data_file.create_dataset('shore_index', \
            (tiles, 1), \
            compression = 'gzip', fillvalue = habitat_nodata, \
            dtype = 'i4')

    limits_group = transect_data_file.create_group('limits')

    indices_limit_dataset = \
        limits_group.create_dataset('indices', \
            (tiles, 2), \
            compression = 'gzip', fillvalue = habitat_nodata, \
            dtype = 'i4')

    coordinates_limits_dataset = \
        limits_group.create_dataset('ij_coordinates', \
            (tiles, 4), \
            compression = 'gzip', fillvalue = habitat_nodata, \
            dtype = 'i4')

    xy_coordinates_limits_dataset = \
        limits_group.create_dataset('xy_coordinates', \
            (tiles, 4), \
            compression = 'gzip', fillvalue = habitat_nodata)

    # TODO: Break this up so we don't use so much memory
    # On a second thought, this might be the best option: the model
    # can run optimally with enough memory, or use the HD otherwise...
    LOGGER.debug('Storing transect_info data')

    #Todo: Remove this by using datasets directly instead of transect_info
    dataset = gdal.Open(args['bathymetry_raster_uri'])

    gt = dataset.GetGeoTransform()
    dataset = None

    for transect in range(len(transect_info)):
        (start, shore, end) = transect_info[transect]['clip_limits']

        positions_dataset[transect, 0, start:end] = \
            transect_info[transect]['raw_positions'][0][start:end]

        positions_dataset[transect, 1, start:end] = \
            transect_info[transect]['raw_positions'][1][start:end]


        # We want pixel center, not corner
        origin = np.array([gt[3]+gt[4]/2.+gt[5]/2., gt[0]+gt[1]/2.+gt[2]/2.])
        step = np.array([gt[4]+gt[5], gt[1]+gt[2]])


        I = transect_info[transect]['raw_positions'][0][start:end] * \
            step[0] + origin[0]
        J = transect_info[transect]['raw_positions'][1][start:end] * \
            step[1] + origin[1]

        xy_positions_dataset[transect, 0, start:end] = J[start:end]
        xy_positions_dataset[transect, 1, start:end] = I[start:end]

        bathymetry_dataset[transect, start:end] = \
            transect_info[transect]['depths'][start:end]


        indices_limit_dataset[transect] = [start, end]

        shore_dataset[transect] = shore

        coordinates_limits_dataset[transect] = [ \
            transect_info[transect]['raw_positions'][0][start], \
            transect_info[transect]['raw_positions'][1][start], \
            transect_info[transect]['raw_positions'][0][end-1], \
            transect_info[transect]['raw_positions'][1][end-1], \
            ]
        xy_coordinates_limits_dataset[transect] = \
            [J[start], I[start], J[end-1], I[end-1]]

        transect_info[transect] = None


    # Going through the bathymetry raster tile-by-tile.
    LOGGER.debug('Saving transect data...')
    single_value = np.array([[0]])


    # Saving the shore
    shore_points = transects.nonzero()  # Extract the shore points
    progress_step = max(shore_points[0].size / 50, 1)
    for segment in range(shore_points[0].size):
        if segment % progress_step == 0:
            print '.',
        single_value[0, 0] = \
            transects[ \
                shore_points[0][segment], \
                shore_points[1][segment]]

        transect_band.WriteArray( \
            single_value, \
            int(shore_points[1][segment]), \
            int(shore_points[0][segment]))
    print('')


    # HDF5 file container
    args['hdf5_files'] = {}
    args['habitat_nodata'] = habitat_nodata


    combine_natural_habitats(args, transect_data_file)
    combine_soil_types(args, transect_data_file)
    store_climatic_forcing(args, transect_data_file)
    store_tidal_information(args, transect_data_file)

    # Saving transects
    progress_step = max(tiles / 50, 1)
    habitat_type_dataset = transect_data_file['habitat_type']
    for transect in range(tiles):
        if transect % progress_step == 0:
            print '.',
        # Extract important positions
        start = indices_limit_dataset[transect,0]
        end = indices_limit_dataset[transect,1]
        shore = shore_dataset[transect]

        for pos in range(end-start):
            # Store bathymetry in the transect

#            single_value[0, 0] = bathymetry_dataset[transect, pos]
            single_value[0, 0] = habitat_type_dataset[transect, pos]

            transect_band.WriteArray( \
                single_value, \
                int(positions_dataset[transect, 1, pos]), \
                int(positions_dataset[transect, 0, pos]))

        # Store inland distance in pixels
        single_value[0, 0] = shore - start

        transect_band.WriteArray( \
            single_value, \
            int(positions_dataset[transect, 1, start]), \
            int(positions_dataset[transect, 0, start]))

        # Store offshore distance in pixels
        single_value[0, 0] = end - shore - 1

        transect_band.WriteArray( \
            single_value, \
            int(positions_dataset[transect, 1, end-1]), \
            int(positions_dataset[transect, 0, end-1]))

        # Store transect ID at the shore location
        single_value[0, 0] = transect

        transect_band.WriteArray( \
            single_value, \
            int(positions_dataset[transect, 1, shore]), \
            int(positions_dataset[transect, 0, shore]))
    print('')


    #Making sure the band and dataset is flushed and not in memory before
    #adding stats
    transect_band.FlushCache()
    transect_band = None
    transect_raster.FlushCache()
    gdal.Dataset.__swig_destroy__(transect_raster)
    transect_raster = None
    pygeoprocessing.geoprocessing.calculate_raster_stats_uri(args['transects_uri'])


    # We're done, we close the file
    transect_data_file.close()

    return args['transect_data_uri']


def export_transect_coordinates_to_CSV(transect_data_uri):
    """Extract transect coordinates to a CSV

        Inputs:
            -transect_data_uri: URI to the HDF5 output file produced
            by the PG tool

        Returns the output CSV's URI that contains the data. """

    output_csv_uri = os.path.splitext(transect_data_uri)[0] + '.csv'

    LOGGER.debug('Exporting transect coordinates to CSV %s.' % output_csv_uri)

    transect_data_file = h5py.File(transect_data_uri)


    # Extract transect position
    shore_id_dataset = transect_data_file['shore_index']
    xy_positions_dataset = transect_data_file['xy_positions']

    # First and last indices of the valid transect points (that are not nodata)
    limit_group = transect_data_file['limits']

    coordinates_limits_dataset = limit_group['xy_coordinates']


    # Save data in CSV
    np.set_printoptions(precision = 20)
    with open(output_csv_uri, 'w') as csv_file:
        writer = csv.writer(csv_file)

        row = ['transect_id', 'x', 'y', 'x_inland', 'y_inland', 'x_offshore', 'y_offshore']
        writer.writerow(row)

        for transect in range(shore_id_dataset.size):

            # Extract transect intersection with shore
            shore_index = shore_id_dataset[transect]
            x_shore = xy_positions_dataset[transect, 0, shore_index]
            y_shore = xy_positions_dataset[transect, 1, shore_index]

            # Extract transect most landward point (first point)
            x_start = coordinates_limits_dataset[transect, 0]
            y_start = coordinates_limits_dataset[transect, 1]

            # Extract transect most seaward point (last point)
            x_end = coordinates_limits_dataset[transect, 2]
            y_end = coordinates_limits_dataset[transect, 3]

            # Create the row
            row = [transect, x_shore, y_shore, x_start, y_start, x_end, y_end]

            # Write it
            writer.writerow(row)


# ----------------------------------------------------
# Loading the transects to exclude from a csv, if any
# ----------------------------------------------------
def load_excluded_transects(args):
    """Load the ecluded tranects from a CSV file in a python set.

        Inputs:
            -args['excluded_transects_uri']: the excluded transect CSV URI

        Returns a set containing the excluded transect IDs as they appear in
            "transect_data.csv".
            If there is no CSV specified, the set is empty.
    """

    args['excluded_transects'] = set()

    if 'excluded_transects_uri' in args:
        if not os.path.isfile(args['excluded_transects_uri']):
            raise IOError, "Can't open transect exclusion file " + \
            args['excluded_transects_uri']

        with open(args['excluded_transects_uri']) as csvfile:
            excluded_transects_reader = csv.reader(csvfile)

            for row in excluded_transects_reader:
                for item in row:

                    # First, try to cast as int:
                    try:
                        transect = int(item)
                        if transect < 0:
                            raise IndexError, \
                                "A transect ID can't be negative (" + \
                                    str(item) + ")"
                        args['excluded_transects'].add(transect)

                    # If it doesn't work, it might be a range:
                    except ValueError:
                        transect_range = item.split('-')
                        if len(transect_range) != 2:
                            raise IndexError, \
                                "Can't interpret CSV token " + item

                        try:
                            start = int(transect_range[0])
                            end = int(transect_range[1])

                            # Enforce start >= end
                            if start < 0:
                                raise IndexError, \
                                    "A transect ID can't be negative (" + \
                                    str(item) + ")"
                            if end < start:
                                raise ValueError,  \
                                    "Invalid range: expected (start >= end), " + \
                                    " got (" + str(start) + "," + str(end) + ")"

                            # It is a range, now add the transects to the set:
                            for transect in range(start, end+1):
                                args['excluded_transects'].add(transect)

                        except ValueError:
                            LOGGER.error("Can't interpret CSV token %s as a range", item)
                            raise

    LOGGER.debug('Found %i transects to exclude', len(args['excluded_transects']))

    return args['excluded_transects']


# ----------------------------------------------
# Nearshore wave and erosion model
# ----------------------------------------------
def compute_nearshore_and_wave_erosion(args):
    LOGGER.debug('Computing nearshore wave and erosion...')

    excluded_transects = load_excluded_transects(args)

    LOGGER.debug('Loading HDF5 files...')

    if not os.path.isfile(args['transect_data_uri']):
        raise IOError, \
            "Can't open the HDF5 file " + args['transect_data_uri']

    f = h5py.File(args['transect_data_uri']) # Open the HDF5 file

    # Average spatial resolution of all transects
    transect_spacing = f['habitat_type'].attrs['transect_spacing']
    # Distance between transect samples
    bathymetry_resolution = f['habitat_type'].attrs['bathymetry_resolution']
    model_resolution = f['habitat_type'].attrs['model_resolution']

#    print('average space between transects:', transect_spacing, 'm')
#    print('bathymetry resolution:', bathymetry_resolution, 'm')
#    print('model resolution:', model_resolution, 'm')

    # ------------------------------------------------
    # Define file contents
    # ------------------------------------------------
    # Values on the transects are that of the closest shapefile point feature

    #--Climatic forcing
    # 5 Fields: Surge, WindSpeed, WavePeriod, WaveHeight
    # Matrix format: transect_count x 5 x max_transect_length
    climatic_forcing_dataset = f['climatic_forcing']

    #--Soil type
    #mud=0, sand=1, gravel=2, unknown=-1
    # Matrix size: transect_count x max_transect_length
    soil_types_dataset = f['soil_type']

    #--Soil properties:
    # 0-mud: DryDensty, ErosionCst
    # 1-sand: SedSize, DuneHeight, BermHeight, BermLength, ForshrSlop
    # 2-gravel: same as sand
    # Matrix size: transect_count x 5 x max_transect_length
    soil_properties_dataset = f['soil_properties']

    #--Habitat types:  !! This is different from the sheet and the content of the file. 2 is seagrass, not reef!!
    #   <0 = no habitats
    #   0 = kelp #   1 = eelgrass #   2 = underwater structure/oyster reef
    #   3 = coral reef #   4 = levee #   5 = beach #   6 = seawall
    #   7 = marsh #   8 = mangrove #   9 = terrestrial structure
    # Matrix size: transect_count x max_transect_length
    habitat_types_dataset = f['habitat_type']

    #--Habitat properties for each habitat type:
    #   <0: no data
    #   0: StemHeight=0, StemDiam=1, StemDensty=2, StemDrag=3, Type=4
    #   1: same as 1
    #   2: ShoreDist=2, Height=3, BaseWidth=4, CrestWidth=5
    #   3: FricCoverd=1, FricUncov=2, SLRKeepUp=3, DegrUncov=4
    #   4: Type=1, Height=2, SideSlope=3, OvertopLim=4
    #   5: SedSize=1, ForshrSlop=2, BermLength=3, BermHeight=4, DuneHeight=5
    #   6: same as 5
    #   7: SurfRough=1, SurRedFact=2, StemHeight=3, StemDiam=4, StemDensty=5, StemDrag=6
    #   8: a lot...
    #   9: Type=1, TransParam=2, Height=3habitat_properties_dataset = f['habitat_properties']
    habitat_properties_dataset = f['habitat_properties']

    #--Bathymetry in meters
    # Matrix size: transect_count x max_transect_length
    bathymetry_dataset = f['bathymetry']
    # row, column (I, J resp.) position of each transect point in the raster matrix when extracted as a numpy array
    # Matrix size: transect_count x 2 (1st index is I, 2nd is J) x max_transect_length

    #--Index of the datapoint that corresponds to the shore pixel
    positions_dataset = f['ij_positions']
    # Matrix size: transect_count x max_transect_length

    # Name of the "subdirectory" that contains the indices and coordinate limits described below
    shore_dataset = f['shore_index']

    # First and last indices of the valid transect points (that are not nodata)
    limit_group = f['limits']
    # First index should be 0, and second is the last index before a nodata point.
    # Matrix size: transect_count x 2 (start, end) x max_transect_length
    indices_limit_dataset = limit_group['indices']

    #--Coordinates of the first transect point (index 0, start point) and the last valid transect point (end point)
    coordinates_limits_dataset = limit_group['ij_coordinates']

    # ------------------------------------------------
    # Extract content
    # ------------------------------------------------

    #-- Climatic forcing
    #transect_count, forcing, transect_length = climatic_forcing_dataset.shape
    Ho=8;To=12;Uo=0;Surge=0; #Default inputs for now

    # 5 Fields: Surge, WindSpeed, WavePeriod, WaveHeight
    # Matrix format: transect_count x 5 x max_transect_length

    #   transect_count: number of transects
    #   max_transect_length: maximum possible length of a transect in pixels
    #   max_habitat_field_count: maximum number of fields for habitats
    #   max_soil_field_count: maximum number of fields for soil types
    transect_count, habitat_fields, transect_length = habitat_properties_dataset.shape
    transect_count, soil_fields, transect_length = soil_properties_dataset.shape

    # Creating the numpy arrays that will store all the information for a single transect
    hab_types = numpy.array(transect_length)
    hab_properties = numpy.array((habitat_fields, transect_length))
    bathymetry = numpy.array(transect_length)
    positions = numpy.array((transect_length, 2))
    soil_types = numpy.array(transect_length)
    soil_properties = numpy.array((soil_fields, transect_length))
    indices_limit = numpy.array((transect_length, 2))
    coordinates_limits = numpy.array((transect_length, 4))

    # Open field indices file, so that we can access a specific field by name
    field_indices_uri = os.path.join(args['intermediate_dir'], 'field_indices')
    field_indices = json.load(open(field_indices_uri)) # Open field indices info

    # Field indices
    for habitat_type in field_indices:
        habitat = field_indices[habitat_type]


    #--------------------------------------------------------------
    #Run the loop
    #--------------------------------------------------------------

    nodata = -99999.0

    #Initialize matrices
    Depth=numpy.zeros((transect_count,transect_length))+nodata
    Wave =numpy.zeros((transect_count,transect_length))+nodata
    WaterLevel=numpy.zeros((transect_count,transect_length))+nodata
    WaterLevel_NoVegetation=numpy.zeros((transect_count,transect_length))+nodata
    VeloBottom=numpy.zeros((transect_count,transect_length))+nodata
    Undertow=numpy.zeros((transect_count,transect_length))+nodata
    SedTrsprt=numpy.zeros((transect_count,transect_length))+nodata

    MErodeLens1 = []    # eroded areas
    Retreats1 = []      # areas of retreat
    transect_ids = []   # transects where erosion and retreat happened

    #Read data for each transect, one at a time
#    for transect in range(2000, 2500): # Debug
    progress_step = max(transect_count / 50, 1)
    for transect in range(transect_count): # Release
        if transect % progress_step == 0:
            print '.',
#        print('')
#        LOGGER.debug('Computing nearshore waves and erosion on transect %i', transect) #transect_count - transect)

        # Extract first and last index of the valid portion of the current transect
        start = indices_limit_dataset[transect,0]   # First index is the most landward point
        end = indices_limit_dataset[transect,1] # Second index is the most seaward point
        # Note: For bad data, the most landard point could be in the ocean or the most
        #   seaward point could be on land!!!
        Length=end-start;Start=start;End=end;
#        print('index limits (start, end):', (Start, End))

        # Extracting the valid portion (Start:End) of habitat properties
        hab_properties = habitat_properties_dataset[transect,:,Start:End]
        # The resulting matrix is of shape transect_count x 5 x max_transect_length
        # The middle index (1) is the maximum number of habitat fields:
#        print('maximum habitat property fields:', hab_properties.shape[1])

        unique_types = numpy.unique(hab_types)  # Compute the unique habitats

        #Bathymetry
        bathymetry = bathymetry_dataset[transect,Start:End]
        max_depth = numpy.amax(bathymetry)
        min_depth = numpy.amin(bathymetry)
        Shore=Indexed(bathymetry,0) #locate zero
        MinDepth=Indexed(bathymetry,min_depth)
#        print('bathymetry (min, max)', (min_depth, max_depth))

        if min_depth>-1 or abs(MinDepth-Shore)<2: #If min depth is too small and there aren't enough points, we don't run
                H=num.zeros(len(bathymetry))
        else: #Run the model
            #------Read habitat
            seagrass = 2

            # Load the habitat types along the valid portion (Start:End) of the current transect
            hab_types = habitat_types_dataset[transect,Start:End]
            habitat_types = numpy.unique(hab_types)#Different types of habitats
            habitat_types = habitat_types[habitat_types>=0]#Remove 'nodata' where theres no habitat

            positions = positions_dataset[transect,Start:End]
            start = [positions[0][0]]
            start.append(positions[0][1])
            end = [positions[-1][0]]
            end.append(positions[-1][1])
            coordinates_limits = coordinates_limits_dataset[transect,:]
#            print('coord limits:', \
#                  (coordinates_limits[0], coordinates_limits[1]), \
#                  (coordinates_limits[2], coordinates_limits[3]))

            #--Collect vegetation properties
            #Zero the phys. char
            RootDiam=numpy.zeros(Length);    RootHeight=numpy.zeros(Length);
            RootDens=numpy.zeros(Length);    RootCd=numpy.zeros(Length);
            TrunkDiam=numpy.zeros(Length);    TrunkHeight=numpy.zeros(Length);
            TrunkDens=numpy.zeros(Length);    TrunkCd=numpy.zeros(Length);
            CanopDiam=numpy.zeros(Length);    CanopHeight=numpy.zeros(Length);
            CanopDens=numpy.zeros(Length);    CanopCd=numpy.zeros(Length)

            if habitat_types.size: #If there is a habitat in the profile
                seagrass=2
                HabType=[];#Collect the names of the different habitats
                if seagrass in habitat_types:
                    HabType.append('Seagrass')
                    seagrass_location = numpy.where(hab_types == seagrass)

                    #Seagrass physical parameters - 'field_indices' dictionary
                    if seagrass_location[0].size:
                        Sg_diameter_id = \
                            field_indices["natural habitats"][str(int(seagrass))]['fields']['stemdiam']
                        Sg_diameters = hab_properties[Sg_diameter_id][seagrass_location]
                        mean_stem_diameter = numpy.average(Sg_diameters)
#                        print('   Seagrass detected. Mean stem diameter: ' + \
#                            str(mean_stem_diameter) + ' m')

                        Sg_height_id = \
                            field_indices["natural habitats"][str(int(seagrass))]['fields']['stemheight']
                        Sg_height = hab_properties[Sg_height_id][seagrass_location]
                        mean_stem_height = numpy.average(Sg_height)
#                        print('                                    Mean stem height: ' + \
#                            str(mean_stem_height) + ' m')

                        Sg_density_id = \
                            field_indices["natural habitats"][str(int(seagrass))]['fields']['stemdensty']
                        Sg_density = hab_properties[Sg_density_id][seagrass_location]
                        mean_stem_density = numpy.average(Sg_density)
#                        print('                                    Mean stem density: ' + \
#                              str(mean_stem_density) + ' #/m^2')

                        Sg_drag_id = \
                            field_indices["natural habitats"][str(int(seagrass))]['fields']['stemdrag']
                        Sg_drag = hab_properties[Sg_drag_id][seagrass_location]
                        mean_stem_drag = numpy.average(Sg_drag)
#                        print('                                    Mean stem drag: ' + \
#                            str(mean_stem_drag) )

                        RootDiam[seagrass_location]=Sg_diameters
                        RootHeight[seagrass_location]=Sg_height
                        RootDens[seagrass_location]=Sg_density
                        RootCd[seagrass_location]=Sg_drag

#                print('unique habitat types:', HabType)

            #Collect reef properties

            #Collect Oyster Reef properties
            Oyster={}

            #Soil types and properties
            soil_types = soil_types_dataset[transect,Start:End]
            soil_properties = soil_properties_dataset[transect,:,Start:End]
#            print('maximum soil property fields:', soil_properties.shape[1])
#            print('soil types', numpy.unique(soil_types)) #, soil_types)

            #Prepare to run the model
            dx=20;
            smoothing_pct=10.0
            smoothing_pct=smoothing_pct/100;

            #Resample the input data
            Xold=range(0,dx*len(bathymetry),dx)
            Xnew=range(0,Xold[-1]+1)
            length=len(Xnew)

            fintp=interpolate.interp1d(Xold,bathymetry, kind='linear')
            bath=fintp(Xnew)
            bath_sm=SignalSmooth.smooth(bath,len(bath)*smoothing_pct,'hanning')
            shore=Indexed(bath_sm,0) #Locate zero in the new vector

            fintp=interpolate.interp1d(Xold,RootDiam, kind='nearest')
            RtDiam=fintp(Xnew)
            fintp=interpolate.interp1d(Xold,RootHeight, kind='nearest')
            RtHeight=fintp(Xnew)
            fintp=interpolate.interp1d(Xold,RootDens, kind='nearest')
            RtDens=fintp(Xnew)
            fintp=interpolate.interp1d(Xold,RootCd, kind='nearest')
            RtCd=fintp(Xnew)

            fintp=interpolate.interp1d(Xold,TrunkDiam, kind='nearest')
            TkDiam=fintp(Xnew)
            fintp=interpolate.interp1d(Xold,TrunkHeight, kind='nearest')
            TkHeight=fintp(Xnew)
            fintp=interpolate.interp1d(Xold,TrunkDens, kind='nearest')
            TkDens=fintp(Xnew)
            fintp=interpolate.interp1d(Xold,TrunkCd, kind='nearest')
            TkCd=fintp(Xnew)

            fintp=interpolate.interp1d(Xold,CanopDiam, kind='nearest')
            CpDiam=fintp(Xnew)
            fintp=interpolate.interp1d(Xold,CanopHeight, kind='nearest')
            CpHeight=fintp(Xnew)
            fintp=interpolate.interp1d(Xold,CanopDens, kind='nearest')
            CpDens=fintp(Xnew)
            fintp=interpolate.interp1d(Xold,CanopCd, kind='nearest')
            CpCd=fintp(Xnew)

            hab_types[hab_types==nodata]=-1
            fintp=interpolate.interp1d(Xold,hab_types, kind='nearest')
            Sr=fintp(Xnew)

            #Check to see if we need to flip the data
            flip=0
            if bath_sm[0]>bath_sm[-1]:
                bath_sm=bath_sm[::-1];        flip=1
                RtDiam=RtDiam[::-1];        RtHeight=RtHeight[::-1]
                RtDens=RtDens[::-1];        RtCd=RtCd[::-1]
                TkDiam=TkDiam[::-1];        TkHeight=TkHeight[::-1]
                TkDens=TkDens[::-1];        TkCd=TkCd[::-1]
                CpDiam=CpDiam[::-1];        CpHeight=CpHeight[::-1]
                CpDens=CpDens[::-1];        CpCd=CpCd[::-1]
                Sr=Sr[::-1]

            #Store hab char. into dic
            PlantsPhysChar={};Roots={};Trunks={};Canops={}
            Roots["RootDiam"]=RtDiam;    Roots["RootHeight"]=RtHeight
            Roots["RootDens"]=RtDens;    Roots["RootCd"]=RtCd
            Trunks["TrunkDiam"]=TkDiam;    Trunks["TrunkHeight"]=TkHeight
            Trunks["TrunkDens"]=TkDens;    Trunks["TrunkCd"]=TkCd
            Canops["CanopDiam"]=CpDiam;    Canops["CanopHeight"]=CpHeight
            Canops["CanopDens"]=CpDens;    Canops["CanopCd"]=CpCd
            #Final dictionary
            PlantsPhysChar['Roots']=Roots.copy()
            PlantsPhysChar['Trunks']=Trunks.copy()
            PlantsPhysChar['Canops']=Canops.copy()
            PlantsPhysChar['Oyster']=Oyster.copy()

            #Define friction coeff
            #   0 = kelp #   1 = eelgrass #   2 = underwater structure/oyster reef
            #   3 = coral reef #   4 = levee #   5 = beach #   6 = seawall
            #   7 = marsh #   8 = mangrove #   9 = terrestrial structure
            Cf=numpy.zeros(length)+.01
            if flip==1:
                Cf=Cf[::-1]

            #Compute Wave Height
            Xnew=num.array(Xnew)
            H,Eta,Etanv,Ubot,Ur,Kt,Ic,Hm,other=WaveRegenWindCD(Xnew,bath_sm,Surge,Ho,To,Uo,Cf,Sr,PlantsPhysChar)

            #Compute maximum wave height
            bath_sm[bath_sm > -0.05] = -0.05

            assert (bath_sm <= -0.05).all()
            k,C,Cg=Fast_k(To,-bath_sm)
            Hmx=0.1*(2.0*pi/k)*tanh((-bath_sm+Surge)*k);#Max wave height - Miche criterion

            #Wave Breaking information
            temp,temp,xb,hb,Hb,temp,temp=Runup_ErCoral(Xnew,-bath_sm,H,Ho,H*0,H*0,To,.2,1.0/10,Sr)
            loc,Hb=FindBreaker(Xnew,-bath_sm,H,To,Sr)
            Transport=nanmean(Ic[loc:-1])

            #Flip the vectors back
            if flip==1:
                H=H[::-1];Eta=Eta[::-1];Etanv=Etanv[::-1];
                Ubot=Ubot[::-1];Ur=Ur[::-1];Ic=Ic[::-1]
                h=bath_sm[::-1];X=Xnew[::-1]

            #Interpolate back to dx and save in matrix
            lx=len(Xold)
            fintp=interpolate.interp1d(X,h, kind='linear')
            h_save=fintp(Xold)
            Depth[transect,0:lx]=h_save

            fintp=interpolate.interp1d(X,H, kind='linear')
            H_save=fintp(Xold)
            Wave[transect,0:lx]=H_save

            fintp=interpolate.interp1d(X,Eta, kind='linear')
            Eta_save=fintp(Xold)
            WaterLevel[transect,0:lx]=Eta_save

            fintp=interpolate.interp1d(X,Etanv, kind='linear')
            Etanv_save=fintp(Xold)
            WaterLevel_NoVegetation[transect,0:lx]=Etanv_save

            fintp=interpolate.interp1d(X,Ubot, kind='linear')
            Ubot_save=fintp(Xold)
            VeloBottom[transect,0:lx]=Ubot_save

            fintp=interpolate.interp1d(X,Ur, kind='linear')
            Ur_save=fintp(Xold)
            Undertow[transect,0:lx]=Ur_save

            fintp=interpolate.interp1d(X,Ic, kind='linear')
            Ic_save=fintp(Xold)
            SedTrsprt[transect,0:lx]=Ic_save

            #Compute beach erosion
            Beach=-1;Struct=-1
            if Beach==1:
                g=9.81;rho=1024.0;Gam=0.78;
                TD=Dur;Lo=g*To**2.0/(2.0*pi);
                Co=g*To/(2.0*pi);#deep water phase speed with period

                Rs0,Rs1,xb,hb,Hb,Etapr,Hpr=Runup_ErCoral(Xnew,-bath_sm,H,Ho,Eta,Eta,To,A,m,Sr)
                #Quick Estimate
                TS=(320.0*(Hb**(3.0/2)/(g**.5*A**3.0))/(1.0+hb/(BermH_P+DuneH_P)+(m*xb)/hb))/3600.0;#erosion response time scale ).
                BetaKD=2.0*pi*(TS/TD)
                expr="num.exp(-2.0*x/BetaKD)-num.cos(2.0*x)+(1.0/BetaKD)*num.sin(2.0*x)" # solve this numerically
                fn=eval("lambda x: "+expr)
                z=FindRootKD(fn,pi,pi/2,BetaKD) # find zero in function,initial guess from K&D
                Ro,Rinfo,m0=Erosion_Quick(Ho,To,Surge[-1]+Rs0,BermH_P,DuneH_P,BermW_P,Dur,m,A,z) #Quick estimate

                #Erosion using waves
                TS=(320.0*(Hb**(3.0/2)/(g**.5*A**3.0))/(1.0+hb/(BermH_P+DuneH_P)+(m*xb)/hb))/3600.0;#erosion response time scale ).
                BetaKD=2.0*pi*(TS/TD)
                z=FindRootKD(fn,pi,pi/2,BetaKD) # find zero in function,initial guess from K&D
                R,Rinf,mo=ErosionFunction(A,m,xb,hb,Hb,Surge1[-1]+Rs0,BermH_P,DuneH_P,BermW_P,Dur,z)

                #Present
                Rsp0,Rsp1,xb1,hb1,Hb1,Etapr1,Hpr1=Runup_ErCoral(X1,-Z1,Hp,Ho,Eta0,Etap,To,A,m,Sr1)
                TS=(320.0*(Hb1**(3.0/2)/(g**.5*A**3.0))/(1.0+hb1/(BermH_P+DuneH_P)+(m*xb1)/hb1))/3600.0;#erosion response time scale ).
                BetaKD=2.0*pi*(TS/TD)
                z=FindRootKD(fn,pi,pi/2,BetaKD) # find zero in function,initial guess from K&D
                R1,Rinf1,m1=ErosionFunction(A,m,xb1,hb1,Hb1,Surge1[-1]+Rsp1,BermH_P,DuneH_P,BermW_P,Dur,z)

                Ro=round(Ro,2);R=round(R,2);R1=round(R1,2);

            #Compute mud scour
            if Beach==0:
                if Mgloc1.any(): #If there is any mangroves present at the site
                    temp1=Mgloc1[0];
                else:
                    temp1=-1

                Mudy1=[];#Location of the muddy bed
                if (temp1)>=0 or (temp2)>=0:
                    MudBeg=min(temp1,temp2)
                    Mudy1=arange(MudBeg,Xend1)

                MErodeVol1=-1; #No mud erosion
                if len(Mudy1)>0:#Calculate mud erosion if there's a muddy bed
                    Ubp=array(Ubp);
                    Retreat1,Trms1,Tc1,Tw1,Te=MudErosion(Ubp[Mudy1]*0,Ubp[Mudy1],-Z1[Mudy1],To,me,Cm)
                    ErodeLoc=find(Trms1>Te[0]); # indices where erosion rate greater than Threshold
                    MErodeLen1=len(ErodeLoc) # erosion rate greater than threshold at each location shoreward of the shoreline (pre-management)
                    if any(ErodeLoc)>0:
                        MErodeVol1=trapz(Retreat1[ErodeLoc]/100.0,Mudy1[ErodeLoc],1.) # volume of mud eroded shoreward of the shoreline (m^3/m)
                    else:
                        MErodeVol1=0
                    MErodeVol1=round(MErodeVol1,2)
                    gp.addmessage('MudScour_Present='+str(MErodeVol1)+' m^3/m')

                    MErodeLens1.append(MErodeLen1)
                    Retreats1.append(MErodeLen1)
                    transect_ids.append(transect)

            if Struct==1:
                #Current conditions
                Qp=150;htoe=round(-Z1[-1]+Etap[-1],1);Hwp=htoe
                while Qp>10:
                    Hwp=Hwp+.1;
                    temp,Qp,msg=SeawallOvertop(Hp[-1],To,htoe,Hwp)
                Fp=ForceSeawall(Hp[-1],To,-Z1[-1])
                Fp=round(Fp,2);Hwp=round(Hwp,2)
                gp.addmessage('Wall_Present='+str(Hwp)+' m; Force on wall='+str(Fp)+' kN/m')
    print('')

    # If the user chose to do valuation
    if args['valuation_group']:

        # Convert types:
        MErodeLens1 = np.array(MErodeLens1)
        Retreats1 = np.array(Retreats1)
        transect_ids = np.array(transect_ids)

        # Fake values for MErodeLen2 and Retreat2:
        MErodeLens2 = numpy.ones(MErodeLens1.size) * -9999
        Retreats2 = numpy.ones(Retreats1.size) * -9999

        # Compute valuation
        compute_valuation( \
            args, MErodeLens1, MErodeLens2, Retreats1, Retreats2, transect_ids)

    # Saving data in HDF5
    args['biophysical_data_uri'] = \
        os.path.join(args['output_dir'], 'output.h5')

    f = h5py.File(args['biophysical_data_uri'], 'w') # Create new HDF5 file

    # Creating the placeholders that will hold the matrices
    Depth_dataset = \
        f.create_dataset("Depth", Depth.shape, \
            compression = 'gzip', fillvalue = nodata)
    Wave_dataset = \
        f.create_dataset("Wave", Wave.shape, \
            compression = 'gzip', fillvalue = nodata)
    WaterLevel_dataset = \
        f.create_dataset("WaterLevel", WaterLevel.shape, \
            compression = 'gzip', fillvalue = nodata)
    WaterLevel_NoVegetation_dataset = \
        f.create_dataset("WaterLevel_NoVegetation", WaterLevel_NoVegetation.shape, \
            compression = 'gzip', fillvalue = nodata)
    VeloBottom_dataset = \
        f.create_dataset("VeloBottom", VeloBottom.shape, \
            compression = 'gzip', fillvalue = nodata)
    Undertow_dataset = \
        f.create_dataset("Undertow", Undertow.shape, \
            compression = 'gzip', fillvalue = nodata)
    SedTrsprt_dataset = \
        f.create_dataset("SedTrsprt", SedTrsprt.shape, \
            compression = 'gzip', fillvalue = nodata)

    # Saving the matrices on file
    Depth_dataset[...] = Depth[...]
    Wave_dataset[...] = Wave[...]
    WaterLevel_dataset[...] = WaterLevel[...]
    WaterLevel_NoVegetation_dataset[...] = WaterLevel_NoVegetation[...]
    VeloBottom_dataset[...] = VeloBottom[...]
    Undertow_dataset[...] = Undertow[...]
    SedTrsprt_dataset[...] = SedTrsprt[...]

    # Close your file at the end, or it could be invalid.
    f.close()

    return args['biophysical_data_uri']


def compute_valuation(args, MErodeLens1, MErodeLens2, Retreats1, Retreats2, transect_ids):
    """ Computes valuation. Tis is Greg's code, unchanged.

        Inputs:
            -args['longshore_extent']: Used to obtain an approximate area of
                land loss associated with retreat/erosion. This is the along
                shore length where the natural habitat types, coverage,
                and management actions, topo/bathy and forcing conditions
                are approximately uniform. Integer number, unit is meter.

            -args['property_value']: Land $ value per square meter (integer)

            -args['storm_return_period']: Return period of storm (int, years)

            -args['discount_rate']: integer percentage point. discount rate to
                adjust the monetary benefits of the natural habitats in future
                years to the present time.

            -args['time_horizon']: integer. Years over which valuation is
                performed.

            -MErodeLens1: numpy array. Length of profile with significant
                erosion from scenario 1, in meters. One value per transect

            -MErodeLens2: Same thing as MErodeLens1, for scenario 2

            -Retreats1: numpy array. Amount of shore retreat, for each transect
                in meters, from scenario 1

            -Retreats2: Same thing as Retreats1, for scenario 2

            -transect_ids: transect indices that correspond to the data points


        Returns a dictionary of numpy arrays from args['valuation'] where:
            -args['valuation']['Eav'][t]: avoided erosion for transect 't'
            -args['valuation']['Dav'][t]: damage value for transect 't'
            -args['valuation']['EPV'][t]: projected habitat value for 't'
    """
    assert MErodeLens1.size == transect_ids.size, \
        "Wrong size for profile erosion array for scenario 1 (" + \
            str(MErodeLens1.size) + " vs " + str(transect_ids.size) + ")"

    assert MErodeLens2.size == transect_ids.size, \
        "Wrong size for profile erosion array for scenario 2 (" + \
            str(MErodeLens2.size) + " vs " + str(transect_ids.size) + ")"

    assert Retreats1.size == transect_ids.size, \
        "Wrong size for shore retreat array for scenario 1 (" + \
            str(MErodeLens1.size) + " vs " + str(transect_ids.size) + ")"

    assert Retreats2.size == transect_ids.size, \
        "Wrong size for shore retreat array for scenario 2 (" + \
            str(MErodeLens2.size) + " vs " + str(transect_ids.size) + ")"


    # function to split thousands with commas
    def splitthousands(s, sep=','):
        if len(s) <= 3:
            return s
        return splitthousands(s[:-3], sep) + sep + s[-3:]


    LOGGER.debug("Computing economic valuation for %i transects...", \
        transect_ids.size)


    # Initialize valuation data structure
    args['valuation'] = {}
    valuation = args['valuation']

    transect_count = transect_ids.size

    valuation['Eav'] = numpy.ones(transect_count) * -9999
    valuation['Dav'] = numpy.ones(transect_count) * -9999
    valuation['EPV'] = numpy.ones(transect_count) * -9999


    progress_step = max(transect_count / 50, 1)
    for transect in range(transect_count):
        if transect % progress_step == 0:
            print '.',

        MErodeLen1 = MErodeLens1[transect]
        MErodeLen2 = MErodeLens2[transect]
        Retreat1 = Retreat1[transect]
        Retreat2 = Retreat2[transect]


        if MErodeLen2==-9999 and Retreat2==-9999: # there's no management action
            message = "You haven't defined a management action." + \
                " We cannot compute an avoided damage value."
            LOGGER.debug(message)
        else:
            if sand==1:
                E1=max([Retreat1,Retreat2])
                E2=min([Retreat1,Retreat2])
            elif mud==1:
                E1=max([MErodeLen1,MErodeLen2])
                E2=min([MErodeLen1,MErodeLen2])

            if E2<0:
                message = "We cannot compute an avoided damage cost " + \
                    " because the retreat amount for one of your scenario" + \
                    " is negative. The biophysical model did not run appropriately."
                LOGGER.debug(message)
            else:
                Eav = valuation['Eav'][transect]
                Dav = valuation['Dav'][transect]
                EPV = valuation['EPV'][transect]


                E1=E1*Longshore;
                E2=E2*Longshore;
                Eav=E1-E2; # avoided erosion
                D1=E1*PropValue;
                D2=E2*PropValue;
                Dav=D1-D2; # avoided erosion
                p=1.0/Tr # return frequency
                temp1=1.0+disc;
                temp2=[(1.0/temp1)**t for t in range(1,int(TimeHoriz)+1)]
                EPV=p*Dav*sum(temp2)
#                LOGGER.debug("...Avoided Erosion between scenarios is %d meters squared.", \
#                    round(Eav))
#                LOGGER.debug("...Avoided Damage Value is $%s (in your local currency)...", \
#                    splitthousands(str(int(Dav))))
#                LOGGER.debug("Expected Projected Value of habitat is $%s (in your local currency)", \
#                    splitthousands(str(int(EPV))))
    print('')



def reconstruct_2D_shore_map(args):
    LOGGER.debug('Reconstructing 2D shore maps...')

    transect_data = h5py.File(args['transect_data_uri'])
    biophysical_data = h5py.File(args['biophysical_data_uri'])

    limit_group = transect_data['limits']
    indices_limit_dataset = limit_group['indices']


    coordinates_dataset = transect_data['ij_positions']
    wave_dataset = biophysical_data['Wave']

    # Shapes should agree
    assert coordinates_dataset.shape[0] == wave_dataset.shape[0], \
        "Wave vector and coordinates vector size mismatch (rows)"
    assert coordinates_dataset.shape[2] == wave_dataset.shape[1], \
        "Wave vector and coordinates vector size mismatch (columns)"

    (transect_count, max_transect_length) = wave_dataset.shape


    wave_coordinates = np.zeros(coordinates_dataset.shape[1])
    wave_array = np.zeros(wave_dataset.shape[1])


    # Find transect intersections
    transect_footprint = set() # Stores (i, j) coord. pairs of occupied cells
    intersection = {} # Stores intersections for each transect
    intersected_transects = {} # Stores intersection information

    intersection_count = 0
    intersecting_transect_count = 0

    for transect in range(transect_count):
#        print('transect', transect_count - transect)

        # Extract and clean the transect from nan values
        start = indices_limit_dataset[transect,0]
        end = indices_limit_dataset[transect,1]

        wave_array = wave_dataset[transect,start:end]

        # Invalid wave data is not supposed to show up.
        invalid_data_count = np.where(wave_array == -99999.)[0].size
        if invalid_data_count:
            continue

        coordinates_array = coordinates_dataset[transect,start:end]

        # Clip the transect at the first occurence of NaN
        first_nan = np.where(np.isnan(wave_array))[0]
        if first_nan.size:
            end = first_nan[0]

        # Loop throught the sample points to compute transect intersections
        for index in range(start,end):

            coord = (coordinates_array[0][index], coordinates_array[1][index])

            # If intersection: store it
            if coord in transect_footprint:
                intersection_count += 1

                # Already an intersection: append to existing list
                if coord in intersection:
                    intersection[coord].append(wave_array[index])

                    if transect in intersected_transects:
                        intersected_transects[transect].append( \
                            (coord, index, wave_array[index]))
                    else:
                        intersected_transects[transect] = \
                            [(coord, index, wave_array[index])]
                        intersecting_transect_count += 1

                # Otherwise, create a new list
                else:
                    intersection[coord] = [wave_array[index]]
                    intersected_transects[transect] = \
                        [(coord, index, wave_array[index])]
                    intersecting_transect_count += 1

            # No intersection here, update footprint
            else:
                transect_footprint.add(coord)

#    print('Detected ' + str(intersection_count) + ' intersections in ' + \
#        str(intersecting_transect_count) + ' transects.')


    # Adjust the intersecting transects so they all agree with each other
    for i in intersection:
        # Compute the mean value at the intersection
        intersection[i] = \
            sum(intersection[i]) / len(intersection[i])


    # Build mask to remove transect portions that are too far
    # from the area we're interested in
    transect_mask = pygeoprocessing.geoprocessing.load_memory_mapped_array( \
        args['bathymetry_raster_uri'], pygeoprocessing.geoprocessing.temporary_filename())

    min_bathy = -10   # Minimum interpolation depth
    max_bathy = 1  # Maximum interpolation depth

    transect_mask[transect_mask <= min_bathy] = min_bathy - 1
    transect_mask[transect_mask >= max_bathy] = min_bathy - 1

    transect_mask = transect_mask != min_bathy - 1


    # Build the interpolation structures
    X = []
    Y = []
    Z = []

    transect_footprint.clear() # used to remove coordinate duplicates

    LOGGER.info("Processing %i transect intersections:", \
        len(intersected_transects))

    progress_step = max(len(intersected_transects) / 50, 1)
    counter = 0
    for transect in intersected_transects:
        if counter % progress_step == 0:
            print '.',
        counter += 1
#        print('intersected transect', transect)

        current_transect = intersected_transects[transect]

        start = indices_limit_dataset[transect,0]
        end = indices_limit_dataset[transect,1]

        # Clip the transect at the first occurence of NaN
        wave_array = wave_dataset[transect,start:end]
        first_nan = np.where(np.isnan(wave_array))[0]
        if first_nan.size:
            end = first_nan[0]

        # Interpolate delta_y: add key values for x and y
        # First value is 0 (no correction)
        delta_y = [0.]  # y
        x = [start] # x


        # Special case for first entry
        coord, index, value = current_transect[0]

        if index == 0:
            delta_y = [intersection[coord] - value]  # y

        for i in range(1, len(current_transect)):

            # Fill in subsequent values using intersection data
            coord, index, value = current_transect[i]

#            print('    ', i, index, value)

            delta_y.append(intersection[coord] - value)
            assert not math.isnan(delta_y[-1]), 'detected nan: ' + str(i)
            x.append(index)

        # Append last value if necessary
        if x[-1] != end-1:
            delta_y.append(0.)
            x.append(end-1)

        assert len(delta_y) == len(x), \
            'Arrays x and delta_y disagree in size (' + \
            str(len(x)) + ' vs ' + str(len(delta_y)) + ')'

        if len(delta_y) == 1:
            continue

        delta_y = np.array(delta_y)
        x = np.array(x)

        if np.isnan(delta_y).any():
            print('current_transect', current_transect)
            print('wave_dataset[transect,start:end]', wave_dataset[transect,start:end])
            print('delta_y', delta_y)
            print('any', np.isnan(delta_y).any(), \
                'count', np.sum(np.isnan(delta_y).any()))
            assert not np.isnan(delta_y).any()

        # Run the interpolation:
        f = interpolate.interp1d(x, delta_y, 'linear')

        interpolation_range = range(start, end)
        interpolated_delta_y = f(interpolation_range)

        if np.isnan(interpolated_delta_y).any():
            print('current_transect', current_transect)
            print('wave_dataset[transect,start:end]', wave_dataset[transect,start:end])
            print('x', x)
            print('delta_y', delta_y)
            print('interpolation_range', interpolation_range)
            print('interpolated_delta_y', interpolated_delta_y)
            print('any', np.isnan(interpolated_delta_y).any(), \
                'count', np.sum(np.isnan(interpolated_delta_y).any()))
            assert not np.isnan(interpolated_delta_y).any()

        # Compute new transect values:
        corrected_transect_values = \
            interpolated_delta_y + wave_dataset[transect,start:end]

#        print('wave_dataset[transect,start:end]', wave_dataset[transect,start:end])


        if np.isnan(corrected_transect_values).any():
            print('current_transect', current_transect)
            print('wave_dataset[transect,start:end]', wave_dataset[transect,start:end])
            print('delta_y', delta_y)
            print('interpolated_delta_y', interpolated_delta_y)
            print('corrected_transect_values', corrected_transect_values)
            print('any', np.isnan(corrected_transect_values).any(), \
                'count', np.sum(np.isnan(corrected_transect_values).any()))
            assert not np.isnan(corrected_transect_values).any()

        # Put the values in structures for 2d interpolation:
        coordinates = coordinates_dataset[transect,:]
        coordinates = (coordinates[0][start:end], coordinates[1][start:end])

#        print('start, end', start, end)
#        print('X size', X.size, 'X coordinates size', coordinates[0].size)
#        print('corrected_transect_values', corrected_transect_values.size)


        # Remove coordinate duplicates
        for index in range(coordinates[0].size):
            coord = (coordinates[0][index], coordinates[1][index])

            # Check if the point is not too far off
            if (coord[0] < 0 ) or (coord[0] >= transect_mask.shape[0]):
                continue

            if (coord[1] < 0 ) or (coord[1] >= transect_mask.shape[1]):
                continue


            if transect_mask[coord]:

                # Check if it's a new point
                if coord not in transect_footprint:
                    X.append(coordinates[0][index])
                    Y.append(coordinates[1][index])
                    assert not math.isnan(corrected_transect_values[index]), \
                        'Found NaN at point ' + str(coord) + \
                        ' while trying removing coordinate duplicates.'
                    Z.append(corrected_transect_values[index])

                    transect_footprint.add(coord)

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    # Now, we're ready to invoke the interpolation function
    # TODO: Fix that...
    print('')
    print('Building the interpolation function for', X.size, 'points...')
    assert not np.isnan(Z).any(), \
        'Found NaN in the values from which to interpolate.'

    F = interpolate.interp2d(X, Y, Z, kind='linear')

    # Build the array of points onto which the function will interpolate:
    # For now, all the points with bathymetry between 1 and 10 meters:
    interp_I, interp_J = np.where(transect_mask)

    interp_I = np.unique(interp_I)
    interp_J = np.unique(interp_J)

    # Compute the actual interpolation
    LOGGER.info('Interpolating %i points...', interp_I.size * interp_J.size)
    surface = F(interp_I, interp_J)

#    print('surface size:', surface.size, 'uniques', np.unique(surface).size, np.unique(surface))

    # Save the values in a raster
    wave_interpolation_uri = os.path.join(args['output_dir'], \
        'wave_interpolation.tif')

    print('Saving data to', wave_interpolation_uri)

    bathymetry_nodata = \
        pygeoprocessing.geoprocessing.get_nodata_from_uri(args['bathymetry_raster_uri'])

    pygeoprocessing.geoprocessing.new_raster_from_base_uri(args['bathymetry_raster_uri'], \
        wave_interpolation_uri, 'GTIFF', bathymetry_nodata, gdal.GDT_Float64)

    wave_raster = gdal.Open(wave_interpolation_uri, gdal.GA_Update)
    wave_band = wave_raster.GetRasterBand(1)
    wave_array = wave_band.ReadAsArray()

    II, JJ = np.meshgrid(interp_I, interp_J)

    wave_array[(II, JJ)] = surface

    wave_array[~transect_mask] = bathymetry_nodata

    wave_array[([0], [0])] = 1.

    wave_band.WriteArray(wave_array)

    wave_array = None
    wave_band = None
    wave_raster.FlushCache()
    gdal.Dataset.__swig_destroy__(wave_raster)
    wave_raster = None

    pygeoprocessing.geoprocessing.calculate_raster_stats_uri(wave_interpolation_uri)



def store_tidal_information(args, transect_data_file):

    LOGGER.info('Processing tidal information...')

    # Create new category
    category = 'tidal information'

    hdf5_files = args['hdf5_files']
    habitat_nodata = args['habitat_nodata']

    if category not in args['shapefiles']:
        LOGGER.info("Couldn't find any %s data. Skip it.", category)
        return

    tidal_forcing_dataset = \
        transect_data_file.create_dataset('tidal forcing', \
            (args['tiles'], args['tidal_forcing_field_count']), \
            compression = 'gzip', fillvalue = 0, \
            dtype = 'i4')

    filenames = args['shapefiles'][category].keys()

    assert len(filenames) == 1, 'Detected more than one tidal data file'

    shp_name = filenames[0]

    limit_group = transect_data_file['limits']
    indices_limit_dataset = limit_group['indices']
    positions_dataset = transect_data_file['ij_positions']

    hdf5_files[category] = []

    for field in args['shapefiles'][category][shp_name]:

        # Retreive the index for this field
        field_id = args['field_index'][category]['fields'][field.lower()]

        # Extract the type for this shapefile
        tidal_file_uri = args['shapefiles'][category][shp_name][field]

        raster = gdal.Open(tidal_file_uri)
        band = raster.GetRasterBand(1)

        source_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(tidal_file_uri)

        raster = gdal.Open(tidal_file_uri)
        band = raster.GetRasterBand(1)

        tiles = args['tiles']

        progress_step = max(tiles / 50, 1)
        for transect in range(tiles):
            if transect % progress_step == 0:
                print '.',

            [start, end] = indices_limit_dataset[transect]

            if end < 0:
                continue

            # Load tidal forcing
            positions = \
                (positions_dataset[transect, 0, start:end], \
                positions_dataset[transect, 1, start:end])

            tidal_forcing = np.ones(end-start)
            for position in range(end-start):
                tidal_forcing[position] = \
                    band.ReadAsArray(int(positions[1][position]), \
                        int(positions[0][position]), 1, 1)[0]

            tidal_forcing = tidal_forcing[tidal_forcing != habitat_nodata]

            if tidal_forcing.size:
                tidal_value = np.average(tidal_forcing)
            else:
                tidal_value = habitat_nodata

            # Copy directly to destination
            tidal_forcing_dataset[transect, field_id] = tidal_value

        # Closing the raster and band before reuse
        band = None
        raster = None

        print('')




def store_climatic_forcing(args, transect_data_file):

    LOGGER.info('Processing climatic forcing...')

    # Create 'climatic forcing' category
    category = 'climatic forcing'

    hdf5_files = args['hdf5_files']
    habitat_nodata = args['habitat_nodata']

    hdf5_files[category] = []

    if category not in args['shapefiles']:
        LOGGER.info("Couldn't find any %s data. Skip it.", category)
        return

    filenames = args['shapefiles'][category].keys()

    assert len(filenames) == 1, 'Detected more than one climatic forcing file'

    shp_name = filenames[0]

    limit_group = transect_data_file['limits']
    indices_limit_dataset = limit_group['indices']
    positions_dataset = transect_data_file['ij_positions']

    climatic_forcing_dataset = \
        transect_data_file.create_dataset('climatic_forcing', \
            (args['tiles'], args['climatic_forcing_field_count']), \
            compression = 'gzip', fillvalue = 0)

    for field in args['shapefiles'][category][shp_name]:

        # Retreive the index for this field
        field_id = args['field_index'][category]['fields'][field.lower()]

        # Extract the type for this shapefile
        file_uri = args['shapefiles'][category][shp_name][field]

        source_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(file_uri)

        raster = gdal.Open(file_uri)
        band = raster.GetRasterBand(1)

        tiles = args['tiles']

        progress_step = max(tiles / 50, 1)
        for transect in range(tiles):
            if transect % progress_step == 0:
                print '.',

            [_, end] = indices_limit_dataset[transect]

            # The climatic data is taken as much offshore as possible
            position = \
                (positions_dataset[transect, 0, end-1], \
                positions_dataset[transect, 1, end-1])

            # Copy directly to destination
            climatic_forcing_dataset[transect, field_id] = \
                band.ReadAsArray(int(position[1]), int(position[0]), 1, 1)[0]

        print('')


def combine_soil_types(args, transect_data_file):

    LOGGER.info('Processing soil types...')

    hdf5_files = args['hdf5_files']
    habitat_nodata = args['habitat_nodata']

    # soil types
    soil_types = { \
        'mud':0, \
        'sand':1, \
        'gravel':2 \
    }


    # List of habitats that must absolutely have mud under them
    mud_only_habitat_names = ['marsh', 'mangrove']

    # Mapping habitat names <=> habitat IDs
    habitat_name_to_ID = dict(zip( \
        [habitat[0] for habitat in args['habitat_information']], \
        range(len(args['habitat_information'])))
    )

    # IDs of habitats that must have mud under them
    mud_only_habitat_ids = \
        [habitat_name_to_ID[name] for name in mud_only_habitat_names]

    # Create 'soil types' category
    category = 'soil type'

    hdf5_files[category] = []

    filenames = args['shapefiles'][category].keys()

    assert len(filenames) == 1, 'Detected more than one soil type file'

    shp_name = filenames[0]

    mask = None
    mask_dict = {}

    # Assert if field 'type' doesn't exist
    field_names_lowercase = \
        [field_name.lower() for field_name in \
            args['shapefiles'][category][shp_name].keys()]

    assert 'type' in field_names_lowercase, \
        "No 'type' field found for soil types."

    type_key = None
    for key in args['shapefiles'][category][shp_name].keys():
        if key.lower() == 'type':
            type_key = key
            break

    # Get rid of the path and the extension
    basename = os.path.splitext(os.path.basename( \
        args['shapefiles'][category][shp_name][type_key]))[0]

    # Extract the type for this shapefile
    type_shapefile_uri = args['shapefiles'][category][shp_name][type_key]

    shapefile_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(type_shapefile_uri)

    raster = gdal.Open(type_shapefile_uri)
    band = raster.GetRasterBand(1)

#    LOGGER.info('Extracting priority information from ' + shp_name)

    limit_group = transect_data_file['limits']
    indices_limit_dataset = limit_group['indices']
    positions_dataset = transect_data_file['ij_positions']
    shore_dataset = transect_data_file['shore_index']

    soil_type_dataset = \
        transect_data_file.create_dataset('soil_type', \
            (args['tiles'], args['max_transect_length']), \
            compression = 'gzip', fillvalue = 0, \
            dtype = 'i4')

    soil_properties_dataset = \
        transect_data_file.create_dataset('soil_properties', \
            (args['tiles'], args['soil_field_count'], \
                args['max_transect_length']), \
            compression = 'gzip', fillvalue = habitat_nodata)


    tiles = args['tiles']

    progress_step = max(tiles / 50, 1)
    for transect in range(tiles):
        if transect % progress_step == 0:
            print '.',

        [start, end] = indices_limit_dataset[transect]

        raw_positions = \
            (positions_dataset[transect, 0, start:end], \
            positions_dataset[transect, 1, start:end])

        #Load the habitats as sampled from the raster
        for position in range(end-start):

            soil_type = \
                band.ReadAsArray(int(raw_positions[1][position]), \
                    int(raw_positions[0][position]), 1, 1)[0]

            dataset_type = soil_type_dataset[transect,position]


            if dataset_type == shapefile_nodata:
                soil_type_dataset[transect,position] = habitat_nodata

            elif dataset_type < soil_type:
                soil_type_dataset[transect,position] = soil_type

            if soil_type in mud_only_habitat_ids:
                soil_type_dataset[transect,position] = soil_types['mud']

    print('')

    # Clean up
    band = None
    raster = None

    for field in args['shapefiles'][category][shp_name]:

        # Skip the field 'type'
        if field.lower() == 'type':
            continue

        # Retreive the index for this field
        field_id = args['field_index'][category]['fields'][field.lower()]

        # Get rid of the path and the extension
        basename = os.path.splitext( \
            os.path.basename(args['shapefiles'][category][shp_name][field]))[0]

        # Extract data from the current raster field
        field_id = args['field_index'][category]['fields'][field.lower()]

        uri = args['shapefiles'][category][shp_name][field]
        raster = gdal.Open(uri)
        band = raster.GetRasterBand(1)

        progress_step = max(tiles / 50, 1)
        for transect in range(tiles):
            if transect % progress_step == 0:
                print '.',

            [start, end] = indices_limit_dataset[transect]

            shore = shore_dataset[transect]

            raw_positions = \
                (positions_dataset[transect, 0, start:end], \
                positions_dataset[transect, 1, start:end])

            for position in range(end-start):
                field_value = \
                    band.ReadAsArray(int(raw_positions[1][position]), \
                        int(raw_positions[0][position]), 1, 1)

                soil_properties_dataset[transect, field_id, position] = \
                    field_value

        print('')

        # Close the raster before proceeding to the next one
        band = None
        raster = None


def combine_natural_habitats(args, transect_data_file):

    LOGGER.info('Processing natural habitats...')

    hdf5_files = args['hdf5_files']
    habitat_nodata = args['habitat_nodata']

    limit_group = transect_data_file['limits']
    indices_limit_dataset = limit_group['indices']
    positions_dataset = transect_data_file['ij_positions']
    shore_dataset = transect_data_file['shore_index']

    habitat_type_dataset = \
        transect_data_file.create_dataset('habitat_type', \
            (args['tiles'], args['max_transect_length']), \
            compression = 'gzip', fillvalue = 0, \
            dtype = 'i4')

    # Add size and model resolution to the attributes
    habitat_type_dataset.attrs.create('transect_spacing', args['i_side_coarse'])
    habitat_type_dataset.attrs.create('model_resolution', args['model_resolution'])
    habitat_type_dataset.attrs.create('bathymetry_resolution', args['i_side_fine'])


    habitat_properties_dataset = \
        transect_data_file.create_dataset('habitat_properties', \
            (args['tiles'], args['habitat_field_count'], args['max_transect_length']), \
            compression = 'gzip', fillvalue = habitat_nodata)

    # A dictionary that maps habitat codes as found on site to a habitat name:
    # {'kelp':1, 'seagrass':2, 'levee':5, etc.}
    # This is useful for the user-specified habitat types such as corals, beach, marsh
    habitat_name_map = {}


    category = 'natural habitats'

    # Create hdf5 category for natural habitats
    hdf5_files[category] = []

    # --------------------------------------------
    # Process each habitat layer
    # --------------------------------------------
    for shp_name in args['shapefiles'][category]:

        LOGGER.info('Extracting information from ' + shp_name)

        # Find habitat_id that will be used to search field position in field_index:
        habitat_type_name = args['shapefile types'][category][shp_name]

        habitat_id = None
        for habitat in args['field_index']['natural habitats']:
            if args['field_index']['natural habitats'][habitat]['name'] == \
                habitat_type_name:
                habitat_id = int(habitat)
                break

        assert habitat_id is not None, \
            "No habitat '" + habitat_type_name + "' found for " + shp_name

        # Build the list of field names in lower case
        field_names_lowercase = \
            [field_name.lower() for field_name in \
                args['shapefiles'][category][shp_name].keys()]

        # Assert if field 'type' doesn't exist
        assert 'type' in field_names_lowercase, "Can't find field 'type' in " \
            + str(field_names_lowercase)

        # Store the key 'type' in its original case
        type_key = None
        for key in args['shapefiles'][category][shp_name].keys():
            if key.lower() == 'type':
                type_key = key
                break


        # Extract the habitat type for this shapefile
        type_shapefile_uri = args['shapefiles'][category][shp_name][type_key]

        # Store the habitat combinations in a set (remove nodata)
        shapefile_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(type_shapefile_uri)
        unique_values = \
            set(pygeoprocessing.geoprocessing.unique_raster_values_uri(type_shapefile_uri))
        if shapefile_nodata in unique_values:
            unique_values.remove(shapefile_nodata)

        # Save all the habitat values in this layer to the field index dictionary
        for code in unique_values:
#            # Check the code is not already used
#            # (2 habitats can't have the same code)
#            assert code not in habitat_name_map, \
#                'code ' + str(unique_values) + ' for ' + habitat_type_name + ' ' + \
#                str(os.path.split(type_shapefile_uri)[1]) + ' ' + \
#                ' is already used by ' + habitat_name_map[code] + \
#                ', habitat map: ' + str(habitat_name_map)

            # Assign this code to the habitat name
            if code not in habitat_name_map:
                habitat_name_map[code] = habitat_type_name

        args['field_index']['natural habitats'][habitat]['habitat_values'] = \
            unique_values


        raster = gdal.Open(type_shapefile_uri)
        band = raster.GetRasterBand(1)

        tiles = args['tiles']

        # ---------------------------------
        # Store habitat type
        # ---------------------------------
        progress_step = max(tiles / 50, 1)
        for transect in range(tiles):
            if transect % progress_step == 0:
                print '.',

            [start, end] = indices_limit_dataset[transect]

            raw_positions = \
                (positions_dataset[transect, 0, start:end], \
                positions_dataset[transect, 1, start:end])

            #Load the habitats as sampled from the raster
            habitat_presence = 0
            for position in range(end-start):

                habitat_type = \
                    band.ReadAsArray(int(raw_positions[1][position]), \
                        int(raw_positions[0][position]), 1, 1)[0]

                dataset_type = habitat_type_dataset[transect,position]

                # Can't decide which habitat to choose
                if dataset_type == habitat_type:
                    raise Exception, \
                        "Can't decide which habitat code to choose (" + \
                        str(dataset_type) + ")"

                # Override with nodata
                if dataset_type == shapefile_nodata:
                    habitat_type_dataset[transect,position] = habitat_nodata

                # Override with higher priority habitat
                elif dataset_type < habitat_type:
                    habitat_type_dataset[transect,position] = habitat_type
                    habitat_presence += 1

#            print habitat_presence,

#            # Load the constraints
#            for hab_id in range(len(args['habitat_information'])):
#                habitat_name = args['habitat_information'][hab_id][0]
#
#                if habitat_name == habitat_type_name:
#                    assert 'constraints' in args['habitat_information'][hab_id][2]
##                    print('Found constraints in', habitat_type_name, \
##                        args['habitat_information'][hab_id][2]['constraints'])
#                    constraints = np.copy(habitat_type)
#                    #constraint_uri =
#                    #constraint_raster = gdal.Open(constraint_uri)
#                    #constraint_band = constraint_raster.GetRasterBand(1)
#                    #for position in range(end-start):
#                    #    constraints[position] = \
#                    #        constraint_band.ReadAsArray( \
#                    #            int(raw_positions[1][position]), \
#                    #            int(raw_positions[0][position]), 1, 1)[0]
##                else:
##                    print('No constraints for', habitat_type_name)

        print('')


        # Clean up
        band = None
        raster = None

        # ---------------------------------
        # Store habitat properties
        # ---------------------------------
        for field in args['shapefiles'][category][shp_name]:

            # Skip the field 'type'
            if field.lower() == 'type':
                continue

            # Get rid of the path and the extension
            basename = os.path.splitext( \
                os.path.basename(args['shapefiles'][category][shp_name][field]))[0]

            # Extract data from the current raster field
            field_id = args['field_index']['natural habitats'][habitat_id]['fields'][field.lower()]

            uri = args['shapefiles'][category][shp_name][field]
            raster = gdal.Open(uri)
            band = raster.GetRasterBand(1)

            # Process each transect individually
            progress_step = max(tiles / 50, 1)
            for transect in range(tiles):
                if transect % progress_step == 0:
                    print '.',

                # Extract raster positions and transect extents
                [start, end] = indices_limit_dataset[transect]

                raw_positions = \
                    (positions_dataset[transect, 0, start:end], \
                    positions_dataset[transect, 1, start:end])


                # Store field value wherever we find the current habitat
                valid_hab_positions = 0
                for position in range(end-start):

                    # Load habitat type, and convert it to string (dict key)
                    habitat_type = \
                        habitat_type_dataset[transect,position]

                    # If current habitat at this position, write the field value
                    if (habitat_type in habitat_name_map) and \
                        habitat_name_map[habitat_type] == habitat_type_name:

                        field_value = \
                            band.ReadAsArray(int(raw_positions[1][position]), \
                                int(raw_positions[0][position]), 1, 1)

                        habitat_properties_dataset[transect, field_id, position] = \
                            field_value

                        valid_hab_positions += 1

#                print valid_hab_positions,

            print('')

            # Close the raster before proceeding to the next one
            band = None
            raster = None

    # Save the habitat index codes that we just used
    with open(os.path.join(args['output_dir'], 'habitat_index_codes.json'), 'w') \
        as habitat_index_codes:
        json.dump(habitat_name_map, habitat_index_codes)


def apply_habitat_constraints(mask, habitat_type, args):
    print('transect size', habitat.size)

    habitat_types = np.unique(habitat).astype(int)

    print('habitat types', habitat_types)

    for habitat_type in habitat_types:
        print('habitat_type', habitat_type)
        if (habitat_type >= 0) and (habitat_type < len(constraints)):
            print('constraints', constraints[habitat_type])

    return habitat


def compute_shore_orientations(shore, shore_pts, i_base, j_base):
    """Compute an estimate of the shore orientation. To do so, the algorithm
        computes the orientation 'O' from two adjascent neighbors for each
        focal shore segment.
        Then, it computes the average of each pixel orientation 'A' by looking
        at the orientation of the two neighboring pixels.
        The orientation at a given pixel is given by the mean:
            1/3 * (2*O + A)

       Inputs:
           -shore: small 2D numpy shore array (1 for shore, 0 otherwise)
           -shore_pts: shore ij coordinates relative to the small shore array
           -i_base, j_base: row/col offsets used to compute absolute
            coordinates from the relative shore array coordinates

        Returns a dictionary of {(shore ij):(orientation vector ij)} pairs"""
    # We don't want to modify the shore passed as argument, so we copy it
    shore = np.copy(shore)
    max_i, max_j = shore.shape

    # Used to swap shore states between steps i and i+1
    updated_shore = shore.astype(int)

    # Compute orientations (O) for each pixel
    # The orientation is undefined if there is less than 1 neighbor or too many:
    #
    #   0 neighbors: shore orientation is undefined

#    print('shore before')
#    print(shore.astype(int))

    orientations = {}
    for coord in zip(shore_pts[0], shore_pts[1]):
        row, col = coord
        if not row or row >= max_i -1:
            updated_shore[row, col] = 0
            continue

        if not col or col >= max_j -1:
            updated_shore[row, col] = 0
            continue

#        print 'shore coord', coord,

        # Compute how many shore pixels there are in the focal pixel's
        # neighborhood and try to clean it up so we get as many valid
        # transect locations as possible:

        # Raw neighborhood before it's cleaned up
        neighborhood = np.copy(shore[row-1:row+2, col-1:col+2])
        # We don't count the focal pixel in the neighborhood
        neighborhood[1, 1] = 0
        # Number of neighboring pixels. Should be 2 ideally.
        neighbor_count = np.sum(neighborhood)

        # First step: try to fix the shore so we get 2 pixels from which
        # to compute a valid orientation:

#        print('neighborhood before', neighborhood)
#        print('neighbor count', neighbor_count)

        # 0 neighbors: can't do anything
        if neighbor_count == 0:
#            print('no neighbors, skip')
            updated_shore[row, col] = 0
            continue
        # 2 neighbors: no changes needed, no 'if' condition for this case
        # 3 and more neighbors: remove the diagonal pixels, it might help.
        elif neighbor_count > 2:
#            print('multiple neighbors')
            neighborhood[0, 0] = 0
            neighborhood[0, 2] = 0
            neighborhood[2, 2] = 0
            neighborhood[2, 0] = 0

        # Update the number of neighbors
        neighbor_count = np.sum(neighborhood)

       # 1 neighbor: re-instate focal point to compute the orientation
        if neighbor_count == 1:
#            print('one neighbor')
            neighborhood[1, 1] = 1

#        print('neighborhood after', neighborhood)

        # Second step: recompute the number of neighbors after cleanup step.
        neighbors = np.where(neighborhood == 1)
        neighbor_count = np.sum(neighborhood)

        # Skip if not 2 neighbors
        if neighbor_count != 2:
#            print('wrong number of neighbors', neighbor_count)
            updated_shore[row, col] = 0
            continue

        # Otherwise, we're good: let's compute the orientation
        orientations[coord] = \
            (neighbors[0][1] - neighbors[0][0], \
            neighbors[1][1] - neighbors[1][0])

#        print('orientation', orientations[coord])

#    print(shore.astype(int))
#    print('shore after')
#    print(updated_shore)

    # Compute average orientations 'A' for each pixel
    # with a valid orientation 'O'
    shore = np.copy(updated_shore)
    average_orientations = {}

    for coord in orientations:
        row, col = coord
#        print 'orientations coord', coord,


        # Same principle as above, but with pixel with valid orientatons only
        # Compute how many valid pixels there are in the focal pixel's
        # neighborhood and try to clean it up so we get as many pixels
        # with valid orientations as possible:

        # Raw neighborhood before it's cleaned up
        neighborhood = np.copy(shore[row-1:row+2, col-1:col+2])
        # We don't count the focal pixel in the neighborhood
        neighborhood[1, 1] = 0
        # Number of neighboring pixels. Should be 2 ideally.
        neighbor_count = np.sum(neighborhood)

        # First step: try to fix the valid pixel shore so we get 2 pixels
        # from which to compute a valid orientation:

        # 0 neighbors: can't do anything
        if neighbor_count == 0:
#            print('no neighbors, skip')
            updated_shore[row, col] = 0
            continue
        # 2 neighbors: no changes needed, no 'if' condition for this case
        # 3 and more neighbors: remove the diagonal pixels, it might help.
        elif neighbor_count > 2:
#            print('multiple neighbors')
            neighborhood[0, 0] = 0
            neighborhood[0, 2] = 0
            neighborhood[2, 2] = 0
            neighborhood[2, 0] = 0

        # Update the number of neighbors
        neighbor_count = np.sum(neighborhood)

        # 1 neighbor: re-instate focal point to compute the orientation
        if neighbor_count == 1:
#            print('one neighbor')
            neighborhood[1, 1] = 1

#        print('neighborhood after', neighborhood)

        # Second step: recompute the number of neighbors after cleanup step.
        neighbors = np.where(neighborhood == 1)
        neighbor_count = np.sum(neighborhood)

        # Skip if not 2 neighbors
        if neighbor_count != 2:
#            print('wrong number of neighbors', neighbor_count)
            updated_shore[row, col] = 0
            continue

        # Make coordinates absolute
        neighbors = (neighbors[0] + row - 1, neighbors[1] + col - 1)

        # Extract points separately
        first = (neighbors[0][0], neighbors[1][0])
        second = (neighbors[0][1], neighbors[1][1])
        assert first in orientations
        assert second in orientations

        # Compute average orientation
        average_orientation = \
            ((orientations[first][0] + orientations[second][0]) / 2,
            (orientations[first][1] + orientations[second][1]) / 2)

        # Store in dictionary
        average_orientations[coord] = average_orientation

#    print('shore at the end')
#    print(updated_shore)

    # Combine orientation (weight==2) with average orientation (weight==1).
    shore_orientation = {}
    for segment in average_orientations:
        O = orientations[segment]
        A = average_orientations[segment]
        composite = np.array([float(2 * O[0] + A[0]), float(2 * O[1] + A[1])])
        norm = (composite[0]**2 + composite[1]**2)**.5

        # Make the key coordiantes absolute
        shore_orientation[(segment[0] + i_base, segment[1] + j_base)] = \
            composite / norm


#    print(shore_orientation)
#    print('')

    return shore_orientation

def select_transect(shore_pts, i_start, j_start, i_end, j_end):
    """Select transect position among shore points, by avoiding positions
        on the bufferred parts of the transect. Buffers overlap between up to
        4 tiles, and cause the same positions to be selected multiple times.

        Inputs:
            -shore_pts: shore point coordinates (absolute rows & cols)
            -i_start, j_start, i_end, j_end: tile extents without the
                buffers. i/j refer to row/col, and start/end refer to
                smallest/largest values.

        Returns a list of [i, j] point coordinates that are not within
            the buffer region"""
    if not len(shore_pts):
        return None

    # Return the transect with the smallest i first, and j second
    sorted_points = sorted(shore_pts, key = lambda p: p[1])
    sorted_points = sorted(sorted_points, key = lambda p: p[0])

    # Remove points in the buffer to avoid 2 transects appear side-by-side
    valid_points = []
    for p in sorted_points:
#        print 'trying point', p,
        if p[0] < i_start:
            continue
        elif p[1] < j_start:
            continue
        elif p[0] > i_end:
            continue
        elif p[1] > j_end:
            continue
#        print('valid')
        valid_points.append(p)

#    if not len(valid_points):
#        print('No valid points in', sorted_points)
#        print('point limits', (i_start, j_start), (i_end, j_end))

    return valid_points

def compute_transect_orientations(positions, orientations, land):
    """Returns transect orientation towards the ocean."""
    # Returned list of all the transect orientations
    transect_orientations = {}

    for position in positions:
        # orientation is perpendicular to the shore
        orientation = orientations[position]
        orientation = np.array([-orientation[1], orientation[0]])
        step = np.array([0., 0.])

        # Scale orientation to 3 pixels for accuracy: minimize roundoff
        # errors due to looking up integer pixel coordinates.
        # 3 was chosen because the orientations are multiples of 1/3
        norm = (np.sum(orientation**2))**.5
        step = 3. * orientation / norm

        # Sample landmass at 1, 2, and 3 pixels away from position
        water_is_1_px_ahead = \
            not land[int(round(position[0] +step[0]/3.)), \
                int(round(position[1] +step[1]/3.))]
        water_is_1_px_behind = \
            not land[int(round(position[0] -step[0]/3.)), \
                int(round(position[1] -step[1]/3.))]

        water_is_2_px_ahead = \
            not land[int(round(position[0] + 2.*step[0]/3.)), \
                int(round(position[1] + 2.*step[1]/3.))]
        water_is_2_px_behind = \
            not land[int(round(position[0] - 2.*step[0]/3.)), \
                int(round(position[1] - 2.*step[1]/3.))]

        water_is_3_px_ahead = not land[ \
            int(round(position[0] +step[0])), \
            int(round(position[1] +step[1]))]
        water_is_3_px_behind = not land[ \
            int(round(position[0] -step[0])), \
            int(round(position[1] -step[1]))]

        # Try 3 pixels in the direction ahead
        if water_is_3_px_ahead:
            # Check if there is no land along the vector 'step'.
            if water_is_1_px_ahead and water_is_2_px_ahead:
                transect_orientations[position] = orientation
                continue

        # Try opposite direction
        elif water_is_3_px_behind:
            # Same thing as above, but for the opposite direction
            if water_is_1_px_behind and water_is_2_px_behind:
                transect_orientations[position] = -orientation
                continue

        # Couldn't detect water 3 pixels away in either direction,
        # look 1 pixel ahead only

        # No land, all good: we store this direction.
        if water_is_1_px_ahead:
            transect_orientations[position] = orientation

        # Still hit land: try the other direction
        elif water_is_1_px_behind:
            transect_orientations[position] = -orientation

    return transect_orientations

def compute_raw_transect_depths(shore_point, \
    direction_vector, bathymetry, bathymetry_uri, landmass, landmass_uri, \
    bathymetry_resolution, max_land_profile_len, max_land_profile_height, \
    max_sea_profile_len):
    """ compute the transect endpoints that will be used to cut transects"""
    # Maximum transect extents
    max_land_len = max_land_profile_len / bathymetry_resolution
    max_sea_len = 1000 * max_sea_profile_len / bathymetry_resolution

    bathymetry_raster = gdal.Open(bathymetry_uri)
    bathymetry_band = bathymetry_raster.GetRasterBand(1)

    habitat_type_raster = gdal.Open(landmass_uri)
    habitat_type_band = habitat_type_raster.GetRasterBand(1)

    # Limits on maximum coordinates
    bathymetry_shape = (bathymetry_band.YSize, bathymetry_band.XSize)

    depths = np.ones((max_land_len + max_sea_len + 1))*-20000

    I = (np.ones(depths.size) * -1).astype(int)
    J = (np.ones(depths.size) * -1).astype(int)

    p_i = shore_point[0]

    # Shore point outside bathymetry
    if (p_i < 0) or (p_i >= bathymetry_shape[0]):
#        print('    shore', shore_point, 'outside bathymetry', bathymetry_shape)
        return (None, None)

    p_j = shore_point[1]

    # Shore point outside bathymetry
    if (p_j < 0) or (p_j >= bathymetry_shape[1]):
#        print('    shore', shore_point, 'outside bathymetry', bathymetry_shape)
        return (None, None)

    d_i = direction_vector[0]
    d_j = direction_vector[1]

    initial_elevation = bathymetry[p_i, p_j]
    depths[max_land_len] = initial_elevation

    I[max_land_len] = p_i
    J[max_land_len] = p_j


    # Compute the landward part of the transect (go backward)
    start_i = p_i - d_i
    start_j = p_j - d_j

    # Offset if the shore is either inland or in the water
    shore_offset = 0

    inland_steps = 0

    # Only compute the landward part of the transect
    elevation = \
        bathymetry_band.ReadAsArray( \
            int(round(p_j)), \
            int(round(p_i)), 1, 1)[0]

    # If no land behind the shore, stop there and report 0
    land = landmass[int(round(start_i)), int(round(start_j))]

    if land:
        # Stop when maximum inland distance is reached
        for inland_steps in range(1, max_land_len):
#            print('    inland_steps', inland_steps)
            # Save last elevation before updating it
            last_elevation = elevation
            elevation = \
                bathymetry_band.ReadAsArray( \
                    int(round(start_j)), int(round(start_i)), 1, 1)[0]

            # Hit either nodata, or some bad data
            if elevation <= -12000:
                inland_steps -= 1
                start_i += d_i
                start_j += d_j
#                print('    Invalid bathymetry')
                break

            # Stop at maximum elevation
            if elevation > 20:
                inland_steps -= 1
                start_i += d_i
                start_j += d_j
#                print('    Hit maximum elevation')
                break

            # Reached the top of a hill, stop there
            if last_elevation > elevation:
                inland_steps -= 1
                start_i += d_i
                start_j += d_j
#                print('    Reached the top of a hill')
                break

            # We can store the depth at this point
            depths[max_land_len - inland_steps] = elevation
            I[max_land_len - inland_steps] = int(round(start_i))
            J[max_land_len - inland_steps] = int(round(start_j))

            # Move backward (inland)
            start_i -= d_i
            start_j -= d_j

            # Stop if outside raster limits
            if (start_i < 0) or (start_j < 0) or \
                (start_i >= bathymetry_shape[0]) or (start_j >= bathymetry_shape[1]):
                # Went too far: backtrack one step
                start_i += d_i
                start_j += d_j
#                print('    Outside raster limits')
                break

        # If the highest inland point is still underwater, this transect is invalid
        highest_elevation = bathymetry_band.ReadAsArray( \
            int(round(start_j)), int(round(start_i)), 1, 1)[0]

        if highest_elevation < 0.:
#            print('    All the transect is underwater')
            return (None, None)

    # No land behind shore, return
    else:
#        print('    No land behind shore')
        return (None, None)


    # Compute the seaward part of the transect
    start_i = p_i + d_i
    start_j = p_j + d_j

    offshore_steps = 0

    last_elevation = \
        bathymetry_band.ReadAsArray( \
            int(round(p_j)), \
            int(round(p_i)), 1, 1)[0]

    # Stop when maximum offshore distance is reached
    for offshore_steps in range(1, max_sea_len):
#        print('    offshore_steps', offshore_steps)

        last_elevation = elevation
        elevation = \
            bathymetry_band.ReadAsArray( \
                int(round(start_j)), int(round(start_i)), 1, 1)[0][0]
        is_land = landmass[int(round(start_i)), int(round(start_j))]

        # Hit either nodata, or some bad data
        if elevation <= -12000:
            offshore_steps -= 1
            start_i -= d_i
            start_j -= d_j
#            print('    Invalid bathymetry')
            break

        # Stop if land is reached
        if is_land:
            offshore_steps -= 1
#            print('    Hit land')
            break

        # If positive elevation:


        # Looking at slope:

        # We're going up...
        if elevation > last_elevation:
            # ...and we're above water: stop there
            if elevation > 0:
                offshore_steps -= 1
                start_i -= d_i
                start_j -= d_j
#                print('    Left water')
                break

        # We can store the depth (elevation) at this point
        depths[max_land_len + offshore_steps] = elevation
        I[max_land_len + offshore_steps] = int(round(start_i))
        J[max_land_len + offshore_steps] = int(round(start_j))

        # Move forward (offshore)
        start_i += d_i
        start_j += d_j

        # Stop if outside raster limits
        if (start_i < 0) or (start_j < 0) or \
            (start_i >= bathymetry_shape[0]) or (start_j >= bathymetry_shape[1]):
#            print('    Outside raster limits', bathymetry_shape, 'when at', (start_i, start_j))
            break

    # If shore borders nodata, offshore_step is -1, set it to 0
    offshore_steps = max(0, offshore_steps)

    start = max_land_len - inland_steps
    end = max_land_len + offshore_steps + 1

#    print('(start, end)', (start, end), \
#        '(inland_steps, offshore_steps)', (inland_steps, offshore_steps), \
#        'max_land_len', max_land_len)
#    print('depths', depths[start:end])
#    print('I', I[start:end])
#    print('J', J[start:end])

    return (depths[start:end], (I[start:end], J[start:end]))


def interpolate_transect(depths, old_resolution, new_resolution, kind = 'linear'):
    """Interpolate transect at a higher resolution"""
    # Minimum entries required for interpolation
    if depths.size < 3:
        return None

    if new_resolution > old_resolution:
        raise Exception, 'New resolution should be finer.'

    x = np.arange(0, depths.size) * old_resolution
    f = interpolate.interp1d(x, depths, kind)
    x_new = \
        np.arange(0, (depths.size-1) * old_resolution / new_resolution) * \
            new_resolution
    interpolated = f(x_new)

    return interpolated


def smooth_transect(transect, window_size_pct):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_size_pct: the dimension of the smoothing window in percent (0.0 to 100.0);

    output:
        the smoothed signal
    """

    if transect.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if (window_size_pct < 0.0) or (window_size_pct > 100.00):
        raise ValueError, "Window size percentage should be between 0.0 and 100.0"

    # Compute window length from window size percentage
    window_length = transect.size * window_size_pct / 100.0

    # Adjust window length to be an odd number
    window_length += int(int(window_length/2) * 2 == window_length)


    if window_length<3:
        return transect

    # Add reflected copies at each end of the signal
    s=np.r_[2*transect[0]-transect[window_length-1::-1], transect, \
        2*transect[-1]-transect[-1:-window_length:-1]]

    w=np.ones(window_length,'d')

    y=np.convolve(w/w.sum(),s,mode='same')


    return y[window_length:-window_length+1]

def clip_transect(transect, max_depth):
    """Clip transect further since the depths have been smoothed out.

        Inputs:
            -transect: numpy array of real-valued depths
            -max_depth: maximum depth constraint beyond which to clip

        Returns nested tuples (clipped_transect, (start, shore, end)) where:
            -clipped_transect is a clipped version of the input numpy array
            -start, end: indices of the transect edges in the array
            -shore: index where the shore is

            If the transect is invalid, returns (None, (None, None, None))"""

    # Transect has to have a minimum size
    if transect.size < 5:
         return (None, (None, None, None))

    # Return if transect is full of zeros, otherwise the transect is invalid
    uniques = np.unique(transect)
    if uniques.size == 1:
        # If depth != 0 everywhere, transect is invalid
        # (either above water or submerged)
        if uniques[0] != 0.0:
            return (None, (None, None, None))

        # Limit case: the transect is valid, but it's really borderline
        return (transect, (0, 0, transect.size))

    # If higher point is negative: can't do anything, everything is underwater
    if transect[0] < 0:
        return (None, (None, None, None))

    # Go along the transect to find the shore
    shore = 1
    while transect[shore] > 0:
        shore += 1
        # End of transect: can't find the shore (first segment in water)
        if shore == transect.size:
            return (None, (None, None, None))

    # Find water extent
    water_extent = 1

    while transect[shore - 1 + water_extent] <= 0:
        # Stop if reached the end of the transect
        if shore + water_extent == transect.size:
            break
        # Stop if reached the maximum depth threshold
        if transect[shore + water_extent] < -max_depth:
            break
        # Else, proceed to the next pixel
        water_extent += 1

    # Compute the extremes on the valid portion only
    # Note:
    #   Numpy's argmin and argmax return the first occurence of the min/max.
    #   We want the longest transect possible, so since argmin and argmax
    #   traverse the transect in increasing index, argmax will indeed return
    #   the highest segment furthest away from the shore.
    #   However, argmin will return the shortest offshore portion. To get the
    #   longest one, we need to revert the offshore transect portion.

    # Highest point is straightforward
    highest_point = np.argmax(transect[:shore])

    # Lowest point has to be searched in the reversed transect portion
    # that is offshore.
    # Find the boundary indices
    rbegin = shore + water_extent-1
    rend = shore-1

    # Extract the offshore slice in reverse order
    reversed_slice = transect[rbegin:rend:-1]

    # Index to the lowest point furthest away from the shore
    lowest_point = np.argmin(reversed_slice)

    # The index is from the reversed slice, adjust it to the original indexing
    lowest_point = shore + reversed_slice.size - 1 - lowest_point

#    print('transect', lowest_point, transect)

    # Testing that lowest_point appears before highest_point
    assert highest_point < lowest_point, \
        'Highest point ' + str(highest_point) + \
        ' appears before lowest point ' + str(lowest_point)

    # Testing that highest_point is higher than lowest_point
    assert transect[highest_point] >= transect[lowest_point], \
        'Highest point ' + str(transect[highest_point]) + ' not >= ' \
        ' lowest point ' + str(transect[lowest_point])

    return (transect[highest_point:lowest_point+1], (highest_point, shore, lowest_point+1))


# improve this docstring!
def detect_shore(land_sea_array, aoi_array, aoi_nodata, buffer_size, connectedness = 8):
    """ Extract the boundary between land and sea from a raster.

        - land_sea_array: numpy array with sea, land and nodata values.
        - tile_limits: 4-tuple of type int with row/col coordinate limits
            of the tile without the buffers: (i_first, j_first, i_last, j_last)
            where i/j is the row/col coords, first/last are lowest/largest
            coordinate values.

        returns a numpy array the same size as the input raster with the shore
        encoded as ones, and zeros everywhere else."""
    # Rich's super-short solution, which uses convolution.
    nodata = -1
    land_sea_array[aoi_array == aoi_nodata] = nodata
    # Don't bother computing anything if there is only land or only sea
    land_size = np.where(land_sea_array > 0)[0].size

    if land_size == 0:
        LOGGER.warning('There is no shore to detect: land area = 0')
        return np.zeros_like(land_sea_array)
    elif land_size == land_sea_array.size:
        LOGGER.warning('There is no shore to detect: sea area = 0')
        return np.zeros_like(land_sea_array)
    else:
        # Shore points are inland (>0), and detected using 8-connectedness
        if connectedness is 8:
            kernel = np.array([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]])
        else:
            kernel = np.array([[ 0, -1,  0],
                               [-1,  4, -1],
                               [ 0, -1,  0]])
        # Generate the nodata shore artifacts
        aoi_array = np.ones_like(land_sea_array)
        aoi_array[land_sea_array == nodata] = nodata
        aoi_borders = (sp.signal.convolve2d(aoi_array, \
                                                kernel, \
                                                mode='same') >0 ).astype('int')
        # Generate all the borders (including data artifacts)
        borders = (sp.signal.convolve2d(land_sea_array, \
                                     kernel, \
                                     mode='same') >0 ).astype('int')
        # Real shore = all borders - shore artifacts
        borders = ((borders - aoi_borders) >0 ).astype('int') * 1.

        return borders



