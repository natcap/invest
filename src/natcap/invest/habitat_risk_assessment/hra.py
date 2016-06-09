'''This will be the preperatory module for HRA. It will take all unprocessed
and pre-processed data from the UI and pass it to the hra_core module.'''

import os
import shutil
import logging
import fnmatch
import functools
import math
import numpy as np

from osgeo import gdal, ogr, osr
from natcap.invest.habitat_risk_assessment import hra_core
from natcap.invest.habitat_risk_assessment import hra_preprocessor
import pygeoprocessing.geoprocessing

LOGGER = logging.getLogger('natcap.invest.habitat_risk_assessment.hra')
logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')


class ImproperCriteriaAttributeName(Exception):
    '''An excepion to pass in hra non core if the criteria provided by the user
    for use in spatially explicit rating do not contain the proper attribute
    name. The attribute should be named 'RATING', and must exist for every
    shape in every layer provided.'''
    pass


class ImproperAOIAttributeName(Exception):
    '''An exception to pass in hra non core if the AOIzone files do not
    contain the proper attribute name for individual indentification. The
    attribute should be named 'name', and must exist for every shape in the
    AOI layer.'''
    pass


class DQWeightNotFound(Exception):
    '''An exception to be passed if there is a shapefile within the spatial
    criteria directory, but no corresponing data quality and weight to support
    it. This would likely indicate that the user is try to run HRA without
    having added the criteria name into hra_preprocessor properly.'''
    pass


def execute(args):
    """Habitat Risk Assessment.

    This function will prepare files passed from the UI to be sent on to the
    hra_core module.

    All inputs are required.

    Args:
        workspace_dir (string): The location of the directory into which
            intermediate and output files should be placed.
        csv_uri (string): The location of the directory containing the CSV
            files of habitat, stressor, and overlap ratings. Will also contain
            a .txt JSON file that has directory locations (potentially) for
            habitats, species, stressors, and criteria.
        grid_size (int): Represents the desired pixel dimensions of both
            intermediate and ouput rasters.
        risk_eq (string): A string identifying the equation that should be used
            in calculating risk scores for each H-S overlap cell. This will be
            either 'Euclidean' or 'Multiplicative'.
        decay_eq (string): A string identifying the equation that should be
            used in calculating the decay of stressor buffer influence. This
            can be 'None', 'Linear', or 'Exponential'.
        max_rating (int): An int representing the highest potential value that
            should be represented in rating, data quality, or weight in the
            CSV table.
        max_stress (int): This is the highest score that is used to rate a
            criteria within this model run. These values would be placed
            within the Rating column of the habitat, species, and stressor
            CSVs.
        aoi_tables (string): A shapefile containing one or more planning
            regions for a given model. This will be used to get the average
            risk value over a larger area. Each potential region MUST contain
            the attribute "name" as a way of identifying each individual shape.

    Example Args Dictionary::

        {
            'workspace_dir': 'path/to/workspace_dir',
            'csv_uri': 'path/to/csv',
            'grid_size': 200,
            'risk_eq': 'Euclidean',
            'decay_eq': 'None',
            'max_rating': 3,
            'max_stress': 4,
            'aoi_tables': 'path/to/shapefile',
        }

    Returns:
        ``None``"""

#    Notes on the internal structure of the model:
#    ---------------------------------------------
#    Intermediate:
#        hra_args['habitats_dir']- The directory location of all habitat
#            shapefiles. These will be parsed though and rasterized to be passed
#            to hra_core module. This may not exist if 'species_dir' exists.
#        hra_args['species_dir']- The directory location of all species
#            shapefiles. These will be parsed though and rasterized to be passed
#            to hra_core module. This may not exist if 'habitats_dir' exists.
#        hra_args['stressors_dir']- The string describing a directory location
#            of all stressor shapefiles. Will be parsed through and rasterized
#            to be passed on to hra_core.
#        hra_args['criteria_dir']- The directory which holds the criteria
#            shapefiles. May not exist if the user does not desire criteria
#            shapefiles. This will be in a VERY specific format, which shall be
#            described in the user's guide.
#        hra_args['buffer_dict']- A dictionary that links the string name of
#            each stressor shapefile to the desired buffering for that shape
#            when rasterized.  This will get unpacked by the hra_preprocessor
#            module.
#
#            Example::
#
#                {
#                    'Stressor 1': 50,
#                    'Stressor 2': ...,
#                }
#
#        hra_args['h_s_c']- A multi-level structure which holds numerical
#            criteria ratings, as well as weights and data qualities for
#            criteria rasters. h-s will hold criteria that apply to habitat and
#            stressor overlaps, and be applied to the consequence score. The
#            structure's outermost keys are tuples of (Habitat, Stressor)
#            names. The overall structure will be as pictured:
#
#            Example::
#
#                {
#                    (Habitat A, Stressor 1):
#                        {'Crit_Ratings':
#                            {'CritName':
#                                {'Rating': 2.0, 'DQ': 1.0, 'Weight': 1.0}
#                            },
#                        'Crit_Rasters':
#                            {'CritName':
#                                {'Weight': 1.0, 'DQ': 1.0}
#                            },
#                        }
#                }
#
#        hra_args['habitats']- Similar to the h-s dictionary, a multi-level
#            dictionary containing all habitat-specific criteria ratings and
#            raster information. The outermost keys are habitat names.
#        hra_args['h_s_e']- Similar to the h_s dictionary, a multi-level
#            dictionary containing habitat-stressor-specific criteria ratings
#            and raster information which should be applied to the exposure\
#            score. The outermost keys are tuples of (Habitat, Stressor) names.
#
#   Output:
#        hra_args- Dictionary containing everything that hra_core will need to
#            complete the rest of the model run. It will contain the following.
#        hra_args['workspace_dir']- Directory in which all data resides. Output
#            and intermediate folders will be subfolders of this one.
#        hra_args['h_s_c']- The same as intermediate/'h-s', but with the
#            addition of a 3rd key 'DS' to the outer dictionary layer. This will
#            map to a dataset URI that shows the potentially buffered overlap
#            between the habitat and stressor. Additionally, any raster criteria
#            will be placed in their criteria name subdictionary. The overall
#            structure will be as pictured:
#
#            Example::
#
#                {
#                    (Habitat A, Stressor 1):
#                        {'Crit_Ratings':
#                            {
#                                'CritName':
#                                    {'Rating': 2.0, 'DQ': 1.0, 'Weight': 1.0}
#                            },
#                        'Crit_Rasters':
#                            {'CritName':
#                                {
#                                    'DS': "CritName Raster URI",
#                                    'Weight': 1.0, 'DQ': 1.0
#                                }
#                            },
#                        'DS':  "A-1 Dataset URI"
#                        }
#                }
#
#        hra_args['habitats']- Similar to the h-s dictionary, a multi-level
#            dictionary containing all habitat-specific criteria ratings and
#            rasters. In this case, however, the outermost key is by habitat
#            name, and habitats['habitatName']['DS'] points to the rasterized
#            habitat shapefile URI provided by the user.
#        hra_args['h_s_e']- Similar to the h_s_c dictionary, a multi-level
#            dictionary containing habitat-stressor-specific criteria ratings
#            and shapes. The same as intermediate/'h-s', but with the addition
#            of a 3rd key 'DS' to the outer dictionary layer. This will map to
#            a dataset URI that shows the potentially buffered overlap between
#            the habitat and stressor. Additionally, any raster criteria will
#            be placed in their criteria name subdictionary.
#        hra_args['risk_eq']- String which identifies the equation to be used
#            for calculating risk.  The core module should check for
#            possibilities, and send to a different function when deciding R
#            dependent on this.
#        hra_args['max_risk']- The highest possible risk value for any given
#            pairing of habitat and stressor.

    hra_args = {}
    inter_dir = os.path.join(args['workspace_dir'], 'intermediate')
    output_dir = os.path.join(args['workspace_dir'], 'output')

    hra_args['workspace_dir'] = args['workspace_dir']

    hra_args['risk_eq'] = args['risk_eq']

    # Depending on the risk calculation equation, this should return the highest
    # possible value of risk for any given habitat-stressor pairing. The highest
    # risk for a habitat would just be this risk value * the number of stressor
    # pairs that apply to it.
    max_r = calc_max_rating(args['risk_eq'], args['max_rating'])
    hra_args['max_risk'] = max_r

    # Pass along the max number of stressors the user believes will overlap one
    # another
    hra_args['max_stress'] = args['max_stress']

    # Create intermediate and output folders. Delete old ones, if they exist.
    for folder in (inter_dir, output_dir):
        if (os.path.exists(folder)):
            shutil.rmtree(folder)

        os.makedirs(folder)

    # If using aoi zones are desired, pass the AOI layer directly to core to be
    # dealt with there.
    if 'aoi_tables' in args:

        # Need to check that this shapefile contains the correct attribute name.
        # Later, this is where the uppercase/lowercase dictionary can be
        # implimented.
        shape = ogr.Open(args['aoi_tables'])
        layer = shape.GetLayer()

        lower_attrib = None
        for feature in layer:
            if lower_attrib is None:
                lower_attrib = dict(
                    zip(map(lambda x: x.lower(), feature.items().keys()),
                        feature.items().keys()))

            if 'name' not in lower_attrib:
                raise ImproperAOIAttributeName("Subregion layer attributes \
                    must contain the attribute \"Name\" in order to be \
                    properly used within the HRA model run.")

        # By this point, we know that the AOI layer contains the 'name' attrib
        # in some form. Pass that on to the core so that the name can be easily
        # pulled from the layers.
        hra_args['aoi_key'] = lower_attrib['name']
        hra_args['aoi_tables'] = args['aoi_tables']

    # Since we need to use the h-s, stressor, and habitat dicts elsewhere, want
    # to use the pre-process module to unpack them and put them into the
    # hra_args dict. Then can modify that within the rest of the code.
    # We will also return a dictionary conatining directory locations for all
    # of the necessary shapefiles. This will be used instead of having users
    # re-enter the locations within args.
    unpack_over_dict(args['csv_uri'], hra_args)

    # Where we will store the burned individual habitat and stressor rasters.
    crit_dir = os.path.join(inter_dir, 'Criteria_Rasters')
    hab_dir = os.path.join(inter_dir, 'Habitat_Rasters')
    stress_dir = os.path.join(inter_dir, 'Stressor_Rasters')
    overlap_dir = os.path.join(inter_dir, 'Overlap_Rasters')

    for folder in (crit_dir, hab_dir, stress_dir, overlap_dir):
        if (os.path.exists(folder)):
            shutil.rmtree(folder)

        os.makedirs(folder)

    # Habitat, Species and Stressor directory paths in the sample data are
    # given as relative paths.  These paths are assumed to be relative to the
    # CSV directory if they are relative paths.  This addresses an issue with
    # the sample data's compatibility with Mac binaries and assumptions about
    # what the CWD is when run from a Windows .bat script (CWD=directory
    # containing the .bat file) and a mac .command script (CWD=the user's home
    # directory, or wherever #!/bin/bash defaults to for the user).
    def _check_relative(path):
        """Verify `path` is relative to the CSV directory or absolute."""
        if not os.path.isabs(path):
            return os.path.abspath(os.path.join(args['csv_uri'], path))
        return path

    # Habitats
    hab_list = []
    for ele in ('habitats_dir', 'species_dir'):
        if ele in hra_args:
            hab_names = listdir(_check_relative(hra_args[ele]))
            hab_list += fnmatch.filter(hab_names, '*.shp')

    # Get all stressor URI's
    stress_names = listdir(_check_relative(hra_args['stressors_dir']))
    stress_list = fnmatch.filter(stress_names, '*.shp')

    # Get the unioned bounding box of all the incoming shapefiles which
    # will be used to define a raster to use for burning vectors onto.
    bounding_box = reduce(
        functools.partial(merge_bounding_boxes, mode="union"),
        [pygeoprocessing.geoprocessing.get_datasource_bounding_box(vector_uri)
            for vector_uri in hab_list + stress_list])

    # If there is an AOI, we want to limit the bounding box to its extents.
    if 'aoi_tables' in args:
        bounding_box = merge_bounding_boxes(
            bounding_box,
            pygeoprocessing.geoprocessing.get_datasource_bounding_box(
                args['aoi_tables']), "intersection")

    # Determine what the maximum buffer is to use in deciding how much
    # a future rasters extents should be expanded below. This will allow for
    # adequate pixel space when running decay functions later.
    max_buffer = reduce(lambda x, y: max(float(x), float(y)),
        hra_args['buffer_dict'].itervalues())

    def _create_raster_from_bb(bbox, pixel_size, wkt_str, out_uri, buff):
        """Create a raster given a bounding box and spatial reference.

        Parameters:
            bbox (list) - a list of values 4 numbers representing a geographic
                bounding box.
            pixel_size (float) - a float value for the size of raster pixels.
            wkt_str (string) - a Well Known Text format for a spatial reference
                to use in setting the rasters projection.
            out_uri (string) - a path on disk for the output raster.

        Returns:
            Nothing
        """
        # The bounding box coming in was set up with the form:
        # [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
        # Convert back into normal Shapefile extent format
        extents = [bbox[0], bbox[2], bbox[3], bbox[1]]

        # Need to create a larger base than the extent that would normally
        # surround the raster, because of some shapefiles needing buffer
        # space for decay equations from stressors on habitat.

        # These have to be expanded by 2 * buffer to account for both sides
        width = abs(extents[1] - extents[0]) + 2 * buff
        height = abs(extents[3] - extents[2]) + 2 * buff
        tiff_width = int(math.ceil(width / pixel_size))
        tiff_height = int(math.ceil(height / pixel_size))

        nodata = -1.0
        driver = gdal.GetDriverByName('GTiff')
        raster = driver.Create(
            grid_raster_path, tiff_width, tiff_height, 1, gdal.GDT_Float32,
            options=['BIGTIFF=IF_SAFER', 'TILED=YES'])
        raster.GetRasterBand(1).SetNoDataValue(nodata)
        # Set the transform based on the upper left corner and given pixel
        # dimensions increasing everything by buffer size
        raster_transform = [
            extents[0] - buff, pixel_size, 0.0, extents[3] + buff, 0.0,
            -pixel_size]

        raster.SetGeoTransform(raster_transform)
        # Use the same projection on the raster as one of the provided vectors
        raster.SetProjection(wkt_str)
        # Initialize everything to nodata
        raster.GetRasterBand(1).Fill(nodata)
        raster.GetRasterBand(1).FlushCache()
        band = None
        raster = None

    grid_raster_path = os.path.join(inter_dir, 'raster_grid_base.tif')
    # Use the first habitat shapefile to set the spatial reference / projection
    spat_ref = pygeoprocessing.geoprocessing.get_spatial_ref_uri(hab_list[0])
    wkt_str = spat_ref.ExportToWkt()

    # Create a raster that has a bounding box which is the UNION of all
    # the incoming shapefiles to be vectorized. This will act as the base
    # base raster to burn all the shapefiles onto, so that they are
    # guaranteed to be aligned upfront.
    _create_raster_from_bb(
        bounding_box, args['grid_size'], wkt_str, grid_raster_path,
        max_buffer)

    LOGGER.info('Rasterizing shapefile layers.')

    # Burn habitat shapefiles onto rasters
    add_hab_rasters(
        hab_dir, hra_args['habitats'], hab_list, args['grid_size'],
        grid_raster_path)

    # Want a super simple dictionary of the stressor rasters we will use for
    # overlap. The local var stress_dir is the location that should be used
    # for rasterized stressor shapefiles.
    stress_dict = make_stress_rasters(
        stress_dir, stress_list, args['grid_size'], args['decay_eq'],
        hra_args['buffer_dict'], grid_raster_path)

    # H_S_C and H_S_E
    # Just add the DS's at the same time to the two dictionaries,
    # since it should be the same keys.
    make_add_overlap_rasters(
        overlap_dir,
        hra_args['habitats'],
        stress_dict,
        hra_args['h_s_c'],
        hra_args['h_s_e'],
        args['grid_size'])

    # Criteria, if they exist.
    if 'criteria_dir' in hra_args:
        c_shape_dict = hra_preprocessor.make_crit_shape_dict(
            hra_args['criteria_dir'])

        add_crit_rasters(
            crit_dir, c_shape_dict, hra_args['habitats'], hra_args['h_s_e'],
            hra_args['h_s_c'], args['grid_size'])

    # No reason to hold the directory paths in memory since all info is now
    # within dictionaries. Can remove them here before passing to core.
    for name in (
        'habitats_dir', 'species_dir', 'stressors_dir', 'criteria_dir'):
        if name in hra_args:
            del hra_args[name]

    hra_core.execute(hra_args)


def merge_bounding_boxes(bb1, bb2, mode):
    """Merge two bounding boxes through union or intersection.

    Parameters:
        bb1 (list) - a list of values for a geographic bounding box set up as:
            [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
        bb2 (list) - a list of values for a geographic bounding box set up as:
            [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
        mode (string) - either 'intersection' or 'union'.

    Returns:
        A list of the merged bounding boxes.
    """

    if mode == "union":
        comparison_ops = [min, max, max, min]
    if mode == "intersection":
        comparison_ops = [max, min, min, max]

    bb_out = [op(x, y) for op, x, y in zip(comparison_ops, bb1, bb2)]
    return bb_out

def make_add_overlap_rasters(dir, habitats, stress_dict, h_s_c, h_s_e, grid_size):
    '''
    For every pair in h_s_c and h_s_e, want to get the corresponding habitat
    and stressor raster, and return the overlap of the two. Should add that as
    the 'DS' entry within each (h, s) pair key in h_s_e and h_s_c.

    Input:
        dir- Directory into which all completed h-s overlap files shoudl be
            placed.
        habitats- The habitats criteria dictionary, which will contain a
            dict[Habitat]['DS']. The structure will be as follows:

            {Habitat A:
                    {'Crit_Ratings':
                        {'CritName':
                            {'Rating': 2.0, 'DQ': 1.0, 'Weight': 1.0}
                        },
                    'Crit_Rasters':
                        {'CritName':
                            {
                                'DS': "CritName Raster URI",
                                'Weight': 1.0, 'DQ': 1.0
                            }
                        },
                    'DS':  "A Dataset URI"
                    }
            }

        stress_dict- A dictionary containing all stressor DS's. The key will be
            the name of the stressor, and it will map to the URI of the
            stressor DS.
        h_s_c- A multi-level structure which holds numerical criteria
            ratings, as well as weights and data qualities for criteria
            rasters. h-s will hold criteria that apply to habitat and stressor
            overlaps, and be applied to the consequence score. The structure's
            outermost keys are tuples of (Habitat, Stressor) names. The overall
            structure will be as pictured:

            {(Habitat A, Stressor 1):
                    {'Crit_Ratings':
                        {'CritName':
                            {'Rating': 2.0, 'DQ': 1.0, 'Weight': 1.0}
                        },
                    'Crit_Rasters':
                        {'CritName':
                            {'Weight': 1.0, 'DQ': 1.0}
                        },
                    }
            }
        h_s_e- Similar to the h_s dictionary, a multi-level
            dictionary containing habitat-stressor-specific criteria ratings
            and raster information which should be applied to the exposure
            score. The outermost keys are tuples of (Habitat, Stressor) names.
        grid_size- The desired pixel size for the rasters that will be created
            for each habitat and stressor.

    Output:
        An edited versions of h_s_e and h_s_c, each of which contains an
        overlap DS at dict[(Hab, Stress)]['DS']. That key will map to the URI
        for the corresponding raster DS.

    Returns nothing.
    '''
    for pair in h_s_c:
        # Check to see if the user has determined this habitat / stressor
        # pair should have no interaction. This means setting no overlap
        compute_overlap = all(h_s_e[pair]['overlap_list'] +
                              h_s_c[pair]['overlap_list'])
        LOGGER.debug("Compute Overlap is set to %s, for pair %s",
                     compute_overlap, pair)

        h, s = pair
        h_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
            habitats[h]['DS'])
        s_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
            stress_dict[s])

        files = [habitats[h]['DS'], stress_dict[s]]

        def add_h_s_pixels(h_pix, s_pix):
            '''Since the stressor is buffered, we actually want to make sure to
            preserve that value. If there is an overlap, return s value.'''
            if compute_overlap:
                # If there is an overlap return the stressor value
                return np.where(
                    ((h_pix != h_nodata) & (s_pix != s_nodata)),
                    s_pix, h_nodata)
            else:
                # Even if there is an overlap, return h_nodata
                return np.where(
                    ((h_pix != h_nodata) & (s_pix != s_nodata)),
                    h_nodata, h_nodata)

        out_uri = os.path.join(dir, 'H[' + h + ']_S[' + s + '].tif')

        pygeoprocessing.geoprocessing.vectorize_datasets(
            files,
            add_h_s_pixels,
            out_uri,
            gdal.GDT_Float32,
            -1.,
            grid_size,
            "union",
            resample_method_list=None,
            dataset_to_align_index=None,
            aoi_uri=None,
            vectorize_op=False)

        h_s_c[pair]['DS'] = out_uri
        h_s_e[pair]['DS'] = out_uri

def make_stress_rasters(dir, stress_list, grid_size, decay_eq, buffer_dict, grid_path):
    '''
    Creating a simple dictionary that will map stressor name to a rasterized
    version of that stressor shapefile. The key will be a string containing
    stressor name, and the value will be the URI of the rasterized shapefile.

    Input:
        dir- The directory into which completed shapefiles should be placed.
        stress_list- A list containing stressor shapefile URIs for all
            stressors desired within the given model run.
        grid_size- The pixel size desired for the rasters produced based on the
            shapefiles.
        decay_eq- A string identifying the equation that should be used
            in calculating the decay of stressor buffer influence.
        buffer_dict- A dictionary that holds desired buffer sizes for each
            stressors. The key is the name of the stressor, and the value is an
            int which correlates to desired buffer size.
        grid_path- A string for a raster file path on disk. Used as a
            universal base raster to create other rasters which to burn
            vectors onto.

    Output:
        A potentially buffered and rasterized version of each stressor
            shapefile provided, which will be stored in 'dir'.

    Returns:
        stress_dict- A simple dictionary which maps a string key of the
            stressor name to the URI for the output raster.

    '''

    stress_dict = {}

    for shape in stress_list:

        # The return of os.path.split is a tuple where everything after the
        # final slash is returned as the 'tail' in the second element of the
        # tuple path.splitext returns a tuple such that the first element is
        # what comes before the file extension, and the second is the extension
        # itself
        name = os.path.splitext(os.path.split(shape)[1])[0]
        out_uri = os.path.join(dir, name + '.tif')

        buff = buffer_dict[name]

        # Want to set this specifically to make later overlap easier.
        nodata = -1.

        # Create a base raster for burning vectors onto from a raster that
        # was set up such that each shapefile is burned onto a consistant
        # grid, ensuring alignment.
        pygeoprocessing.geoprocessing.new_raster_from_base_uri(
            grid_path, out_uri, 'GTiff', -1., gdal.GDT_Float32, fill_value=nodata)

        # Burn polygon land values onto newly constructed raster
        pygeoprocessing.geoprocessing.rasterize_layer_uri(
            out_uri, shape, burn_values=[1], option_list=['ALL_TOUCHED=TRUE'])

        nodata_to_zero_uri = pygeoprocessing.geoprocessing.temporary_filename()

        def nodata_to_zero(chunk):
            """vectorize_dataset operation to turn nodata values
                 to 0s

                chunk - a numpy array

                returns - a numpy array
            """
            return np.where(chunk == nodata, 0., chunk)

        cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(out_uri)
        # Convert nodata values to 0 to prep for distance transform
        pygeoprocessing.geoprocessing.vectorize_datasets(
            [out_uri],
            nodata_to_zero,
            nodata_to_zero_uri,
            gdal.GDT_Float32,
            nodata,
            cell_size,
            "intersection",
            vectorize_op=False)

        dist_trans_uri = pygeoprocessing.geoprocessing.temporary_filename()
        pygeoprocessing.geoprocessing.distance_transform_edt(
            nodata_to_zero_uri, dist_trans_uri)

        new_buff_uri = os.path.join(dir, name + '_buff.tif')

        # Do buffering protocol specified
        if buff == 0:
            make_zero_buff_decay_array(dist_trans_uri, new_buff_uri, nodata)
        elif decay_eq == 'None':
            make_no_decay_array(dist_trans_uri, new_buff_uri, buff, nodata)
        elif decay_eq == 'Exponential':
            make_exp_decay_array(dist_trans_uri, new_buff_uri, buff, nodata)
        elif decay_eq == 'Linear':
            make_lin_decay_array(dist_trans_uri, new_buff_uri, buff, nodata)

        # Now, write the buffered version of the stressor to the stressors
        # dictionary
        stress_dict[name] = new_buff_uri

    return stress_dict

def make_zero_buff_decay_array(dist_trans_uri, out_uri, nodata):
    '''
    Creates a raster in the case of a zero buffer width, where we should
    have is land and nodata values.

    Input:
        dist_trans_uri- uri to a gdal raster where each pixel value represents
            the distance to the closest piece of land.
        out_uri- uri for the gdal raster output with the buffered outputs
        nodata- The value which should be placed into anything that is not
            land.
    Returns: Nothing
    '''

    def zero_buff_op(chunk):
        """vecorize_dataset operation to replace 0s with 1s
            and mask out anything not land

            chunk - numpy block

            returns - numpy array with buffered values
        """
        # Since we know anything that is land is currently represented as 0's,
        # want to turn that back into 1's. everything else will just be nodata
        return np.where(chunk == 0., 1, nodata)

    cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(dist_trans_uri)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [dist_trans_uri],
        zero_buff_op,
        out_uri,
        gdal.GDT_Float32,
        nodata,
        cell_size,
        "intersection",
        vectorize_op=False)


def make_lin_decay_array(dist_trans_uri, out_uri, buff, nodata):
    '''
    Should create a raster where the area around land is a function of
    linear decay from the values representing the land.

    Input:
        dist_trans_uri- uri to a gdal raster where each pixel value represents
            the distance to the closest piece of land.
        out_uri- uri for the gdal raster output with the buffered outputs
        buff- The distance surrounding the land that the user desires to buffer
            with linearly decaying values.
        nodata- The value which should be placed into anything not land or
            buffer area.
    Returns: Nothing
    '''
    buff = float(buff)
    cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(dist_trans_uri)

    def lin_decay_op(chunk):
        """vecorize_dataset operation to create a buffer around
            land masses based on a linear rate of decay

            chunk - numpy block

            returns - numpy array with buffered values
        """
        chunk_met = chunk * cell_size
        # The decay rate should be approximately -1/distance we want 0 to be at
        # We add one to have a proper y-intercept.
        lin_decay_chunk = -chunk_met / buff + 1.0
        return np.where(lin_decay_chunk < 0.0, nodata, lin_decay_chunk)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [dist_trans_uri],
        lin_decay_op,
        out_uri,
        gdal.GDT_Float32,
        nodata,
        cell_size,
        "intersection",
        vectorize_op=False)


def make_exp_decay_array(dist_trans_uri, out_uri, buff, nodata):
    '''
    Should create a raster where the area around the land is a function of
    exponential decay from the land values.

    Input:
        dist_trans_uri- uri to a gdal raster where each pixel value represents
            the distance to the closest piece of land.
        out_uri- uri for the gdal raster output with the buffered outputs
        buff- The distance surrounding the land that the user desires to buffer
            with exponentially decaying values.
        nodata- The value which should be placed into anything not land or
            buffer area.
    Returns: Nothing
    '''

    # Want a cutoff for the decay amount after which we will say things are
    # equivalent to nodata, since we don't want to have values outside the
    # buffer zone.
    cutoff = 0.01

    # Need a value representing the decay rate for the exponential decay
    rate = -math.log(cutoff) / buff
    cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(dist_trans_uri)

    def exp_decay_op(chunk):
        """vecorize_dataset operation to create a buffer around
            land masses based on an exponential rate of decay

            chunk - numpy block

            returns - numpy array with buffered values
        """
        chunk_met = chunk * cell_size
        exp_decay_chunk = np.exp(-rate * chunk_met)
        return np.where(exp_decay_chunk < cutoff, nodata, exp_decay_chunk)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [dist_trans_uri],
        exp_decay_op,
        out_uri,
        gdal.GDT_Float32,
        nodata,
        cell_size,
        "intersection",
        vectorize_op=False)


def make_no_decay_array(dist_trans_uri, out_uri, buff, nodata):
    '''
    Should create a raster where the buffer zone surrounding the land is
    buffered with the same values as the land, essentially creating an equally
    weighted larger landmass.

    Input:
        dist_trans_uri- uri to a gdal raster where each pixel value represents
            the distance to the closest piece of land.
        out_uri- uri for the gdal raster output with the buffered outputs
        buff- The distance surrounding the land that the user desires to buffer
            with land data values.
        nodata- The value which should be placed into anything not land or
            buffer area.
    Returns: Nothing
    '''

    buff = float(buff)
    cell_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(dist_trans_uri)

    def no_decay_op(chunk):
        """vecorize_dataset operation to create a constant buffer

            chunk - numpy block

            returns - numpy array with buffered values
        """
        chunk_met = chunk * cell_size
        # Setting anything within the buffer zone to 1, and anything outside
        # that distance to nodata.
        return np.where(chunk_met <= buff, 1., nodata)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [dist_trans_uri],
        no_decay_op,
        out_uri,
        gdal.GDT_Float32,
        nodata,
        cell_size,
        "intersection",
        vectorize_op=False)

def add_hab_rasters(dir, habitats, hab_list, grid_size, grid_path):
    '''
    Want to get all shapefiles within any directories in hab_list, and burn
    them to a raster.

    Input:
        dir- Directory into which all completed habitat rasters should be
            placed.
        habitats- A multi-level dictionary containing all habitat and
            species-specific criteria ratings and rasters.
        hab_list- File URI's for all shapefile in habitats dir, species dir, or
            both.
        grid_size- Int representing the desired pixel dimensions of
            both intermediate and ouput rasters.
        grid_path- A string for a raster file path on disk. Used as a
            universal base raster to create other rasters which to burn
            vectors onto.

    Output:
        A modified version of habitats, into which we have placed the URI to
            the rasterized version of the habitat shapefile. It will be placed
            at habitats[habitatName]['DS'].
   '''

    for shape in hab_list:

        # The return of os.path.split is a tuple where everything after the
        # final slash is returned as the 'tail' in the second element of the
        # tuple path.splitext returns a tuple such that the first element is
        # what comes before the file extension, and the second is the extension
        # itself
        name = os.path.splitext(os.path.split(shape)[1])[0]

        out_uri = os.path.join(dir, name + '.tif')
        # Create a base raster for burning vectors onto from a raster that
        # was set up such that each shapefile is burned onto a consistent
        # grid, ensuring alignment.
        pygeoprocessing.geoprocessing.new_raster_from_base_uri(
            grid_path, out_uri, 'GTiff', -1., gdal.GDT_Float32, fill_value=-1.)

        pygeoprocessing.geoprocessing.rasterize_layer_uri(
            out_uri, shape, burn_values=[1], option_list=['ALL_TOUCHED=TRUE'])

        habitats[name]['DS'] = out_uri

def calc_max_rating(risk_eq, max_rating):
    '''
    Should take in the max possible risk, and return the highest possible
    per pixel risk that would be seen on a H-S raster pixel.

    Input:
        risk_eq- The equation that will be used to determine risk.
        max_rating- The highest possible value that could be given as a
            criteria rating, data quality, or weight.

    Returns:
        An int representing the highest possible risk value for any given h-s
        overlap raster.
    '''

    # The max_rating ends up being the simplified result of each of the E and
    # C equations when the same value is used in R/DQ/W. Thus for E and C, their
    # max value is equivalent to the max_rating.

    if risk_eq == 'Multiplicative':
        max_r = max_rating * max_rating

    elif risk_eq == 'Euclidean':
        under_rt = (max_rating - 1)**2 + (max_rating - 1)**2
        max_r = math.sqrt(under_rt)

    return max_r


def listdir(path):
    '''
    A replacement for the standar os.listdir which, instead of returning
    only the filename, will include the entire path. This will use os as a
    base, then just lambda transform the whole list.

    Input:
        path- The location container from which we want to gather all files.

    Returns:
        A list of full URIs contained within 'path'.
    '''
    LOGGER.debug("PATH: %s", path)
    file_names = os.listdir(path)
    uris = map(lambda x: os.path.join(path, x), file_names)

    return uris


def add_crit_rasters(dir, crit_dict, habitats, h_s_e, h_s_c, grid_size):
    '''
    This will take in the dictionary of criteria shapefiles, rasterize them,
    and add the URI of that raster to the proper subdictionary within h/s/h-s.

    Input:
        dir- Directory into which the raserized criteria shapefiles should be
            placed.
        crit_dict- A multi-level dictionary of criteria shapefiles. The
            outermost keys refer to the dictionary they belong with. The
            structure will be as follows:

            {'h':
                {'HabA':
                    {'CriteriaName: "Shapefile Datasource URI"...}, ...
                },
             'h_s_c':
                {('HabA', 'Stress1'):
                    {'CriteriaName: "Shapefile Datasource URI", ...}, ...
                },
             'h_s_e'
                {('HabA', 'Stress1'):
                    {'CriteriaName: "Shapefile Datasource URI", ...}, ...
                }
            }
        h_s_c- A multi-level structure which holds numerical criteria
            ratings, as well as weights and data qualities for criteria
            rasters. h-s will hold only criteria that apply to habitat and
            stressor overlaps. The structure's outermost keys are tuples of
            (Habitat, Stressor) names. The overall structure will be as
            pictured:

            {(Habitat A, Stressor 1):
                    {'Crit_Ratings':
                        {'CritName':
                            {'Rating': 2.0, 'DQ': 1.0, 'Weight': 1.0}
                        },
                    'Crit_Rasters':
                        {'CritName':
                            {'Weight': 1.0, 'DQ': 1.0}
                        },
                    },
                    'DS': "HabitatStressor Raster URI"
            }
        habitats- Similar to the h-s dictionary, a multi-level
            dictionary containing all habitat-specific criteria ratings and
            raster information. The outermost keys are habitat names. Within
            the dictionary, the habitats['habName']['DS'] will be the URI of
            the raster of that habitat.
        h_s_e- Similar to the h-s dictionary, a multi-level dictionary
            containing all stressor-specific criteria ratings and
            raster information. The outermost keys are tuples of
            (Habitat, Stressor) names.
        grid_size- An int representing the desired pixel size for the criteria
            rasters.
    Output:
        A set of rasterized criteria files. The criteria shapefiles will be
            burned based on their 'Rating' attribute. These will be placed in
            the 'dir' folder.

        An appended version of habitats, h_s_e, and h_s_c which will include
        entries for criteria rasters at 'Rating' in the appropriate dictionary.
        'Rating' will map to the URI of the corresponding criteria dataset.

    Returns nothing.
    '''
    # H-S-C
    for pair in crit_dict['h_s_c']:

        for c_name, c_path in crit_dict['h_s_c'][pair].iteritems():

            # The path coming in from the criteria should be of the form
            # dir/h_s_critname.shp.
            filename = os.path.splitext(os.path.split(c_path)[1])[0]
            shape = ogr.Open(c_path)
            layer = shape.GetLayer()

            # Since all features will contain the same set of attributes,
            # and if it passes this loop, will definitely contain a 'rating', we
            # can just use the last feature queried to figure out how 'rating'
            # was used.
            lower_attrib = None

            for feature in layer:

                if lower_attrib is None:
                    lower_attrib = dict(
                        zip(map(lambda x: x.lower(), feature.items().keys()),
                            feature.items().keys()))

                if 'rating' not in lower_attrib:
                    raise ImproperCriteriaAttributeName("Criteria layer must \
                        contain the attribute \"Rating\" in order to be \
                        properly used within the HRA model run.")

            out_uri_pre_overlap = os.path.join(
                dir, filename + '_pre_overlap.tif')

            pygeoprocessing.geoprocessing.create_raster_from_vector_extents_uri(
                c_path, grid_size, gdal.GDT_Int32, -1, out_uri_pre_overlap)

            pygeoprocessing.geoprocessing.rasterize_layer_uri(
                out_uri_pre_overlap, c_path,
                option_list=['ATTRIBUTE=' + lower_attrib[
                             'rating'], 'ALL_TOUCHED=TRUE'])

            # Want to do a vectorize with the base layer, to make sure we're not
            # adding in values where there should be none
            base_uri = h_s_c[pair]['DS']
            base_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(base_uri)

            def overlap_hsc_spat_crit(base_pix, spat_pix):

                # If there is no overlap, just return whatever is underneath.
                # It will either be the value of that patial region or nodata
                return np.where(
                    (base_pix != base_nodata), spat_pix, base_nodata)

            out_uri = os.path.join(dir, filename + '.tif')

            pygeoprocessing.geoprocessing.vectorize_datasets(
                [base_uri, out_uri_pre_overlap],
                overlap_hsc_spat_crit,
                out_uri,
                gdal.GDT_Float32,
                -1.,
                grid_size,
                "union",
                resample_method_list=None,
                dataset_to_align_index=0,
                aoi_uri=None,
                vectorize_op=False)

            if c_name in h_s_c[pair]['Crit_Rasters']:
                h_s_c[pair]['Crit_Rasters'][c_name]['DS'] = out_uri
            else:
                raise DQWeightNotFound("All spatial criteria desired within \
                    the model run require corresponding Data Quality and \
                    Weight information. Please run HRA Preprocessor again to \
                    include all relavant criteria data.")

    # Habs
    for h in crit_dict['h']:

        for c_name, c_path in crit_dict['h'][h].iteritems():

            # The path coming in from the criteria should be of the form
            # dir/h_critname.shp.
            filename = os.path.splitext(os.path.split(c_path)[1])[0]
            shape = ogr.Open(c_path)
            layer = shape.GetLayer()

            # Since all features will contain the same set of attributes,
            # and if it passes this loop, will definitely contain a 'rating', we
            # can just use the last feature queried to figure out how 'rating'
            # was used.
            lower_attrib = None

            for feature in layer:

                if lower_attrib is None:
                    lower_attrib = dict(zip(
                        map(lambda x: x.lower(), feature.items().keys()),
                        feature.items().keys()))

                if 'rating' not in lower_attrib:
                    raise ImproperCriteriaAttributeName("Criteria layer must \
                        contain the attribute \"Rating\" in order to be \
                        properly used within the HRA model run.")

            out_uri_pre_overlap = os.path.join(
                dir, filename + '_pre_overlap.tif')

            pygeoprocessing.geoprocessing.create_raster_from_vector_extents_uri(
                c_path, grid_size, gdal.GDT_Int32, -1, out_uri_pre_overlap)

            pygeoprocessing.geoprocessing.rasterize_layer_uri(
                out_uri_pre_overlap, c_path,
                option_list=['ATTRIBUTE=' + lower_attrib['rating'],
                             'ALL_TOUCHED=TRUE'])

            # Want to do a vectorize with the base layer to make sure we're not
            # adding in values where there should be none
            base_uri = habitats[h]['DS']
            base_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(base_uri)

            def overlap_h_spat_crit(base_pix, spat_pix):

                # If there is no overlap, just return whatever is underneath.
                # It will either be the value of that patial region or nodata
                return np.where(
                    (base_pix != base_nodata), spat_pix, base_nodata)

            out_uri = os.path.join(dir, filename + '.tif')

            pygeoprocessing.geoprocessing.vectorize_datasets(
                [base_uri, out_uri_pre_overlap],
                overlap_h_spat_crit,
                out_uri,
                gdal.GDT_Float32,
                -1.,
                grid_size,
                "union",
                resample_method_list=None,
                dataset_to_align_index=0, aoi_uri=None,
                vectorize_op=False)

            if c_name in habitats[h]['Crit_Rasters']:
                habitats[h]['Crit_Rasters'][c_name]['DS'] = out_uri
            else:
                raise DQWeightNotFound("All spatial criteria desired within \
                    the model run require corresponding Data Quality and \
                    Weight information. Please run HRA Preprocessor again to \
                    include all relavant criteria data.")
    # H-S-E
    for pair in crit_dict['h_s_e']:

        for c_name, c_path in crit_dict['h_s_e'][pair].iteritems():

            # The path coming in from the criteria should be of the form
            # dir/h_s_critname.shp.
            filename = os.path.splitext(os.path.split(c_path)[1])[0]
            shape = ogr.Open(c_path)
            layer = shape.GetLayer()

            # Since all features will contain the same set of attributes,
            # and if it passes this loop, will definitely contain a 'rating', we
            # can just use the last feature queried to figure out how 'rating'
            # was used.
            lower_attrib = None

            for feature in layer:

                if lower_attrib is None:
                    lower_attrib = dict(zip(
                        map(lambda x: x.lower(), feature.items().keys()),
                        feature.items().keys()))

                if 'rating' not in lower_attrib:
                    raise ImproperCriteriaAttributeName("Criteria layer must \
                        contain the attribute \"Rating\" in order to be \
                        properly used within the HRA model run.")

            out_uri_pre_overlap = os.path.join(
                dir, filename + '_pre_overlap.tif')

            pygeoprocessing.geoprocessing.create_raster_from_vector_extents_uri(
                c_path, grid_size, gdal.GDT_Int32, -1, out_uri_pre_overlap)

            pygeoprocessing.geoprocessing.rasterize_layer_uri(
                out_uri_pre_overlap, c_path,
                option_list=['ATTRIBUTE=' + lower_attrib['rating'],
                             'ALL_TOUCHED=TRUE'])

            # Want to do a vectorize with the base layer, to make sure we're not
            # adding in values where there should be none
            base_uri = h_s_e[pair]['DS']
            base_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(base_uri)

            def overlap_hse_spat_crit(base_pix, spat_pix):

                # If there is no overlap, just return whatever is underneath.
                # It will either be the value of that patial region or nodata
                return np.where(
                    (base_pix != base_nodata), spat_pix, base_nodata)

            out_uri = os.path.join(dir, filename + '.tif')

            pygeoprocessing.geoprocessing.vectorize_datasets(
                [base_uri, out_uri_pre_overlap],
                overlap_hse_spat_crit,
                out_uri,
                gdal.GDT_Float32,
                -1.,
                grid_size,
                "union",
                resample_method_list=None,
                dataset_to_align_index=0, aoi_uri=None,
                vectorize_op=False)

            if c_name in h_s_e[pair]['Crit_Rasters']:
                h_s_e[pair]['Crit_Rasters'][c_name]['DS'] = out_uri
            else:
                raise DQWeightNotFound("All spatial criteria desired within \
                    the model run require corresponding Data Quality and \
                    Weight information. Please run HRA Preprocessor again to \
                    include all relavant criteria data.")


def unpack_over_dict(csv_uri, args):
    '''
    This throws the dictionary coming from the pre-processor into the
    equivalent dictionaries in args so that they can be processed before being
    passed into the core module.

    Input:
        csv_uri- Reference to the folder location of the CSV tables containing
            all habitat and stressor rating information.
        args- The dictionary into which the individual ratings dictionaries
            should be placed.
    Output:
        A modified args dictionary containing dictionary versions of the CSV
        tables located in csv_uri. The dictionaries should be of the forms as
        follows.

        h_s_c- A multi-level structure which will hold all criteria ratings,
            both numerical and raster that apply to habitat and stressor
            overlaps. The structure, whose keys are tuples of
            (Habitat, Stressor) names and map to an inner dictionary will have
            2 outer keys containing numeric-only criteria, and raster-based
            criteria. At this time, we should only have two entries in a
            criteria raster entry, since we have yet to add the rasterized
            versions of the criteria.

            {(Habitat A, Stressor 1):
                    {'Crit_Ratings':
                        {'CritName':
                            {'Rating': 2.0, 'DQ': 1.0, 'Weight': 1.0}
                        },
                    'Crit_Rasters':
                        {'CritName':
                            {'Weight': 1.0, 'DQ': 1.0}
                        },
                    }
            }
        habitats- Similar to the h-s dictionary, a multi-level
            dictionary containing all habitat-specific criteria ratings and
            weights and data quality for the rasters.
        h_s_e- Similar to the h-s dictionary, a multi-level dictionary
            containing habitat stressor-specific criteria ratings and
            weights and data quality for the rasters.
    Returns nothing.
    '''
    dicts = hra_preprocessor.parse_hra_tables(csv_uri)

    for dict_name in dicts:
        args[dict_name] = dicts[dict_name]
