"""Carbon Storage and Sequestration."""
from __future__ import absolute_import
import logging
import os

from osgeo import gdal
from osgeo import ogr

from . import validation
from . import utils

LOGGER = logging.getLogger(__name__)


def execute(args):
    """InVEST Urban Flood Risk Mitigation.

    The model computes the peak flow attenuation for each pixel, delineates
    areas benefiting from this service, then calculates the monetary value of
    potential avoided damage to built infrastructure.

    Parameters:
        args['workspace_dir'] (string): a path to the directory that will
            write output and other temporary files during calculation.
        args['results_suffix'] (string): appended to any output file name.
        args['dem_path'] (string): path to the DEM that will be used to
            delineate watersheds.
        args['aoi_watersheds_path'] (string): path to a shapefile of
            (sub)watersheds or sewersheds used to indicate spatial area of
            interest.
        args['rainfall_depth'] (float): depth of rainfall in mm.
        args['lulc_path'] (string): path to a landcover raster.
        args['soils_hydrological_group_raster_path'] (string): Raster with
            values equal to 1, 2, 3, 4, corresponding to soil hydrologic group
            A, B, C, or D, respectively (used to derive the CN number).
        args['curve_number_table_path'] (string): path to a CSV table that
            contains at least the headers 'lucode', 'CN_A', 'CN_B', 'CN_C',
            'CN_D'.
        args['flood_prone_areas_vector_path'] (string): path to vector of
            polygon areas of known occurrence of flooding where peakflow
            retention will be more critical.
        args['built_infrastructure_vector_path'] (string): path to a vector
            with built infrastructure footprints. Attribute table contains a
            column 'Type' with integers (e.g. 1=residential, 2=office, etc.).
        args['infrastructure_damage_loss_table_path'] (string): path to a
            a CSV table with columns 'Type' and 'Damage' with values of built
            infrastructure type from the 'Type' field in
            `args['built_infrastructure_vector_path']` and potential damage
            loss (in $/m^2).

    Returns:
        None.

    """
    pass


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to `execute`'s contract.

    Parameters:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in `execute` docstring.
        limit_to (str): (optional) if not None indicates that validation
            should only occur on the args[limit_to] value. The intent that
            individual key validation could be significantly less expensive
            than validating the entire `args` dictionary.

    Returns:
        list of ([invalid key_a, invalid_keyb, ...], 'warning/error message')
            tuples. Where an entry indicates that the invalid keys caused
            the error message in the second part of the tuple. This should
            be an empty list if validation succeeds.
    """
    missing_key_list = []
    no_value_list = []
    validation_error_list = []

    required_keys = [
        'workspace_dir',
        ]

    for key in required_keys:
        if limit_to is None or limit_to == key:
            if key not in args:
                missing_key_list.append(key)
            elif args[key] in ['', None]:
                no_value_list.append(key)

    if len(missing_key_list) > 0:
        # if there are missing keys, we have raise KeyError to stop hard
        raise KeyError(
            "The following keys were expected in `args` but were missing " +
            ', '.join(missing_key_list))

    if len(no_value_list) > 0:
        validation_error_list.append(
            (no_value_list, 'parameter has no value'))

    file_type_list = [
        ]

    # check that existing/optional files are the correct types
    with utils.capture_gdal_logging():
        for key, key_type in file_type_list:
            if ((limit_to is None or limit_to == key) and
                    key in args and key in required_keys):
                if not os.path.exists(args[key]):
                    validation_error_list.append(
                        ([key], 'not found on disk'))
                    continue
                if key_type == 'raster':
                    raster = gdal.Open(args[key])
                    if raster is None:
                        validation_error_list.append(
                            ([key], 'not a raster'))
                    del raster
                elif key_type == 'vector':
                    vector = ogr.Open(args[key])
                    if vector is None:
                        validation_error_list.append(
                            ([key], 'not a vector'))
                    del vector

    return validation_error_list
