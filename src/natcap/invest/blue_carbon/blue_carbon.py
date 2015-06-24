"""
"""
from osgeo import gdal, ogr, osr
#gdal.UseExceptions()
from pygeoprocessing import geoprocessing as raster_utils

import logging
import pprint as pp

import blue_carbon_io as io
import blue_carbon_model as model

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.blue_carbon.blue_carbon')


def execute(args):
    """Entry point for the blue carbon model.

    :param args["workspace_dir"]: The directory to hold output from a particular model run
    :type args["workspace_dir"]: str
    :param args["lulc_uri_1"]: The land use land cover raster for time 1.
    :type args["lulc_uri_1"]: str
    :param args["year_1"]: The year for the land use land cover raster for time 1.
    :type args["year_1"]: int
    :param args["lulc_uri_2"]: The land use land cover raster for time 2.
    :type args["lulc_uri_2"]: str
    :param args["year_2"]: The year for the land use land cover raster for time 2.
    :type args["year_2"]: int
    :param args["lulc_uri_3"]: The land use land cover raster for time 3.
    :type args["lulc_uri_3"]: str
    :param args["year_3"]: The year for the land use land cover raster for time 3.
    :type args["year_3"]: int
    :param args["lulc_uri_4"]: The land use land cover raster for time 4.
    :type args["lulc_uri_4"]: str
    :param args["year_4"]: The year for the land use land cover raster for time 4.
    :type args["year_4"]: int
    :param args["lulc_uri_5"]: The land use land cover raster for time 5.
    :type args["lulc_uri_5"]: str
    :param args["year_5"]: The year for the land use land cover raster for time 5.
    :type args["year_5"]: int

    """
    vars_dict = io.fetch_args(args)

    # with open('output.txt', 'wt') as out:
    #     pp.pprint(vars_dict, stream=out)
    # return

    # Biophysical Component
    vars_dict = model.run_biophysical(vars_dict)

    # Valuation Component

    if args["do_private_valuation"]:
        model.run_valuation(vars_dict)
