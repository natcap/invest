"""
Recreation server intial run.
"""
import sys
import os
import copy
import logging
import json
import subprocess
import datetime
import traceback

import psycopg2
from osgeo import osr
from osgeo import ogr

import recreation_server_core as rs_core

#logging.basic_config(format = '%(asctime)s %(name)-20s %(levelname)-8s \
#%(message)s', level = logging.DEBUG, datefmt = '%m/%d/%Y %H:%M:%S ')
#
#LOGGER = logging.get_logger('recreation')


def execute(args, config):
    """
    This function invokes the recreation model.

    :param string args['aoi_file_name']: a shapefile for the area of interest
        (required)
    :param float? args['cell_size']: the cell size for the grid in meters
        (required)
    :param string args['grid_file_name']: a shapefile for the area of
        interest (required)
    :param string args['flickr_file_name']: a path for the Flickr summary
    :param float? config['min_cell_size']: the minimum cell size allowed in a
        grid
    :param float? config['max_grid_size']: the maximum total size for a grid

    Returns:
        None

    Example Args::

        args = {
            'aoi_file_name': 'filename',
            'cell_size': '1',
            'grid_file_name': 'filename',
            'flickr_file_name': 'filename',
        }

        config = {
            'min_cell_size': '1.2',
            'max_grid_size': '3.4',
        }
    """
    LOGGER.debug("Beginning execute(args).")

    geometry_column_name = "way"
    grid_column_name = "cell"

    standard_predictors = [
        config["postgis"]["table"]["names"]["landscan_name"],
        config["postgis"]["table"]["names"]["point_name"],
        config["postgis"]["table"]["names"]["line_name"],
        config["postgis"]["table"]["names"]["poly_name"],
        config["postgis"]["table"]["names"]["protected_name"],
        config["postgis"]["table"]["names"]["lulc_1_name"],
        config["postgis"]["table"]["names"]["lulc_2_name"],
        config["postgis"]["table"]["names"]["lulc_3_name"],
        config["postgis"]["table"]["names"]["lulc_4_name"],
        config["postgis"]["table"]["names"]["lulc_5_name"],
        config["postgis"]["table"]["names"]["lulc_6_name"],
        config["postgis"]["table"]["names"]["lulc_7_name"],
        config["postgis"]["table"]["names"]["lulc_8_name"],
        config["postgis"]["table"]["names"]["mangrove_name"],
        config["postgis"]["table"]["names"]["reef_name"],
        config["postgis"]["table"]["names"]["seagrass_name"]]

    simple_predictors = [
        config["postgis"]["table"]["names"]["landscan_name"],
        config["postgis"]["table"]["names"]["protected_name"],
        config["postgis"]["table"]["names"]["lulc_1_name"],
        config["postgis"]["table"]["names"]["lulc_2_name"],
        config["postgis"]["table"]["names"]["lulc_3_name"],
        config["postgis"]["table"]["names"]["lulc_4_name"],
        config["postgis"]["table"]["names"]["lulc_5_name"],
        config["postgis"]["table"]["names"]["lulc_6_name"],
        config["postgis"]["table"]["names"]["lulc_7_name"],
        config["postgis"]["table"]["names"]["lulc_8_name"],
        config["postgis"]["table"]["names"]["mangrove_name"],
        config["postgis"]["table"]["names"]["reef_name"],
        config["postgis"]["table"]["names"]["seagrass_name"]]

    column_alias = {
        config["postgis"]["table"]["names"]
              ["landscan_name"]: "landscan",
        config["postgis"]["table"]["names"]
              ["protected_name"]: "protected",
        config["postgis"]["table"]["names"]
              ["mangrove_name"]: "mangrove",
        config["postgis"]["table"]["names"]
              ["reef_name"]: "reef",
        config["postgis"]["table"]["names"]
              ["seagrass_name"]: "seagrass"}

    compound_predictors = [
        config["postgis"]["table"]["names"]["point_name"],
        config["postgis"]["table"]["names"]["line_name"],
        config["postgis"]["table"]["names"]["poly_name"]]

    compound_predictor_classes = [4, 4, 4, 8]

    osm_srid = 900913
    flickr_srid = 4326

    predictor_srid = {}
    predictor_srid[config["postgis"]["table"]["names"]
                   ["flickr_name"]] = flickr_srid

    predictor_srid[config["postgis"]["table"]["names"]
                   ["point_name"]] = osm_srid
    predictor_srid[config["postgis"]["table"]["names"]
                   ["line_name"]] = osm_srid
    predictor_srid[config["postgis"]["table"]["names"]
                   ["poly_name"]] = osm_srid

    predictor_srid[config["postgis"]["table"]["names"]
                   ["landscan_name"]] = 4326
    predictor_srid[config["postgis"]["table"]["names"]
                   ["lulc_1_name"]] = 4326
    predictor_srid[config["postgis"]["table"]["names"]
                   ["lulc_2_name"]] = 4326
    predictor_srid[config["postgis"]["table"]["names"]
                   ["lulc_3_name"]] = 4326
    predictor_srid[config["postgis"]["table"]["names"]
                   ["lulc_4_name"]] = 4326
    predictor_srid[config["postgis"]["table"]["names"]
                   ["lulc_5_name"]] = 4326
    predictor_srid[config["postgis"]["table"]["names"]
                   ["lulc_6_name"]] = 4326
    predictor_srid[config["postgis"]["table"]["names"]
                   ["lulc_7_name"]] = 4326
    predictor_srid[config["postgis"]["table"]["names"]
                   ["lulc_8_name"]] = 4326

    predictor_srid[config["postgis"]["table"]["names"]
                   ["mangrove_name"]] = 4326
    predictor_srid[config["postgis"]["table"]["names"]
                   ["reef_name"]] = 4326
    predictor_srid[config["postgis"]["table"]["names"]
                   ["seagrass_name"]] = 4326
    predictor_srid[config["postgis"]["table"]["names"]
                   ["protected_name"]] = 4326

    LOGGER.debug("Parsing database connection string.")
    #database definitions
    dbase_file_name = os.path.join(
        os.path.abspath(os.path.dirname(sys.argv[0])),
        "postgis.db")
    dbase_file = open(dbase_file_name, "r")
    dbase_post_gis = dbase_file.read().strip().replace("\n", " ")
    dbase_file.close()

    LOGGER.debug("Assigning user parameters locally.")
    #parameters
    aoi_file_name = str(args["aoi_file_name"])

    LOGGER.debug(
        "Getting linear units of shapefile %s." % repr(aoi_file_name).replace(".","||"))
    datasource = ogr.Open(aoi_file_name)
    layer = datasource.GetLayer()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(layer.GetSpatialRef().ExportToWkt())
    linear_units = srs.GetLinearUnits()
    srs = None
    layer = None
    datasource = None

    LOGGER.info(
        "The map unit coversion to meters is %s." % str(linear_units).replace(".","||"))

    units = {
        'kilometre': 1000,
        #'Gold Coast foot': 0,
        'German legal metre': 1.0000135965,
        #'degree': 0,
        #'British yard (Sears 1922)':,
        #"Clarke''s link":,
        #'British foot (Sears 1922)':,
        'metre': 1,
        'Meter': 1,
        #'Indian yard':,
        #'British chain (Sears 1922 truncated)':,
        'US survey foot': 1200./3937,
        #'British chain (Benoit 1895 B)':,
        #'British chain (Sears 1922)':,
        'foot': 0.3048
        #"Clarke''s foot":,
        #'grad':,
        #'link':
        }

    if args["grid"]:
        cell_size = args["cell_size"]

        if (cell_size * linear_units) < config["min_cell_size"]:
            LOGGER.info("Your sepecified cell size in meters is approximately %i." % (cell_size * linear_units))
            err_msg = ("The cell size of %i is less than the minimum "
                       "allowed size of %i.") % (cell_size, config["min_cell_size"])
            LOGGER.error(err_msg)
            raise ValueError(err_msg)
    else:
        LOGGER.debug("The customized grid will be checked for cell size after insertion.")

    grid_file_name = args["grid_file_name"]
    flickr_file_name = args["flickr_file_name"]
    data_dir = args["data_dir"]
    landscan = args["landscan"]
    osm_point = args["osm_point"]
    osm_line = args["osm_line"]
    osm_poly = args["osm_poly"]
    protected = args["protected"]
    mangroves = args["mangroves"]
    reefs = args["reefs"]
    grass = args["grass"]

    standard_mask = [
        landscan,
        osm_point,
        osm_line,
        osm_poly,
        protected,
        args["lulc_1"],
        args["lulc_2"],
        args["lulc_3"],
        args["lulc_4"],
        args["lulc_5"],
        args["lulc_6"],
        args["lulc_7"],
        args["lulc_8"],
        mangroves,
        reefs,
        grass]

    LOGGER.debug("Stadard mask: %s", repr(standard_mask).replace(", ", "|"))

    LOGGER.debug("Setting intermediate table suffixes.")
    #intermediate table names
    aoi_name = "aoi"
    aoi4326_name = "%s_4326" % aoi_name
    aoi_union_name = "%s_union" % aoi_name
    aoi_transformed_name = "prj"
    grid_name = "%s_grid" % aoi_transformed_name
    grid_union_name = "%s_union" % grid_name
    results_name = "results"
    join_name = "results"

    #column names
    result_column = "result"

    #suffixes
    union = "_union"
    clip = "_clip"
    results = "_result"
    category = "_user_category"

    #table names
    standard_class_format = "%s_class"
    standard_category_format = "%s_category"
    class_format = "%s_user_class"
    category_format = "%s" + category
    union_format = "%s" + union
    clip_format = "%s" + clip
    projected_format = "%s_%i"
    results_format = "%s" + results

    try:
        database = psycopg2.connect(dbase_post_gis)
        cur = database.cursor()

        #processing user provided predictors
        #create list from directory
        user_simple_predictors = []
        user_compound_predictors = []
        if not data_dir == "":
            for file_path in os.listdir(data_dir):
                file_name, file_extension = os.path.splitext(file_path)
                if file_extension == ".shp":
                    if os.path.exists(data_dir+file_name+".shx") and \
                    os.path.exists(data_dir+file_name+".shp") and \
                    os.path.exists(data_dir+file_name+".prj"):
                        if os.path.exists(data_dir+file_name+".tsv"):
                            LOGGER.info("Found compound %s predictor.", file_name)
                            user_compound_predictors.append(file_name)
                        else:
                            LOGGER.info("Found %s predictor.", file_name)
                            user_simple_predictors.append(file_name)
                    else:
                        LOGGER.info("Predictor %s is missing file(s).", data_dir+file_name)
                        LOGGER.error("Predictor %s is missing file(s).", file_name)
                        raise ValueError("Predictor %s is missing file(s)." % (file_name))

        #creating list of model predictors
        model_simple_predictors = copy.copy(user_simple_predictors)
        model_compound_predictors = copy.copy(user_compound_predictors)
        LOGGER.debug("Standard predictor mask %s.", repr(
            standard_mask).replace(", ", "|"))
        for include, predictor in zip(standard_mask, standard_predictors):
            if include:
                try:
                    simple_predictors.index(predictor)
                    LOGGER.info("Incuding simple predictor %s.", predictor)
                    model_simple_predictors.append(predictor)
                except ValueError:
                    LOGGER.info("Incuding compound predictor %s.", predictor)
                    model_compound_predictors.append(predictor)

        LOGGER.info(
            "Checking for custom categorization for standard predictors.")
        LOGGER.debug("Actually checking for all categorization tables.")
        user_categorization = []
        user_categorization_dict = {}
        user_class_dict = {}
        if not data_dir == "":
            for file_path in os.listdir(data_dir):
                file_name, file_extension = os.path.splitext(file_path)
                if file_extension == ".tsv":
                    try:
                        model_compound_predictors.index(file_name)
                        user_categorization.append(file_name)
                        LOGGER.info("Found custom categorization for %s.",
                                    file_name)
                    except ValueError:
                        try:
                            compound_predictors.index(file_name)
                            LOGGER.warn(("Categorization table found for "
                                         "excluded predictor %s."), file_name)
                        except:
                            LOGGER.warn(("Categorization table found for "
                                         "a missing predictor %s."), file_name)

            if len(user_categorization):
                LOGGER.info("Validating categorization tables.")
                for tsv in user_categorization:
                    LOGGER.info("Validating categorization table %s.", tsv)
                    categories, classes = rs_core.category_dict(
                        "%s%s.tsv" % (data_dir, tsv))
                    user_categorization_dict[tsv] = categories
                    user_class_dict[tsv] = classes

        #warn if none
        if len(user_simple_predictors) == 0:
            if len(user_compound_predictors) == 0:
                LOGGER.info("No user provided predictors found.")
            else:
                LOGGER.info("Only user provided compound predictors found.")
        elif len(user_compound_predictors) == 0:
            LOGGER.info("Only user provided simple predictors found.")
        else:
            LOGGER.info("User provided simple and compound predictors found.")

        #loading data into tables
        LOGGER.debug("Processing user data.")
        LOGGER.info("Importing AOI %s.", aoi_file_name)
        aoi_srid = rs_core.temp_shapefile_db(cur, aoi_file_name, aoi_name)
        if rs_core.not_valid_count_execute(
                cur, aoi_name, geometry_column_name) > 0:
            msg = "The AOI contains invalid geometry."
            LOGGER.warning(msg)

            msg = "Attempting to fix the AOI geometry."
            LOGGER.warning(msg)

            rs_core.make_valid_execute(
                cur, aoi_name, geometry_column_name)

            if rs_core.not_valid_count_execute(
                    cur, aoi_name, geometry_column_name) > 0:
                msg = "The AOI contains invalid geometry that could not be automatically fixed."
                LOGGER.error(msg)
                raise ValueError(msg)
            else:
                msg = "AOI geometry now valid."
                LOGGER.info(msg)

        LOGGER.info("Imported AOI.")
        LOGGER.debug("Imported AOI with SRID %i", aoi_srid)

        if not args["grid"]:
            sql = "SELECT MIN(ST_Area(way)) FROM %s" % aoi_name
            cur.execute(sql)
            area, = cur.fetchone()
            area_hex = (3.0 / 2.0) * (3 ** 0.5) * ((config["min_cell_size"] / 2.0) ** 2) * (1 - 5e-5)
            if area * (linear_units ** 2) < area_hex:
                msg = "The custom grid contains cells smaller than %i square meters." % area_hex
                LOGGER.error(msg)
                raise ValueError(msg)
            else:
                LOGGER.info("All cells in the custom grid meet the minimum area requirement.")

            sql = "SELECT COUNT(*) FROM %s" % aoi_name
            cur.execute(sql)
            count, = cur.fetchone()
            sql = "SELECT SUM(Cast(ST_Crosses(x.way, y.way) AS integer)) FROM %s AS x, %s AS y" % (aoi_name, aoi_name)
            cur.execute(sql)
            intersects, = cur.fetchone()
            if intersects > count:
                msg = "Custom grids cannot have overlapping polygons."
                LOGGER.info("The custom grid contains %i intersections." % (intersects - count))
                LOGGER.error(msg)
                raise ValueError(msg)

        #create tables
        LOGGER.info("Importing user supplied predictor variables.")
        for table_name in user_simple_predictors:
            table_file_name = data_dir+table_name+".shp"
            LOGGER.debug(
                "Creating table %s from %s.", table_name, table_file_name)
            predictor_srid[table_name] = rs_core.temp_shapefile_db(
                cur, table_file_name, table_name)

        LOGGER.info("Importing user supplied compound variables.")
        for table_name in user_compound_predictors:
            table_file_name = data_dir+table_name+".shp"
            LOGGER.debug(
                "Creating table %s from %s.", table_name, table_file_name)
            predictor_srid[table_name] = rs_core.temp_shapefile_db(
                cur, table_file_name, table_name)

        halt = False
        for table_name in user_simple_predictors + user_compound_predictors:
            if rs_core.not_valid_count_execute(
                    cur, table_name, geometry_column_name) > 0:
                LOGGER.warn("Predictor %s contains invalid geometry." % table_name)

                msg = "Attempting to fix the geometry of predictor %s."
                LOGGER.warn(msg, table_name)

                rs_core.make_valid_execute(
                    cur, table_name, geometry_column_name)
                if rs_core.not_valid_count_execute(
                        cur, table_name, geometry_column_name) > 0:
                    msg = "Predictor %s contains geometry that could ot be automatically fixed."
                    LOGGER.error(msg, table_name)
                    halt = True

                else:
                    msg = "Predictor %s geometry now valid."
                    LOGGER.info(msg, table_name)

        if halt:
            msg = "One or more predictors contain invalid geometry that could not be automatically fixed."
            LOGGER.error(msg)
            raise ValueError(msg)

        #processing AOI
        #merge multiple parts
        LOGGER.info("Merging AOI if multiple parts.")
        rs_core.union_execute(
            cur, aoi_name, aoi_union_name, geometry_column_name)
        aoi_transformed_name = aoi_union_name

        #find location
        LOGGER.info("Transforming AOI to Latitude and Longitude.")
        rs_core.transform_execute(
            cur, aoi_union_name, aoi4326_name,
            geometry_column_name, 4326)

        #count intersection and coverage with administrative areas
        intersects, covers = rs_core.get_intersects_covers(
            cur,
            aoi4326_name,
            config["postgis"]["table"]["names"]["borders_name"])

        if intersects == 1 and covers == 1:
            LOGGER.info(("The AOI is ideally located completely within one "
                         "administrative area."))
        else:
            LOGGER.warn(("The AOI intersects %i and is covered by %i "
                         "administrative area(s)."), intersects, covers)

        sql = "SELECT ST_SRID(%s) FROM %s LIMIT 1" % (geometry_column_name,
                                                      aoi_union_name)
        cur.execute(sql)
        output_srid, = cur.fetchone()

        #create grid
        if args["grid"]:
            if args["rectangular_grid"]:
                LOGGER.info(("Creating recatangular grid %s from %s using "
                             "cell size %s."), grid_name, aoi_transformed_name,
                            str(cell_size))
                rs_core.temp_grid_db(
                    cur,
                    aoi_transformed_name,
                    geometry_column_name,
                    grid_name,
                    grid_column_name,
                    cell_size)
            else:
                LOGGER.info(("Creating hexagonal grid %s from %s using "
                             "cell size %s."), grid_name, aoi_transformed_name,
                            str(cell_size))
                rs_core.hex_grid(
                    cur,
                    aoi_transformed_name,
                    geometry_column_name,
                    grid_name,
                    grid_column_name,
                    cell_size)
        else:
            LOGGER.debug("Duplicating table to allow for custom grids.")
            sql = "CREATE TEMPORARY TABLE %s AS (SELECT row_number() OVER (ORDER BY ST_YMin(box2d(way)), ST_XMin(box2d(way)) ASC) AS id, way AS cell FROM %s)" % (grid_name, aoi_name)
            cur.execute(sql)

        #counting grid cells
        sql = "SELECT COUNT(*) FROM %s"
        sql = sql % grid_name
        cur.execute(sql)
        cells, = cur.fetchone()
        if cells < 4:
            LOGGER.error("%i cells is too few for the grid.", cells)
            raise ValueError("%i cells is too few for the grid." % cells)
        else:
            LOGGER.info("The grid contains %i cells.", cells)

        #check grid cell count relative to parameters
        predictors = 0
        predictors += len(model_simple_predictors)

       # LOGGER.debug("Standard mask: %s", str(standard_mask).replace(",","|").replace(".","||"))
       # LOGGER.debug("Compound predictors: %s", str(compound_predictors).replace(",","|").replace(".","||"))
        if standard_mask[1] and not compound_predictors[0] in user_categorization_dict:
            predictors += compound_predictor_classes[0]
        if standard_mask[2] and not user_categorization_dict.has_key(compound_predictors[1]):
            predictors += compound_predictor_classes[1]
        if standard_mask[3] and not user_categorization_dict.has_key(compound_predictors[2]):
            predictors += compound_predictor_classes[2]
       # if standard_mask[5] and not user_categorization_dict.has_key(compound_predictors[3]):
       #     predictors += compound_predictor_classes[3]
        predictors += sum([len(user_categorization_dict[category_key]) for category_key in user_categorization_dict])
        if cells <= predictors:
            LOGGER.debug(
                "There are %i grid cells and %i predictors.", cells, predictors)
            msg = "The number of predictors exceeds the number of grid cells and will likely result in invalid estimations."
            LOGGER.error(msg)
            raise ValueError(msg)

        grid_union_name = union_format % (grid_name)
        rs_core.union_execute(
            cur, grid_name, grid_union_name, grid_column_name)

        #check grid size
        if rs_core.single_area_execute(
                cur, grid_union_name, grid_column_name) > config["max_grid_size"]:
            LOGGER.error("The grid is too large please use a smailler AOI.")
            raise ValueError("The grid is too large please use a smaller AOI.")
        else:
            LOGGER.info("The AOI meets the maximum size requirement.")

        #project grid for clips
        LOGGER.info("Projecting the grid for clips.")
        LOGGER.debug(
            "predictor_srid dictionary: %s", repr(predictor_srid).replace(", ", "|").replace(".", "||"))
        for srid in set([predictor_srid[predictor_key] for predictor_key in predictor_srid]):
            LOGGER.debug(
                "Projecting grid to SRID %i.", srid)
            rs_core.transform_execute(
                cur, grid_union_name, projected_format % (grid_union_name, srid), grid_column_name, srid)

        #clipping predictors
        LOGGER.info("Clipping simple predictors.")
        for predictor in model_simple_predictors:
            LOGGER.info("Clipping %s.", predictor)
            rs_core.clip_execute(
                cur, predictor, geometry_column_name, projected_format % (grid_union_name, predictor_srid[predictor]), grid_column_name, clip_format % (predictor))

        LOGGER.info("Clipping compound predictors.")
        for predictor in model_compound_predictors:
            LOGGER.info("Clipping %s.", predictor)
            if predictor == "planet_osm_point" or predictor == "planet_osm_line" or predictor == "planet_osm_polygon":
                extra_columns = ["osm_id"]
            else:
                extra_columns = ["id"]
            #append categorization columns if needed
            if user_categorization_dict.has_key(predictor):
                cat_columns = user_categorization_dict[predictor].keys()
                cat_columns.sort()
                #remove default category value
                cat_columns.pop(0)
                extra_columns.extend(cat_columns)
            LOGGER.debug("Including columns: %s.", str(
                extra_columns).replace(", ", "|"))
            rs_core.clip_execute(
                cur,
                predictor,
                geometry_column_name,
                projected_format % (
                    grid_union_name, predictor_srid[predictor]),
                grid_column_name,
                clip_format % (predictor), extra_columns)

        #categorizing compound predictors
        for predictor in user_categorization_dict.keys():
            LOGGER.info("Categorizing %s.", predictor)
            rs_core.categorize_execute(
                cur,
                clip_format % predictor,
                user_categorization_dict[predictor],
                user_class_dict[predictor],
                category_format,
                class_format)

        #splitting compound predictors
        LOGGER.info("Converting compound predictors to simple predictors.")
        model_split_predictors = []
        cat_column = "cat"
        for predictor in model_compound_predictors:
            LOGGER.info("Processing compound predictor %s.", predictor)
            if predictor == "planet_osm_point" or predictor == "planet_osm_line" or predictor == "planet_osm_polygon":
                id_column = "osm_id"
            else:
                id_column = "id"
            sql = "SELECT %s, field FROM %s"
            try:
                user_categorization.index(predictor)
                sql = sql % (id_column, class_format % (
                    clip_format % predictor))
            except ValueError:
                LOGGER.info("Using default classification for %s.", predictor)
                sql = sql % (id_column, standard_class_format % predictor)
            cur.execute(sql)
            class_table = cur.fetchall()
            for category, table_name in class_table:
                LOGGER.info("Processing category %s.", table_name)
                sql = "CREATE TABLE %s AS (SELECT %s FROM %s AS layer LEFT JOIN %s AS filter ON layer.%s = filter.%s WHERE filter.%s = %i)"
                try:
                    user_categorization.index(predictor)
                    sql = sql % (clip_format % table_name.lower(), geometry_column_name, clip_format % predictor, category_format % (clip_format % predictor), id_column, id_column, cat_column, category)
                except ValueError:
                    sql = sql % (clip_format % table_name.lower(), geometry_column_name, clip_format % predictor, standard_category_format % predictor, id_column, id_column, cat_column, category)
                cur.execute(sql)
                model_split_predictors.append(table_name.lower())

        #transforming predictors
        LOGGER.info("Projecting simple predictors.")
        for predictor in model_simple_predictors+model_split_predictors:
            LOGGER.info("Projecting %s.", predictor)
            rs_core.transform_execute(
                cur, clip_format % (predictor), projected_format % (predictor, output_srid), geometry_column_name, output_srid)

        #aggregating simple predictors
        join_tables = []
        for predictor in model_simple_predictors+model_split_predictors:
            LOGGER.info("Aggregating %s.", predictor)
            geo_type = rs_core.dimension_execute(
                cur,
                projected_format % (predictor, output_srid),
                geometry_column_name)
            LOGGER.debug("Predictor %s has dimensionality %i.", predictor, geo_type)
            projected_name = projected_format % (predictor, output_srid)
            results_name = results_format % predictor
            if geo_type == -1:
                LOGGER.warn("Predictor %s contains no features inside the grid.", predictor)
                LOGGER.debug("Processing empty predictor %s.", predictor)
                sql = "CREATE TABLE %s (%s int, id int)"% (results_format%predictor, "result")
                LOGGER.debug("Executing SQL: %s", sql.replace(", ", "|").replace(".", "||"))
                cur.execute(sql)
            elif geo_type == 0:
                LOGGER.info("Processing point predictor %s.", predictor)
                rs_core.grid_point_execute(
                    cur, grid_name, projected_name, results_name)
            elif geo_type == 1:
                LOGGER.info("Processing line predictor %s.", predictor)
                rs_core.grid_line_execute(
                    cur, grid_name, projected_name, results_name)
            elif geo_type == 2:
                LOGGER.info("Processing polygon predictor %s.", predictor)
                rs_core.grid_polygon_execute(
                    cur, grid_name, projected_name, results_name)
            else:
                raise ValueError("Predictor %s has an unknown geometry type." % predictor)

            join_tables.append(predictor+results)

        #joining results
        LOGGER.info("Joining results.")
        rs_core.join_results_execute(
            cur,
            model_simple_predictors+model_split_predictors,
            grid_name,
            results_format,
            result_column,
            join_name)

        ignore_category = set()

        #osm patch
        if args["osm"]:
            sql = "ALTER TABLE %s DROP COLUMN %s"
            if not args["osm_1"]:
                LOGGER.debug("Removing OSM information for cultural features.")
                cur.execute(sql % ("results", "pointCult"))
                ignore_category.add("pointcult")
                cur.execute(sql % ("results", "lineCult"))
                ignore_category.add("linecult")
                cur.execute(sql % ("results", "polyCult"))
                ignore_category.add("polycult")
            if not args["osm_2"]:
                LOGGER.debug(
                    "Removing OSM information for industrial features.")
                cur.execute(sql % ("results", "pointIndus"))
                ignore_category.add("pointindus")
                cur.execute(sql % ("results", "lineIndus"))
                ignore_category.add("lineindus")
                cur.execute(sql % ("results", "polyIndus"))
                ignore_category.add("polyindus")
            if not args["osm_3"]:
                LOGGER.debug("Removing OSM information for natural features.")
                cur.execute(sql % ("results", "pointNat"))
                ignore_category.add("pointnat")
                cur.execute(sql % ("results", "lineNat"))
                ignore_category.add("linenat")
                cur.execute(sql % ("results", "polyNat"))
                ignore_category.add("polynat")
            if not args["osm_4"]:
                LOGGER.debug(
                    "Removing OSM information for superstructure features.")
                cur.execute(sql % ("results", "pointStruc"))
                ignore_category.add("pointstruc")
                cur.execute(sql % ("results", "lineStruc"))
                ignore_category.add("linestruc")
                cur.execute(sql % ("results", "polyStruc"))
                ignore_category.add("polystruc")
            if not args["osm_0"]:
                LOGGER.debug(
                    "Removing OSM information for miscellaneous features.")
                cur.execute(sql % ("results", "pointMisc"))
                ignore_category.add("pointmisc")
                cur.execute(sql % ("results", "lineMisc"))
                ignore_category.add("linemisc")
                cur.execute(sql % ("results", "polyMisc"))
                ignore_category.add("polymisc")

        #writing predictor table
        LOGGER.info("Creating data shapefile.")
        rs_core.dump_execute(
            cur, join_name, grid_file_name, column_alias)

        #save data for download
        if args["download"]:
            LOGGER.info("Saving predictors to disk.")
            downloads = set(model_simple_predictors+model_split_predictors)
            downloads.discard(
                config["postgis"]["table"]["names"]["landscan_name"])
            downloads.difference_update(ignore_category)
            for predictor in downloads:
                LOGGER.info("Saving predictor %s for downloading.", predictor)
                if column_alias.has_key(predictor):
                    predictor_file_name = os.path.abspath(os.path.join(os.path.dirname(grid_file_name),os.path.join("download","%s.shp") % column_alias[predictor]))
                else:
                    predictor_file_name = os.path.abspath(os.path.join(os.path.dirname(grid_file_name),os.path.join("download","%s.shp") % predictor))
                rs_core.dump_execute(
                    cur,
                    projected_format % (predictor, output_srid),
                    predictor_file_name)

        #Flickr
        LOGGER.info("Transforming grid to Flickr projection.")
        rs_core.grid_transform(
            cur,
            grid_name,
            projected_format % (
                grid_name, predictor_srid[config["postgis"]["table"]["names"]["flickr_name"]]),
            predictor_srid[config["postgis"]["table"]["names"]["flickr_name"]])
        LOGGER.info("Creating Flickr summary table.")
        LOGGER.debug("Saving Flickr summary table to %s.", flickr_file_name)
        rs_core.flickr_grid_table(
            cur,
            projected_format % (grid_name, predictor_srid[config["postgis"]["table"]["names"]["flickr_name"]]), config["postgis"]["table"]["names"]["flickr_name"],
            flickr_file_name)

        #house keeping
        LOGGER.info("Dropping intermediate tables.")
        temp_tables = [aoi_name,
                       #aoi_union_name,
                       aoi4326_name,
                       aoi_transformed_name,
                       union_format % (grid_name),
                       join_name,
                       grid_name,
                       projected_format % (grid_name, predictor_srid[config["postgis"]["table"]["names"]["flickr_name"]])]

        for srid in set([predictor_srid[predictor_key] for predictor_key in predictor_srid]):
            temp_tables.append(
                projected_format % (union_format % grid_name, srid))

        for predictor in user_simple_predictors:
            temp_tables.append(predictor)

        for predictor in model_simple_predictors:
            temp_tables.append(clip_format % predictor)
            temp_tables.append(projected_format % (predictor, output_srid))
            temp_tables.append(results_format % predictor)

        for predictor in user_compound_predictors:
            temp_tables.append(predictor)

        for predictor in user_categorization:
            temp_tables.append(category_format % (clip_format % predictor))
            temp_tables.append(class_format % (clip_format % predictor))

        for predictor in model_compound_predictors:
            temp_tables.append(clip_format % predictor)

        for predictor in model_split_predictors:
            temp_tables.append(clip_format % predictor)
            temp_tables.append(projected_format % (predictor, output_srid))
            temp_tables.append(results_format % predictor)

        LOGGER.debug("Dropping tables: %s.", str(temp_tables).replace(", ", "|"))

        for table_name in temp_tables:
            cur.execute("DROP TABLE %s" % (table_name))
            LOGGER.debug("Dropped table %s.", table_name)

        LOGGER.info("Dropped intermediate tables.")

        cur.close()
        database.commit()
        database.close()
    except:
        e = sys.exc_info()[1]
        t = sys.exc_info()[2]
        msg = str(e)
        if len(msg) == 0:
            msg = repr(e)

        msg = msg + ": " + traceback.format_exc(t)

        msg = msg.replace(",", "").replace(".", "")
        if msg[-1] != ".":
            msg = msg + "."

        LOGGER.error(msg)
        raise e


if __name__ == "__main__":
    LOGGER = logging.getLogger("natcap.invest.recreation.server_init")
    #LOGGER.remove_handler(LOGGER.handlers[0])
    formatter = logging.Formatter(
        "%(asctime)s, %(levelname)s, %(message)s", "%m/%d/%Y %H:%M:%S")
    LOGGER.setLevel(logging.DEBUG)

    #load configuration
    LOGGER.info("Loading server configuration file.")
    config_file = open(
        os.path.abspath(os.path.dirname(
            sys.argv[0]))+os.sep+"recreation_server_config.json", 'r')
    config = json.loads(config_file.read())
    config_file.close()

    #load model paramters
    LOGGER.info("The length of sys argv is %i.", len(sys.argv))
    if len(sys.argv) > 1:
        model_file = open(sys.argv[1], 'r')
    else:
        model_file = open(os.path.abspath(
            os.path.join(os.path.dirname(sys.argv[0]), "default_init.json")))

    model = json.loads(model_file.read())
    model_file.close()
    session_path = os.path.join(
        config["paths"]["absolute"]["userroot"], config["paths"]["relative"]["data"] + model["sessid"])

    if model["sessid"] == "init":
        os.system("rm %s*" % session_path)
        os.system("rm %s*" % os.path.join(
            session_path, config["paths"]["relative"]["predictors"]))
        os.system("rm %s*" % os.path.join(
            session_path, config["paths"]["relative"]["download"]))
        os.system("mkdir %s" % session_path)
        os.system("mkdir %s" % os.path.join(
            session_path, config["paths"]["relative"]["predictors"]))
        os.system("mkdir %s" % os.path.join(
            session_path, config["paths"]["relative"]["download"]))
        os.system("cp %s %s" % (model["aoi_file_name"], os.path.join(
            session_path, config["files"]["aoi"]["shp"])))
        os.system("cp %s %s" % (model["aoi_file_name"][:-3]+"shx",
                                os.path.join(session_path, config["files"]["aoi"]["shx"])))
        os.system("cp %s %s" % (model["aoi_file_name"][:-3]+"dbf",
                                os.path.join(session_path, config["files"]["aoi"]["dbf"])))
        os.system("cp %s %s" % (model["aoi_file_name"][:-3]+"prj",
                                os.path.join(session_path, config["files"]["aoi"]["prj"])))
        os.system("cp %s* %s" % (model["data_dir"], os.path.join(
            session_path, config["paths"]["relative"]["data"])))

    #log to file
    handler = logging.FileHandler(os.path.join(
        session_path, config["files"]["log"]))
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)

    LOGGER.info("Running server side model with user provided parameters.")
    LOGGER.debug("Running server side model with parameters: %s.", repr(
        sys.argv).replace(", ", "|").replace(".", "||"))

    args = {}
    args["aoi_file_name"] = os.path.join(
        session_path, config["files"]["aoi"]["shp"])
    args["grid"] = model["grid"]
    if model["grid"]:
        args["rectangular_grid"] = (model["grid_type"] == "0")
        args["cell_size"] = float(model["cell_size"])
    args["grid_file_name"] = os.path.join(
        session_path, config["files"]["grid"]["shp"])
    args["flickr_file_name"] = os.path.join(
        session_path, config["files"]["flickr"])
    args["data_dir"] = os.path.join(
        session_path, config["paths"]["relative"]["predictors"])
    args["download"] = model["download"]

    #check for landscan categoirization
    if os.path.exists(args["data_dir"]+"landscan.tsv"):
        LOGGER.error("The categorization of the Landscan data is not allowed.")
        raise ValueError("The categorization of the Landscan data "
                         "is not allowed.")

    #change dev version to release version
    if not model["is_release"]:
        model["is_release"] = True
        model["version_info"] = "2.5.5"

    if model["is_release"]:

        args["version_info"] = model["version_info"]
        if args["version_info"] == "2.5.4":
            if model["global_data"]:
                args["landscan"] = model["landscan"]
                args["osm_point"] = model["osm_point"]
                args["osm_line"] = model["osm_line"]
                args["osm_poly"] = model["osm_poly"]
                args["protected"] = model["protected"]

                args["lulc"] = model["lulc"]
                if args["lulc"]:
                    args["lulc_1"] = model["lulc_1"]
                    args["lulc_2"] = model["lulc_2"]
                    args["lulc_3"] = model["lulc_3"]
                    args["lulc_4"] = model["lulc_4"]
                    args["lulc_5"] = model["lulc_5"]
                    args["lulc_6"] = model["lulc_6"]
                    args["lulc_7"] = model["lulc_7"]
                    args["lulc_8"] = model["lulc_8"]
                else:
                    args["lulc_1"] = False
                    args["lulc_2"] = False
                    args["lulc_3"] = False
                    args["lulc_4"] = False
                    args["lulc_5"] = False
                    args["lulc_6"] = False
                    args["lulc_7"] = False
                    args["lulc_8"] = False

                args["mangroves"] = model["mangroves"]
                args["reefs"] = model["reefs"]
                args["grass"] = model["grass"]
            else:
                args["landscan"] = False
                args["osm_point"] = False
                args["osm_line"] = False
                args["osm_poly"] = False
                args["protected"] = False

                args["lulc"] = False
                args["lulc_1"] = False
                args["lulc_2"] = False
                args["lulc_3"] = False
                args["lulc_4"] = False
                args["lulc_5"] = False
                args["lulc_6"] = False
                args["lulc_7"] = False
                args["lulc_8"] = False

                args["mangroves"] = False
                args["reefs"] = False
                args["grass"] = False
        elif args["version_info"] >= "2.5.5":
            if model["global_data"]:
                args["landscan"] = model["landscan"]
                args["protected"] = model["protected"]

                args["osm"] = model["osm"]
                if args["osm"]:
                    args["osm_point"] = True
                    args["osm_line"] = True
                    args["osm_poly"] = True

                    args["osm_0"] = model["osm_0"]
                    args["osm_1"] = model["osm_1"]
                    args["osm_2"] = model["osm_2"]
                    args["osm_3"] = model["osm_3"]
                    args["osm_4"] = model["osm_4"]
                else:
                    args["osm_point"] = False
                    args["osm_line"] = False
                    args["osm_poly"] = False

                args["lulc"] = model["lulc"]
                if args["lulc"]:
                    args["lulc_1"] = model["lulc_1"]
                    args["lulc_2"] = model["lulc_2"]
                    args["lulc_3"] = model["lulc_3"]
                    args["lulc_4"] = model["lulc_4"]
                    args["lulc_5"] = model["lulc_5"]
                    args["lulc_6"] = model["lulc_6"]
                    args["lulc_7"] = model["lulc_7"]
                    args["lulc_8"] = model["lulc_8"]
                else:
                    args["lulc_1"] = False
                    args["lulc_2"] = False
                    args["lulc_3"] = False
                    args["lulc_4"] = False
                    args["lulc_5"] = False
                    args["lulc_6"] = False
                    args["lulc_7"] = False
                    args["lulc_8"] = False

                args["ouoc"] = model["ouoc"]
                if args["ouoc"]:
                    args["mangroves"] = model["mangroves"]
                    args["reefs"] = model["reefs"]
                    args["grass"] = model["grass"]
                else:
                    args["mangroves"] = False
                    args["reefs"] = False
                    args["grass"] = False

            else:
                args["landscan"] = False

                args["osm"] = False
                args["osm_point"] = False
                args["osm_line"] = False
                args["osm_poly"] = False

                args["protected"] = False

                args["lulc"] = False
                args["lulc_1"] = False
                args["lulc_2"] = False
                args["lulc_3"] = False
                args["lulc_4"] = False
                args["lulc_5"] = False
                args["lulc_6"] = False
                args["lulc_7"] = False
                args["lulc_8"] = False

                args["mangroves"] = False
                args["reefs"] = False
                args["grass"] = False

        else:
            LOGGER.error("Unsupported version.")
            raise ValueError("Unsupported version.")
    else:
        LOGGER.info("Getting timestamp of developer version.")
        revision = model["version_info"].split(
            "[")[-1][:-1]
        LOGGER.debug("Revison %s.", revision)
        pipe = subprocess.Popen(
            "hg log -R ~/workspace/invest3/ -r ffa218dbe978", stdout=subprocess.PIPE, shell=True).communicate()[0]
        stamp = pipe.strip().split("\n")[-2][5:-5].strip()
        LOGGER.debug("Developer version committed on %s.", str(stamp))
        when = datetime.datetime.strptime(stamp, "%a %b %d %H:%M:%S %Y")
        if when < datetime.datetime.now():
            LOGGER.error("Developer version.")
            raise ValueError("Developer version not supported.")

    LOGGER.debug("Calling execute(%s).",
                 repr(args).replace(", ", "|").replace(".", "||"))
    execute(args, config)
