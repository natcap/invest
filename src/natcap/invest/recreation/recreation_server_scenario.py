import sys
import os
import psycopg2
import logging
import json
import recreation_server_core as rs_core


def loggify(string):
    '''
    Replaces commas and periods with vertical bars

    Args:
        string (string): string with periods and commas

    Returns:
        string (string): string with vertical bars in place of any periods and
            commas
    '''
    return string.replace(", ", "|").replace(".", "||")


def execute(args):
    '''
    Entry point into Recreation Server Scenario program

    :param str args['data_dir']:
    :param str args['grid_file_name']:
    :param str args['scenario_file_name']:

    Returns:
        None

    Example Args::

        args = {
            'data_dir': '/path/to/data_dir/',
            'grid_file_name': 'filename',
            'scenario_file_name': 'filename',
        }

    '''
    #parameters
    LOGGER.debug("Processing parameters.")
    grid_file_name = args["grid_file_name"]
    data_dir = args["data_dir"]

    #column names
    geometry_column_name = "way"
    grid_column_name = "cell"

    # #OSM
    # point_osm_name = "planet_osm_point"
    # line_osm_name = "planet_osm_line"
    # poly_osm_name = "planet_osm_polygon"

    # #OSM classification
    # point_category_name = "category_point"
    # line_category_name = "category_line"
    # poly_category_name = "category_polygon"

    # #social media
    # flickr_name = "photos_gis"

    #population
    landscan_name = "predictor_landscan"

    #osm
    point_name = "planet_osm_point"
    line_name = "planet_osm_line"
    polygon_name = "planet_osm_polygon"

   # #areas
   # borders_name = "borders"
    protected_name = "predictor_protected"

    #coverages
    lulc_name = "predictor_lulc"

    #habitas
    mangrove_name = "predictor_mangrove"
    reef_name = "predictor_reef"
    seagrass_name = "predictor_seagrass"

    # standard_predictors = [landscan_name,
    #                       point_name,
    #                       line_name,
    #                       polygon_name,
    #                       protected_name,
    #                       lulc_name,
    #                       mangrove_name,
    #                       reef_name,
    #                       seagrass_name]

    osm_srid = 900913
    world = 4326

    standard_srid = {
        landscan_name: world,
        point_name: osm_srid,
        line_name: osm_srid,
        polygon_name: osm_srid,
        protected_name: osm_srid,
        lulc_name: world,
        mangrove_name: world,
        reef_name: world,
        seagrass_name: world}

    simple_predictors = [
        landscan_name,
        protected_name,
        mangrove_name,
        reef_name,
        seagrass_name]

    # column_alias = {landscan_name: "landscan",
    #                protected_name: "protected",
    #                mangrove_name: "mangrove",
    #                reef_name: "reef",
    #                seagrass_name: "seagrass"}

    # table_alias = {"landscan": landscan_name,
    #               "protected": protected_name,
    #               "mangrove": mangrove_name,
    #               "reef": reef_name,
    #               "seagrass": seagrass_name}

    compound_predictors = [
        point_name,
        line_name,
        polygon_name,
        lulc_name]

    #intermediate table names
    grid_name = "grid"
    grid_union_name = "%s_union" % grid_name
    results_name = "results"
    join_name = "results"

    #column names
    result_column = "result"
    grid_column_name = "cell"
    geometry_column_name = "way"

    #suffixes
    union = "_union"
    clip = "_clip"
   # tansform = "_transform"
    results = "_result"
    category = "_user_category"

   # #table names
   # standard_class_format = "%s_class"
   # standard_category_format = "%s_category"
    class_format = "%s_user_class"
    category_format = "%s" + category
    union_format = "%s" + union
    clip_format = "%s" + clip
    projected_format = "%s_%i"
    results_format = "%s" + results

    LOGGER.debug("Parsing database connection string.")
    #database definitions
    dbase_file_name = os.path.join(os.path.abspath(
        os.path.dirname(sys.argv[0])), "postgis.db")
    dbase_file = open(dbase_file_name, "r")
    dbase_postgis = dbase_file.read().strip().replace("\n", " ")
    dbase_file.close()

    try:
        LOGGER.info("Connecting to database.")
        database = psycopg2.connect(dbase_postgis)
        cur = database.cursor()

        #load grid
        LOGGER.info("Loading grid.")
        LOGGER.debug("Loading grid from %s.",
                     loggify(grid_file_name))
        grid_srid = rs_core.temp_shapefile_db(
            cur,
            grid_file_name,
            grid_name,
            True)

        #rename id column
        sql = "ALTER TABLE %s RENAME COLUMN %s to %s"
        sql = sql % (grid_name, "cellid", "id")
        cur.execute(sql)
        sql = "ALTER TABLE %s RENAME COLUMN %s to %s"
        sql = sql % (grid_name, geometry_column_name, grid_column_name)
        cur.execute(sql)

        #build predictor list
        LOGGER.info("Getting list of predictors in grid.")
        sql = "SELECT * FROM %s LIMIT 0"
        sql = sql % (grid_name)
        cur.execute(sql)
        grid_predictors = set([desc[0] for desc in cur.description])
        ignore_columns = ["cell", "cellarea", "id"]
        for column_name in ignore_columns:
            grid_predictors.discard(column_name)
        grid_predictors = list(grid_predictors)
        LOGGER.debug("Grid contains the following columns: %s",
                     loggify(str(grid_predictors)))

        #processing user provided predictors
        #create list from directory
        LOGGER.info("Processing user uploaded files.")
        user_simple_predictors = []
        user_compound_predictors = []
        model_compound_predictors = []
        # user_categorization = []
        user_categorization_dict = {}
        user_class_dict = {}
        predictor_srid = {}
        # predictorList= copy.copy(grid_predictors)
        if not data_dir == "":
            for file_name in os.listdir(data_dir):
                file_stem, file_extension = os.path.splitext(file_name)
                if file_extension == ".shp" and \
                   os.path.exists(data_dir + file_stem + ".shx") and \
                   os.path.exists(data_dir + file_stem + ".shp") and \
                   os.path.exists(data_dir + file_stem + ".prj"):
                    if os.path.exists(data_dir + file_stem + ".tsv"):
                        LOGGER.info("Found compound predictor %s.", file_stem)
                        include = False
                        LOGGER.info("Processing categorization table.")
                        categories, classes = rs_core.category_dict(
                            data_dir + file_stem + ".tsv")
                        user_categorization_dict[file_stem] = categories
                        user_class_dict[file_stem] = classes
                        for predictor in classes.keys():
                            try:
                                grid_predictors.index(predictor)
                                LOGGER.info("Found simple predictor %s in %s.",
                                            predictor,
                                            file_stem)
                                include = True
                            except ValueError:
                                LOGGER.warn(
                                    "Simple predictor %s from %s in not in the grid.",
                                    predictor,
                                    file_stem)
                        if include:
                            LOGGER.info(
                                "Adding compound predictor %s to processing queue.",
                                file_stem)
                            user_compound_predictors.append(file_stem)
                            LOGGER.info("Importing compound predictor %s.",
                                        file_stem)
                            predictor_srid[file_stem] = rs_core.temp_shapefile_db(
                                cur,
                                data_dir + file_stem + ".shp",
                                file_stem,
                                True)
                        else:
                            LOGGER.warn(
                                "_compound predictor %s does not contain any simple predictors in the grid.")
                    else:
                        LOGGER.info("Found simple predictor %s.", file_stem)
                        try:
                            grid_predictors.index(file_stem)
                            LOGGER.debug("Found predictor %s in grid.",
                                         file_stem)
                            LOGGER.info(
                                "Adding simple predictor %s to processing queue.",
                                file_stem)
                            user_simple_predictors.append(file_stem)
                            LOGGER.info("Importing simple predictor %s.",
                                        file_stem)
                            LOGGER.debug("Importing simple predictor %s.",
                                         (data_dir + file_stem + ".shp"))
                            predictor_srid[file_stem] = rs_core.temp_shapefile_db(cur, data_dir + file_stem + ".shp", file_stem)
                        except ValueError:
                            LOGGER.warn("Predictor %s is not in the grid.",
                                        file_stem)
                elif file_extension == ".tsv" and not os.path.exists(data_dir + file_stem + ".shp"):
                    LOGGER.debug("Found category table %s without shapefile.",
                                 file_stem)
                    try:
                        simple_predictors.index(file_stem)
                        try:
                            #this should never be reached, right?
                            grid_predictors.index(file_stem)
                            LOGGER.info(
                                "Adding simple predictor %s to processing queue.",
                                file_stem)
                            user_simple_predictors.append(file_stem)
                            predictor_srid[file_stem] = standard_srid[file_stem]
                        except ValueError:
                            LOGGER.warn("Predictor %s is not in the grid.",
                                        file_stem)
                    except ValueError:
                        try:
                            compound_predictors.index(file_stem)
                            include = False
                            LOGGER.info("Processing categorization table.")
                            categories, classes = rs_core.category_dict(
                                data_dir + file_stem + ".tsv")
                            user_categorization_dict[file_stem] = categories
                            user_class_dict[file_stem] = classes
                            for predictor in classes.keys():
                                try:
                                    grid_predictors.index(predictor)
                                    LOGGER.info(
                                        "Found simple predictor %s in %s.",
                                        predictor, file_stem)
                                    include = True
                                except ValueError:
                                    LOGGER.warn(
                                        "Simple predictor %s from %s in not in the grid.",
                                        predictor, file_stem)
                            if include:
                                LOGGER.info(
                                    "Adding compound predictor %s to processing queue.",
                                    file_stem)
                                model_compound_predictors.append(file_stem)
                                predictor_srid[file_stem] = standard_srid[file_stem]
                            else:
                                LOGGER.warn(
                                    "Compound predictor %s does not contain any simple predictors in the grid.",
                                    file_stem)
                        except ValueError:
                            LOGGER.warn(
                                "Categorization table %s is not part of the grid.",
                                file_stem)
        else:
            LOGGER.error("Scenario runs must have additional data.")
            raise ValueError("Scenario runs must have additional data.")

        halt = False
        for table_name in user_simple_predictors + user_compound_predictors:
            if rs_core.not_valid_count_execute(
                    cur, table_name, geometry_column_name) > 0:
                LOGGER.warn(
                    "Predictor %s contains invalid geometry." % table_name)
                halt = True
        if halt:
            msg = "One or more predictors contain invalid geometry."
            LOGGER.error(msg)
            raise ValueError(msg)

        LOGGER.debug("The following simple predictors will be updated: %s.",
                     loggify(repr(user_simple_predictors)))
        LOGGER.debug("The following compound predictors will be updated: %s.",
                     loggify(repr(user_compound_predictors)))

        #union grid
        grid_union_name = union_format % (grid_name)
        rs_core.union_execute(
            cur,
            grid_name,
            grid_union_name,
            grid_column_name)

        #project grid for clips
        LOGGER.info("Projecting the grid for clips.")
        LOGGER.debug("SRID dictionary: %s",
                     loggify(repr(predictor_srid)))
        for srid in set([predictor_srid[key] for key in predictor_srid.keys()]):
            LOGGER.debug("Projecting grid to srid %i.", srid)
            rs_core.transform_execute(
                cur,
                grid_union_name,
                projected_format % (grid_union_name, srid),
                grid_column_name,
                srid)

        #clipping predictors
        LOGGER.info("Clipping simple predictors.")
        for predictor in user_simple_predictors:
            LOGGER.debug("Clipping %s.", predictor)
            rs_core.clip_execute(
                cur,
                predictor,
                geometry_column_name,
                projected_format % (
                    grid_union_name, predictor_srid[predictor]),
                grid_column_name,
                clip_format % (predictor))

        LOGGER.info("Clipping compound predictors.")
        for predictor in user_compound_predictors + model_compound_predictors:
            LOGGER.debug("Clipping %s.", predictor)
            if predictor == "planet_osm_point" or predictor == "planet_osm_line" or predictor == "planet_osm_polygon":
                extra_columns = ["osm_id"]
            else:
                extra_columns = ["id"]
            #append categorization columns if needed
            if predictor in user_categorization_dict:
                cat_columns = user_categorization_dict[predictor].keys()
                cat_columns.sort()
                #remove default category value
                cat_columns.pop(0)
                extra_columns.extend(cat_columns)
            LOGGER.debug("Including columns: %s.",
                         loggify(repr(extra_columns)))
            rs_core.clip_execute(
                cur,
                predictor,
                geometry_column_name,
                projected_format % (
                    grid_union_name, predictor_srid[predictor]),
                grid_column_name,
                clip_format % (predictor),
                extra_columns)

        #categorizing compound predictors
        for predictor in user_compound_predictors + model_compound_predictors:
            LOGGER.debug("Categorizing %s.", predictor)
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
        for predictor in user_compound_predictors + model_compound_predictors:
            LOGGER.info("Processing compound predictor %s.", predictor)
            if predictor == "planet_osm_point" or predictor == "planet_osm_line" or predictor == "planet_osm_polygon":
                id_column = "osm_id"
            else:
                id_column = "id"

            sql = "SELECT %s, field FROM %s"
            sql = sql % (id_column, class_format % (clip_format % predictor))
            cur.execute(sql)
            class_table = cur.fetchall()

            for category, table_name in class_table:
                LOGGER.info("Processing category %s.", table_name)
                sql = "CREATE TABLE %s AS (SELECT %s FROM %s AS layer "
                "LEFT JOIN %s AS filter ON layer.%s = filter.%s WHERE "
                "filter.%s = %i)"
                sql = sql % (clip_format % table_name.lower(),
                             geometry_column_name,
                             clip_format % predictor,
                             category_format % (clip_format % predictor),
                             id_column, id_column, cat_column, category)
                cur.execute(sql)
                model_split_predictors.append(table_name.lower())

        #transforming predictors
        LOGGER.info("Projecting simple predictors.")
        for predictor in user_simple_predictors + model_split_predictors:
            LOGGER.debug("Projecting %s.", predictor)
            rs_core.transform_execute(
                cur,
                clip_format % (predictor),
                projected_format % (predictor, grid_srid),
                geometry_column_name,
                grid_srid)

        #aggregating simple predictors
        join_tables = []
        for predictor in user_simple_predictors + model_split_predictors:
            LOGGER.debug("Aggregating %s.", predictor)
            geo_type = rs_core.dimension_execute(
                cur,
                projected_format % (predictor, grid_srid),
                geometry_column_name)
            LOGGER.debug("Predictor %s has dimensionality %i.", predictor,
                         geo_type)
            projected_name = projected_format % (predictor, grid_srid)
            results_name = results_format % predictor
            if geo_type == -1:
                LOGGER.debug("Processing empty predictor %s.", predictor)
                sql = "CREATE TABLE %s (%s int, id int)" % (results_format % predictor, "result")
                LOGGER.debug("Executing sql: %s", loggify(sql))
                cur.execute(sql)
            elif geo_type == 0:
                LOGGER.debug("Processing point predictor %s.", predictor)
                rs_core.grid_point_execute(
                    cur,
                    grid_name,
                    projected_name,
                    results_name)
            elif geo_type == 1:
                LOGGER.debug("Processing line predictor %s.", predictor)
                rs_core.grid_line_execute(
                    cur,
                    grid_name,
                    projected_name,
                    results_name)
            elif geo_type == 2:
                LOGGER.debug("Processing polygon predictor %s.", predictor)
                rs_core.grid_polygon_execute(
                    cur,
                    grid_name,
                    projected_name,
                    results_name)
            else:
                raise ValueError("Predictor %s has an unknown geometry type.",
                                 predictor)

            join_tables.append(predictor + results)

        #join
        LOGGER.info("New data for columns %s.",
                     loggify(repr(user_simple_predictors + model_split_predictors)))
        if len(user_simple_predictors + model_split_predictors) == 0:
            LOGGER.error("There will be no modified predictors in the scenario.")

        grid_predictors = set(grid_predictors)
        grid_predictors.difference_update(
            set(user_simple_predictors + model_split_predictors))
        LOGGER.info(
            "Old data for columns %s.",
            loggify(repr(grid_predictors)))

        LOGGER.info("Preserving old columns.")
        if not len(grid_predictors):
            sql = "CREATE TABLE %s AS (SELECT %s, %s FROM %s)" % ("newgrid", grid_column_name, "id", grid_name)
        else:
            sql = "CREATE TABLE %s AS (SELECT %s, %s, %s from %s)" % ("newgrid", grid_column_name, "id", ", ".join(grid_predictors), grid_name)
        LOGGER.debug("Executing sql statement : %s.", loggify(sql))
        cur.execute(sql)

        LOGGER.info("Joining new columns.")
        rs_core.join_results_execute(
            cur,
            user_simple_predictors + model_split_predictors,
            "newgrid",
            results_format,
            result_column,
            join_name,
            grid_predictors)

        #writing predictor table
        LOGGER.info("Creating data shapefile.")
        rs_core.dump_execute(
            cur,
            join_name,
            args["scenario_file_name"],
            {})

        LOGGER.debug("Shapefile written to disk.")

        #drop
        temp_tables = [grid_name,
                       "newgrid",
                       grid_union_name,
                       "results"]

        LOGGER.debug("Appending projected table names.")
        for srid in set([predictor_srid[key] for key in predictor_srid.keys()]):
            temp_tables.append(projected_format % (grid_union_name, srid))

        LOGGER.debug("Appending user simple predictor table names.")
        for predictor in user_simple_predictors:
            temp_tables.append(predictor)
            temp_tables.append(clip_format % predictor)
            temp_tables.append(projected_format % (predictor, grid_srid))
            temp_tables.append(results_format % predictor)

        LOGGER.debug("Appending user compound predictor table names.")
        for predictor in user_compound_predictors:
            temp_tables.append(predictor)
            temp_tables.append(clip_format % predictor)

        LOGGER.debug("Appending model compound predictors table names.")
        for predictor in model_compound_predictors:
            temp_tables.append(clip_format % predictor)
            temp_tables.append(category_format % (clip_format % predictor))
            temp_tables.append(class_format % (clip_format % predictor))

        LOGGER.debug("Appending model split predictor table names.")
        for predictor in model_split_predictors:
            temp_tables.append(clip_format % predictor)
            temp_tables.append(projected_format % (predictor, grid_srid))
            temp_tables.append(results_format % predictor)

        LOGGER.debug("Dropping intermediate tables.")
        for table_name in temp_tables:
            cur.execute("DROP TABLE %s" % table_name)
            LOGGER.debug("Dropped table %s.", table_name)

        LOGGER.info("Dropped intermediate tables.")

        cur.close()
        database.commit()
        database.close()

    except Exception, msg:
        msg = str(msg).replace(", ", "")
        if msg[-1] != ".":
            msg = msg + "."
        LOGGER.error(msg)

if __name__ == "__main__":
    LOGGER = logging.getLogger("natcap.invest.recreation.server_scenario")
    formatter = logging.Formatter("%(asctime)s, %(levelname)s, %(message)s",
                                  "%m/%d/%Y %H:%M:%S")
    LOGGER.setLevel(logging.DEBUG)

    #load configuration
    LOGGER.info("Loading server configuration file.")
    dir_name = os.path.abspath(os.path.dirname(sys.argv[0]))
    config_file = open(os.path.join(dir_name,
                                    "recreation_server_config.json"), 'r')
    config = json.loads(config_file.read())
    config_file.close()

    #load model parameters
    LOGGER.info("The length of sys argv is %i.", len(sys.argv))
    if len(sys.argv) > 1:
        scenario_file = open(sys.argv[1], 'r')
        initial_file = open(sys.argv[2], 'r')
    else:
        scenario_file = open(os.path.join(
            dir_name, "default_scenario.json"), 'r')
        initial_file = open(os.path.join(
            dir_name, "default_initial.json"), 'r')

    #load scenario parameters
    scenario = json.loads(scenario_file.read())
    scenario_file.close()
    session_path = "".join([config["paths"]["absolute"]["userroot"],
                           config["paths"]["relative"]["data"],
                           scenario["sessid"],
                           os.sep])
    LOGGER.debug("Scenario path: %s.", session_path)

    #print scenario parameters
    scenario_keys = scenario.keys()
    scenario_keys.sort()

    for k in scenario_keys:
        LOGGER.debug("Scenario parameter %s has value: %s.",
                     k, loggify(repr(scenario[k])))

    #load initial parameters
    initial = json.loads(initial_file.read())
    initial_file.close()
    initial_path = "".join((config["paths"]["absolute"]["userroot"],
                            config["paths"]["relative"]["data"],
                            initial["sessid"],
                            os.sep))
    LOGGER.debug("Initial path: %s.", initial_path)

    #print initial parameters
    initial_keys = initial.keys()
    initial_keys.sort()

    for k in initial_keys:
        LOGGER.debug("Initial parameter %s has value: %s.",
                     k, loggify(repr(initial[k])))

    #construct directory structure for default runs
    if scenario["sessid"] == "scenario":
        os.system("rm %s*" % session_path)
        os.system("rm %s*" % (
            session_path + config["paths"]["relative"]["predictors"]))
        os.system("rm %s*" % (
            session_path + config["paths"]["relative"]["download"]))
        os.system("mkdir %s" % session_path)
        os.system("mkdir %s" % (
            session_path + config["paths"]["relative"]["predictors"]))
        os.system("mkdir %s" % (
            session_path + config["paths"]["relative"]["download"]))
        os.system("cp %s %s" % (
            scenario["grid_file_name"], os.path.join(
                session_path,
                config["files"]["grid"]["shp"])))
        file_stem, file_extension = os.path.splitext(
            scenario["grid_file_name"])
        os.system("cp %s %s" % (
            file_stem + ".shx",
            os.path.join(
                session_path,
                config["files"]["grid"]["shx"])))
        os.system("cp %s %s" % (
            file_stem + ".dbf",
            os.path.join(
                session_path,
                config["files"]["grid"]["dbf"])))
        os.system("cp %s %s" % (
            file_stem + ".prj",
            os.path.join(
                session_path,
                config["files"]["grid"]["prj"])))
        os.system("cp %s* %s" % (
            scenario["data_dir"],
            session_path + config["paths"]["relative"]["data"]))

    #log to file
    hdlr = logging.FileHandler(str(session_path + config["files"]["log"]))
    hdlr.setFormatter(formatter)
    LOGGER.addHandler(hdlr)

    LOGGER.info("Running server side scenario with user provided parameters.")
    LOGGER.debug("Running server side scenario with parameters: %s.",
                 loggify(repr(sys.argv)))

    LOGGER.debug("Constructing args dictionary.")
    args = {}
    args["grid_file_name"] = str(initial_path + config["files"]["grid"]["shp"])
    args["data_dir"] = os.path.join(session_path,
                                    config["paths"]["relative"]["predictors"])

    #check for landscan categoirization
    if os.path.exists(args["data_dir"] + "landscan.tsv"):
        msg = "The categorization of the Landscan data is not allowed."
        LOGGER.error(msg)
        raise ValueError(msg)

    args["scenario_file_name"] = os.path.join(session_path,
                                              config["files"]["grid"]["shp"])

    LOGGER.debug("Calling execute(%s).", loggify(repr(args)))
    execute(args)
