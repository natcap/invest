"""Urban Flood Risk Mitigation model."""
import logging
import os

import numpy
import pygeoprocessing
import rtree
import scipy
import shapely.prepared
import shapely.wkb
import taskgraph
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from . import gettext
from . import spec
from . import utils
from . import validation
from .unit_registry import u
from .file_registry import FileRegistry

LOGGER = logging.getLogger(__name__)

MODEL_SPEC = spec.ModelSpec(
    model_id="urban_flood_risk_mitigation",
    model_title=gettext("Urban Flood Risk Mitigation"),
    userguide="urban_flood_mitigation.html",
    validate_spatial_overlap=True,
    different_projections_ok=True,
    aliases=("ufrm",),
    module_name=__name__,
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["aoi_watersheds_path", "rainfall_depth"],
        ["lulc_path", "curve_number_table_path", "soils_hydrological_group_raster_path"],
        ["built_infrastructure_vector_path", "infrastructure_damage_loss_table_path"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.AOI.model_copy(update=dict(id="aoi_watersheds_path")),
        spec.NumberInput(
            id="rainfall_depth",
            name=gettext("rainfall depth"),
            about=gettext("Depth of rainfall for the design storm of interest."),
            units=u.millimeter,
            expression="value > 0"
        ),
        spec.SingleBandRasterInput(
            id="lulc_path",
            name=gettext("land use/land cover"),
            about=gettext(
                "Map of LULC. All values in this raster must have corresponding entries"
                " in the Biophysical Table."
            ),
            data_type=int,
            units=None,
            projected=True
        ),
        spec.SOIL_GROUP.model_copy(update=dict(
            id="soils_hydrological_group_raster_path",
            projected=True
        )),
        spec.CSVInput(
            id="curve_number_table_path",
            name=gettext("biophysical table"),
            about=gettext(
                "Table of curve number data for each LULC class. All LULC codes in the"
                " LULC raster must have corresponding entries in this table for each soil"
                " group."
            ),
            columns=[
                spec.IntegerInput(
                    id="lucode",
                    about=gettext(
                        "LULC codes from the LULC raster. Each code must be a unique"
                        " integer."
                    )
                ),
                spec.NumberInput(
                    id="cn_a",
                    about=gettext(
                        "The curve number value for this LULC type in the soil group"
                        " code A. Curve numbers must be greater than 0 and less than"
                        " or equal to 100."
                    ),
                    expression="0 < value <= 100",
                    units=u.none
                ),
                spec.NumberInput(
                    id="cn_b",
                    about=gettext(
                        "The curve number value for this LULC type in the soil group"
                        " code B. Curve numbers must be greater than 0 and less than"
                        " or equal to 100."
                    ),
                    expression="0 < value <= 100",
                    units=u.none
                ),
                spec.NumberInput(
                    id="cn_c",
                    about=gettext(
                        "The curve number value for this LULC type in the soil group"
                        " code C. Curve numbers must be greater than 0 and less than"
                        " or equal to 100."
                    ),
                    expression="0 < value <= 100",
                    units=u.none
                ),
                spec.NumberInput(
                    id="cn_d",
                    about=gettext(
                        "The curve number value for this LULC type in the soil group"
                        " code D. Curve numbers must be greater than 0 and less than"
                        " or equal to 100."
                    ),
                    expression="0 < value <= 100",
                    units=u.none
                )
            ],
            index_col="lucode"
        ),
        spec.VectorInput(
            id="built_infrastructure_vector_path",
            name=gettext("built infrastructure"),
            about=gettext("Map of building footprints."),
            required=False,
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[
                spec.IntegerInput(
                    id="type",
                    about=gettext(
                        "Code indicating the building type. These codes must match those"
                        " in the Damage Loss Table."
                    )
                )
            ],
            projected=None
        ),
        spec.CSVInput(
            id="infrastructure_damage_loss_table_path",
            name=gettext("damage loss table"),
            about=(
                "Table of potential damage loss data for each building type. All values"
                " in the Built Infrastructure vector 'type' field must have corresponding"
                " entries in this table. Required if the Built Infrastructure vector is"
                " provided."
            ),
            required="built_infrastructure_vector_path",
            columns=[
                spec.IntegerInput(id="type", about=gettext("Building type code.")),
                spec.NumberInput(
                    id="damage",
                    about=gettext("Potential damage loss for this building type."),
                    units=u.currency / u.meter**2
                )
            ],
            index_col="type"
        )
    ],
    outputs=[
        spec.SingleBandRasterOutput(
            id="runoff_retention_index",
            path="Runoff_retention_index.tif",
            about=gettext("Map of runoff retention index."),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="runoff_retention_m3",
            path="Runoff_retention_m3.tif",
            about=gettext("Map of runoff retention volume."),
            data_type=float,
            units=u.meter**3
        ),
        spec.SingleBandRasterOutput(
            id="q_mm",
            path="Q_mm.tif",
            about=gettext("Map of runoff."),
            data_type=float,
            units=u.millimeter
        ),
        spec.VectorOutput(
            id="flood_risk_service",
            path="flood_risk_service.shp",
            about=gettext("Aggregated results for each area of interest."),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[
                spec.NumberOutput(
                    id="rnf_rt_idx",
                    about=gettext("Average runoff retention index."),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="rnf_rt_m3",
                    about=gettext("Average runoff retention volume."),
                    units=u.meter**3
                ),
                spec.NumberOutput(
                    id="flood_vol", about=gettext("Total flood volume"), units=u.meter**3
                ),
                spec.NumberOutput(
                    id="aff_bld",
                    about=gettext("Total potential damage to built infrastructure."),
                    created_if="built_infrastructure_vector_path",
                    units=u.currency
                ),
                spec.NumberOutput(
                    id="serv_blt",
                    about=gettext("Total service value of built infrastructure."),
                    created_if="built_infrastructure_vector_path",
                    units=u.currency * u.meter**3
                )
            ]
        ),
        spec.SingleBandRasterOutput(
            id="q_m3",
            path="intermediate_files/Q_m3.tif",
            about=gettext("Map of runoff volume."),
            data_type=float,
            units=u.meter**3
        ),
        spec.VectorOutput(
            id="reprojected_aoi",
            path="intermediate_files/reprojected_aoi.shp",
            about=gettext(
                "Copy of AOI vector reprojected to the same spatial reference as"
                " the LULC."
            ),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[]
        ),
        spec.VectorOutput(
            id="structures_reprojected",
            path="intermediate_files/structures_reprojected.shp",
            about=gettext(
                "Copy of built infrastructure vector reprojected to the same"
                " spatial reference as the LULC."
            ),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[]
        ),
        spec.SingleBandRasterOutput(
            id="aligned_lulc",
            path="intermediate_files/aligned_lulc.tif",
            about=gettext("Aligned and clipped copy of the LULC."),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="aligned_soils_hydrological_group",
            path="intermediate_files/aligned_soils_hydrological_group.tif",
            about=gettext("Aligned and clipped copy of the soils map."),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="cn_raster",
            path="intermediate_files/cn_raster.tif",
            about=gettext("Map of curve number."),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="s_max",
            path="intermediate_files/s_max.tif",
            about=gettext("Map of potential retention."),
            data_type=float,
            units=u.millimeter
        ),
        spec.TASKGRAPH_CACHE
    ]
)


def execute(args):
    """Urban Flood Risk Mitigation.

    The model computes the peak flow attenuation for each pixel, delineates
    areas benefiting from this service, then calculates the monetary value of
    potential avoided damage to built infrastructure.

    Args:
        args['workspace_dir'] (string): a path to the directory that will
            write output and other temporary files during calculation.
        args['results_suffix'] (string): appended to any output file name.
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
        args['built_infrastructure_vector_path'] (string): (optional) path to
            a vector with built infrastructure footprints. Attribute table
            contains a column 'Type' with integers (e.g. 1=residential,
            2=office, etc.).
        args['infrastructure_damage_loss_table_path'] (string): (optional)
            path to a CSV table with columns 'Type' and 'Damage' with values
            of built infrastructure type from the 'Type' field in
            ``args['built_infrastructure_vector_path']`` and potential damage
            loss (in currency/m^2).
        args['n_workers'] (int): (optional) if present, indicates how many
            worker processes should be used in parallel processing. -1
            indicates single process mode, 0 is single process but
            non-blocking mode, and >= 1 is number of processes.

    Returns:
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths

    """
    args, file_registry, task_graph = MODEL_SPEC.setup(args)

    # Align LULC with soils
    lulc_raster_info = pygeoprocessing.get_raster_info(
        args['lulc_path'])
    target_pixel_size = lulc_raster_info['pixel_size']
    pixel_area = abs(target_pixel_size[0] * target_pixel_size[1])
    target_sr_wkt = lulc_raster_info['projection_wkt']

    soil_raster_info = pygeoprocessing.get_raster_info(
        args['soils_hydrological_group_raster_path'])

    align_raster_stack_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(
            [args['lulc_path'], args['soils_hydrological_group_raster_path']],
            [file_registry['aligned_lulc'],
            file_registry['aligned_soils_hydrological_group']],
            ['mode', 'mode'],
            target_pixel_size, 'intersection'),
        kwargs={
            'target_projection_wkt': target_sr_wkt,
            'base_vector_path_list': [args['aoi_watersheds_path']],
            'raster_align_index': 0},
        target_path_list=[file_registry['aligned_lulc'], file_registry['aligned_soils_hydrological_group']],
        task_name='align raster stack')

    # Load CN table
    cn_df = MODEL_SPEC.get_input(
        'curve_number_table_path').get_validated_dataframe(
        args['curve_number_table_path'])

    # make cn_table into a 2d array where first dim is lucode, second is
    # 0..3 to correspond to CN_A..CN_D
    data = []
    row_ind = []
    col_ind = []
    for lucode, row in cn_df.iterrows():
        data.extend([row[f'cn_{soil_id}'] for soil_id in ['a', 'b', 'c', 'd']])
        row_ind.extend([lucode] * 4)
    col_ind = [0, 1, 2, 3] * (len(row_ind) // 4)

    lucode_to_cn_table = scipy.sparse.csr_matrix((data, (row_ind, col_ind)))

    cn_nodata = -1
    lucode_nodata = lulc_raster_info['nodata'][0]
    soil_type_nodata = soil_raster_info['nodata'][0]

    align_raster_stack_task.join()

    cn_raster_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(file_registry['aligned_lulc'], 1),
             (file_registry['aligned_soils_hydrological_group'], 1),
             (lucode_nodata, 'raw'), (soil_type_nodata, 'raw'),
             (cn_nodata, 'raw'), (lucode_to_cn_table, 'raw')], _lu_to_cn_op,
            file_registry['cn_raster'], gdal.GDT_Float32, cn_nodata),
        target_path_list=[file_registry['cn_raster']],
        dependent_task_list=[align_raster_stack_task],
        task_name='create Curve Number raster')

    # Generate S_max
    s_max_nodata = -9999
    s_max_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(file_registry['cn_raster'], 1), (cn_nodata, 'raw'), (s_max_nodata, 'raw')],
            _s_max_op, file_registry['s_max'], gdal.GDT_Float32, s_max_nodata),
        target_path_list=[file_registry['s_max']],
        dependent_task_list=[cn_raster_task],
        task_name='create S_max')

    # Generate Qpi
    q_pi_nodata = -9999
    q_pi_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(args['rainfall_depth'], 'raw'), (file_registry['s_max'], 1),
             (s_max_nodata, 'raw'), (q_pi_nodata, 'raw')], _q_pi_op,
            file_registry['q_mm'], gdal.GDT_Float32, q_pi_nodata),
        target_path_list=[file_registry['q_mm']],
        dependent_task_list=[s_max_task],
        task_name='create Q_mm.tif')

    # Generate Runoff Retention
    runoff_retention_nodata = -9999
    runoff_retention_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([
            (file_registry['q_mm'], 1), (args['rainfall_depth'], 'raw'),
            (q_pi_nodata, 'raw'), (runoff_retention_nodata, 'raw')],
            _runoff_retention_op, file_registry['runoff_retention_index'],
            gdal.GDT_Float32, runoff_retention_nodata),
        target_path_list=[file_registry['runoff_retention_index']],
        dependent_task_list=[q_pi_task],
        task_name='generate runoff retention')

    # calculate runoff retention volume
    runoff_retention_vol_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([
            (file_registry['runoff_retention_index'], 1),
            (runoff_retention_nodata, 'raw'),
            (args['rainfall_depth'], 'raw'),
            (abs(target_pixel_size[0]*target_pixel_size[1]), 'raw'),
            (runoff_retention_nodata, 'raw')], _runoff_retention_vol_op,
            file_registry['runoff_retention_m3'], gdal.GDT_Float32,
            runoff_retention_nodata),
        target_path_list=[file_registry['runoff_retention_m3']],
        dependent_task_list=[runoff_retention_task],
        task_name='calculate runoff retention vol')

    # calculate flood vol raster
    flood_vol_nodata = -1
    flood_vol_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(file_registry['q_mm'], 1), (q_pi_nodata, 'raw'),
             (pixel_area, 'raw'), (flood_vol_nodata, 'raw')],
            _flood_vol_op, file_registry['q_m3'], gdal.GDT_Float32,
            flood_vol_nodata),
        target_path_list=[file_registry['q_m3']],
        dependent_task_list=[q_pi_task],
        task_name='calculate service built raster')

    reprojected_aoi_task = task_graph.add_task(
        func=pygeoprocessing.reproject_vector,
        args=(
            args['aoi_watersheds_path'],
            target_sr_wkt,
            file_registry['reprojected_aoi']),
        kwargs={'driver_name': 'ESRI Shapefile'},
        target_path_list=[file_registry['reprojected_aoi']],
        task_name='reproject aoi/watersheds')

    # Determine flood_volume over the watershed
    flood_volume_in_aoi_task = task_graph.add_task(
        func=pygeoprocessing.zonal_statistics,
        args=(
            (file_registry['q_m3'], 1),
            file_registry['reprojected_aoi']),
        store_result=True,
        dependent_task_list=[flood_vol_task, reprojected_aoi_task],
        task_name='zonal_statistics over the flood_volume raster')

    runoff_retention_stats_task = task_graph.add_task(
        func=pygeoprocessing.zonal_statistics,
        args=(
            (file_registry['runoff_retention_index'], 1),
            file_registry['reprojected_aoi']),
        store_result=True,
        dependent_task_list=[runoff_retention_task, reprojected_aoi_task],
        task_name='zonal_statistics over runoff_retention raster')

    runoff_retention_volume_stats_task = task_graph.add_task(
        func=pygeoprocessing.zonal_statistics,
        args=(
            (file_registry['runoff_retention_m3'], 1),
            file_registry['reprojected_aoi']),
        store_result=True,
        dependent_task_list=[runoff_retention_vol_task, reprojected_aoi_task],
        task_name='zonal_statistics over runoff_retention_volume raster')

    damage_per_aoi_stats = None
    flood_volume_stats = flood_volume_in_aoi_task.get()
    summary_tasks = [
        flood_volume_in_aoi_task,
        runoff_retention_stats_task,
        runoff_retention_volume_stats_task]
    if args['built_infrastructure_vector_path']:
        # Reproject the built infrastructure vector to the target SRS.
        reproject_built_infrastructure_task = task_graph.add_task(
            func=pygeoprocessing.reproject_vector,
            args=(args['built_infrastructure_vector_path'],
                  target_sr_wkt,
                  file_registry['structures_reprojected']),
            kwargs={'driver_name': 'ESRI Shapefile'},
            target_path_list=[file_registry['structures_reprojected']],
            task_name='reproject built infrastructure to target SRS')

        # determine the total damage to all infrastructure in the watershed/AOI
        damage_to_infrastructure_in_aoi_task = task_graph.add_task(
            func=_calculate_damage_to_infrastructure_in_aoi,
            args=(file_registry['reprojected_aoi'],
                  file_registry['structures_reprojected'],
                  args['infrastructure_damage_loss_table_path']),
            store_result=True,
            dependent_task_list=[
                reprojected_aoi_task,
                reproject_built_infrastructure_task],
            task_name='calculate damage to infrastructure in aoi')

        damage_per_aoi_stats = damage_to_infrastructure_in_aoi_task.get()

        # It isn't strictly necessary for us to append this task to
        # ``summary_tasks`` here, since the ``.get()`` calls below will block
        # until those tasks complete.  I'm adding these tasks ere anyways
        # "just in case".
        summary_tasks.append(damage_to_infrastructure_in_aoi_task)

    _ = task_graph.add_task(
        func=_write_summary_vector,
        args=(file_registry['reprojected_aoi'],
              file_registry['flood_risk_service']),
        kwargs={
            'runoff_ret_stats': runoff_retention_stats_task.get(),
            'runoff_ret_vol_stats': runoff_retention_volume_stats_task.get(),
            'damage_per_aoi_stats': damage_per_aoi_stats,
            'flood_volume_stats': flood_volume_stats,
        },
        target_path_list=[file_registry['flood_risk_service']],
        task_name='write summary stats to flood_risk_service.shp',
        dependent_task_list=summary_tasks)

    task_graph.close()
    task_graph.join()
    return file_registry.registry


def _write_summary_vector(
        source_aoi_vector_path, target_vector_path, runoff_ret_stats,
        runoff_ret_vol_stats, flood_volume_stats, damage_per_aoi_stats=None):
    """Write a vector with summary statistics.

    This vector will always contain three fields::

        * ``'flood_vol'``: The volume of flood (runoff), in m3, per watershed.
        * ``'rnf_rt_idx'``: Average of runoff retention values per watershed
        * ``'rnf_rt_m3'``: Sum of runoff retention volumes, in m3,
          per watershed.

    If ``damage_per_aoi_stats`` is provided, then these additional columns will
    be written to the vector::

        * ``'aff_bld'``: Potential damage to built infrastructure in currency
          units, per watershed.
        * ``'serv_blt'``: Spatial indicator of the importance of the runoff
          retention service

    Args:
        source_aoi_vector_path (str): The path to a GDAL vector that exists on
            disk.
        target_vector_path (str): The path to a vector that will be
            created.  If a file already exists at this path, it will be deleted
            before the new file is created.  This filepath must end with the
            extension ``.shp``, as the file created will be an ESRI Shapefile.
        runoff_ret_stats (dict): A dict representing summary statistics of the
            runoff raster. If provided, it must be a dictionary mapping feature
            IDs from ``source_aoi_vector_path`` to dicts with ``'count'`` and
            ``'sum'`` keys.
        runoff_ret_vol_stats (dict): A dict representing summary statistics of
            the runoff volume raster. If provided, it must be a dictionary
            mapping feature IDs from ``source_aoi_vector_path`` to dicts with
            ``'count'`` and ``'sum'`` keys.
        flood_volume_stats(dict): A dict mapping feature IDs from
            ``source_aoi_vector_path`` to float values representing the flood
            volume over the AOI.
        damage_per_aoi_stats (dict): A dict mapping feature IDs from
            ``source_aoi_vector_path`` to float values representing the total
            damage to built infrastructure in that watershed.

    Returns:
        ``None``
    """
    source_aoi_vector = gdal.OpenEx(source_aoi_vector_path, gdal.OF_VECTOR)
    esri_driver = gdal.GetDriverByName('ESRI Shapefile')
    esri_driver.CreateCopy(target_vector_path, source_aoi_vector)
    target_watershed_vector = gdal.OpenEx(target_vector_path,
                                          gdal.OF_VECTOR | gdal.GA_Update)
    target_watershed_layer = target_watershed_vector.GetLayer()

    target_fields = ['rnf_rt_idx', 'rnf_rt_m3', 'flood_vol']
    if damage_per_aoi_stats is not None:
        target_fields += ['aff_bld', 'serv_blt']

    for field_name in target_fields:
        field_def = ogr.FieldDefn(field_name, ogr.OFTReal)
        field_def.SetWidth(36)
        field_def.SetPrecision(11)
        target_watershed_layer.CreateField(field_def)

    target_watershed_layer.ResetReading()
    for target_feature in target_watershed_layer:
        feature_id = target_feature.GetFID()

        pixel_count = runoff_ret_stats[feature_id]['count']
        if pixel_count > 0:
            mean_value = (
                runoff_ret_stats[feature_id]['sum'] / float(pixel_count))
            target_feature.SetField('rnf_rt_idx', float(mean_value))

        target_feature.SetField(
            'rnf_rt_m3', float(
                runoff_ret_vol_stats[feature_id]['sum']))

        if damage_per_aoi_stats is not None:
            pixel_count = runoff_ret_vol_stats[feature_id]['count']
            if pixel_count > 0:
                damage_sum = damage_per_aoi_stats[feature_id]
                target_feature.SetField('aff_bld', damage_sum)

                # This is the service_built equation.
                target_feature.SetField(
                    'serv_blt', float(
                        damage_sum * runoff_ret_vol_stats[feature_id]['sum']))

        target_feature.SetField(
            'flood_vol', float(flood_volume_stats[feature_id]['sum']))

        target_watershed_layer.SetFeature(target_feature)

    target_watershed_layer.SyncToDisk()
    target_watershed_layer = None
    target_watershed_vector = None


def _calculate_damage_to_infrastructure_in_aoi(
        aoi_vector_path, structures_vector_path, structures_damage_table):
    """Determine the damage to infrastructure in each AOI feature.

    Args:
        aoi_vector_path (str): Path to a GDAL vector of AOI or watershed
            polygons.  Must be in the same projection as
            ``structures_vector_path``.
        structures_vector_path (str): Path to a GDAL vector of built
            infrastructure polygons.  Must be in the same projection as
            ``aoi_vector_path``.  Must have a ``Type`` column matching a type
            in the ``structures_damage_table`` table.
        structures_damage_table (str): Path to a CSV containing information
            about the damage to each type of structure. This table must have
            the ``Type`` and ``Damage`` columns.

    Returns:
        A ``dict`` mapping the FID of geometries in ``aoi_vector_path`` with
        the ``float`` total damage to infrastructure in that AOI/watershed.
    """
    infrastructure_vector = gdal.OpenEx(structures_vector_path, gdal.OF_VECTOR)
    infrastructure_layer = infrastructure_vector.GetLayer()

    damage_type_map = MODEL_SPEC.get_input(
        'infrastructure_damage_loss_table_path').get_validated_dataframe(
        structures_damage_table)['damage'].to_dict()

    infrastructure_layer_defn = infrastructure_layer.GetLayerDefn()
    type_index = -1
    for field_defn in infrastructure_layer.schema:
        field_name = field_defn.GetName()
        if field_name.lower() == 'type':
            type_index = infrastructure_layer_defn.GetFieldIndex(field_name)
            break

    if type_index == -1:
        raise ValueError(
            f"Could not find field 'Type' in {structures_vector_path}")

    structures_index = rtree.index.Index(interleaved=True)
    for infrastructure_feature in infrastructure_layer:
        infrastructure_geometry = infrastructure_feature.GetGeometryRef()

        # We've had a case on the forums where a user provided an
        # infrastructure vector with either invalid or missing geometries. This
        # allows us to handle these in the model run itself.
        if not infrastructure_geometry:
            LOGGER.debug(
                f'Infrastructure feature {infrastructure_feature.GetFID()} has '
                'no geometry; skipping.')
            continue

        shapely_geometry = shapely.wkb.loads(
            bytes(infrastructure_geometry.ExportToWkb()))

        structures_index.insert(
            infrastructure_feature.GetFID(), shapely_geometry.bounds)

    aoi_vector = gdal.OpenEx(aoi_vector_path, gdal.OF_VECTOR)
    aoi_layer = aoi_vector.GetLayer()

    aoi_damage = {}
    for aoi_feature in aoi_layer:
        aoi_geometry = aoi_feature.GetGeometryRef()
        aoi_geometry_shapely = shapely.wkb.loads(
            bytes(aoi_geometry.ExportToWkb()))
        aoi_geometry_prep = shapely.prepared.prep(aoi_geometry_shapely)

        total_damage = 0
        for infrastructure_fid in structures_index.intersection(
                aoi_geometry_shapely.bounds):
            infrastructure_feature = infrastructure_layer.GetFeature(
                infrastructure_fid)
            infrastructure_geometry = shapely.wkb.loads(
                bytes(infrastructure_feature.GetGeometryRef().ExportToWkb()))
            if aoi_geometry_prep.intersects(infrastructure_geometry):
                intersection_geometry = aoi_geometry_shapely.intersection(
                    infrastructure_geometry)
                damage_type = int(infrastructure_feature.GetField(type_index))
                total_damage += (
                    intersection_geometry.area * damage_type_map[damage_type])

        aoi_damage[aoi_feature.GetFID()] = total_damage

    return aoi_damage


def _flood_vol_op(
        q_pi_array, q_pi_nodata, pixel_area, target_nodata):
    """Calculate vol of flood water.

    Parmeters:
        rainfall_depth (float): depth of rainfall in mm.
        q_pi_array (numpy.ndarray): quick flow array.
        q_pi_nodata (float): nodata for q_pi.
        pixel_area (float): area of pixel in m^2.
        target_nodata (float): output nodata value.

    Returns:
        numpy array of flood volume per pixel in m^3.

    """
    result = numpy.empty(q_pi_array.shape, dtype=numpy.float32)
    result[:] = target_nodata
    valid_mask = ~pygeoprocessing.array_equals_nodata(q_pi_array, q_pi_nodata)
    # 0.001 converts mm (quickflow) to m (pixel area units)
    result[valid_mask] = (
        q_pi_array[valid_mask] * pixel_area * 0.001)
    return result


def _runoff_retention_vol_op(
        runoff_retention_array, runoff_retention_nodata, p_value,
        cell_area, target_nodata):
    """Calculate peak flow retention as a vol.

    Args:
        runoff_retention_array (numpy.ndarray): proportion of pixel retention.
        runoff_retention_nodata (float): nodata value for corresponding array.
        p_value (float): precipitation depth in mm.
        cell_area (float): area of cell in m^2.
        target_nodata (float): target nodata to write.

    Returns:
        (runoff_retention * p_value * pixel_area * 10e-3)

    """
    result = numpy.empty(runoff_retention_array.shape, dtype=numpy.float32)
    result[:] = target_nodata
    valid_mask = ~pygeoprocessing.array_equals_nodata(
        runoff_retention_array, runoff_retention_nodata)
    # the 1e-3 converts the mm of p_value to meters.
    result[valid_mask] = (
        runoff_retention_array[valid_mask] * p_value * cell_area * 1e-3)
    return result


def _runoff_retention_op(q_pi_array, p_value, q_pi_nodata, result_nodata):
    """Calculate peak flow retention.

    Args:
        q_pi_array (numpy.ndarray): quick flow array.
        p_value (float): precipition in mm.
        q_pi_nodata (float): nodata for q_pi.
        pixel_area (float): area of pixel in m^2.
        target_nodata (float): output nodata value.

    Returns:
        1 - q_pi/p

    """
    result = numpy.empty_like(q_pi_array)
    result[:] = result_nodata
    valid_mask = numpy.ones(q_pi_array.shape, dtype=bool)
    if q_pi_nodata is not None:
        valid_mask[:] = ~pygeoprocessing.array_equals_nodata(q_pi_array, q_pi_nodata)
    result[valid_mask] = 1 - (q_pi_array[valid_mask] / p_value)
    return result


def _q_pi_op(p_value, s_max_array, s_max_nodata, result_nodata):
    """Calculate peak flow Q (mm) with the Curve Number method.

    Args:
        p_value (float): precipitation in mm.
        s_max_array (numpy.ndarray): max S value per pixel.
        s_max_nodata (float): nodata value for s_max_array.
        result_nodata (float): return value nodata.

    Returns:
        ndarray of peak flow.

    """
    lam = 0.2  # this value of lambda is hard-coded in the design doc.
    result = numpy.empty_like(s_max_array)
    result[:] = result_nodata

    zero_mask = (p_value <= lam * s_max_array)
    non_nodata_mask = numpy.ones(s_max_array.shape, dtype=bool)
    if s_max_nodata is not None:
        non_nodata_mask[:] = ~pygeoprocessing.array_equals_nodata(
            s_max_array, s_max_nodata)

    # valid if not nodata and not going to be set to 0.
    valid_mask = non_nodata_mask & ~zero_mask
    result[valid_mask] = (
        p_value - lam * s_max_array[valid_mask])**2 / (
            p_value + (1 - lam) * s_max_array[valid_mask])
    # any non-nodata result that should be zero is set so.
    result[zero_mask & non_nodata_mask] = 0
    return result


def _s_max_op(cn_array, cn_nodata, result_nodata):
    """Calculate S_max from the curve number.

    Args:
        cn_array (numpy.ndarray): curve number array.
        cn_nodata (float): nodata value for cn_array.
        result_nodata (float): output nodata value.

    Return:
        ndarray of Smax calcualted from curve number.

    """
    result = numpy.empty_like(cn_array, dtype=numpy.float32)
    result[:] = result_nodata
    zero_mask = cn_array == 0
    valid_mask = ~zero_mask
    if cn_nodata is not None:
        valid_mask[:] &= ~pygeoprocessing.array_equals_nodata(cn_array, cn_nodata)
    result[valid_mask] = 25400 / cn_array[valid_mask] - 254
    # Curve Number of 0 means infitite retention so set s_max to a value
    # higher than any possible storm depth. Largest storm depth is recorded
    # at 6,433mm. 
    result[zero_mask] = 100000
    return result


def _lu_to_cn_op(
        lucode_array, soil_type_array, lucode_nodata, soil_type_nodata,
        cn_nodata, lucode_to_cn_table):
    """Map combination landcover soil type map to curve number raster.

    Args:
        lucode_array (numpy.ndarray): array of landcover codes.
        soil_type_array (numpy.ndarray): array of soil type values.
        lucode_nodata  (float): nodata value for corresponding array.
        soil_type_nodata (float): nodata value for corresponding array.
        cn_nodata (float): nodata value for return value array.
        lucode_to_cn_table (scipy.sparse.csr.csr_matrix):

    Returns:
        ndarray of curve numbers by looking up landcover type to soil type
        to then soil value.

    """
    result = numpy.empty_like(lucode_array, dtype=numpy.float32)
    result[:] = cn_nodata
    valid_mask = numpy.ones(lucode_array.shape, dtype=bool)
    if lucode_nodata is not None:
        valid_mask[:] &= ~pygeoprocessing.array_equals_nodata(
            lucode_array, lucode_nodata)
    if soil_type_nodata is not None:
        valid_mask[:] &= ~pygeoprocessing.array_equals_nodata(
            soil_type_array, soil_type_nodata)

    # this is an array where each column represents a valid landcover
    # pixel and the rows are the curve number index for the landcover
    # type under that pixel (0..3 are CN_A..CN_D and 4 is "unknown")
    valid_lucode_array = lucode_array[valid_mask].astype(int)

    try:
        cn_matrix = lucode_to_cn_table[valid_lucode_array]
    except IndexError:
        # Find the code that raised the IndexError, and possibly
        # any others that also would have.
        lucodes = numpy.unique(valid_lucode_array)
        missing_codes = lucodes[lucodes >= lucode_to_cn_table.shape[0]]
        raise ValueError(
            f'The biophysical table is missing a row for lucode(s) '
            f'{missing_codes.tolist()}')

    # Even without an IndexError, still must guard against
    # lucodes that can index into the sparse matrix but were
    # missing from the biophysical table. Do this by intersecting
    # rows with no stored values with the lulc array
    empty_rows = cn_matrix.getnnz(1) == 0
    if empty_rows.any():
        missing_codes = numpy.intersect1d(valid_lucode_array, empty_rows)
        raise ValueError(
            f'The biophysical table is missing a row for lucode(s) '
            f'{missing_codes.tolist()}')

    per_pixel_cn_array = (
        cn_matrix.toarray().reshape(
            (-1, 4))).transpose()

    # this is the soil type array with values ranging from 0..4 that will
    # choose the appropriate row for each pixel colum in
    # `per_pixel_cn_array`
    soil_choose_array = (
        soil_type_array[valid_mask].astype(numpy.int8))-1

    # soil arrays are 1 - 4, remap to 0 - 3 and choose from the per
    # pixel CN array
    try:
        result[valid_mask] = numpy.choose(
            soil_choose_array,
            per_pixel_cn_array)
    except ValueError as error:
        err_msg = (
            'Check that the Soil Group raster does not contain values '
            'other than (1, 2, 3, 4)')
        raise ValueError(str(error) + '\n' + err_msg)

    return result


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to ``execute``'s contract.

    Args:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in ``execute`` docstring.
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
    validation_warnings = validation.validate(args, MODEL_SPEC)

    sufficient_keys = validation.get_sufficient_keys(args)
    invalid_keys = validation.get_invalid_keys(validation_warnings)

    if ("curve_number_table_path" not in invalid_keys and
            "curve_number_table_path" in sufficient_keys):
        # Load CN table. Resulting DF has index and CN_X columns only.
        cn_df = MODEL_SPEC.get_input(
            'curve_number_table_path').get_validated_dataframe(
            args['curve_number_table_path'])
        # Check for NaN values.
        nan_mask = cn_df.isna()
        if nan_mask.any(axis=None):
            nan_lucodes = nan_mask[nan_mask.any(axis=1)].index
            # Convert numpy dtype values to native python types
            lucode_list = [i.item() for i in nan_lucodes.values]
            validation_warnings.append((
                ['curve_number_table_path'],
                f'Missing curve numbers for lucode(s) {lucode_list}'))

    return validation_warnings
