"""InVEST Sediment Delivery Ratio (SDR) module.

The SDR method in this model is based on:
    Winchell, M. F., et al. "Extension and validation of a geographic
    information system-based method for calculating the Revised Universal
    Soil Loss Equation length-slope factor for erosion risk assessments in
    large watersheds." Journal of Soil and Water Conservation 63.3 (2008):
    105-111.
"""
import logging
import os

import numpy
import pygeoprocessing
import pygeoprocessing.routing
import taskgraph
from osgeo import gdal
from osgeo import ogr

from .. import gettext
from .. import spec_utils
from .. import urban_nature_access
from .. import utils
from .. import validation
from ..model_metadata import MODEL_METADATA
from ..unit_registry import u
from . import sdr_core

LOGGER = logging.getLogger(__name__)

MODEL_SPEC = {
    "model_name": MODEL_METADATA["sdr"].model_title,
    "pyname": MODEL_METADATA["sdr"].pyname,
    "userguide": MODEL_METADATA["sdr"].userguide,
    "args_with_spatial_overlap": {
        "spatial_keys": ["dem_path", "erosivity_path", "erodibility_path",
                         "lulc_path", "drainage_path", "watersheds_path", ],
        "different_projections_ok": False,
    },
    "args": {
        "workspace_dir": spec_utils.WORKSPACE,
        "results_suffix": spec_utils.SUFFIX,
        "n_workers": spec_utils.N_WORKERS,
        "dem_path": {
            **spec_utils.DEM,
            "projected": True
        },
        "erosivity_path": {
            "type": "raster",
            "bands": {1: {
                "type": "number",
                "units": u.megajoule*u.millimeter/(u.hectare*u.hour*u.year)}},
            "projected": True,
            "about": gettext(
                "Map of rainfall erosivity, reflecting the intensity and "
                "duration of rainfall in the area of interest."),
            "name": gettext("erosivity")
        },
        "erodibility_path": {
            "type": "raster",
            "bands": {1: {
                "type": "number",
                "units": u.metric_ton*u.hectare*u.hour/(u.hectare*u.megajoule*u.millimeter)}},
            "projected": True,
            "about": gettext(
                "Map of soil erodibility, the susceptibility of soil "
                "particles to detachment and transport by rainfall and "
                "runoff."),
            "name": gettext("soil erodibility")
        },
        "lulc_path": {
            **spec_utils.LULC,
            "projected": True,
            "about": spec_utils.LULC['about'] + " " + gettext(
                "All values in this raster must "
                "have corresponding entries in the Biophysical Table.")
        },
        "watersheds_path": {
            "type": "vector",
            "geometries": spec_utils.POLYGONS,
            "projected": True,
            "fields": {},
            "about": gettext(
                "Map of the boundaries of the watershed(s) over which to "
                "aggregate results. Each watershed should contribute to a "
                "point of interest where water quality will be analyzed."),
            "name": gettext("Watersheds")
        },
        "biophysical_table_path": {
            "type": "csv",
            "index_col": "lucode",
            "columns": {
                "lucode": spec_utils.LULC_TABLE_COLUMN,
                "usle_c": {
                    "type": "ratio",
                    "about": gettext("Cover-management factor for the USLE")},
                "usle_p": {
                    "type": "ratio",
                    "about": gettext("Support practice factor for the USLE")}
            },
            "about": gettext(
                "A table mapping each LULC code to biophysical properties of "
                "that LULC class. All values in the LULC raster must have "
                "corresponding entries in this table."),
            "name": gettext("biophysical table")
        },
        "threshold_flow_accumulation": spec_utils.THRESHOLD_FLOW_ACCUMULATION,
        "k_param": {
            "type": "number",
            "units": u.none,
            "about": gettext("Borselli k parameter."),
            "name": gettext("Borselli k parameter")
        },
        "sdr_max": {
            "type": "ratio",
            "about": gettext("The maximum SDR value that a pixel can have."),
            "name": gettext("maximum SDR value")
        },
        "ic_0_param": {
            "type": "number",
            "units": u.none,
            "about": gettext("Borselli IC0 parameter."),
            "name": gettext("Borselli IC0 parameter")
        },
        "l_max": {
            "type": "number",
            "expression": "value > 0",
            "units": u.none,
            "about": gettext(
                "The maximum allowed value of the slope length parameter (L) "
                "in the LS factor."),
            "name": gettext("maximum l value"),
        },
        "drainage_path": {
            "type": "raster",
            "bands": {1: {"type": "integer"}},
            "required": False,
            "about": gettext(
                "Map of locations of artificial drainages that drain to the "
                "watershed. Pixels with 1 are drainages and are treated like "
                "streams. Pixels with 0 are not drainages."),
            "name": gettext("drainages")
        }
    },
    "outputs": {
        "avoided_erosion.tif": {
            "about": "The contribution of vegetation to keeping soil from eroding from each pixel. (Eq. (82))",
            "bands": {1: {
                "type": "number",
                "units": u.metric_ton/u.pixel
            }}
        },
        "avoided_export.tif": {
            "about": "The contribution of vegetation to keeping erosion from entering a stream. This combines local/on-pixel sediment retention with trapping of erosion from upslope of the pixel. (Eq. (83))",
            "bands": {1: {
                "type": "number",
                "units": u.metric_ton/u.pixel
            }}
        },
        "rkls.tif": {
            "bands": {1: {
                "type": "number",
                "units": u.metric_ton/u.pixel
            }},
            "about": "Total potential soil loss per pixel in the original land cover from the RKLS equation. Equivalent to the soil loss for bare soil. (Eq. (68), without applying the C or P factors)."
        },
        "sed_deposition.tif": {
            "about": "The total amount of sediment deposited on the pixel from upslope sources as a result of trapping. (Eq. (80))",
            "bands": {1: {
                "type": "number",
                "units": u.metric_ton/u.pixel
            }}
        },
        "sed_export.tif": {
            "about": "The total amount of sediment exported from each pixel that reaches the stream. (Eq. (76))",
            "bands": {1: {
                "type": "number",
                "units": u.metric_ton/u.pixel
            }}
        },
        "stream.tif": spec_utils.STREAM,
        "stream_and_drainage.tif": {
            "created_if": "drainage_path",
            "about": "This raster is the union of that layer with the calculated stream layer(Eq. (85)). Values of 1 represent streams, values of 0 are non-stream pixels.",
            "bands": {1: {"type": "integer"}}
        },
        "usle.tif": {
            "about": "Total potential soil loss per pixel in the original land cover calculated from the USLE equation. (Eq. (68))",
            "bands": {1: {
                "type": "number",
                "units": u.metric_ton/u.pixel
            }}
        },
        "watershed_results_sdr.shp": {
            "about": "Table containing biophysical values for each watershed",
            "geometries": spec_utils.POLYGONS,
            "fields": {
                "sed_export": {
                    "type": "number",
                    "units": u.metric_ton,
                    "about": "Total amount of sediment exported to the stream per watershed. (Eq. (77) with sum calculated over the watershed area)"
                },
                "usle_tot": {
                    "type": "number",
                    "units": u.metric_ton,
                    "about": "Total amount of potential soil loss in each watershed calculated by the USLE equation. (Sum of USLE from (68) over the watershed area)"
                },
                "avoid_exp": {
                    "type": "number",
                    "units": u.metric_ton,
                    "about": "The sum of avoided export in the watershed."
                },
                "avoid_eros": {
                    "type": "number",
                    "units": u.metric_ton,
                    "about": "The sum of avoided local erosion in the watershed"
                },
                "sed_dep": {
                    "type": "number",
                    "units": u.metric_ton,
                    "about": "Total amount of sediment deposited on the landscape in each watershed, which does not enter the stream."
                }
            }
        },
        "intermediate_outputs": {
            "type": "directory",
            "contents": {
                "cp.tif": {
                    "about": gettext(
                        "CP factor derived by mapping usle_c and usle_p from "
                        "the biophysical table to the LULC raster."),
                    "bands": {1: {"type": "ratio"}}
                },
                "d_dn.tif": {
                    "about": gettext(
                        "Downslope factor of the index of connectivity (Eq. (74))"),
                    "bands": {1: {"type": "number", "units": u.none}}
                },
                "d_up.tif": {
                    "about": gettext(
                        "Upslope factor of the index of connectivity (Eq. (73))"),
                    "bands": {1: {"type": "number", "units": u.none}}
                },
                "e_prime.tif": {
                    "about": gettext(
                        "Sediment downslope deposition, the amount of sediment "
                        "from a given pixel that does not reach a stream (Eq. (78))"),
                    "bands": {1: {
                        "type": "number",
                        "units": u.metric_ton/(u.hectare*u.year)
                    }}
                },
                "f.tif": {
                    "about": gettext(
                        "Map of sediment flux for sediment that does not "
                        "reach the stream (Eq. (81))"),
                    "bands": {1: {
                        "type": "number",
                        "units": u.metric_ton/(u.hectare*u.year)
                    }}
                },
                "flow_accumulation.tif": spec_utils.FLOW_ACCUMULATION,
                "flow_direction.tif": spec_utils.FLOW_DIRECTION,
                "ic.tif": {
                    "about": gettext("Index of connectivity (Eq. (70))"),
                    "bands": {1: {
                        "type": "number",
                        "units": u.none
                    }}
                },
                "ls.tif": {
                    "about": gettext("LS factor for USLE (Eq. (69))"),
                    "bands": {1: {
                        "type": "number",
                        "units": u.none
                    }}
                },
                "pit_filled_dem.tif": spec_utils.FILLED_DEM,
                "s_accumulation.tif": {
                    "about": gettext(
                        "Flow accumulation weighted by the thresholded slope. "
                        "Used in calculating s_bar."),
                    "bands": {1: {
                        "type": "number",
                        "units": u.none
                    }}
                },
                "s_bar.tif": {
                    "about": gettext(
                        "Mean thresholded slope gradient of the upslope "
                        "contributing area (in eq. (73))"),
                    "bands": {1: {
                        "type": "number",
                        "units": u.none
                    }}
                },
                "sdr_factor.tif": {
                    "about": gettext("Sediment delivery ratio (Eq. (75))"),
                    "bands": {1: {"type": "ratio"}}
                },
                "slope.tif": spec_utils.SLOPE,
                "slope_threshold.tif": {
                    "about": gettext(
                        "Percent slope, thresholded to be no less than 0.005 "
                        "and no greater than 1 (eq. (71)). 1 is equivalent to "
                        "a 45 degree slope."),
                    "bands": {1: {"type": "ratio"}}
                },
                "w_accumulation.tif": {
                    "about": gettext(
                        "Flow accumulation weighted by the thresholded "
                        "cover-management factor. Used in calculating w_bar."),
                    "bands": {1: {
                        "type": "number",
                        "units": u.none
                    }}
                },
                "w_bar.tif": {
                    "about": gettext(
                        "Mean thresholded cover-management factor for upslope "
                        "contributing area (in eq. (73))"),
                    "bands": {1: {"type": "ratio"}}
                },
                "w.tif": {
                    "about": gettext(
                        "Cover-management factor derived by mapping usle_c "
                        "from the biophysical table to the LULC raster."),
                    "bands": {1: {"type": "ratio"}}
                },
                "w_threshold.tif": {
                    "about": gettext(
                        "Cover-management factor thresholded to be no less "
                        "than 0.001 (eq. (72))"),
                    "bands": {1: {"type": "ratio"}}
                },
                "weighted_avg_aspect.tif": {
                    "about": gettext(
                        "Average aspect weighted by flow direction (in eq. (69))"),
                    "bands": {1: {"type": "number", "units": u.none}}
                },
                "what_drains_to_stream.tif": {
                    "about": gettext(
                        "Map of which pixels drain to a stream. A value of "
                        "1 means that at least some of the runoff from that "
                        "pixel drains to a stream in stream.tif. A value of 0 "
                        "means that it does not drain at all to any stream "
                        "in stream.tif."),
                    "bands": {1: {"type": "integer"}}
                },
                "ws_inverse.tif": {
                    "about": gettext(
                        "Inverse of the thresholded cover-management factor "
                        "times the thresholded slope (in eq. (74))"),
                    "bands": {1: {"type": "ratio"}}
                },
                "aligned_dem.tif": {
                    "about": gettext(
                        "Copy of the input DEM, clipped to the extent "
                        "of the other raster inputs."),
                    "bands": {1: {
                        "type": "number",
                        "units": u.meter
                    }}
                },
                "aligned_drainage.tif": {
                    "about": gettext(
                        "Copy of the input drainage map, clipped to "
                        "the extent of the other raster inputs and "
                        "aligned to the DEM."),
                    "bands": {1: {"type": "integer"}},
                },
                "aligned_erodibility.tif": {
                    "about": gettext(
                        "Copy of the input erodibility map, clipped to "
                        "the extent of the other raster inputs and "
                        "aligned to the DEM."),
                    "bands": {1: {
                        "type": "number",
                        "units": u.metric_ton*u.hectare*u.hour/(u.hectare*u.megajoule*u.millimeter)
                    }}
                },
                "aligned_erosivity.tif": {
                    "about": gettext(
                        "Copy of the input erosivity map, clipped to "
                        "the extent of the other raster inputs and "
                        "aligned to the DEM."),
                    "bands": {1: {
                        "type": "number",
                        "units": u.megajoule*u.millimeter/(u.hectare*u.hour*u.year)
                    }}
                },
                "aligned_lulc.tif": {
                    "about": gettext(
                        "Copy of the input Land Use Land Cover map, clipped to "
                        "the extent of the other raster inputs and "
                        "aligned to the DEM."),
                    "bands": {1: {"type": "integer"}},
                },
                "mask.tif": {
                    "about": gettext(
                        "A raster aligned to the DEM and clipped to the "
                        "extent of the other raster inputs. Pixel values "
                        "indicate where a nodata value exists in the stack "
                        "of aligned rasters (pixel value of 0), or if all "
                        "values in the stack of rasters at this pixel "
                        "location are valid."),
                    "bands": {1: {"type": "integer"}},
                },
                "masked_dem.tif": {
                    "about": gettext(
                        "A copy of the aligned DEM, masked using the mask raster."
                    ),
                    "bands": {1: {
                        "type": "number",
                        "units": u.meter}},
                },
                "masked_drainage.tif": {
                    "about": gettext(
                        "A copy of the aligned drainage map, masked using the "
                        "mask raster."
                    ),
                    "bands": {1: {"type": "integer"}}
                },
                "masked_erodibility.tif": {
                    "about": gettext(
                        "A copy of the aligned erodibility map, masked using "
                        "the mask raster."
                    ),
                    "bands": {1: {
                        "type": "number",
                        "units": u.metric_ton*u.hectare*u.hour/(u.hectare*u.megajoule*u.millimeter)
                    }},
                },
                "masked_erosivity.tif": {
                    "about": gettext(
                        "A copy of the aligned erosivity map, masked using "
                        "the mask raster."),
                    "bands": {1: {
                        "type": "number",
                        "units": u.megajoule*u.millimeter/(u.hectare*u.hour*u.year)
                    }},
                },
                "masked_lulc.tif": {
                    "about": gettext(
                        "A copy of the aligned Land Use Land Cover map, "
                        "masked using the mask raster."),
                    "bands": {1: {"type": "integer"}},
                }
            },
        },
        "taskgraph_cache": spec_utils.TASKGRAPH_DIR
    }
}

_OUTPUT_BASE_FILES = {
    'rkls_path': 'rkls.tif',
    'sed_export_path': 'sed_export.tif',
    'sed_deposition_path': 'sed_deposition.tif',
    'stream_and_drainage_path': 'stream_and_drainage.tif',
    'stream_path': 'stream.tif',
    'usle_path': 'usle.tif',
    'watershed_results_sdr_path': 'watershed_results_sdr.shp',
    'avoided_export_path': 'avoided_export.tif',
    'avoided_erosion_path': 'avoided_erosion.tif',
}

INTERMEDIATE_DIR_NAME = 'intermediate_outputs'

_INTERMEDIATE_BASE_FILES = {
    'aligned_dem_path': 'aligned_dem.tif',
    'aligned_drainage_path': 'aligned_drainage.tif',
    'aligned_erodibility_path': 'aligned_erodibility.tif',
    'aligned_erosivity_path': 'aligned_erosivity.tif',
    'aligned_lulc_path': 'aligned_lulc.tif',
    'mask_path': 'mask.tif',
    'masked_dem_path': 'masked_dem.tif',
    'masked_drainage_path': 'masked_drainage.tif',
    'masked_erodibility_path': 'masked_erodibility.tif',
    'masked_erosivity_path': 'masked_erosivity.tif',
    'masked_lulc_path': 'masked_lulc.tif',
    'cp_factor_path': 'cp.tif',
    'd_dn_path': 'd_dn.tif',
    'd_up_path': 'd_up.tif',
    'f_path': 'f.tif',
    'flow_accumulation_path': 'flow_accumulation.tif',
    'flow_direction_path': 'flow_direction.tif',
    'ic_path': 'ic.tif',
    'ls_path': 'ls.tif',
    'pit_filled_dem_path': 'pit_filled_dem.tif',
    's_accumulation_path': 's_accumulation.tif',
    's_bar_path': 's_bar.tif',
    'sdr_path': 'sdr_factor.tif',
    'slope_path': 'slope.tif',
    'thresholded_slope_path': 'slope_threshold.tif',
    'thresholded_w_path': 'w_threshold.tif',
    'w_accumulation_path': 'w_accumulation.tif',
    'w_bar_path': 'w_bar.tif',
    'w_path': 'w.tif',
    'ws_inverse_path': 'ws_inverse.tif',
    'e_prime_path': 'e_prime.tif',
    'drainage_mask': 'what_drains_to_stream.tif',
}


# Target nodata is for general rasters that are positive, and _IC_NODATA are
# for rasters that are any range
_TARGET_NODATA = -1.0
_BYTE_NODATA = 255
_IC_NODATA = float(numpy.finfo('float32').min)


def execute(args):
    """Sediment Delivery Ratio.

    This function calculates the sediment export and retention of a landscape
    using the sediment delivery ratio model described in the InVEST user's
    guide.

    Args:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output file names
        args['dem_path'] (string): path to a digital elevation raster
        args['erosivity_path'] (string): path to rainfall erosivity index
            raster
        args['erodibility_path'] (string): a path to soil erodibility raster
        args['lulc_path'] (string): path to land use/land cover raster
        args['watersheds_path'] (string): path to vector of the watersheds
        args['biophysical_table_path'] (string): path to CSV file with
            biophysical information of each land use classes.  contain the
            fields 'usle_c' and 'usle_p'
        args['threshold_flow_accumulation'] (number): number of upslope pixels
            on the dem to threshold to a stream.
        args['k_param'] (number): k calibration parameter
        args['sdr_max'] (number): max value the SDR
        args['ic_0_param'] (number): ic_0 calibration parameter
        args['drainage_path'] (string): (optional) path to drainage raster that
            is used to add additional drainage areas to the internally
            calculated stream layer
        args['l_max'] (number): the maximum allowed value of the slope length
            parameter (L) in the LS factor. If the calculated value of L
            exceeds 'l_max' it will be clamped to this value.
        args['n_workers'] (int): if present, indicates how many worker
            processes should be used in parallel processing. -1 indicates
            single process mode, 0 is single process but non-blocking mode,
            and >= 1 is number of processes.

    Returns:
        None.

    """
    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    biophysical_df = validation.get_validated_dataframe(
        args['biophysical_table_path'],
        **MODEL_SPEC['args']['biophysical_table_path'])

    # Test to see if c or p values are outside of 0..1
    for key in ['usle_c', 'usle_p']:
        for lulc_code, row in biophysical_df.iterrows():
            if row[key] < 0 or row[key] > 1:
                raise ValueError(
                    f'A value in the biophysical table is not a number '
                    f'within range 0..1. The offending value is in '
                    f'column "{key}", lucode row "{lulc_code}", '
                    f'and has value "{row[key]}"')

    intermediate_output_dir = os.path.join(
        args['workspace_dir'], INTERMEDIATE_DIR_NAME)
    output_dir = os.path.join(args['workspace_dir'])
    utils.make_directories([output_dir, intermediate_output_dir])

    f_reg = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir)], file_suffix)

    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous mode.
    task_graph = taskgraph.TaskGraph(
        os.path.join(output_dir, 'taskgraph_cache'),
        n_workers, reporting_interval=5.0)

    base_list = []
    aligned_list = []
    masked_list = []
    input_raster_key_list = ['dem', 'lulc', 'erosivity', 'erodibility']
    for file_key in input_raster_key_list:
        base_list.append(args[f"{file_key}_path"])
        aligned_list.append(f_reg[f"aligned_{file_key}_path"])
        masked_list.append(f_reg[f"masked_{file_key}_path"])
    # all continuous rasters can use bilinear, but lulc should be mode
    interpolation_list = ['bilinear', 'mode', 'bilinear', 'bilinear']

    drainage_present = False
    if 'drainage_path' in args and args['drainage_path'] != '':
        drainage_present = True
        input_raster_key_list.append('drainage')
        base_list.append(args['drainage_path'])
        aligned_list.append(f_reg['aligned_drainage_path'])
        masked_list.append(f_reg['masked_drainage_path'])
        interpolation_list.append('near')

    dem_raster_info = pygeoprocessing.get_raster_info(args['dem_path'])
    min_pixel_size = numpy.min(numpy.abs(dem_raster_info['pixel_size']))
    target_pixel_size = (min_pixel_size, -min_pixel_size)

    target_sr_wkt = dem_raster_info['projection_wkt']
    vector_mask_options = {
        'mask_vector_path': args['watersheds_path'],
    }
    align_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(
            base_list, aligned_list, interpolation_list,
            target_pixel_size, 'intersection'),
        kwargs={
            'target_projection_wkt': target_sr_wkt,
            'base_vector_path_list': (args['watersheds_path'],),
            'raster_align_index': 0,
            'vector_mask_options': vector_mask_options,
        },
        target_path_list=aligned_list,
        task_name='align input rasters')

    mutual_mask_task = task_graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs={
            'op': _create_mutual_mask_op,
            'rasters': aligned_list,
            'target_path': f_reg['mask_path'],
            'target_nodata': 0,
        },
        target_path_list=[f_reg['mask_path']],
        dependent_task_list=[align_task],
        task_name='create mask')

    mask_tasks = {}  # use a dict so we can put these in a loop
    for key, aligned_path, masked_path in zip(input_raster_key_list,
                                              aligned_list, masked_list):
        mask_tasks[f"masked_{key}"] = task_graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs={
                'op': _mask_single_raster_op,
                'rasters': [aligned_path, f_reg['mask_path']],
                'target_path': masked_path,
            },
            target_path_list=[masked_path],
            dependent_task_list=[mutual_mask_task, align_task],
            task_name=f'mask {key}')

    pit_fill_task = task_graph.add_task(
        func=pygeoprocessing.routing.fill_pits,
        args=(
            (f_reg['masked_dem_path'], 1),
            f_reg['pit_filled_dem_path']),
        target_path_list=[f_reg['pit_filled_dem_path']],
        dependent_task_list=[mask_tasks['masked_dem']],
        task_name='fill pits')

    slope_task = task_graph.add_task(
        func=pygeoprocessing.calculate_slope,
        args=(
            (f_reg['pit_filled_dem_path'], 1),
            f_reg['slope_path']),
        dependent_task_list=[pit_fill_task],
        target_path_list=[f_reg['slope_path']],
        task_name='calculate slope')

    threshold_slope_task = task_graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=threshold_slope_op,
            rasters=[f_reg['slope_path']],
            target_path=f_reg['thresholded_slope_path']),
        target_path_list=[f_reg['thresholded_slope_path']],
        dependent_task_list=[slope_task],
        task_name='threshold slope')

    flow_dir_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_dir_mfd,
        args=(
            (f_reg['pit_filled_dem_path'], 1),
            f_reg['flow_direction_path']),
        target_path_list=[f_reg['flow_direction_path']],
        dependent_task_list=[pit_fill_task],
        task_name='flow direction calculation')

    flow_accumulation_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_accumulation_mfd,
        args=(
            (f_reg['flow_direction_path'], 1),
            f_reg['flow_accumulation_path']),
        target_path_list=[f_reg['flow_accumulation_path']],
        dependent_task_list=[flow_dir_task],
        task_name='flow accumulation calculation')

    ls_factor_task = task_graph.add_task(
        func=_calculate_ls_factor,
        args=(
            f_reg['flow_accumulation_path'],
            f_reg['slope_path'],
            float(args['l_max']),
            f_reg['ls_path']),
        target_path_list=[f_reg['ls_path']],
        dependent_task_list=[
            flow_accumulation_task, slope_task],
        task_name='ls factor calculation')

    stream_task = task_graph.add_task(
        func=pygeoprocessing.routing.extract_streams_mfd,
        args=(
            (f_reg['flow_accumulation_path'], 1),
            (f_reg['flow_direction_path'], 1),
            float(args['threshold_flow_accumulation']),
            f_reg['stream_path']),
        kwargs={'trace_threshold_proportion': 0.7},
        target_path_list=[f_reg['stream_path']],
        dependent_task_list=[flow_accumulation_task],
        task_name='extract streams')

    if drainage_present:
        drainage_task = task_graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs=dict(
                op=add_drainage_op,
                rasters=[f_reg['stream_path'], f_reg['masked_drainage_path']],
                target_path=f_reg['stream_and_drainage_path'],
                target_dtype=numpy.uint8),
            target_path_list=[f_reg['stream_and_drainage_path']],
            dependent_task_list=[stream_task, mask_tasks['masked_drainage']],
            task_name='add drainage')
        drainage_raster_path_task = (
            f_reg['stream_and_drainage_path'], drainage_task)
    else:
        drainage_raster_path_task = (
            f_reg['stream_path'], stream_task)

    lulc_to_c = biophysical_df['usle_c'].to_dict()
    threshold_w_task = task_graph.add_task(
        func=_calculate_w,
        args=(
            lulc_to_c, f_reg['masked_lulc_path'], f_reg['w_path'],
            f_reg['thresholded_w_path']),
        target_path_list=[f_reg['w_path'], f_reg['thresholded_w_path']],
        dependent_task_list=[mask_tasks['masked_lulc']],
        task_name='calculate W')

    lulc_to_cp = (biophysical_df['usle_c'] * biophysical_df['usle_p']).to_dict()
    cp_task = task_graph.add_task(
        func=_calculate_cp,
        args=(
            lulc_to_cp, f_reg['masked_lulc_path'],
            f_reg['cp_factor_path']),
        target_path_list=[f_reg['cp_factor_path']],
        dependent_task_list=[mask_tasks['masked_lulc']],
        task_name='calculate CP')

    rkls_task = task_graph.add_task(
        func=_calculate_rkls,
        args=(
            f_reg['ls_path'],
            f_reg['masked_erosivity_path'],
            f_reg['masked_erodibility_path'],
            drainage_raster_path_task[0],
            f_reg['rkls_path']),
        target_path_list=[f_reg['rkls_path']],
        dependent_task_list=[
            mask_tasks['masked_erosivity'], mask_tasks['masked_erodibility'],
            drainage_raster_path_task[1], ls_factor_task],
        task_name='calculate RKLS')

    usle_task = task_graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=usle_op,
            rasters=[f_reg['rkls_path'], f_reg['cp_factor_path']],
            target_path=f_reg['usle_path']),
        target_path_list=[f_reg['usle_path']],
        dependent_task_list=[rkls_task, cp_task],
        task_name='calculate USLE')

    bar_task_map = {}
    for factor_path, factor_task, accumulation_path, out_bar_path, bar_id in [
            (f_reg['thresholded_w_path'], threshold_w_task,
             f_reg['w_accumulation_path'],
             f_reg['w_bar_path'],
             'w_bar'),
            (f_reg['thresholded_slope_path'], threshold_slope_task,
             f_reg['s_accumulation_path'],
             f_reg['s_bar_path'],
             's_bar')]:
        bar_task = task_graph.add_task(
            func=_calculate_bar_factor,
            args=(
                f_reg['flow_direction_path'], factor_path,
                f_reg['flow_accumulation_path'],
                accumulation_path, out_bar_path),
            target_path_list=[accumulation_path, out_bar_path],
            dependent_task_list=[
                factor_task, flow_accumulation_task, flow_dir_task],
            task_name='calculate %s' % bar_id)
        bar_task_map[bar_id] = bar_task

    d_up_task = task_graph.add_task(
        func=_calculate_d_up,
        args=(
            f_reg['w_bar_path'], f_reg['s_bar_path'],
            f_reg['flow_accumulation_path'], f_reg['d_up_path']),
        target_path_list=[f_reg['d_up_path']],
        dependent_task_list=[
            bar_task_map['s_bar'], bar_task_map['w_bar'],
            flow_accumulation_task],
        task_name='calculate Dup')

    inverse_ws_factor_task = task_graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=inverse_ws_op,
            rasters=[f_reg['thresholded_w_path'],
                     f_reg['thresholded_slope_path']],
            target_path=f_reg['ws_inverse_path']),
        target_path_list=[f_reg['ws_inverse_path']],
        dependent_task_list=[threshold_slope_task, threshold_w_task],
        task_name='calculate inverse ws factor')

    d_dn_task = task_graph.add_task(
        func=pygeoprocessing.routing.distance_to_channel_mfd,
        args=(
            (f_reg['flow_direction_path'], 1),
            (drainage_raster_path_task[0], 1),
            f_reg['d_dn_path']),
        kwargs={'weight_raster_path_band': (f_reg['ws_inverse_path'], 1)},
        target_path_list=[f_reg['d_dn_path']],
        dependent_task_list=[
            flow_dir_task, drainage_raster_path_task[1],
            inverse_ws_factor_task],
        task_name='calculating d_dn')

    ic_task = task_graph.add_task(
        func=_calculate_ic,
        args=(
            f_reg['d_up_path'], f_reg['d_dn_path'], f_reg['ic_path']),
        target_path_list=[f_reg['ic_path']],
        dependent_task_list=[d_up_task, d_dn_task],
        task_name='calculate ic')

    sdr_task = task_graph.add_task(
        func=_calculate_sdr,
        args=(
            float(args['k_param']), float(args['ic_0_param']),
            float(args['sdr_max']), f_reg['ic_path'],
            drainage_raster_path_task[0], f_reg['sdr_path']),
        target_path_list=[f_reg['sdr_path']],
        dependent_task_list=[ic_task],
        task_name='calculate sdr')

    sed_export_task = task_graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=numpy.multiply,  # export = USLE * SDR
            rasters=[f_reg['usle_path'], f_reg['sdr_path']],
            target_path=f_reg['sed_export_path']),
        target_path_list=[f_reg['sed_export_path']],
        dependent_task_list=[usle_task, sdr_task],
        task_name='calculate sed export')

    e_prime_task = task_graph.add_task(
        func=_calculate_e_prime,
        args=(
            f_reg['usle_path'], f_reg['sdr_path'],
            drainage_raster_path_task[0], f_reg['e_prime_path']),
        target_path_list=[f_reg['e_prime_path']],
        dependent_task_list=[usle_task, sdr_task],
        task_name='calculate export prime')

    sed_deposition_task = task_graph.add_task(
        func=sdr_core.calculate_sediment_deposition,
        args=(
            f_reg['flow_direction_path'], f_reg['e_prime_path'],
            f_reg['f_path'], f_reg['sdr_path'],
            f_reg['sed_deposition_path']),
        dependent_task_list=[e_prime_task, sdr_task, flow_dir_task],
        target_path_list=[f_reg['sed_deposition_path'], f_reg['f_path']],
        task_name='sediment deposition')

    avoided_erosion_task = task_graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=numpy.subtract,  # avoided erosion = rkls - usle
            rasters=[f_reg['rkls_path'], f_reg['usle_path']],
            target_path=f_reg['avoided_erosion_path']),
        dependent_task_list=[rkls_task, usle_task],
        target_path_list=[f_reg['avoided_erosion_path']],
        task_name='calculate avoided erosion')

    avoided_export_task = task_graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=_avoided_export_op,
            rasters=[f_reg['avoided_erosion_path'],
                     f_reg['sdr_path'],
                     f_reg['sed_deposition_path']],
            target_path=f_reg['avoided_export_path']),
        dependent_task_list=[avoided_erosion_task, sdr_task,
                             sed_deposition_task],
        target_path_list=[f_reg['avoided_export_path']],
        task_name='calculate total retention')

    _ = task_graph.add_task(
        func=_calculate_what_drains_to_stream,
        args=(f_reg['flow_direction_path'], f_reg['d_dn_path'],
              f_reg['drainage_mask']),
        target_path_list=[f_reg['drainage_mask']],
        dependent_task_list=[flow_dir_task, d_dn_task],
        task_name='write mask of what drains to stream')

    _ = task_graph.add_task(
        func=_generate_report,
        args=(
            args['watersheds_path'], f_reg['usle_path'],
            f_reg['sed_export_path'], f_reg['sed_deposition_path'],
            f_reg['avoided_export_path'], f_reg['avoided_erosion_path'],
            f_reg['watershed_results_sdr_path']),
        target_path_list=[f_reg['watershed_results_sdr_path']],
        dependent_task_list=[
            usle_task, sed_export_task, avoided_export_task,
            sed_deposition_task, avoided_erosion_task],
        task_name='generate report')

    task_graph.close()
    task_graph.join()


# raster_map op for building a mask where all pixels in the stack are valid.
def _create_mutual_mask_op(*arrays): return 1


# raster_map op for using a mask raster to mask out another raster.
def _mask_single_raster_op(source_array, mask_array): return source_array


def _avoided_export_op(avoided_erosion, sdr, sed_deposition):
    """raster_map equation: calculate total retention.

    Args:
        avoided_erosion (numpy.array): Avoided erosion values.
        sdr (numpy.array): SDR values.
        sed_deposition (numpy.array): Sediment deposition values.

    Returns:
        A ``numpy.array`` of computed total retention matching the shape of
        the input numpy arrays.
    """
    # avoided_erosion represents RLKS - RKLSCP (where RKLSCP is also
    # known as a modified USLE)
    return avoided_erosion * sdr + sed_deposition

def add_drainage_op(stream, drainage):
    """raster_map equation: add drainage mask to stream layer.

    Args:
        stream (numpy.array): binary array where 1 indicates
            a stream, and 0 is a valid landscape pixel but not a stream.
        drainage (numpy.array): binary array where 1 indicates any water
            reaching that pixel drains to a stream.

    Returns:
        numpy.array combination of stream and drainage
    """
    return numpy.where(drainage == 1, 1, stream)

# raster_map equation: calculate USLE
def usle_op(rkls, cp_factor): return rkls * cp_factor

# raster_map equation: calculate the inverse ws factor
def inverse_ws_op(w_factor, s_factor): return 1 / (w_factor * s_factor)


def _calculate_what_drains_to_stream(
        flow_dir_mfd_path, dist_to_channel_mfd_path, target_mask_path):
    """Create a mask indicating regions that do or do not drain to a stream.

    This is useful because ``pygeoprocessing.distance_to_stream_mfd`` may leave
    some unexpected regions as nodata if they do not drain to a stream.  This
    may be confusing behavior, so this mask is intended to locate what drains
    to a stream and what does not. A pixel doesn't drain to a stream if it has
    a defined flow direction but undefined distance to stream.

    Args:
        flow_dir_mfd_path (string): The path to an MFD flow direction raster.
            This raster must have a nodata value defined.
        dist_to_channel_mfd_path (string): The path to an MFD
            distance-to-channel raster.  This raster must have a nodata value
            defined.
        target_mask_path (string): The path to where the mask raster should be
            written.

    Returns:
        ``None``
    """
    flow_dir_mfd_nodata = pygeoprocessing.get_raster_info(
        flow_dir_mfd_path)['nodata'][0]
    dist_to_channel_nodata = pygeoprocessing.get_raster_info(
        dist_to_channel_mfd_path)['nodata'][0]

    def _what_drains_to_stream(flow_dir_mfd, dist_to_channel):
        """Determine which pixels do and do not drain to a stream.

        Args:
            flow_dir_mfd (numpy.array): A numpy array of MFD flow direction
                values.
            dist_to_channel (numpy.array): A numpy array of calculated
                distances to the nearest channel.

        Returns:
            A ``numpy.array`` of dtype ``numpy.uint8`` with pixels where:

                * ``255`` where ``flow_dir_mfd`` is nodata (and thus
                  ``dist_to_channel`` is also nodata).
                * ``0`` where ``flow_dir_mfd`` has data and ``dist_to_channel``
                  does not
                * ``1`` where ``flow_dir_mfd`` has data, and
                  ``dist_to_channel`` also has data.
        """
        drains_to_stream = numpy.full(
            flow_dir_mfd.shape, _BYTE_NODATA, dtype=numpy.uint8)
        valid_flow_dir = ~pygeoprocessing.array_equals_nodata(
            flow_dir_mfd, flow_dir_mfd_nodata)
        valid_dist_to_channel = (
            ~pygeoprocessing.array_equals_nodata(
                dist_to_channel, dist_to_channel_nodata) &
            valid_flow_dir)

        # Nodata where both flow_dir and dist_to_channel are nodata
        # 1 where flow_dir and dist_to_channel have values (drains to stream)
        # 0 where flow_dir has data and dist_to_channel doesn't (doesn't drain)
        drains_to_stream[valid_flow_dir & valid_dist_to_channel] = 1
        drains_to_stream[valid_flow_dir & ~valid_dist_to_channel] = 0
        return drains_to_stream

    pygeoprocessing.raster_calculator(
        [(flow_dir_mfd_path, 1), (dist_to_channel_mfd_path, 1)],
        _what_drains_to_stream, target_mask_path, gdal.GDT_Byte, _BYTE_NODATA)


def _calculate_ls_factor(
        flow_accumulation_path, slope_path, l_max,
        target_ls_factor_path):
    """Calculate LS factor.

    Calculates the LS factor using Equation 3 from "Extension and
    validation of a geographic information system-based method for calculating
    the Revised Universal Soil Loss Equation length-slope factor for erosion
    risk assessments in large watersheds".

    The equation for this is::

                 (upstream_area + pixel_area)^(m+1) - upstream_area^(m+1)
        LS = S * --------------------------------------------------------
                       (pixel_area^(m+2)) * aspect_dir * 22.13^(m)

    Where

        * ``S`` is the slope factor defined in equation 4 from the same paper,
          calculated by the following where ``b`` is the slope in radians:

          * ``S = 10.8 * sin(b) + 0.03`` where slope < 9%
          * ``S = 16.8 * sin(b) - 0.50`` where slope >= 9%

        * ``upstream_area`` is interpreted as the square root of the
          catchment area, to match SAGA-GIS's method for calculating LS
          Factor.
        * ``pixel_area`` is the area of the pixel in square meters.
        * ``m`` is the slope-length exponent of the RUSLE LS-factor,
          which, as discussed in Oliveira et al. 2013 is a function of the
          on-pixel slope theta:

          * ``m = 0.2`` when ``theta <= 1%``
          * ``m = 0.3`` when ``1% < theta <= 3.5%``
          * ``m = 0.4`` when ``3.5% < theta <= 5%``
          * ``m = 0.5`` when ``5% < theta <= 9%``
          * ``m = (beta / (1+beta)`` when ``theta > 9%``, where
            ``beta = (sin(theta) / 0.0896) / (3*sin(theta)^0.8 + 0.56)``

        * ``aspect_dir`` is calculated by ``|sin(alpha)| + |cos(alpha)|``
          for the given pixel.

    Oliveira et al can be found at:

        Oliveira, A.H., Silva, M.A. da, Silva, M.L.N., Curi, N., Neto, G.K.,
        Freitas, D.A.F. de, 2013. Development of Topographic Factor Modeling
        for Application in Soil Erosion Models, in: Intechopen (Ed.), Soil
        Processes and Current Trends in Quality Assessment. p. 28.

    Args:
        flow_accumulation_path (string): path to raster, pixel values are the
            contributing upslope area at that cell. Pixel size is square.
        slope_path (string): path to slope raster as a percent
        l_max (float): if the calculated value of L exceeds this value
            it is clamped to this value.
        target_ls_factor_path (string): path to output ls_prime_factor
            raster

    Returns:
        None

    """
    cell_size = abs(pygeoprocessing.get_raster_info(
        flow_accumulation_path)['pixel_size'][0])
    cell_area = cell_size ** 2

    def ls_factor_function(percent_slope, flow_accumulation):
        """Calculate the LS factor.

        Args:
            percent_slope (numpy.ndarray): slope in percent
            flow_accumulation (numpy.ndarray): upslope pixels
            l_max (float): max L factor, clamp to this value if L exceeds it

        Returns:
            ls_factor

        """
        # Although Desmet & Govers (1996) discusses "upstream contributing
        # area", this is not strictly defined. We decided to use the square
        # root of the upstream contributing area here as an estimate, which
        # matches the SAGA LS Factor option "square root of catchment area".
        # See the InVEST ADR-0001 for more information.
        # We subtract 1 from the flow accumulation because FA includes itself
        # in its count of pixels upstream and our LS factor equation wants only
        # those pixels that are strictly upstream.
        contributing_area = numpy.sqrt((flow_accumulation - 1) * cell_area)
        slope_in_radians = numpy.arctan(percent_slope / 100)

        aspect_length = (numpy.fabs(numpy.sin(slope_in_radians)) +
                         numpy.fabs(numpy.cos(slope_in_radians)))

        # From Equation 4 in "Extension and validation of a geographic
        # information system ..."
        slope_factor = numpy.where(
            percent_slope < 9,
            10.8 * numpy.sin(slope_in_radians) + 0.03,
            16.8 * numpy.sin(slope_in_radians) - 0.5)

        beta = (
            (numpy.sin(slope_in_radians) / 0.0896) /
            (3 * numpy.sin(slope_in_radians)**0.8 + 0.56))

        # Set m value via lookup table: Table 1 in
        # InVEST Sediment Model_modifications_10-01-2012_RS.docx
        # note slope_table in percent
        slope_table = numpy.array([1., 3.5, 5., 9.])
        m_table = numpy.array([0.2, 0.3, 0.4, 0.5])
        # mask where slopes are larger than lookup table
        big_slope_mask = percent_slope > slope_table[-1]
        m_indexes = numpy.digitize(
            percent_slope[~big_slope_mask], slope_table, right=True)
        m_exp = numpy.empty(big_slope_mask.shape, dtype=numpy.float32)
        m_exp[big_slope_mask] = (
            beta[big_slope_mask] / (1 + beta[big_slope_mask]))
        m_exp[~big_slope_mask] = m_table[m_indexes]

        l_factor = (
            ((contributing_area + cell_area)**(m_exp+1) -
             contributing_area ** (m_exp+1)) /
            ((cell_size ** (m_exp + 2)) * (aspect_length**m_exp) *
             (22.13**m_exp)))

        # threshold L factor to l_max
        l_factor[l_factor > l_max] = l_max

        return l_factor * slope_factor

    pygeoprocessing.raster_map(
        op=ls_factor_function,
        rasters=[slope_path, flow_accumulation_path],
        target_path=target_ls_factor_path)


def _calculate_rkls(
        ls_factor_path, erosivity_path, erodibility_path, stream_path,
        rkls_path):
    """Calculate potential soil loss (tons / (pixel * year)) using RKLS.

    (revised universal soil loss equation with no C or P).

    Args:
        ls_factor_path (string): path to LS raster that has square pixels in
            meter units.
        erosivity_path (string): path to erosivity raster
            (MJ * mm / (ha * hr * yr))
        erodibility_path (string): path to erodibility raster
            (t * ha * hr / (MJ * ha * mm))
        stream_path (string): path to drainage raster
            (1 is drainage, 0 is not)
        rkls_path (string): path to RKLS raster

    Returns:
        None

    """
    erosivity_nodata = pygeoprocessing.get_raster_info(
        erosivity_path)['nodata'][0]
    erodibility_nodata = pygeoprocessing.get_raster_info(
        erodibility_path)['nodata'][0]
    stream_nodata = pygeoprocessing.get_raster_info(
        stream_path)['nodata'][0]

    cell_size = abs(
        pygeoprocessing.get_raster_info(ls_factor_path)['pixel_size'][0])
    cell_area_ha = cell_size**2 / 10000.0  # hectares per pixel

    def rkls_function(ls_factor, erosivity, erodibility, stream):
        """Calculate the RKLS equation.

        Args:
            ls_factor (numpy.ndarray): length/slope factor. unitless.
            erosivity (numpy.ndarray): related to peak rainfall events. units:
                MJ * mm / (ha * hr * yr)
            erodibility (numpy.ndarray): related to the potential for soil to
                erode. units: t * ha * hr / (MJ * ha * mm)
            stream (numpy.ndarray): stream mask (1 stream, 0 no stream)

        Returns:
            numpy.ndarray of RKLS values in tons / (pixel * year))
        """
        rkls = numpy.empty(ls_factor.shape, dtype=numpy.float32)
        nodata_mask = (
            ~pygeoprocessing.array_equals_nodata(ls_factor, _TARGET_NODATA) &
            ~pygeoprocessing.array_equals_nodata(stream, stream_nodata))
        if erosivity_nodata is not None:
            nodata_mask &= ~pygeoprocessing.array_equals_nodata(
                erosivity, erosivity_nodata)
        if erodibility_nodata is not None:
            nodata_mask &= ~pygeoprocessing.array_equals_nodata(
                erodibility, erodibility_nodata)

        valid_mask = nodata_mask & (stream == 0)
        rkls[:] = _TARGET_NODATA

        rkls[valid_mask] = (           # rkls units are tons / (pixel * year)
            ls_factor[valid_mask] *    # unitless
            erosivity[valid_mask] *    # MJ * mm / (ha * hr * yr)
            erodibility[valid_mask] *  # t * ha * hr / (MJ * ha * mm)
            cell_area_ha)              # ha / pixel
        return rkls

    # aligning with index 3 that's the stream and the most likely to be
    # aligned with LULCs
    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [
            ls_factor_path, erosivity_path, erodibility_path, stream_path]],
        rkls_function, rkls_path, gdal.GDT_Float32, _TARGET_NODATA)


def threshold_slope_op(slope):
    """raster_map equation: convert slope to m/m and clamp at 0.005 and 1.0.

    As desribed in Cavalli et al., 2013.
    """
    slope_m = slope / 100
    slope_m[slope_m < 0.005] = 0.005
    slope_m[slope_m > 1] = 1
    return slope_m


def _calculate_w(
        lulc_to_c, lulc_path, w_factor_path,
        out_thresholded_w_factor_path):
    """W factor: map C values from LULC and lower threshold to 0.001.

    W is a factor in calculating d_up accumulation for SDR.

    Args:
        lulc_to_c (dict): mapping of LULC codes to C values
        lulc_path (string): path to LULC raster
        w_factor_path (string): path to outputed raw W factor
        out_thresholded_w_factor_path (string): W factor from `w_factor_path`
            thresholded to be no less than 0.001.

    Returns:
        None

    """
    if pygeoprocessing.get_raster_info(lulc_path)['nodata'][0] is None:
        # will get a case where the raster might be masked but nothing to
        # replace so 0 is used by default. Ensure this exists in lookup.
        if 0 not in lulc_to_c:
            lulc_to_c = lulc_to_c.copy()
            lulc_to_c[0] = 0.0

    reclass_error_details = {
        'raster_name': 'LULC', 'column_name': 'lucode',
        'table_name': 'Biophysical'}

    utils.reclassify_raster(
        (lulc_path, 1), lulc_to_c, w_factor_path, gdal.GDT_Float32,
        _TARGET_NODATA, reclass_error_details)

    pygeoprocessing.raster_map(
        op=lambda w_val: numpy.where(w_val < 0.001, 0.001, w_val),
        rasters=[w_factor_path],
        target_path=out_thresholded_w_factor_path)


def _calculate_cp(lulc_to_cp, lulc_path, cp_factor_path):
    """Map LULC to C*P value.

    Args:
        lulc_to_cp (dict): mapping of lulc codes to CP values
        lulc_path (string): path to LULC raster
        cp_factor_path (string): path to output raster of LULC mapped to C*P
            values

    Returns:
        None

    """
    if pygeoprocessing.get_raster_info(lulc_path)['nodata'][0] is None:
        # will get a case where the raster might be masked but nothing to
        # replace so 0 is used by default. Ensure this exists in lookup.
        if 0 not in lulc_to_cp:
            lulc_to_cp[0] = 0.0

    reclass_error_details = {
        'raster_name': 'LULC', 'column_name': 'lucode',
        'table_name': 'Biophysical'}

    utils.reclassify_raster(
        (lulc_path, 1), lulc_to_cp, cp_factor_path, gdal.GDT_Float32,
        _TARGET_NODATA, reclass_error_details)


def _calculate_bar_factor(
        flow_direction_path, factor_path, flow_accumulation_path,
        accumulation_path, out_bar_path):
    """Route user defined source across DEM.

    Used for calculating S and W bar in the SDR operation.

    Args:
        dem_path (string): path to DEM raster
        factor_path (string): path to arbitrary factor raster
        flow_accumulation_path (string): path to flow accumulation raster
        flow_direction_path (string): path to flow direction path (in radians)
        accumulation_path (string): path to a raster that can be used to
            save the accumulation of the factor.  Temporary file.
        out_bar_path (string): path to output raster that is the result of
            the factor accumulation raster divided by the flow accumulation
            raster.

    Returns:
        None.

    """
    LOGGER.debug("doing flow accumulation mfd on %s", factor_path)
    # manually setting compression to DEFLATE because we got some LZW
    # errors when testing with large data.
    pygeoprocessing.routing.flow_accumulation_mfd(
        (flow_direction_path, 1), accumulation_path,
        weight_raster_path_band=(factor_path, 1),
        raster_driver_creation_tuple=('GTIFF', [
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE',
            'PREDICTOR=3']))

    pygeoprocessing.raster_map(
        op=lambda base_accum, flow_accum: base_accum / flow_accum,
        rasters=[accumulation_path, flow_accumulation_path],
        target_path=out_bar_path)


def _calculate_d_up(
        w_bar_path, s_bar_path, flow_accumulation_path, out_d_up_path):
    """Calculate w_bar * s_bar * sqrt(flow accumulation * cell area)."""
    cell_area = abs(
        pygeoprocessing.get_raster_info(w_bar_path)['pixel_size'][0])**2
    pygeoprocessing.raster_map(
        op=lambda w_bar, s_bar, flow_accum: (
            w_bar * s_bar * numpy.sqrt(flow_accum * cell_area)),
        rasters=[w_bar_path, s_bar_path, flow_accumulation_path],
        target_path=out_d_up_path)


def _calculate_d_up_bare(
        s_bar_path, flow_accumulation_path, out_d_up_bare_path):
    """Calculate s_bar * sqrt(flow accumulation * cell area)."""
    cell_area = abs(
        pygeoprocessing.get_raster_info(s_bar_path)['pixel_size'][0])**2
    pygeoprocessing.raster_map(
        op=lambda s_bar, flow_accum: (
            numpy.sqrt(flow_accum * cell_area) * s_bar),
        rasters=[s_bar_path, flow_accumulation_path],
        target_path=out_d_up_bare_path)


def _calculate_ic(d_up_path, d_dn_path, out_ic_factor_path):
    """Calculate log10(d_up/d_dn)."""
    # ic can be positive or negative, so float.min is a reasonable nodata value
    d_dn_nodata = pygeoprocessing.get_raster_info(d_dn_path)['nodata'][0]

    def ic_op(d_up, d_dn):
        """Calculate IC factor."""
        valid_mask = (
            ~pygeoprocessing.array_equals_nodata(d_up, _TARGET_NODATA) &
            ~pygeoprocessing.array_equals_nodata(d_dn, d_dn_nodata) & (d_dn != 0) &
            (d_up != 0))
        ic_array = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        ic_array[:] = _IC_NODATA
        ic_array[valid_mask] = numpy.log10(
            d_up[valid_mask] / d_dn[valid_mask])
        return ic_array

    pygeoprocessing.raster_calculator(
        [(d_up_path, 1), (d_dn_path, 1)], ic_op, out_ic_factor_path,
        gdal.GDT_Float32, _IC_NODATA)


def _calculate_sdr(
        k_factor, ic_0, sdr_max, ic_path, stream_path, out_sdr_path):
    """Derive SDR from k, ic0, ic; 0 on the stream and clamped to sdr_max."""
    def sdr_op(ic_factor, stream):
        """Calculate SDR factor."""
        valid_mask = (
            ~pygeoprocessing.array_equals_nodata(ic_factor, _IC_NODATA) & (stream != 1))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = (
            sdr_max / (1+numpy.exp((ic_0-ic_factor[valid_mask])/k_factor)))
        result[stream == 1] = 0.0
        return result

    pygeoprocessing.raster_calculator(
        [(ic_path, 1), (stream_path, 1)], sdr_op, out_sdr_path,
        gdal.GDT_Float32, _TARGET_NODATA)


def _calculate_e_prime(usle_path, sdr_path, stream_path, target_e_prime):
    """Calculate USLE * (1-SDR)."""
    def e_prime_op(usle, sdr, streams):
        """Wash that does not reach stream."""
        valid_mask = (
            ~pygeoprocessing.array_equals_nodata(usle, _TARGET_NODATA) &
            ~pygeoprocessing.array_equals_nodata(sdr, _TARGET_NODATA))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = _TARGET_NODATA
        result[valid_mask] = usle[valid_mask] * (1-sdr[valid_mask])
        # set to 0 on streams, to prevent nodata propagating up/down slope
        # in calculate_sediment_deposition. This makes sense intuitively:
        # E'_i represents the sediment export from pixel i that does not
        # reach a stream, which is 0 if pixel i is already in a stream.
        result[streams == 1] = 0
        return result

    pygeoprocessing.raster_calculator(
        [(usle_path, 1), (sdr_path, 1), (stream_path, 1)], e_prime_op,
        target_e_prime, gdal.GDT_Float32, _TARGET_NODATA)


def _generate_report(
        watersheds_path, usle_path, sed_export_path,
        sed_deposition_path, avoided_export_path, avoided_erosion_path,
        watershed_results_sdr_path):
    """Create summary vector with totals for rasters.

    Args:
        watersheds_path (string): The path to the watersheds vector.
        usle_path (string): The path to the computed USLE raster.
        sed_export_path (string): The path to the sediment export raster.
        sed_deposition_path (string): The path to the sediment deposition
            raster.
        avoided_export_path (string): The path to the total retention raster.
        avoided_erosion_path (string): The path to the avoided local
            erosion raster.
        watershed_results_sdr_path (string): The path to where the watersheds
            vector will be created.  This path must end in ``.shp`` as it will
            be written as an ESRI Shapefile.

    Returns:
        ``None``
    """
    original_datasource = gdal.OpenEx(watersheds_path, gdal.OF_VECTOR)
    if os.path.exists(watershed_results_sdr_path):
        LOGGER.warning(f'overwriting results at {watershed_results_sdr_path}')
        os.remove(watershed_results_sdr_path)
    driver = gdal.GetDriverByName('ESRI Shapefile')
    target_vector = driver.CreateCopy(
        watershed_results_sdr_path, original_datasource)

    target_layer = target_vector.GetLayer()
    target_layer.SyncToDisk()

    # It's worth it to check if the geometries don't significantly overlap.
    # On large rasters, this can save a TON of time rasterizing even a
    # relatively simple vector.
    geometries_might_overlap = urban_nature_access._geometries_overlap(
        watershed_results_sdr_path)
    fields_and_rasters = [
        ('usle_tot', usle_path), ('sed_export', sed_export_path),
        ('sed_dep', sed_deposition_path), ('avoid_exp', avoided_export_path),
        ('avoid_eros', avoided_erosion_path)]

    # Using the list option for raster path bands so that we can reduce
    # rasterizations, which are costly on large datasets.
    zonal_stats_results = pygeoprocessing.zonal_statistics(
        [(raster_path, 1) for (_, raster_path) in fields_and_rasters],
        watershed_results_sdr_path,
        polygons_might_overlap=geometries_might_overlap)

    field_summaries = {
        field: stats for ((field, _), stats) in
        zip(fields_and_rasters, zonal_stats_results)}

    for field_name in field_summaries:
        field_def = ogr.FieldDefn(field_name, ogr.OFTReal)
        field_def.SetWidth(24)
        field_def.SetPrecision(11)
        target_layer.CreateField(field_def)

    target_layer.ResetReading()
    for feature in target_layer:
        feature_id = feature.GetFID()
        for field_name in field_summaries:
            feature.SetField(
                field_name,
                float(field_summaries[field_name][feature_id]['sum']))
        target_layer.SetFeature(feature)
    target_vector = None
    target_layer = None


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to `execute`'s contract.

    Args:
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
    return validation.validate(
        args, MODEL_SPEC['args'], MODEL_SPEC['args_with_spatial_overlap'])
