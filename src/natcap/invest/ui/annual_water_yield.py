# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest.model_metadata import MODEL_METADATA
from natcap.invest import annual_water_yield


class AnnualWaterYield(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=MODEL_METADATA['annual_water_yield'].model_title,
            target=annual_water_yield.execute,
            validator=annual_water_yield.validate,
            localdoc=MODEL_METADATA['annual_water_yield'].userguide)

        self.precipitation = inputs.File(
            args_key='precipitation_path',
            helptext=(
                "A GDAL-supported raster file containing non-zero, "
                "average annual precipitation values for each cell. "
                "The precipitation values should be in millimeters "
                "(mm)."),
            label='Precipitation (Raster)',
            validator=self.validator)
        self.add_input(self.precipitation)
        self.potential_evapotranspiration = inputs.File(
            args_key='eto_path',
            helptext=(
                "A GDAL-supported raster file containing annual "
                "average reference evapotranspiration values for each "
                "cell.  The reference evapotranspiration values should "
                "be in millimeters (mm)."),
            label='Reference Evapotranspiration (Raster)',
            validator=self.validator)
        self.add_input(self.potential_evapotranspiration)
        self.depth_to_root_rest_layer = inputs.File(
            args_key='depth_to_root_rest_layer_path',
            helptext=(
                "A GDAL-supported raster file containing an average "
                "root restricting layer depth value for each cell. "
                "The root restricting layer depth value should be in "
                "millimeters (mm)."),
            label='Depth To Root Restricting Layer (Raster)',
            validator=self.validator)
        self.add_input(self.depth_to_root_rest_layer)
        self.plant_available_water_fraction = inputs.File(
            args_key='pawc_path',
            helptext=(
                "A GDAL-supported raster file containing plant "
                "available water content values for each cell.  The "
                "plant available water content fraction should be a "
                "value between 0 and 1."),
            label='Plant Available Water Fraction (Raster)',
            validator=self.validator)
        self.add_input(self.plant_available_water_fraction)
        self.land_use = inputs.File(
            args_key='lulc_path',
            helptext=(
                "A GDAL-supported raster file containing LULC code "
                "(expressed as integers) for each cell."),
            label='Land Use (Raster)',
            validator=self.validator)
        self.add_input(self.land_use)
        self.watersheds = inputs.File(
            args_key='watersheds_path',
            helptext=(
                "An OGR-supported vector file containing one polygon "
                "per watershed.  Each polygon that represents a "
                "watershed is required to have a field 'ws_id' that is "
                "a unique integer which identifies that watershed."),
            label='Watersheds (Vector)',
            validator=self.validator)
        self.add_input(self.watersheds)
        self.sub_watersheds = inputs.File(
            args_key='sub_watersheds_path',
            helptext=(
                "An OGR-supported vector file with one polygon per "
                "sub-watershed within the main watersheds specified in "
                "the Watersheds shapefile.  Each polygon that "
                "represents a sub-watershed is required to have a "
                "field 'subws_id' that is a unique integer which "
                "identifies that sub-watershed."),
            label='Sub-Watersheds (Vector) (Optional)',
            validator=self.validator)
        self.add_input(self.sub_watersheds)
        self.biophysical_table = inputs.File(
            args_key='biophysical_table_path',
            helptext=(
                "A CSV table of land use/land cover (LULC) classes, "
                "containing data on biophysical coefficients used in "
                "this model.  The following columns are required: "
                "'lucode' (integer), 'root_depth' (mm), 'Kc' "
                "(coefficient)."),
            label='Biophysical Table (CSV)',
            validator=self.validator)
        self.add_input(self.biophysical_table)
        self.seasonality_constant = inputs.Text(
            args_key='seasonality_constant',
            helptext=(
                "Floating point value on the order of 1 to 30 "
                "corresponding to the seasonal distribution of "
                "precipitation."),
            label='Z parameter',
            validator=self.validator)
        self.add_input(self.seasonality_constant)
        self.scarcity_valuation_container = inputs.Container(
            args_key='do_scarcity_and_valuation',
            expandable=True,
            expanded=False,
            label='Water Scarcity and Valuation')
        self.add_input(self.scarcity_valuation_container)
        self.demand_table = inputs.File(
            args_key='demand_table_path',
            helptext=(
                "A CSV table of LULC classes, showing consumptive "
                "water use for each land-use/land-cover type.  The "
                "table requires two column fields: 'lucode' and "
                "'demand'. The demand values should be the estimated "
                "average consumptive water use for each land-use/land- "
                "cover type.  Water use should be given in cubic "
                "meters per year for a pixel in the land-use/land- "
                "cover map.  NOTE: the accounting for pixel area is "
                "important since larger areas will consume more water "
                "for the same land-cover type."),
            label='Water Demand Table (CSV)',
            validator=self.validator)
        self.scarcity_valuation_container.add_input(self.demand_table)
        self.hydropower_valuation_table = inputs.File(
            args_key='valuation_table_path',
            helptext=(
                "A CSV table of hydropower stations with associated "
                "model values.  The table should have the following "
                "column fields: 'ws_id', 'efficiency', 'fraction', "
                "'height', 'kw_price', 'cost', 'time_span', and "
                "'discount'."),
            label='Hydropower Valuation Table (CSV) (Optional)',
            validator=self.validator)
        self.scarcity_valuation_container.add_input(self.hydropower_valuation_table)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.precipitation.args_key: self.precipitation.value(),
            self.potential_evapotranspiration.args_key:
                self.potential_evapotranspiration.value(),
            self.depth_to_root_rest_layer.args_key:
                self.depth_to_root_rest_layer.value(),
            self.plant_available_water_fraction.args_key:
                self.plant_available_water_fraction.value(),
            self.land_use.args_key: self.land_use.value(),
            self.watersheds.args_key: self.watersheds.value(),
            self.sub_watersheds.args_key: self.sub_watersheds.value(),
            self.biophysical_table.args_key: self.biophysical_table.value(),
            self.seasonality_constant.args_key:
                self.seasonality_constant.value(),
            self.scarcity_valuation_container.args_key:
                self.scarcity_valuation_container.value(),
        }

        if self.scarcity_valuation_container.value():
            args[self.demand_table.args_key] = self.demand_table.value()
            args[self.hydropower_valuation_table.args_key] = (
                self.hydropower_valuation_table.value())

        return args
