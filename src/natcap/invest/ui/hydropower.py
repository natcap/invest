# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest.hydropower import hydropower_water_yield


class HydropowerWaterYield(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'Hydropower Water Yield',
            target=hydropower_water_yield.execute,
            validator=hydropower_water_yield.validate,
            localdoc=u'../documentation/reservoirhydropowerproduction.html')

        self.precipitation = inputs.File(
            args_key=u'precipitation_path',
            helptext=(
                u"A GDAL-supported raster file containing non-zero, "
                u"average annual precipitation values for each cell. "
                u"The precipitation values should be in millimeters "
                u"(mm)."),
            label=u'Precipitation (Raster)',
            validator=self.validator)
        self.add_input(self.precipitation)
        self.potential_evapotranspiration = inputs.File(
            args_key=u'eto_path',
            helptext=(
                u"A GDAL-supported raster file containing annual "
                u"average reference evapotranspiration values for each "
                u"cell.  The reference evapotranspiration values should "
                u"be in millimeters (mm)."),
            label=u'Reference Evapotranspiration (Raster)',
            validator=self.validator)
        self.add_input(self.potential_evapotranspiration)
        self.depth_to_root_rest_layer = inputs.File(
            args_key=u'depth_to_root_rest_layer_path',
            helptext=(
                u"A GDAL-supported raster file containing an average "
                u"root restricting layer depth value for each cell. "
                u"The root restricting layer depth value should be in "
                u"millimeters (mm)."),
            label=u'Depth To Root Restricting Layer (Raster)',
            validator=self.validator)
        self.add_input(self.depth_to_root_rest_layer)
        self.plant_available_water_fraction = inputs.File(
            args_key=u'pawc_path',
            helptext=(
                u"A GDAL-supported raster file containing plant "
                u"available water content values for each cell.  The "
                u"plant available water content fraction should be a "
                u"value between 0 and 1."),
            label=u'Plant Available Water Fraction (Raster)',
            validator=self.validator)
        self.add_input(self.plant_available_water_fraction)
        self.land_use = inputs.File(
            args_key=u'lulc_path',
            helptext=(
                u"A GDAL-supported raster file containing LULC code "
                u"(expressed as integers) for each cell."),
            label=u'Land Use (Raster)',
            validator=self.validator)
        self.add_input(self.land_use)
        self.watersheds = inputs.File(
            args_key=u'watersheds_path',
            helptext=(
                u"An OGR-supported vector file containing one polygon "
                u"per watershed.  Each polygon that represents a "
                u"watershed is required to have a field 'ws_id' that is "
                u"a unique integer which identifies that watershed."),
            label=u'Watersheds (Vector)',
            validator=self.validator)
        self.add_input(self.watersheds)
        self.sub_watersheds = inputs.File(
            args_key=u'sub_watersheds_path',
            helptext=(
                u"An OGR-supported vector file with one polygon per "
                u"sub-watershed within the main watersheds specified in "
                u"the Watersheds shapefile.  Each polygon that "
                u"represents a sub-watershed is required to have a "
                u"field 'subws_id' that is a unique integer which "
                u"identifies that sub-watershed."),
            label=u'Sub-Watersheds (Vector) (Optional)',
            validator=self.validator)
        self.add_input(self.sub_watersheds)
        self.biophysical_table = inputs.File(
            args_key=u'biophysical_table_path',
            helptext=(
                u"A CSV table of land use/land cover (LULC) classes, "
                u"containing data on biophysical coefficients used in "
                u"this model.  The following columns are required: "
                u"'lucode' (integer), 'root_depth' (mm), 'Kc' "
                u"(coefficient)."),
            label=u'Biophysical Table (CSV)',
            validator=self.validator)
        self.add_input(self.biophysical_table)
        self.seasonality_constant = inputs.Text(
            args_key=u'seasonality_constant',
            helptext=(
                u"Floating point value on the order of 1 to 30 "
                u"corresponding to the seasonal distribution of "
                u"precipitation."),
            label=u'Z parameter',
            validator=self.validator)
        self.add_input(self.seasonality_constant)
        self.scarcity_valuation_container = inputs.Container(
            args_key=u'do_scarcity_and_valuation',
            expandable=True,
            expanded=False,
            label=u'Water Scarcity and Valuation')
        self.add_input(self.scarcity_valuation_container)
        self.demand_table = inputs.File(
            args_key=u'demand_table_path',
            helptext=(
                u"A CSV table of LULC classes, showing consumptive "
                u"water use for each land-use/land-cover type.  The "
                u"table requires two column fields: 'lucode' and "
                u"'demand'. The demand values should be the estimated "
                u"average consumptive water use for each land-use/land- "
                u"cover type.  Water use should be given in cubic "
                u"meters per year for a pixel in the land-use/land- "
                u"cover map.  NOTE: the accounting for pixel area is "
                u"important since larger areas will consume more water "
                u"for the same land-cover type."),
            label=u'Water Demand Table (CSV)',
            validator=self.validator)
        self.scarcity_valuation_container.add_input(self.demand_table)
        self.hydropower_valuation_table = inputs.File(
            args_key=u'valuation_table_path',
            helptext=(
                u"A CSV table of hydropower stations with associated "
                u"model values.  The table should have the following "
                u"column fields: 'ws_id', 'efficiency', 'fraction', "
                u"'height', 'kw_price', 'cost', 'time_span', and "
                u"'discount'."),
            label=u'Hydropower Valuation Table (CSV) (Optional)',
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
