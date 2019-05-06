# coding=UTF-8

from natcap.invest.ui import model, inputs
import natcap.invest.forest_carbon_edge_effect


class ForestCarbonEdgeEffect(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'Forest Carbon Edge Effect Model',
            target=natcap.invest.forest_carbon_edge_effect.execute,
            validator=natcap.invest.forest_carbon_edge_effect.validate,
            localdoc=u'forest_carbon_edge_effect.html')

        self.lulc_raster_path = inputs.File(
            args_key=u'lulc_raster_path',
            helptext=(
                u"A GDAL-supported raster file, with an integer LULC "
                u"code for each cell."),
            label=u'Land-Use/Land-Cover Map (raster)',
            validator=self.validator)
        self.add_input(self.lulc_raster_path)
        self.biophysical_table_path = inputs.File(
            args_key=u'biophysical_table_path',
            helptext=(
                u"A CSV table containing model information "
                u"corresponding to each of the land use classes in the "
                u"LULC raster input.  It must contain the fields "
                u"'lucode', 'is_tropical_forest', 'c_above'.  If the "
                u"user selects 'all carbon pools' the table must also "
                u"contain entries for 'c_below', 'c_soil', and "
                u"'c_dead'.  See the InVEST Forest Carbon User's Guide "
                u"for more information about these fields."),
            label=u'Biophysical Table (csv)',
            validator=self.validator)
        self.add_input(self.biophysical_table_path)
        self.pools_to_calculate = inputs.Dropdown(
            args_key=u'pools_to_calculate',
            helptext=(
                u"If 'all carbon pools' is selected then the headers "
                u"'c_above', 'c_below', 'c_dead', 'c_soil' are used in "
                u"the carbon pool calculation.  Otherwise only "
                u"'c_above' is considered."),
            label=u'Carbon Pools to Calculate',
            options=[u'all carbon pools', u'above ground only'],
            return_value_map={'all carbon pools': 'all',
                              'above ground only': 'above_ground'})
        self.add_input(self.pools_to_calculate)
        self.compute_forest_edge_effects = inputs.Checkbox(
            args_key=u'compute_forest_edge_effects',
            helptext=(
                u"If selected, will use the Chaplin-Kramer, et.  al "
                u"method to account for above ground carbon stocks in "
                u"tropical forest types indicated by a '1' in the "
                u"'is_tropical_forest' field in the biophysical table."),
            label=u'Compute forest edge effects')
        self.add_input(self.compute_forest_edge_effects)
        self.tropical_forest_edge_carbon_model_vector_path = inputs.File(
            args_key=u'tropical_forest_edge_carbon_model_vector_path',
            helptext=(
                u"A shapefile with fields 'method', 'theta1', "
                u"'theta2', 'theta3' describing the global forest "
                u"carbon edge models.  Provided as default data for the "
                u"model."),
            interactive=False,
            label=u'Global forest carbon edge regression models (vector)',
            validator=self.validator)
        self.add_input(self.tropical_forest_edge_carbon_model_vector_path)
        self.n_nearest_model_points = inputs.Text(
            args_key=u'n_nearest_model_points',
            helptext=(
                u"Used when calculating the biomass in a pixel.  This "
                u"number determines the number of closest regression "
                u"models that are used when calculating the total "
                u"biomass.  Each local model is linearly weighted by "
                u"distance such that the biomass in the pixel is a "
                u"function of each of these points with the closest "
                u"point having the highest effect."),
            interactive=False,
            label=u'Number of nearest model points to average',
            validator=self.validator)
        self.add_input(self.n_nearest_model_points)
        self.biomass_to_carbon_conversion_factor = inputs.Text(
            args_key=u'biomass_to_carbon_conversion_factor',
            helptext=(
                u"Number by which to scale forest edge biomass to "
                u"convert to carbon.  Default value is 0.47 (according "
                u"to IPCC 2006). This pertains to forest classes only; "
                u"values in the biophysical table for non-forest "
                u"classes should already be in terms of carbon, not "
                u"biomass."),
            interactive=False,
            label=u'Forest Edge Biomass to Carbon Conversion Factor',
            validator=self.validator)
        self.add_input(self.biomass_to_carbon_conversion_factor)
        self.aoi_vector_path = inputs.File(
            args_key=u'aoi_vector_path',
            helptext=(
                u"This is a set of polygons that will be used to "
                u"aggregate carbon values at the end of the run if "
                u"provided."),
            label=u'Service areas of interest <em>(optional)</em> (vector)',
            validator=self.validator)
        self.add_input(self.aoi_vector_path)

        # Set interactivity, requirement as input sufficiency changes
        self.compute_forest_edge_effects.sufficiency_changed.connect(
            self.tropical_forest_edge_carbon_model_vector_path.set_interactive)
        self.compute_forest_edge_effects.sufficiency_changed.connect(
            self.n_nearest_model_points.set_interactive)
        self.compute_forest_edge_effects.sufficiency_changed.connect(
            self.biomass_to_carbon_conversion_factor.set_interactive)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.lulc_raster_path.args_key: self.lulc_raster_path.value(),
            self.biophysical_table_path.args_key:
                self.biophysical_table_path.value(),
            self.pools_to_calculate.args_key:
                self.pools_to_calculate.value(),
            self.compute_forest_edge_effects.args_key:
                self.compute_forest_edge_effects.value(),
            self.tropical_forest_edge_carbon_model_vector_path.args_key:
                self.tropical_forest_edge_carbon_model_vector_path.value(),
            self.n_nearest_model_points.args_key:
                self.n_nearest_model_points.value(),
            self.biomass_to_carbon_conversion_factor.args_key:
                self.biomass_to_carbon_conversion_factor.value(),
            self.aoi_vector_path.args_key: self.aoi_vector_path.value(),
        }

        return args
