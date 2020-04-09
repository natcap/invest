# coding=UTF-8

from natcap.invest.ui import model, inputs
import natcap.invest.forest_carbon_edge_effect


class ForestCarbonEdgeEffect(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label='Forest Carbon Edge Effect Model',
            target=natcap.invest.forest_carbon_edge_effect.execute,
            validator=natcap.invest.forest_carbon_edge_effect.validate,
            localdoc='forest_carbon_edge_effect.html')

        self.lulc_raster_path = inputs.File(
            args_key='lulc_raster_path',
            helptext=(
                "A GDAL-supported raster file, with an integer LULC "
                "code for each cell."),
            label='Land-Use/Land-Cover Map (raster)',
            validator=self.validator)
        self.add_input(self.lulc_raster_path)
        self.biophysical_table_path = inputs.File(
            args_key='biophysical_table_path',
            helptext=(
                "A CSV table containing model information "
                "corresponding to each of the land use classes in the "
                "LULC raster input.  It must contain the fields "
                "'lucode', 'is_tropical_forest', 'c_above'.  If the "
                "user selects 'all carbon pools' the table must also "
                "contain entries for 'c_below', 'c_soil', and "
                "'c_dead'.  See the InVEST Forest Carbon User's Guide "
                "for more information about these fields."),
            label='Biophysical Table (csv)',
            validator=self.validator)
        self.add_input(self.biophysical_table_path)
        self.pools_to_calculate = inputs.Dropdown(
            args_key='pools_to_calculate',
            helptext=(
                "If 'all carbon pools' is selected then the headers "
                "'c_above', 'c_below', 'c_dead', 'c_soil' are used in "
                "the carbon pool calculation.  Otherwise only "
                "'c_above' is considered."),
            label='Carbon Pools to Calculate',
            options=['all carbon pools', 'above ground only'],
            return_value_map={'all carbon pools': 'all',
                              'above ground only': 'above_ground'})
        self.add_input(self.pools_to_calculate)
        self.compute_forest_edge_effects = inputs.Checkbox(
            args_key='compute_forest_edge_effects',
            helptext=(
                "If selected, will use the Chaplin-Kramer, et.  al "
                "method to account for above ground carbon stocks in "
                "tropical forest types indicated by a '1' in the "
                "'is_tropical_forest' field in the biophysical table."),
            label='Compute forest edge effects')
        self.add_input(self.compute_forest_edge_effects)
        self.tropical_forest_edge_carbon_model_vector_path = inputs.File(
            args_key='tropical_forest_edge_carbon_model_vector_path',
            helptext=(
                "A shapefile with fields 'method', 'theta1', "
                "'theta2', 'theta3' describing the global forest "
                "carbon edge models.  Provided as default data for the "
                "model."),
            interactive=False,
            label='Global forest carbon edge regression models (vector)',
            validator=self.validator)
        self.add_input(self.tropical_forest_edge_carbon_model_vector_path)
        self.n_nearest_model_points = inputs.Text(
            args_key='n_nearest_model_points',
            helptext=(
                "Used when calculating the biomass in a pixel.  This "
                "number determines the number of closest regression "
                "models that are used when calculating the total "
                "biomass.  Each local model is linearly weighted by "
                "distance such that the biomass in the pixel is a "
                "function of each of these points with the closest "
                "point having the highest effect."),
            interactive=False,
            label='Number of nearest model points to average',
            validator=self.validator)
        self.add_input(self.n_nearest_model_points)
        self.biomass_to_carbon_conversion_factor = inputs.Text(
            args_key='biomass_to_carbon_conversion_factor',
            helptext=(
                "Number by which to scale forest edge biomass to "
                "convert to carbon.  Default value is 0.47 (according "
                "to IPCC 2006). This pertains to forest classes only; "
                "values in the biophysical table for non-forest "
                "classes should already be in terms of carbon, not "
                "biomass."),
            interactive=False,
            label='Forest Edge Biomass to Carbon Conversion Factor',
            validator=self.validator)
        self.add_input(self.biomass_to_carbon_conversion_factor)
        self.aoi_vector_path = inputs.File(
            args_key='aoi_vector_path',
            helptext=(
                "This is a set of polygons that will be used to "
                "aggregate carbon values at the end of the run if "
                "provided."),
            label='Service areas of interest <em>(optional)</em> (vector)',
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
