# coding=UTF-8

from natcap.invest.ui import model, inputs
import natcap.invest.scenario_gen_proximity
import natcap.invest.scenario_generator.scenario_generator

from osgeo import ogr


class ScenarioGenProximity(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'Scenario Generator: Proximity Based',
            target=natcap.invest.scenario_gen_proximity.execute,
            validator=natcap.invest.scenario_gen_proximity.validate,
            localdoc=u'../documentation/scenario_gen_proximity.html')

        self.base_lulc_path = inputs.File(
            args_key=u'base_lulc_path',
            label=u'Base Land Use/Cover (Raster)',
            validator=self.validator)
        self.add_input(self.base_lulc_path)
        self.aoi_path = inputs.File(
            args_key=u'aoi_path',
            helptext=(
                u"This is a set of polygons that will be used to "
                u"aggregate carbon values at the end of the run if "
                u"provided."),
            label=u'Area of interest (Vector) (optional)',
            validator=self.validator)
        self.add_input(self.aoi_path)
        self.area_to_convert = inputs.Text(
            args_key=u'area_to_convert',
            label=u'Max area to convert (Ha)',
            validator=self.validator)
        self.add_input(self.area_to_convert)
        self.focal_landcover_codes = inputs.Text(
            args_key=u'focal_landcover_codes',
            label=u'Focal Landcover Codes (list)',
            validator=self.validator)
        self.add_input(self.focal_landcover_codes)
        self.convertible_landcover_codes = inputs.Text(
            args_key=u'convertible_landcover_codes',
            label=u'Convertible Landcover Codes (list)',
            validator=self.validator)
        self.add_input(self.convertible_landcover_codes)
        self.replacment_lucode = inputs.Text(
            args_key=u'replacment_lucode',
            label=u'Replacement Landcover Code (int)',
            validator=self.validator)
        self.add_input(self.replacment_lucode)
        self.convert_farthest_from_edge = inputs.Checkbox(
            args_key=u'convert_farthest_from_edge',
            helptext=(
                u"This scenario converts the convertible landcover "
                u"codes starting at the furthest pixel from the closest "
                u"base landcover codes and moves inward."),
            label=u'Farthest from edge')
        self.add_input(self.convert_farthest_from_edge)
        self.convert_nearest_to_edge = inputs.Checkbox(
            args_key=u'convert_nearest_to_edge',
            helptext=(
                u"This scenario converts the convertible landcover "
                u"codes starting at the closest pixel in the base "
                u"landcover codes and moves outward."),
            label=u'Nearest to edge')
        self.add_input(self.convert_nearest_to_edge)
        self.n_fragmentation_steps = inputs.Text(
            args_key=u'n_fragmentation_steps',
            helptext=(
                u"This parameter is used to divide the conversion "
                u"simulation into equal subareas of the requested max "
                u"area.  During each sub-step the distance transform is "
                u"recalculated from the base landcover codes.  This can "
                u"affect the final result if the base types are also "
                u"convertible types."),
            label=u'Number of Steps in Conversion',
            validator=self.validator)
        self.add_input(self.n_fragmentation_steps)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.base_lulc_path.args_key: self.base_lulc_path.value(),
            self.aoi_path.args_key: self.aoi_path.value(),
            self.area_to_convert.args_key: self.area_to_convert.value(),
            self.focal_landcover_codes.args_key:
                self.focal_landcover_codes.value(),
            self.convertible_landcover_codes.args_key:
                self.convertible_landcover_codes.value(),
            self.replacment_lucode.args_key: self.replacment_lucode.value(),
            self.convert_farthest_from_edge.args_key:
                self.convert_farthest_from_edge.value(),
            self.convert_nearest_to_edge.args_key:
                self.convert_nearest_to_edge.value(),
            self.n_fragmentation_steps.args_key:
                self.n_fragmentation_steps.value(),
        }

        return args


class ScenarioGenerator(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'Scenario Generator',
            target=natcap.invest.scenario_generator.scenario_generator.execute,
            validator=natcap.invest.scenario_generator.scenario_generator.validate,
            localdoc=u'../documentation/scenario_generator.html',
            suffix_args_key='suffix',
        )
        self.landcover = inputs.File(
            args_key=u'landcover',
            helptext=u'A GDAL-supported raster file representing land-use/land-cover.',
            label=u'Land Cover (Raster)',
            validator=self.validator)
        self.add_input(self.landcover)
        self.transition = inputs.File(
            args_key=u'transition',
            helptext=(
                u"This table contains the land-cover transition "
                u"likelihoods, priority of transitions, area change, "
                u"proximity suitiblity, proximity effect distance, seed "
                u"size, short name, and patch size."),
            label=u'Transition Table (CSV)',
            validator=self.validator)
        self.add_input(self.transition)
        self.calculate_priorities = inputs.Checkbox(
            args_key=u'calculate_priorities',
            helptext=(
                u"This option enables calculation of the land-cover "
                u"priorities using analytical hierarchical processing. "
                u"A matrix table must be entered below.  Optionally, "
                u"the priorities can manually be entered in the "
                u"priority column of the land attributes table."),
            interactive=False,
            label=u'Calculate Priorities')
        self.add_input(self.calculate_priorities)
        self.priorities_csv_uri = inputs.File(
            args_key=u'priorities_csv_uri',
            helptext=(
                u"This table contains a matrix of land-cover type "
                u"pairwise priorities used to calculate land-cover "
                u"priorities."),
            interactive=False,
            label=u'Priorities Table (CSV)',
            validator=self.validator)
        self.add_input(self.priorities_csv_uri)
        self.calculate_proximity = inputs.Container(
            args_key=u'calculate_proximity',
            expandable=True,
            expanded=True,
            label=u'Proximity')
        self.add_input(self.calculate_proximity)
        self.calculate_transition = inputs.Container(
            args_key=u'calculate_transition',
            expandable=True,
            expanded=True,
            label=u'Specify Transitions')
        self.add_input(self.calculate_transition)
        self.calculate_factors = inputs.Container(
            args_key=u'calculate_factors',
            expandable=True,
            expanded=True,
            label=u'Use Factors')
        self.add_input(self.calculate_factors)
        self.suitability_folder = inputs.Folder(
            args_key=u'suitability_folder',
            label=u'Factors Folder',
            validator=self.validator)
        self.calculate_factors.add_input(self.suitability_folder)
        self.suitability = inputs.File(
            args_key=u'suitability',
            helptext=(
                u"This table lists the factors that determine "
                u"suitability of the land-cover for change, and "
                u"includes: the factor name, layer name, distance of "
                u"influence, suitability value, weight of the factor, "
                u"distance breaks, and applicable land-cover."),
            label=u'Factors Table',
            validator=self.validator)
        self.calculate_factors.add_input(self.suitability)
        self.weight = inputs.Text(
            args_key=u'weight',
            helptext=(
                u"The factor weight is a value between 0 and 1 which "
                u"determines the weight given to the factors vs.  the "
                u"expert opinion likelihood rasters.  For example, if a "
                u"weight of 0.3 is entered then 30% of the final "
                u"suitability is contributed by the factors and the "
                u"likelihood matrix contributes 70%.  This value is "
                u"entered on the tool interface."),
            label=u'Factor Weight',
            validator=self.validator)
        self.calculate_factors.add_input(self.weight)
        self.factor_inclusion = inputs.Dropdown(
            args_key=u'factor_inclusion',
            helptext=u'',
            interactive=False,
            label=u'Rasterization Method',
            options=[u'All touched pixels', u'Only pixels with covered center points'])
        self.calculate_factors.add_input(self.factor_inclusion)
        self.calculate_constraints = inputs.Container(
            args_key=u'calculate_constraints',
            expandable=True,
            label=u'Constraints Layer')
        self.add_input(self.calculate_constraints)
        self.constraints = inputs.File(
            args_key=u'constraints',
            helptext=(
                u"An OGR-supported vector file.  This is a vector "
                u"layer which indicates the parts of the landscape that "
                u"are protected of have constraints to land-cover "
                u"change.  The layer should have one field named "
                u"'porosity' with a value between 0 and 1 where 0 means "
                u"its fully protected and 1 means its fully open to "
                u"change."),
            label=u'Constraints Layer (Vector)',
            validator=self.validator)
        self.calculate_constraints.add_input(self.constraints)
        self.constraints.sufficiency_changed.connect(
            self._load_colnames_constraints)
        self.constraints_field = inputs.Dropdown(
            args_key=u'constraints_field',
            helptext=(
                u"The field from the override table that contains the "
                u"value for the override."),
            interactive=False,
            options=('UNKNOWN',),
            label=u'Constraints Field')
        self.calculate_constraints.add_input(self.constraints_field)
        self.override_layer = inputs.Container(
            args_key=u'override_layer',
            expandable=True,
            expanded=True,
            label=u'Override Layer')
        self.add_input(self.override_layer)
        self.override = inputs.File(
            args_key=u'override',
            helptext=(
                u"An OGR-supported vector file.  This is a vector "
                u"(polygon) layer with land-cover types in the same "
                u"scale and projection as the input land-cover.  This "
                u"layer is used to override all the changes and is "
                u"applied after the rule conversion is complete."),
            label=u'Override Layer (Vector)',
            validator=self.validator)
        self.override_layer.add_input(self.override)
        self.override.sufficiency_changed.connect(
            self._load_colnames_override)
        self.override_field = inputs.Dropdown(
            args_key=u'override_field',
            helptext=(
                u"The field from the override table that contains the "
                u"value for the override."),
            interactive=False,
            options=('UNKNOWN',),
            label=u'Override Field')
        self.override_layer.add_input(self.override_field)
        self.override_inclusion = inputs.Dropdown(
            args_key=u'override_inclusion',
            helptext=u'',
            interactive=False,
            label=u'Rasterization Method',
            options=[u'All touched pixels', u'Only pixels with covered center points'])
        self.override_layer.add_input(self.override_inclusion)
        self.seed = inputs.Text(
            args_key=u'seed',
            helptext=(
                u"Seed must be an integer or blank.  <br/><br/>Under "
                u"normal conditions, parcels with the same suitability "
                u"are picked in a random order.  Setting the seed value "
                u"allows the scenario generator to randomize the order "
                u"in which parcels are picked, but two runs with the "
                u"same seed will pick parcels in the same order."),
            label=u'Seed for random parcel selection (optional)',
            validator=self.validator)
        self.add_input(self.seed)

        # Set interactivity, requirement as input sufficiency changes
        self.transition.sufficiency_changed.connect(
            self.calculate_priorities.set_interactive)
        self.calculate_priorities.sufficiency_changed.connect(
            self.priorities_csv_uri.set_interactive)
        self.calculate_factors.sufficiency_changed.connect(
            self.factor_inclusion.set_interactive)
        self.constraints.sufficiency_changed.connect(
            self.constraints_field.set_interactive)
        self.override.sufficiency_changed.connect(
            self.override_field.set_interactive)
        self.override_field.sufficiency_changed.connect(
            self.override_inclusion.set_interactive)

    def _load_colnames_constraints(self, new_interactivity):
        self._load_colnames(new_interactivity,
                            self.constraints,
                            self.constraints_field)

    def _load_colnames_override(self, new_interactivity):
        self._load_colnames(new_interactivity,
                            self.override,
                            self.override_field)

    def _load_colnames(self, new_interactivity, vector_input, dropdown_input):
        if new_interactivity:
            vector_path = vector_input.value()
            vector = gdal.OpenEx(vector_path)
            layer = vector.GetLayer()
            colnames = [defn.GetName() for defn in layer.schema]
            dropdown_input.set_options(colnames)
            dropdown_input.set_interactive(True)
        else:
            dropdown_input.set_options([])

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.landcover.args_key: self.landcover.value(),
            self.transition.args_key: self.transition.value(),
            self.calculate_priorities.args_key: self.calculate_priorities.value(),
            self.priorities_csv_uri.args_key: self.priorities_csv_uri.value(),
            self.calculate_proximity.args_key: self.calculate_proximity.value(),
            self.calculate_transition.args_key: self.calculate_transition.value(),
            self.calculate_factors.args_key: self.calculate_factors.value(),
            self.calculate_constraints.args_key: self.calculate_constraints.value(),
            self.override_layer.args_key: self.override_layer.value(),
            self.seed.args_key: self.seed.value(),
        }

        if self.calculate_factors.value():
            args[self.suitability_folder.args_key] = self.suitability_folder.value()
            args[self.suitability.args_key] = self.suitability.value()
            args[self.weight.args_key] = self.weight.value()
            args[self.factor_inclusion.args_key] = self.factor_inclusion.value()

        if self.calculate_constraints.value():
            args[self.constraints.args_key] = self.constraints.value()
            args[self.constraints_field.args_key] = self.constraints_field.value()

        if self.override_layer.value():
            args[self.override.args_key] = self.override.value()
            args[self.override_field.args_key] = self.override_field.value()
            args[self.override_inclusion.args_key] = self.override_inclusion.value()

        return args
