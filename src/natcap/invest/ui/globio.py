# coding=UTF-8

from natcap.invest.ui import model, inputs
import natcap.invest.globio


class GLOBIO(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'GLOBIO',
            target=natcap.invest.globio.execute,
            validator=natcap.invest.globio.validate,
            localdoc=u'../documentation/globio.html')

        self.lulc_to_globio_table_path = inputs.File(
            args_key=u'lulc_to_globio_table_path',
            helptext=(
                u"A CSV table containing model information "
                u"corresponding to each of the land use classes in the "
                u"LULC raster input.  It must contain the fields "
                u"'lucode', 'usle_c', and 'usle_p'.  See the InVEST "
                u"Sediment User's Guide for more information about "
                u"these fields."),
            label=u'Landcover to GLOBIO Landcover Table (CSV)',
            validator=self.validator)
        self.add_input(self.lulc_to_globio_table_path)
        self.aoi_path = inputs.File(
            args_key=u'aoi_path',
            helptext=(
                u"This is a set of polygons that can be used to "
                u"aggregate MSA sum and mean to a polygon."),
            label=u'AOI (Vector) (optional)',
            validator=self.validator)
        self.add_input(self.aoi_path)
        self.land_use = inputs.File(
            args_key=u'lulc_path',
            label=u'Land Use/Cover (Raster)',
            validator=self.validator)
        self.add_input(self.land_use)
        self.infrastructure_dir = inputs.Folder(
            args_key=u'infrastructure_dir',
            label=u'Infrastructure Directory',
            validator=self.validator)
        self.add_input(self.infrastructure_dir)
        self.pasture_path = inputs.File(
            args_key=u'pasture_path',
            label=u'Pasture (Raster)',
            validator=self.validator)
        self.add_input(self.pasture_path)
        self.potential_vegetation_path = inputs.File(
            args_key=u'potential_vegetation_path',
            label=u'Potential Vegetation (Raster)',
            validator=self.validator)
        self.add_input(self.potential_vegetation_path)
        self.primary_threshold = inputs.Text(
            args_key=u'primary_threshold',
            label=u'Primary Threshold',
            validator=self.validator)
        self.add_input(self.primary_threshold)
        self.pasture_threshold = inputs.Text(
            args_key=u'pasture_threshold',
            label=u'Pasture Threshold',
            validator=self.validator)
        self.add_input(self.pasture_threshold)
        self.intensification_fraction = inputs.Text(
            args_key=u'intensification_fraction',
            helptext=(
                u"A value between 0 and 1 denoting proportion of total "
                u"agriculture that should be classified as 'high "
                u"input'."),
            label=u'Proportion of of Agriculture Intensified',
            validator=self.validator)
        self.add_input(self.intensification_fraction)
        self.msa_parameters_path = inputs.File(
            args_key=u'msa_parameters_path',
            helptext=(
                u"A CSV table containing MSA threshold values as "
                u"defined in the user's guide.  Provided for advanced "
                u"users that may wish to change those values."),
            label=u'MSA Parameter Table (CSV)',
            validator=self.validator)
        self.add_input(self.msa_parameters_path)
        self.predefined_globio = inputs.Container(
            args_key=u'predefined_globio',
            expandable=True,
            expanded=False,
            label=u'Predefined land use map for GLOBIO')
        self.add_input(self.predefined_globio)
        self.globio_land_use = inputs.File(
            args_key=u'globio_lulc_path',
            label=u'GLOBIO Classified Land Use (Raster)',
            validator=self.validator)
        self.predefined_globio.add_input(self.globio_land_use)

        # Set interactivity, requirement as input sufficiency changes
        self.predefined_globio.sufficiency_changed.connect(
            self._predefined_globio_toggled)

    def _predefined_globio_toggled(self, use_predefined):
        self.lulc_to_globio_table_path.set_interactive(not use_predefined)
        self.land_use.set_interactive(not use_predefined)
        self.pasture_path.set_interactive(not use_predefined)
        self.potential_vegetation_path.set_interactive(not use_predefined)
        self.primary_threshold.set_interactive(not use_predefined)
        self.pasture_threshold.set_interactive(not use_predefined)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.lulc_to_globio_table_path.args_key:
                self.lulc_to_globio_table_path.value(),
            self.aoi_path.args_key: self.aoi_path.value(),
            self.land_use.args_key: self.land_use.value(),
            self.infrastructure_dir.args_key: self.infrastructure_dir.value(),
            self.pasture_path.args_key: self.pasture_path.value(),
            self.potential_vegetation_path.args_key:
                self.potential_vegetation_path.value(),
            self.primary_threshold.args_key: self.primary_threshold.value(),
            self.pasture_threshold.args_key: self.pasture_threshold.value(),
            self.intensification_fraction.args_key:
                self.intensification_fraction.value(),
            self.msa_parameters_path.args_key: self.msa_parameters_path.value(),
            self.predefined_globio.args_key: self.predefined_globio.value(),
        }

        if self.predefined_globio.value():
            args[self.globio_land_use.args_key] = self.globio_land_use.value()

        return args
