from __future__ import absolute_import
import logging

from . import inputs
from .model import Model
from ..pollination import pollination

LOGGER = logging.getLogger(__name__)
_validate = lambda args, limit_to: []


class Pollination(Model):
    label = pollination.LABEL
    target = staticmethod(pollination.execute)
    validator = staticmethod(_validate)
    localdoc = 'croppollination.html'

    def __init__(self):
        Model.__init__(self)

        self.landcover_container = inputs.Container(label='Land Use / Land Cover')
        self.add_input(self.landcover_container)
        self.cur_lulc_raster = inputs.File(
            args_key=u'landuse_cur_uri',
            helptext=u'A GDAL-supported vector file.',
            label=u'Current Land Cover Scenario (Raster)',
            required=True)
        self.landcover_container.add_input(self.cur_lulc_raster)
        self.landcover_attribute_table = inputs.File(
            args_key=u'landuse_attributes_uri',
            helptext=u'A CSV table of land-cover attributes.',
            label=u'Land Cover Attributes Table (CSV)',
            required=True)
        self.landcover_container.add_input(self.landcover_attribute_table)
        self.fut_lulc_raster = inputs.File(
            args_key=u'landuse_fut_uri',
            helptext=(u'Optional. An GDAL-supported raster file representing '
                      u'a future land-cover scenario.<br/><br/>Providing a '
                      u'future land-cover scenario will cause pollinator '
                      u'supply and abundance to be calculated for both the '
                      u'current and future scenarios.  The future scenario '
                      u'land cover raster should use the same land cover '
                      u'attribute table as the current land cover raster.'),
            hideable=True,
            label=u'Calculate Future Scenario (Raster)',
            required=False)
        self.landcover_container.add_input(self.fut_lulc_raster)
        self.valuation_container = inputs.Container(
            args_key='do_valuation',
            expandable=True,
            label='Valuation Options(enable to trigger valuation)')
        self.add_input(self.valuation_container)
        self.half_saturation_const = inputs.Text(
            args_key=u'half_saturation',
            helptext=(u'This should be a number between 0 and 1.  It '
                      u'represents the abundance of pollinators required to '
                      u'reach 50% of pollinator-dependent yield.'),
            label=u'Half-saturation constant',
            required=True)
        self.valuation_container.add_input(self.half_saturation_const)
        self.wild_pollination_proportion = inputs.Text(
            args_key=u'wild_pollination_proportion',
            helptext=(u'This should be a number between 0 and 1.  It '
                      u'represents the proportion of all crop yield '
                      u'attributed to wild pollinators on this landscape.'),
            label=u'Proportion of Total Yield Due to Wild Pollinators',
            required=True)
        self.valuation_container.add_input(self.wild_pollination_proportion)
        self.guilds = inputs.File(
            args_key=u'guilds_uri',
            helptext=(u"A CSV table containing information specific to the "
                      u"various pollinators to be modeled.  Please see the "
                      u"documentation for details on the structure of this "
                      u"table.<br/><br/><b>Optional:</b><br/>If aggregating "
                      u"by crops, the table should contain fields matching "
                      u"'crp_*', where the value is either 1 or 0."),
            label=u'Guilds Table (CSV)',
            required=True)
        self.add_input(self.guilds)
        self.ag_classes = inputs.Text(
            args_key=u'ag_classes',
            helptext=(u'A space-separated list of agricultural land-cover '
                      u'classes.<br/><br/>Example:<br/>3 7 34 35 68<br/><br/>'
                      u'This input is optional.  If agricultural classes are '
                      u'not provided here, the entire land-cover raster will '
                      u'be considered as agricultural.'),
            label=u'Agricultural Classes (space-separated)',
            required=False)
        self.add_input(self.ag_classes)

    def assemble_args(self):
        return {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.cur_lulc_raster.args_key: self.cur_lulc_raster.value(),
            self.landcover_attribute_table.args_key: (
                self.landcover_attribute_table.value()),
            self.fut_lulc_raster.args_key: self.fut_lulc_raster.value(),
            self.valuation_container.args_key: (
                self.valuation_container.value()),
            self.half_saturation_const.args_key: (
                self.half_saturation_const.value()),
            self.wild_pollination_proportion.args_key: (
                self.wild_pollination_proportion.value()),
            self.guilds.args_key: self.guilds.value(),
            self.ag_classes.args_key: self.ag_classes.value(),
        }
