#TODO: all the other UI modules have a # coding=UTF-8 here, does this one need it too?
from __future__ import absolute_import
import logging

from . import inputs
from . import model
from .. import pollination

LOGGER = logging.getLogger(__name__)


class Pollination(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'Crop Pollination',
            target=pollination.execute,
            validator=pollination.validate,
            localdoc=u'croppollination.html')

        self.landcover_raster_path = inputs.File(
            args_key=u'landcover_raster_path',
            helptext=(
                u"This is the landcover map that's used to map "
                u"biophyiscal properties about habitat and floral "
                u"resources of landcover types to a spatial layout."),
            label=u'Land Cover Map (Raster)',
            validator=self.validator)
        self.add_input(self.landcover_raster_path)
        self.landcover_biophysical_table_path = inputs.File(
            args_key=u'landcover_biophysical_table_path',
            helptext=(
                u"A CSV table mapping landcover codes in the landcover "
                u"raster to indexes of nesting availability for each "
                u"nesting substrate referenced in guilds table as well "
                u"as indexes of abundance of floral resources on that "
                u"landcover type per season in the bee activity columns "
                u"of the guild table.<br/>All indexes are in the range "
                u"[0.0, 1.0].<br/>Columns in the table must be at "
                u"least<br/>* 'lucode': representing all the unique "
                u"landcover codes in the raster st "
                u"`args['landcover_path']`<br/>* For every nesting "
                u"matching _NESTING_SUITABILITY_PATTERN in the guild "
                u"stable, a column matching the pattern in "
                u"`_LANDCOVER_NESTING_INDEX_HEADER`.<br/>* For every "
                u"season matching _FORAGING_ACTIVITY_PATTERN in the "
                u"guilds table, a column matching the pattern in "
                u"`_LANDCOVER_FLORAL_RESOURCES_INDEX_HEADER`."),
            label=u'Land Cover Biophysical Table (CSV)',
            validator=self.validator)
        self.add_input(self.landcover_biophysical_table_path)
        self.guild_table_path = inputs.File(
            args_key=u'guild_table_path',
            helptext=(
                u"A table indicating the bee species to analyze in "
                u"this model run.  Table headers must include:<br/>* "
                u"'species': a bee species whose column string names "
                u"will be referred to in other tables and the model "
                u"will output analyses per species.<br/> * any number "
                u"of columns matching _NESTING_SUITABILITY_PATTERN with "
                u"values in the range [0.0, 1.0] indicating the "
                u"suitability of the given species to nest in a "
                u"particular substrate.<br/>* any number of "
                u"_FORAGING_ACTIVITY_PATTERN columns with values in the "
                u"range [0.0, 1.0] indicating the relative level of "
                u"foraging activity for that species during a "
                u"particular season.<br/>* 'alpha': the sigma average "
                u"flight distance of that bee species in meters.<br/>* "
                u"'relative_abundance': a weight indicating the "
                u"relative abundance of the particular species with "
                u"respect to the sum of all relative abundance weights "
                u"in the table."),
            label=u'Guild Table (CSV)',
            validator=self.validator)
        self.add_input(self.guild_table_path)
        self.farm_vector_path = inputs.File(
            args_key=u'farm_vector_path',
            helptext=(
                u"This is a layer of polygons representing farm sites "
                u"to be analyzed.  The shapefile must have at least the "
                u"following fields:<br/><br/>* season (string): season "
                u"in which the farm needs pollination.<br/>* half_sat "
                u"(float): a real in the range [0.0, 1.0] representing "
                u"the proportion of wild pollinators to achieve a 50% "
                u"yield of that crop.<br/>* p_wild_dep (float): a "
                u"number in the range [0.0, 1.0] representing the "
                u"proportion of yield dependent on pollinators.<br/>* "
                u"p_managed (float): proportion of pollinators that "
                u"come from non-native/managed hives.<br/>* f_[season] "
                u"(float): any number of fields that match this pattern "
                u"such that `season` also matches the season headers in "
                u"the biophysical and guild table.  Any areas that "
                u"overlap the landcover map will replace seasonal "
                u"floral resources with this value.  Ranges from "
                u"0..1.<br/>* n_[substrate] (float): any number of "
                u"fields that match this pattern such that `substrate` "
                u"also matches the nesting substrate headers in the "
                u"biophysical and guild table.  Any areas that overlap "
                u"the landcover map will replace nesting substrate "
                u"suitability with this value.  Ranges from 0..1."),
            label=u'Farm Vector (Vector) (optional)',
            validator=self.validator)
        self.add_input(self.farm_vector_path)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.landcover_raster_path.args_key:
                self.landcover_raster_path.value(),
            self.landcover_biophysical_table_path.args_key:
                self.landcover_biophysical_table_path.value(),
            self.guild_table_path.args_key: self.guild_table_path.value(),
            self.farm_vector_path.args_key: self.farm_vector_path.value(),
        }

        return args
