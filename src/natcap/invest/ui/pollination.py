#TODO: all the other UI modules have a # coding=UTF-8 here, does this one need it too?
import logging

from . import inputs
from . import model
from .. import pollination

LOGGER = logging.getLogger(__name__)


class Pollination(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label='Crop Pollination',
            target=pollination.execute,
            validator=pollination.validate,
            localdoc='croppollination.html')

        self.landcover_raster_path = inputs.File(
            args_key='landcover_raster_path',
            helptext=(
                "This is the landcover map that's used to map "
                "biophyiscal properties about habitat and floral "
                "resources of landcover types to a spatial layout."),
            label='Land Cover Map (Raster)',
            validator=self.validator)
        self.add_input(self.landcover_raster_path)
        self.landcover_biophysical_table_path = inputs.File(
            args_key='landcover_biophysical_table_path',
            helptext=(
                "A CSV table mapping landcover codes in the landcover "
                "raster to indexes of nesting availability for each "
                "nesting substrate referenced in guilds table as well "
                "as indexes of abundance of floral resources on that "
                "landcover type per season in the bee activity columns "
                "of the guild table.<br/>All indexes are in the range "
                "[0.0, 1.0].<br/>Columns in the table must be at "
                "least<br/>* 'lucode': representing all the unique "
                "landcover codes in the raster st "
                "`args['landcover_path']`<br/>* For every nesting "
                "matching _NESTING_SUITABILITY_PATTERN in the guild "
                "stable, a column matching the pattern in "
                "`_LANDCOVER_NESTING_INDEX_HEADER`.<br/>* For every "
                "season matching _FORAGING_ACTIVITY_PATTERN in the "
                "guilds table, a column matching the pattern in "
                "`_LANDCOVER_FLORAL_RESOURCES_INDEX_HEADER`."),
            label='Land Cover Biophysical Table (CSV)',
            validator=self.validator)
        self.add_input(self.landcover_biophysical_table_path)
        self.guild_table_path = inputs.File(
            args_key='guild_table_path',
            helptext=(
                "A table indicating the bee species to analyze in "
                "this model run.  Table headers must include:<br/>* "
                "'species': a bee species whose column string names "
                "will be referred to in other tables and the model "
                "will output analyses per species.<br/> * any number "
                "of columns matching _NESTING_SUITABILITY_PATTERN with "
                "values in the range [0.0, 1.0] indicating the "
                "suitability of the given species to nest in a "
                "particular substrate.<br/>* any number of "
                "_FORAGING_ACTIVITY_PATTERN columns with values in the "
                "range [0.0, 1.0] indicating the relative level of "
                "foraging activity for that species during a "
                "particular season.<br/>* 'alpha': the sigma average "
                "flight distance of that bee species in meters.<br/>* "
                "'relative_abundance': a weight indicating the "
                "relative abundance of the particular species with "
                "respect to the sum of all relative abundance weights "
                "in the table."),
            label='Guild Table (CSV)',
            validator=self.validator)
        self.add_input(self.guild_table_path)
        self.farm_vector_path = inputs.File(
            args_key='farm_vector_path',
            helptext=(
                "This is a layer of polygons representing farm sites "
                "to be analyzed.  The shapefile must have at least the "
                "following fields:<br/><br/>* season (string): season "
                "in which the farm needs pollination.<br/>* half_sat "
                "(float): a real in the range [0.0, 1.0] representing "
                "the proportion of wild pollinators to achieve a 50% "
                "yield of that crop.<br/>* p_wild_dep (float): a "
                "number in the range [0.0, 1.0] representing the "
                "proportion of yield dependent on pollinators.<br/>* "
                "p_managed (float): proportion of pollinators that "
                "come from non-native/managed hives.<br/>* f_[season] "
                "(float): any number of fields that match this pattern "
                "such that `season` also matches the season headers in "
                "the biophysical and guild table.  Any areas that "
                "overlap the landcover map will replace seasonal "
                "floral resources with this value.  Ranges from "
                "0..1.<br/>* n_[substrate] (float): any number of "
                "fields that match this pattern such that `substrate` "
                "also matches the nesting substrate headers in the "
                "biophysical and guild table.  Any areas that overlap "
                "the landcover map will replace nesting substrate "
                "suitability with this value.  Ranges from 0..1."),
            label='Farm Vector (Vector) (optional)',
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
