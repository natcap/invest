# coding=UTF-8

from natcap.invest.ui import model, inputs
from natcap.invest.coastal_vulnerability import coastal_vulnerability


class CoastalVulnerability(model.InVESTModel):
    def __init__(self):
        model.InVESTModel.__init__(
            self,
            label=u'Coastal Vulnerability Assessment Tool',
            target=coastal_vulnerability.execute,
            validator=coastal_vulnerability.validate,
            localdoc=u'coastal_vulnerability.html',
            suffix_args_key='suffix'
        )

        self.general_tab = inputs.Container(
            interactive=True,
            label=u'General')
        self.add_input(self.general_tab)
        self.area_computed = inputs.Dropdown(
            args_key=u'area_computed',
            helptext=(
                u"Determine if the output data is about all the coast "
                u"or about sheltered segments only."),
            label=u'Output Area: Sheltered/Exposed?',
            options=[u'both', u'sheltered'])
        self.general_tab.add_input(self.area_computed)
        self.area_of_interest = inputs.File(
            args_key=u'aoi_uri',
            helptext=(
                u"An OGR-supported, single-feature polygon vector "
                u"file. Must have a projected coordinate system. "
                u"All outputs will be in the AOI's projection."),
            label=u'Area of Interest (Vector)',
            validator=self.validator)
        self.general_tab.add_input(self.area_of_interest)
        self.landmass_uri = inputs.File(
            args_key=u'landmass_uri',
            helptext=(
                u"An OGR-supported vector file containing a landmass "
                u"polygon from where the coastline will be extracted. "
                u"The default is the global land polygon."),
            label=u'Land Polygon (Vector)',
            validator=self.validator)
        self.general_tab.add_input(self.landmass_uri)
        self.bathymetry_layer = inputs.File(
            args_key=u'bathymetry_uri',
            helptext=(
                u"A GDAL-supported raster of the terrain elevation in "
                u"the area of interest.  Used to compute depths along "
                u"fetch rays, relief and surge potential."),
            label=u'Bathymetry Layer (Raster)',
            validator=self.validator)
        self.general_tab.add_input(self.bathymetry_layer)
        self.bathymetry_constant = inputs.Text(
            args_key=u'bathymetry_constant',
            helptext=(
                u"Integer value between 1 and 5. If layer associated "
                u"to this field is omitted, replace all shore points "
                u"for this layer with a constant rank value in the "
                u"computation of the coastal vulnerability index.  If "
                u"both the file and value for the layer are omitted, "
                u"the layer is skipped altogether."),
            label=u'Layer Value if Path Omitted',
            validator=self.validator)
        self.general_tab.add_input(self.bathymetry_constant)
        self.relief = inputs.File(
            args_key=u'relief_uri',
            helptext=(
                u"A GDAL-supported raster file containing the land "
                u"elevation used to compute the average land elevation "
                u"within a user-defined radius (see Elevation averaging "
                u"radius)."),
            label=u'Relief (Raster)',
            validator=self.validator)
        self.general_tab.add_input(self.relief)
        self.relief_constant = inputs.Text(
            args_key=u'relief_constant',
            helptext=(
                u"Integer value between 1 and 5. If layer associated "
                u"to this field is omitted, replace all shore points "
                u"for this layer with a constant rank value in the "
                u"computation of the coastal vulnerability index.  If "
                u"both the file and value for the layer are omitted, "
                u"the layer is skipped altogether."),
            label=u'Layer Value If Path Omitted',
            validator=self.validator)
        self.general_tab.add_input(self.relief_constant)
        self.cell_size = inputs.Text(
            args_key=u'cell_size',
            helptext=(
                u"Cell size measured in the projection units of the AOI. "
                u"For example, UTM projections use meters. "
                u"The higher the value, the faster the computation, "
                u"but the coarser the output rasters produced by the model."),
            label=u'Model Resolution (Segment Size)',
            validator=self.validator)
        self.general_tab.add_input(self.cell_size)
        self.depth_threshold = inputs.Text(
            args_key=u'depth_threshold',
            helptext=(
                u"Depth in meters (integer) cutoff to determine if "
                u"fetch rays project over deep areas."),
            label=u'Depth Threshold (meters)',
            validator=self.validator)
        self.depth_threshold.set_value('0')
        self.general_tab.add_input(self.depth_threshold)
        self.exposure_proportion = inputs.Text(
            args_key=u'exposure_proportion',
            helptext=(
                u"Minimum proportion of rays that project over exposed "
                u"and/or deep areas need to classify a shore segment as "
                u"exposed."),
            label=u'Exposure Proportion',
            validator=self.validator)
        self.exposure_proportion.set_value('0.8')
        self.general_tab.add_input(self.exposure_proportion)
        self.geomorphology_uri = inputs.File(
            args_key=u'geomorphology_uri',
            helptext=(
                u"A OGR-supported polygon vector file that has a field "
                u"called 'RANK' with values between 1 and 5 in the "
                u"attribute table."),
            label=u'Geomorphology (Vector)',
            validator=self.validator)
        self.general_tab.add_input(self.geomorphology_uri)
        self.geomorphology_constant = inputs.Text(
            args_key=u'geomorphology_constant',
            helptext=(
                u"Integer value between 1 and 5. If layer associated "
                u"to this field is omitted, replace all shore points "
                u"for this layer with a constant rank value in the "
                u"computation of the coastal vulnerability index.  If "
                u"both the file and value for the layer are omitted, "
                u"the layer is skipped altogether."),
            label=u'Layer Value if Path Omitted',
            validator=self.validator)
        self.general_tab.add_input(self.geomorphology_constant)
        self.habitats_directory_uri = inputs.Folder(
            args_key=u'habitats_directory_uri',
            helptext=(
                u"Directory containing OGR-supported polygon vectors "
                u"associated with natural habitats.  The name of these "
                u"shapefiles should be suffixed with the ID that is "
                u"specified in the natural habitats CSV file provided "
                u"along with the habitats."),
            label=u'Natural Habitats Directory',
            validator=self.validator)
        self.general_tab.add_input(self.habitats_directory_uri)
        self.habitats_csv_uri = inputs.File(
            args_key=u'habitats_csv_uri',
            helptext=(
                u"A CSV file listing the attributes for each habitat. "
                u"For more information, see 'Habitat Data Layer' "
                u"section in the model's documentation.</a>."),
            interactive=False,
            label=u'Natural Habitats Table (CSV)',
            validator=self.validator)
        self.general_tab.add_input(self.habitats_csv_uri)
        self.habitats_constant = inputs.Text(
            args_key=u'habitat_constant',
            helptext=(
                u"Integer value between 1 and 5. If layer associated "
                u"to this field is omitted, replace all shore points "
                u"for this layer with a constant rank value in the "
                u"computation of the coastal vulnerability index.  If "
                u"both the file and value for the layer are omitted, "
                u"the layer is skipped altogether."),
            label=u'Layer Value if Path Omitted',
            validator=self.validator)
        self.general_tab.add_input(self.habitats_constant)
        self.climatic_forcing_uri = inputs.File(
            args_key=u'climatic_forcing_uri',
            helptext=(
                u"An OGR-supported vector containing both wind and "
                u"wave information across the region of interest."),
            label=u'Climatic Forcing Grid (Vector)',
            validator=self.validator)
        self.general_tab.add_input(self.climatic_forcing_uri)
        self.climatic_forcing_constant = inputs.Text(
            args_key=u'climatic_forcing_constant',
            helptext=(
                u"Integer value between 1 and 5. If layer associated "
                u"to this field is omitted, replace all shore points "
                u"for this layer with a constant rank value in the "
                u"computation of the coastal vulnerability index.  If "
                u"both the file and value for the layer are omitted, "
                u"the layer is skipped altogether."),
            label=u'Layer Value if Path Omitted',
            validator=self.validator)
        self.general_tab.add_input(self.climatic_forcing_constant)
        self.continental_shelf_uri = inputs.File(
            args_key=u'continental_shelf_uri',
            helptext=(
                u"An OGR-supported polygon vector delineating the "
                u"edges of the continental shelf.  Default is global "
                u"continental shelf shapefile.  If omitted, the user "
                u"can specify depth contour.  See entry below."),
            label=u'Continental Shelf (Vector)',
            validator=self.validator)
        self.general_tab.add_input(self.continental_shelf_uri)
        self.depth_contour = inputs.Text(
            args_key=u'depth_contour',
            helptext=(
                u"Used to delineate shallow and deep areas. "
                u"Continental shelf limit is at about 150 meters."),
            label=u'Depth Countour Level (meters)',
            validator=self.validator)
        self.general_tab.add_input(self.depth_contour)
        self.sea_level_rise_uri = inputs.File(
            args_key=u'sea_level_rise_uri',
            helptext=(
                u"An OGR-supported point or polygon vector file "
                u"containing features with 'Trend' fields in the "
                u"attributes table."),
            label=u'Sea Level Rise (Vector)',
            validator=self.validator)
        self.general_tab.add_input(self.sea_level_rise_uri)
        self.sea_level_rise_constant = inputs.Text(
            args_key=u'sea_level_rise_constant',
            helptext=(
                u"Integer value between 1 and 5. If layer associated "
                u"to this field is omitted, replace all shore points "
                u"for this layer with a constant rank value in the "
                u"computation of the coastal vulnerability index.  If "
                u"both the file and value for the layer are omitted, "
                u"the layer is skipped altogether."),
            label=u'Layer Value if Path Omitted',
            validator=self.validator)
        self.general_tab.add_input(self.sea_level_rise_constant)
        self.structures_uri = inputs.File(
            args_key=u'structures_uri',
            helptext=(
                u"An OGR-supported vector file containing rigid "
                u"structures used to identify the portions of the coast "
                u"that is armored."),
            label=u'Structures (Vectors)',
            validator=self.validator)
        self.general_tab.add_input(self.structures_uri)
        self.structures_constant = inputs.Text(
            args_key=u'structures_constant',
            helptext=(
                u"Integer value between 1 and 5. If layer associated "
                u"to this field is omitted, replace all shore points "
                u"for this layer with a constant rank value in the "
                u"computation of the coastal vulnerability index.  If "
                u"both the file and value for the layer are omitted, "
                u"the layer is skipped altogether."),
            label=u'Layer Value if Path Omitted',
            validator=self.validator)
        self.general_tab.add_input(self.structures_constant)
        self.population_uri = inputs.File(
            args_key=u'population_uri',
            helptext=(
                u'A GDAL-supported raster file representing the population '
                u'density.'),
            label=u'Population Layer (Raster)',
            validator=self.validator)
        self.general_tab.add_input(self.population_uri)
        self.urban_center_threshold = inputs.Text(
            args_key=u'urban_center_threshold',
            helptext=(
                u"Minimum population required to consider the shore "
                u"segment a population center."),
            label=u'Min. Population in Urban Centers',
            validator=self.validator)
        self.general_tab.add_input(self.urban_center_threshold)
        self.additional_layer_uri = inputs.File(
            args_key=u'additional_layer_uri',
            helptext=(
                u"An OGR-supported vector file representing sea level "
                u"rise, and will be used in the computation of coastal "
                u"vulnerability and coastal vulnerability without "
                u"habitat."),
            label=u'Additional Layer (Vector)',
            validator=self.validator)
        self.general_tab.add_input(self.additional_layer_uri)
        self.additional_layer_constant = inputs.Text(
            args_key=u'additional_layer_constant',
            helptext=(
                u"Integer value between 1 and 5. If layer associated "
                u"to this field is omitted, replace all shore points "
                u"for this layer with a constant rank value in the "
                u"computation of the coastal vulnerability index.  If "
                u"both the file and value for the layer are omitted, "
                u"the layer is skipped altogether."),
            label=u'Layer Value if Path Omitted',
            validator=self.validator)
        self.general_tab.add_input(self.additional_layer_constant)
        self.advanced_tab = inputs.Container(
            interactive=True,
            label=u'Advanced')
        self.add_input(self.advanced_tab)
        self.elevation_averaging_radius = inputs.Text(
            args_key=u'elevation_averaging_radius',
            helptext=(
                u"Distance in meters (integer). Each pixel average "
                u"elevation will be computed within this radius."),
            label=u'Elevation Averaging Radius (meters)',
            validator=self.validator)
        self.elevation_averaging_radius.set_value('5000')
        self.advanced_tab.add_input(self.elevation_averaging_radius)
        self.mean_sea_level_datum = inputs.Text(
            args_key=u'mean_sea_level_datum',
            helptext=(
                u"Height in meters (integer). This input is the "
                u"elevation of Mean Sea Level (MSL) datum relative to "
                u"the datum of the bathymetry layer.  The model "
                u"transforms all depths to MSL datum.  A positive value "
                u"means the MSL is higher than the bathymetry's zero "
                u"(0) elevation, so the value is subtracted from the "
                u"bathymetry."),
            label=u'Mean Sea Level Datum (meters)',
            validator=self.validator)
        self.mean_sea_level_datum.set_value('0')
        self.advanced_tab.add_input(self.mean_sea_level_datum)
        self.rays_per_sector = inputs.Text(
            args_key=u'rays_per_sector',
            helptext=(
                u"Number of rays used to subsample the fetch distance "
                u"within each of the 16 sectors."),
            label=u'Rays per Sector',
            validator=self.validator)
        self.rays_per_sector.set_value('1')
        self.advanced_tab.add_input(self.rays_per_sector)
        self.max_fetch = inputs.Text(
            args_key=u'max_fetch',
            helptext=(
                u'Maximum fetch distance computed by the model '
                u'(&gt;=60,000m).'),
            label=u'Maximum Fetch Distance (meters)',
            validator=self.validator)
        self.max_fetch.set_value('12000')
        self.advanced_tab.add_input(self.max_fetch)
        self.spread_radius = inputs.Text(
            args_key=u'spread_radius',
            helptext=(
                u"Integer multiple of 'cell size'. The coast from the "
                u"geomorphology layer could be of a better resolution "
                u"than the global landmass, so the shores do not "
                u"necessarily overlap.  To make them coincide, the "
                u"shore from the geomorphology layer is widened by 1 or "
                u"more pixels.  The value should be a multiple of 'cell "
                u"size' that indicates how many pixels the coast from "
                u"the geomorphology layer is widened.  The widening "
                u"happens on each side of the coast (n pixels landward, "
                u"and n pixels seaward)."),
            label=u'Coastal Overlap (meters)',
            validator=self.validator)
        self.spread_radius.set_value('250')
        self.advanced_tab.add_input(self.spread_radius)
        self.population_radius = inputs.Text(
            args_key=u'population_radius',
            helptext=(
                u"Radius length in meters used to count the number of "
                u"people leaving close to the coast."),
            label=u'Coastal Neighborhood (radius in meters)',
            validator=self.validator)
        self.population_radius.set_value('1000')
        self.advanced_tab.add_input(self.population_radius)

        # Set interactivity, requirement as input sufficiency changes
        self.habitats_directory_uri.sufficiency_changed.connect(
            self.habitats_csv_uri.set_interactive)

    def assemble_args(self):
        args = {
            self.workspace.args_key: self.workspace.value(),
            self.suffix.args_key: self.suffix.value(),
            self.area_computed.args_key: self.area_computed.value(),
            self.area_of_interest.args_key: self.area_of_interest.value(),
            self.landmass_uri.args_key: self.landmass_uri.value(),
            self.cell_size.args_key: self.cell_size.value(),
            self.exposure_proportion.args_key: self.exposure_proportion.value(),
            self.habitats_csv_uri.args_key: self.habitats_csv_uri.value(),
            self.population_uri.args_key: self.population_uri.value(),
            self.urban_center_threshold.args_key: (
                self.urban_center_threshold.value()),
            self.rays_per_sector.args_key: self.rays_per_sector.value(),
            self.spread_radius.args_key: self.spread_radius.value(),
            self.population_radius.args_key: self.population_radius.value(),
            self.bathymetry_layer.args_key: self.bathymetry_layer.value(),
            self.relief.args_key: self.relief.value(),
        }
        if self.bathymetry_constant.value():
            args[self.bathymetry_constant.args_key] = (
                self.bathymetry_constant.value())
        if self.relief_constant.value():
            args[self.relief_constant.args_key] = self.relief_constant.value()
        if self.depth_threshold.value():
            args[self.depth_threshold.args_key] = self.depth_threshold.value()
        if self.geomorphology_uri.value():
            args[self.geomorphology_uri.args_key] = (
                self.geomorphology_uri.value())
        if self.geomorphology_constant.value():
            args[self.geomorphology_constant.args_key] = self.geomorphology_constant.value()
        if self.habitats_directory_uri.value():
            args[self.habitats_directory_uri.args_key] = self.habitats_directory_uri.value()
        if self.habitats_constant.value():
            args[self.habitats_constant.args_key] = self.habitats_constant.value()
        if self.climatic_forcing_uri.value():
            args[self.climatic_forcing_uri.args_key] = self.climatic_forcing_uri.value()
        if self.climatic_forcing_constant.value():
            args[self.climatic_forcing_constant.args_key] = self.climatic_forcing_constant.value()
        if self.continental_shelf_uri.value():
            args[self.continental_shelf_uri.args_key] = self.continental_shelf_uri.value()
        if self.depth_contour.value():
            args[self.depth_contour.args_key] = self.depth_contour.value()
        if self.sea_level_rise_uri.value():
            args[self.sea_level_rise_uri.args_key] = self.sea_level_rise_uri.value()
        if self.sea_level_rise_constant.value():
            args[self.sea_level_rise_constant.args_key] = self.sea_level_rise_constant.value()
        if self.structures_uri.value():
            args[self.structures_uri.args_key] = self.structures_uri.value()
        if self.structures_constant.value():
            args[self.structures_constant.args_key] = self.structures_constant.value()
        if self.additional_layer_uri.value():
            args[self.additional_layer_uri.args_key] = self.additional_layer_uri.value()
        if self.additional_layer_constant.value():
            args[self.additional_layer_constant.args_key] = self.additional_layer_constant.value()
        if self.elevation_averaging_radius.value():
            args[self.elevation_averaging_radius.args_key] = self.elevation_averaging_radius.value()
        if self.mean_sea_level_datum.value():
            args[self.mean_sea_level_datum.args_key] = self.mean_sea_level_datum.value()
        if self.max_fetch.value():
            args[self.max_fetch.args_key] = self.max_fetch.value()

        return args
