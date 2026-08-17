from pydantic import BaseModel, ConfigDict


class Keyword(BaseModel):
    """Base model for all keywords."""

    model_config = ConfigDict(validate_assignment=True,
                              extra='forbid',
                              use_attribute_docstrings=True)

    value: str
    """The keyword string."""
    vocabulary: str
    """The name of a controlled vocabulary where this keyword is defined."""
    aliases: list[str] = []
    """A list of alternatives to the ``value`` string that carry the same meaning.

    It is useful to alias some GCMD keywords when they include special characters
    that are not usable in some contexts (e.g. as CKAN tags).
    """

    def __new__(cls, *args, **kwargs):
        if cls.__name__ == 'Keyword':
            raise TypeError(
                'Keyword is an abstract class and cannot be instantiated directly.')
        return super().__new__(cls)


class GCMDKeyword(Keyword):
    """A Keyword from the NASA Global Change Master Directory (GCMD)"""
    uuid: str
    """The unique id from the GCMD."""
    full_path: str
    """The full hierarchy of they keyword within the GCMD."""
    vocabulary: str = 'Global Change Master Directory (GCMD) Keywords'


class InvestKeyword(Keyword):
    definition: str = ''
    """The meaning of the keyword."""
    vocabulary: str = 'InVEST Keywords'


AGRICULTURE_PRODUCTION = GCMDKeyword(
    value='AGRICULTURE PRODUCTION',
    uuid='83741fb9-6f86-4670-abbb-c1f3b14a939d',
    full_path='EARTH SCIENCE > HUMAN DIMENSIONS > ECONOMIC RESOURCES > AGRICULTURE PRODUCTION')

ALBEDO = GCMDKeyword(
    value='ALBEDO',
    uuid='136b1de3-4b2e-49e6-80cd-cf2e9bac2c48',
    full_path='EARTH SCIENCE > LAND SURFACE > SURFACE RADIATIVE PROPERTIES > ALBEDO')

BATHYMETRY = GCMDKeyword(
    value='BATHYMETRY',
    uuid='80d79c7e-6c64-4ada-bfcc-4093969758a5',
    full_path='EARTH SCIENCE > OCEANS > BATHYMETRY/SEAFLOOR TOPOGRAPHY > BATHYMETRY')

BIOPHYSICAL_TABLE = InvestKeyword(
    value='BIOPHYSICAL TABLE',
    definition=(
        'A table used by InVEST models typically used to define relationships'
        ' between land cover classes and biophysical properties.'))

BUILDINGS = GCMDKeyword(
    value='BUILDINGS',
    uuid='d7742082-5461-4610-9ced-e0ec3bb64697',
    full_path='EARTH SCIENCE > HUMAN DIMENSIONS > INFRASTRUCTURE > BUILDINGS')

BUILDING_COOLING_ENERGY = InvestKeyword(
    value='BUILDING COOLING ENERGY',
    definition='The energy consumption required for cooling a building')

CARBON = GCMDKeyword(
    value='CARBON',
    uuid='6f6537f5-773f-4df1-862b-d9ab80eb5e04',
    full_path='EARTH SCIENCE > BIOSPHERE > VEGETATION > CARBON')

CARBON_REGRESSION_PARAMETERS = InvestKeyword(
    value='CARBON REGRESSION PARAMETERS',
    definition=(
        'Parameters used by the regression model in the InVEST Forest Carbon'
        ' Edge Effect model.'))

CARBON_SEQUESTRATION = GCMDKeyword(
    value='CARBON SEQUESTRATION',
    uuid='e58872a8-6104-4ff8-bbca-4b00ba4b38e8',
    full_path='EARTH SCIENCE > BIOSPHERE > ECOLOGICAL DYNAMICS > ECOSYSTEM FUNCTIONS > CARBON SEQUESTRATION')

CLIMATE_ZONES = InvestKeyword(
    value='CLIMATE ZONES',
    definition=(
        'Geographic distributions of climate groups based on seasonality,'
        ' temperature, and precipitation'))

COASTAL_LANDFORMS = GCMDKeyword(
    value='COASTAL LANDFORMS',
    uuid='c58320e6-3f1d-4c36-9bee-6bad73404c21',
    full_path='EARTH SCIENCE > SOLID EARTH > GEOMORPHIC LANDFORMS/PROCESSES > COASTAL LANDFORMS')

CONTOUR_MAPS = GCMDKeyword(
    value='CONTOUR MAPS',
    uuid='120f9132-a756-4f6f-a74c-78e94dfcd2a1',
    full_path='EARTH SCIENCE > LAND SURFACE > TOPOGRAPHY > TERRAIN ELEVATION > CONTOUR MAPS')

CONTINENTAL_MARGINS = GCMDKeyword(
    value='CONTINENTAL MARGINS',
    uuid='a91a00f7-05ed-4633-9fac-1772a48b6342',
    full_path='EARTH SCIENCE > OCEANS > BATHYMETRY/SEAFLOOR TOPOGRAPHY > CONTINENTAL MARGINS')

CROP_CLIMATE_SUITABILITY = InvestKeyword(
    value='CROP CLIMATE SUITABILITY',
    definition=(
        'An index describing the suitability of a location for a specific crop,'
        ' based on relevant climate parameters'))

CROP_COEFFICIENT = InvestKeyword(value='CROP COEFFICIENT')

CROP_PLANT_YIELDS = GCMDKeyword(
    value='CROP/PLANT YIELDS',
    uuid='f12d8026-f24a-4413-91d0-4704c243c9e7',
    full_path='EARTH SCIENCE > AGRICULTURE > AGRICULTURAL PLANT SCIENCE > CROP/PLANT YIELDS',
    aliases=['CROP YIELDS'])

CROP_TYPE = GCMDKeyword(
    value='CROP TYPE',
    uuid='e210bb78-19b6-453c-8dcb-81e389601329',
    full_path='EARTH SCIENCE > AGRICULTURE > AGRICULTURAL PLANT SCIENCE > CROP TYPE')

DEM = GCMDKeyword(
    value='DIGITAL ELEVATION/TERRAIN MODEL (DEM)',
    uuid='395372ad-2883-4b6a-a481-6383a310ca47',
    full_path='EARTH SCIENCE > LAND SURFACE > TOPOGRAPHY > TERRAIN ELEVATION > DIGITAL ELEVATION/TERRAIN MODEL (DEM)',
    aliases=['DEM', 'DIGITAL ELEVATION MODEL'])

DRAINAGE = GCMDKeyword(
    value='DRAINAGE',
    uuid='269c7277-fa8f-4c1c-bd8b-ab772c1df4e5',
    full_path='EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > SURFACE WATER > SURFACE WATER PROCESSES/MEASUREMENTS > DRAINAGE')

ELECTRICITY_COST = InvestKeyword(
    value='ELECTRICITY COST',
    definition='The monetary cost of electricity, typically per kilowatt-hour')

EVAPOTRANSPIRATION = GCMDKeyword(
    value='EVAPOTRANSPIRATION',
    uuid='26fc4850-7ba9-44d8-a156-5c623e17b72f',
    full_path='EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC WATER VAPOR > WATER VAPOR PROCESSES > EVAPOTRANSPIRATION')

EVENT_MEAN_CONCENTRATION = InvestKeyword(
    value='EVENT MEAN CONCENTRATION',
    definition='The mean pollutant concentration in the discharge of a storm event')

FERTILIZERS = GCMDKeyword(
    value='FERTILIZERS',
    uuid='18a8197e-3a3f-408c-9c51-e9fe89dd6b45',
    full_path='EARTH SCIENCE > AGRICULTURE > AGRICULTURAL CHEMICALS > FERTILIZERS')

FLOOD_DAMAGE_VALUE = InvestKeyword(
    value='FLOOD DAMAGE VALUE',
    definition='The economic value of flood damage to a property.')

FLOOR_AREA_RATIO = InvestKeyword(
    value='FLOOR AREA RATIO',
    definition='The ratio of the total floor area of a building to the area of the land parcel')

HABITAT_SENSITIVITY = InvestKeyword(
    value='HABITAT SENSITIVITY',
    definition='Parameters related to the sensitivity of habitat to specific threats.')

HABITAT_STRESSOR_CRITERIA = InvestKeyword(
    value='HABITAT STRESSOR CRITERIA',
    definition='Criteria related to how specific stressors impact specific habitats.')

HABITAT_THREATS = InvestKeyword(
    value='HABITAT THREATS',
    definition='Geographic distribution and parameters related to threats to habitat.')

HABITATS_STRESSORS = InvestKeyword(
    value='HABITATS AND STRESSORS',
    definition='Geographic distribution and parameters related to habitats and their stressors.')

HEIGHT = InvestKeyword(
    value='HEIGHT',
    definition='Distance above the ground.')

HYDROELECTRIC_ENERGY = GCMDKeyword(
    value='HYDROELECTRIC ENERGY PRODUCTION/USE',
    uuid='7eba0eef-3a30-4282-a162-1f483370ddc4',
    full_path='EARTH SCIENCE > HUMAN DIMENSIONS > ECONOMIC RESOURCES > ENERGY PRODUCTION/USE > HYDROELECTRIC ENERGY PRODUCTION/USE')

HYDROLOGIC_SOIL_GROUPS = InvestKeyword(
    value='HYDROLOGIC SOIL GROUPS',
    definition='Groups representing rainfall runoff potential. A component of the USDA curve number method.')

INFILTRATION_AMOUNT = GCMDKeyword(
    value='INFILTRATION AMOUNT',
    uuid='59ce52b5-0386-4b51-b5ac-049a0862e9cd',
    full_path='EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > GROUND WATER > GROUND WATER PROCESSES/MEASUREMENTS > INFILTRATION > INFILTRATION AMOUNT')

INFILTRATION_FREQUENCY = GCMDKeyword(
    value='INFILTRATION FREQUENCY',
    uuid='55642a14-2ff4-4892-b61a-ae3ece7fbcd7',
    full_path='EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > GROUND WATER > GROUND WATER PROCESSES/MEASUREMENTS > INFILTRATION > INFILTRATION FREQUENCY')

LANDMASS = InvestKeyword(
    value='LANDMASS',
    definition='Polygonal area of land surface above sea level.')

LULC = GCMDKeyword(
    value='LAND USE/LAND COVER',
    uuid='e5815f58-8232-4c7f-b50d-ea71d73891a9',
    full_path='EARTH SCIENCE > LAND SURFACE > LAND USE/LAND COVER',
    aliases=['LAND USE LAND COVER'])

# TODO: this is not currently in use. Instead biophysical tables are
# tagged with LULC. Should we use this tag instead? It would not currently
# match any tags in use on the DH.
LULC_CLASSES = GCMDKeyword(
    value='LAND USE/LAND COVER CLASSES',
    uuid='fe2f8240-4d8e-4b1f-b869-29fee59692f7',
    full_path='EARTH SCIENCE > LAND SURFACE > LAND USE/LAND COVER > LAND USE/LAND COVER CLASSES',
    aliases=['LAND USE LAND COVER CLASSES'])

NITROGEN = GCMDKeyword(
    value='NITROGEN',
    uuid='bf03dba8-2881-44ac-abfc-ba3353f67a24',
    full_path='EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > WATER QUALITY/WATER CHEMISTRY > NUTRIENTS > NITROGEN')

NUTRIENTS = GCMDKeyword(
    value='NUTRIENTS',
    uuid='9bcb805c-718e-42c3-913d-174bdf06d4c1',
    full_path='EARTH SCIENCE > BIOSPHERE > VEGETATION > NUTRIENTS')

PAWC = InvestKeyword(
    value='PLANT AVAILABLE WATER CONTENT',
    definition='The quantity of water in the soil that is available to plants.')

PERCOLATION = GCMDKeyword(
    value='PERCOLATION',
    uuid='d64094ae-774b-4435-8f2e-a54d114e5555',
    full_path='EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > GROUND WATER > GROUND WATER PROCESSES/MEASUREMENTS > PERCOLATION')

PHOSPHOROUS = GCMDKeyword(
    value='PHOSPHOROUS',
    uuid='846d2db9-41cd-4ae8-b4ff-a34a9efb7428',
    full_path='EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > WATER QUALITY/WATER CHEMISTRY > NUTRIENTS > PHOSPHOROUS')

PLANT_COMMODITIES = GCMDKeyword(
    value='PLANT COMMODITIES',
    uuid='d6560f20-3bef-41c6-8eec-9f913329b9ac',
    full_path='EARTH SCIENCE > AGRICULTURE > PLANT COMMODITIES')

POLLINATION_SEASON = InvestKeyword(
    value='POLLINATION SEASON',
    definition='The season in which a crop is pollinated.')

POLLINATOR_ABUNDANCE = InvestKeyword(
    value='POLLINATOR ABUNDANCE',
    definition=(
        'The abundance of a population of a pollinator species/guild.'
        ' Sometimes expressed as a proportion of the total'
        ' amount of pollinators in the landscape.'))

POLLINATOR_DEPENDENCE = InvestKeyword(
    value='POLLINATOR DEPENDENCE',
    definition=(
        'Parameters related to the dependence of a crop on pollinators.'
        ' For example, the proportion of farm output that is dependent on pollinators.'
        ' Or the proportion of pollinators that are wild vs. managed.'))

POLLINATOR_FLORAL_RESOURCES = InvestKeyword(
    value='POLLINATOR FLORAL RESOURCES',
    definition=(
        'Parameters related to the availability of floral resources for'
        ' pollinators. This may refer to seasonality of floral resources and'
        ' the abundance of floral resources across the landscape.'))

POLLINATOR_FORAGE_ACTIVITY = InvestKeyword(
    value='POLLINATOR FORAGE ACTIVITY',
    definition=(
        'Parameters related to the foraging activites of a pollinator species.'
        ' This may refer to seasonality of foraging and distance travelled by'
        ' foragers.'))

POLLINATOR_NESTING_SUITABILITY = InvestKeyword(
    value='POLLINATOR NESTING SUITABILITY',
    definition=(
        'Parameters related to the suitability of a habitat type for'
        ' pollinator nesting. This may refer to land cover classes that provide'
        ' suitable nesting substrate.'))

POLLINATOR_SPECIES = GCMDKeyword(
    value='POLLINATOR SPECIES',
    uuid='45950ee6-adc2-4f39-96a7-c00bacd1ba9e',
    full_path='EARTH SCIENCE > BIOSPHERE > ECOLOGICAL DYNAMICS > SPECIES/POPULATION INTERACTIONS > POLLINATOR SPECIES')

POPULATION_SIZE = GCMDKeyword(
    value='POPULATION SIZE',
    uuid='dd0b8bc9-90b3-4e7d-a021-e91dc676d622',
    full_path='EARTH SCIENCE > HUMAN DIMENSIONS > POPULATION > POPULATION SIZE')

PRECIPITATION = GCMDKeyword(
    value='PRECIPITATION',
    uuid='1532e590-a62d-46e3-8d03-2351bc48166a',
    full_path='EARTH SCIENCE > ATMOSPHERE > PRECIPITATION')

PRECIPITATION_RATE = GCMDKeyword(
    value='PRECIPITATION RATE',
    uuid='ac50c468-df2f-429c-8394-9d63efcc6f9d',
    full_path='EARTH SCIENCE > ATMOSPHERE > PRECIPITATION > PRECIPITATION_RATE')

PROTECTED_AREA = InvestKeyword(
    value='PROTECTED AREA',
    definition='An area managed for preservation of natural resources.')

RAINFALL_EROSIVITY = InvestKeyword(
    value='RAINFALL EROSIVITY',
    definition='The capacity of rainfall to cause soil erosion.')

REFERENCE_EVAPOTRANSPIRATION = InvestKeyword(
    value='REFERENCE EVAPOTRANSPIRATION',
    definition='Evapotranspiration of a reference vegetation such as grass or alfalfa.')

ROADS = GCMDKeyword(
    value='ROADS',
    uuid='648e6116-5f0c-4314-86a8-ede6dda6bbc1',
    full_path='EARTH SCIENCE > HUMAN DIMENSIONS > INFRASTRUCTURE > TRANSPORTATION > ROADS')

RUNOFF = GCMDKeyword(
    value='RUNOFF',
    uuid='f6a54329-486b-4d5f-b105-c639cec42351',
    full_path='EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > SURFACE WATER > SURFACE WATER PROCESSES/MEASUREMENTS > RUNOFF')

RUNOFF_COEFICIENT = InvestKeyword(
    value='RUNOFF COEFFICIENT',
    definition='The ratio between direct runoff and precipitation, typically as a function of soil, cover type, and hydrologic and antecedent conditions.')

RUNOFF_CURVE_NUMBER = InvestKeyword(
    value='RUNOFF CURVE NUMBER (CN)',
    definition='An index representing runoff as a function of soil, cover type, and hydrologic and antecedent conditions.')

SEA_LEVEL_RISE = GCMDKeyword(
    value='SEA LEVEL RISE',
    uuid='536a86bd-3dd1-4f4a-9b4a-222a12746db5',
    full_path='EARTH SCIENCE > CLIMATE INDICATORS > ATMOSPHERIC/OCEAN INDICATORS > SEA LEVEL RISE')

SHADE_FRACTION = GCMDKeyword(
    value='SHADE FRACTION',
    uuid='87be8c1f-9f92-45f9-ba9b-f50efa5bcca0',
    full_path='EARTH SCIENCE > LAND SURFACE > LAND USE/LAND COVER > SHADE FRACTION')

SHORELINES = GCMDKeyword(
    value='SHORELINES',
    uuid='1d3b4eb7-9931-44bf-8457-26847051b7a8',
    full_path='EARTH SCIENCE > OCEANS > COASTAL PROCESSES > SHORELINES',
    aliases=['COASTLINE'])

SOIL_ERODIBILITY = InvestKeyword(
    value='SOIL ERODIBILITY',
    definition='The susceptibility of soil to erosion as a function of soil properties.')

SOIL_EROSION = GCMDKeyword(
    value='SOIL EROSION',
    uuid='6eef914d-ff9f-44b0-a3a6-3dcf911023d4',
    full_path='EARTH SCIENCE > LAND SURFACE > SOILS > SOIL EROSION')

SOIL_ROOTING_DEPTH = GCMDKeyword(
    value='SOIL ROOTING DEPTH',
    uuid='1b475201-a032-4a66-a3aa-a35605affaee',
    full_path='EARTH SCIENCE > LAND SURFACE > SOILS > SOIL ROOTING DEPTH',
    aliases=['DEPTH TO BEDROCK', 'ROOT RESTRICTING LAYER DEPTH'])

USLE_C_FACTOR = InvestKeyword(
    value='USLE C FACTOR',
    definition='Cover-management factor for the Universal Soil Loss Equation')

USLE_P_FACTOR = InvestKeyword(
    value='USLE P FACTOR',
    definition='Support-practice factor for the Universal Soil Loss Equation')

VIEWSHED_OBSTRUCTION = InvestKeyword(
    value='VIEWSHED OBSTRUCTION',
    definition='Geographic features that obstruct the viewshed of an observer.')

WATERSHED_BOUNDARIES = GCMDKeyword(
    value='WATERSHED BOUNDARIES',
    uuid='b98123fc-6a87-4396-8e1a-ae7406e76ff6',
    full_path='EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > SURFACE WATER > WATERSHED CHARACTERISTICS > WATERSHED BOUNDARIES')

WATERSHED_DRAINAGE = GCMDKeyword(
    value='WATERSHED DRAINAGE',
    uuid='ae36ad48-85f2-42a0-958f-efec71c34cc0',
    full_path='EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > SURFACE WATER > WATERSHED CHARACTERISTICS > WATERSHED DRAINAGE')

WATER_BUDGET = GCMDKeyword(
    value='WATER BUDGET',
    uuid='f8702aed-a0ae-46f0-89eb-abde858bc6ac',
    full_path='EARTH SCIENCE > TERRESTRIAL HYDROSPHERE > WATER BUDGET')

WAVE_DIRECTION = GCMDKeyword(
    value='WAVE DIRECTION',
    uuid='037ce518-b71f-4599-b37f-feab9cc9809d',
    full_path='EARTH SCIENCE > OCEANS > OCEAN WAVES > WAVE DIRECTION')

WAVE_FETCH = GCMDKeyword(
    value='WAVE FETCH',
    uuid='09b326df-79b3-41b8-8998-e06344b0fe0d',
    full_path='EARTH SCIENCE > OCEANS > OCEAN WAVES > WAVE FETCH')

WAVE_POWER = InvestKeyword(
    value='WAVE POWER',
    definition='The power of surface ocean waves.')

WEATHER_EVENTS = GCMDKeyword(
    value='WEATHER EVENTS',
    uuid='b7d562cf-9b9b-4461-900b-50423a8c4d29',
    full_path='EARTH SCIENCE > ATMOSPHERE > WEATHER EVENTS')

WIND_DIRECTION = GCMDKeyword(
    value='WIND DIRECTION',
    uuid='d78e5503-d78e-466d-97bb-e68d6e768a9d',
    full_path='EARTH SCIENCE > OCEANS > OCEAN WINDS > SURFACE WINDS > WIND DIRECTION')

WIND_SPEED = GCMDKeyword(
    value='WIND SPEED',
    uuid='a7ce84a3-8329-4eb7-b5de-72d2dea8c6bf',
    full_path='EARTH SCIENCE > OCEANS > OCEAN WINDS > SURFACE WINDS > WIND SPEED')


def to_list():
    keywords = []
    for attr in globals().values():
        if isinstance(attr, Keyword):
            keywords.append(attr)
    return keywords
