"""init module for natcap.invest."""
import builtins
import collections
import gettext
import logging
import os
import sys

import pkg_resources

# location of our translation message catalog directory
LOCALE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'translations/locales')

LOGGER = logging.getLogger('natcap.invest')
LOGGER.addHandler(logging.NullHandler())
__all__ = ['local_dir', ]

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    # package is not installed.  Log the exception for debugging.
    LOGGER.exception('Could not load natcap.invest version information')


# Check if the function _() is available
# If not, define it as the identity function
# _() is installed into builtins by gettext when we set up to translate
# It wraps every string in every model that we want to translate
# Make sure it's defined so that natcap.invest modules are importable whether
# or not gettext has been installed in the importing namespace
if not callable(getattr(builtins, '_', None)):
    print('_() not already defined; setting to identity')
    def identity(x): return x
    builtins.__dict__['_'] = identity
else:
    print('_() already defined')


_UIMETA = collections.namedtuple('UIMeta', 'humanname pyname gui aliases')
MODEL_UIS = {
    'carbon': _UIMETA(
        humanname=_('Carbon Storage and Sequestration'),
        pyname='natcap.invest.carbon',
        gui='carbon.Carbon',
        aliases=()),
    'coastal_blue_carbon': _UIMETA(
        humanname=_('Coastal Blue Carbon'),
        pyname='natcap.invest.coastal_blue_carbon.coastal_blue_carbon',
        gui='cbc.CoastalBlueCarbon',
        aliases=('cbc',)),
    'coastal_blue_carbon_preprocessor': _UIMETA(
        humanname=_('Coastal Blue Carbon: Preprocessor'),
        pyname='natcap.invest.coastal_blue_carbon.preprocessor',
        gui='cbc.CoastalBlueCarbonPreprocessor',
        aliases=('cbc_pre',)),
    'coastal_vulnerability': _UIMETA(
        humanname=_('Coastal Vulnerability'),
        pyname='natcap.invest.coastal_vulnerability',
        gui='coastal_vulnerability.CoastalVulnerability',
        aliases=('cv',)),
    'crop_production_percentile': _UIMETA(
        humanname=_('Crop Production: Percentile Model'),
        pyname='natcap.invest.crop_production_percentile',
        gui='crop_production.CropProductionPercentile',
        aliases=('cpp',)),
    'crop_production_regression': _UIMETA(
        humanname=_('Crop Production: Regression Model'),
        pyname='natcap.invest.crop_production_regression',
        gui='crop_production.CropProductionRegression',
        aliases=('cpr',)),
    'delineateit': _UIMETA(
        humanname=_('DelineateIt'),
        pyname='natcap.invest.delineateit.delineateit',
        gui='delineateit.Delineateit',
        aliases=()),
    'finfish_aquaculture': _UIMETA(
        humanname=_('Marine Finfish Aquaculture Production'),
        pyname='natcap.invest.finfish_aquaculture.finfish_aquaculture',
        gui='finfish.FinfishAquaculture',
        aliases=()),
    'fisheries': _UIMETA(
        humanname=_('Fisheries'),
        pyname='natcap.invest.fisheries.fisheries',
        gui='fisheries.Fisheries',
        aliases=()),
    'fisheries_hst': _UIMETA(
        humanname=_('Fisheries: Habitat Scenario Tool'),
        pyname='natcap.invest.fisheries.fisheries_hst',
        gui='fisheries.FisheriesHST',
        aliases=()),
    'forest_carbon_edge_effect': _UIMETA(
        humanname=_('Forest Carbon Edge Effect'),
        pyname='natcap.invest.forest_carbon_edge_effect',
        gui='forest_carbon.ForestCarbonEdgeEffect',
        aliases=('fc',)),
    'globio': _UIMETA(
        humanname=_('GLOBIO'),
        pyname='natcap.invest.globio',
        gui='globio.GLOBIO',
        aliases=()),
    'habitat_quality': _UIMETA(
        humanname=_('Habitat Quality'),
        pyname='natcap.invest.habitat_quality',
        gui='habitat_quality.HabitatQuality',
        aliases=('hq',)),
    'habitat_risk_assessment': _UIMETA(
        humanname=_('Habitat Risk Assessment'),
        pyname='natcap.invest.hra',
        gui='hra.HabitatRiskAssessment',
        aliases=('hra',)),
    'hydropower_water_yield': _UIMETA(
        humanname=_('Annual Water Yield'),
        pyname='natcap.invest.hydropower.hydropower_water_yield',
        gui='hydropower.HydropowerWaterYield',
        aliases=('hwy',)),
    'ndr': _UIMETA(
        humanname=_('NDR: Nutrient Delivery Ratio'),
        pyname='natcap.invest.ndr.ndr',
        gui='ndr.Nutrient',
        aliases=()),
    'pollination': _UIMETA(
        humanname=_('Pollinator Abundance: Crop Pollination'),
        pyname='natcap.invest.pollination',
        gui='pollination.Pollination',
        aliases=()),
    'recreation': _UIMETA(
        humanname=_('Visitation: Recreation and Tourism'),
        pyname='natcap.invest.recreation.recmodel_client',
        gui='recreation.Recreation',
        aliases=()),
    'routedem': _UIMETA(
        humanname=_('RouteDEM'),
        pyname='natcap.invest.routedem',
        gui='routedem.RouteDEM',
        aliases=()),
    'scenario_generator_proximity': _UIMETA(
        humanname=_('Scenario Generator: Proximity Based'),
        pyname='natcap.invest.scenario_gen_proximity',
        gui='scenario_gen.ScenarioGenProximity',
        aliases=('sgp',)),
    'scenic_quality': _UIMETA(
        humanname=_('Unobstructed Views: Scenic Quality Provision'),
        pyname='natcap.invest.scenic_quality.scenic_quality',
        gui='scenic_quality.ScenicQuality',
        aliases=('sq',)),
    'sdr': _UIMETA(
        humanname=_('SDR: Sediment Delivery Ratio'),
        pyname='natcap.invest.sdr.sdr',
        gui='sdr.SDR',
        aliases=()),
    'seasonal_water_yield': _UIMETA(
        humanname=_('Seasonal Water Yield'),
        pyname='natcap.invest.seasonal_water_yield.seasonal_water_yield',
        gui='seasonal_water_yield.SeasonalWaterYield',
        aliases=('swy',)),
    'wind_energy': _UIMETA(
        humanname=_('Offshore Wind Energy Production'),
        pyname='natcap.invest.wind_energy',
        gui='wind_energy.WindEnergy',
        aliases=()),
    'wave_energy': _UIMETA(
        humanname=_('Wave Energy Production'),
        pyname='natcap.invest.wave_energy',
        gui='wave_energy.WaveEnergy',
        aliases=()),
    'urban_flood_risk_mitigation': _UIMETA(
        humanname=_('Urban Flood Risk Mitigation'),
        pyname='natcap.invest.urban_flood_risk_mitigation',
        gui='urban_flood_risk_mitigation.UrbanFloodRiskMitigation',
        aliases=('ufrm',)),
    'urban_cooling_model': _UIMETA(
        humanname=_('Urban Cooling'),
        pyname='natcap.invest.urban_cooling_model',
        gui='urban_cooling_model.UrbanCoolingModel',
        aliases=('ucm',)),
}


def install_language(language_code):
    # globally install the _() function for the requested language
    # fall back to a NullTranslation, which returns the English messages
    print(LOCALE_DIR)
    language = gettext.translation(
        'messages',
        languages=[language_code],
        localedir=LOCALE_DIR,
        fallback=True)
    language.install()
    LOGGER.debug(f'Installed language "{language_code}"')
    print('installed language', language_code)


def local_dir(source_file):
    """Return the path to where `source_file` would be on disk.

    If this is frozen (as with PyInstaller), this will be the folder with the
    executable in it.  If not, it'll just be the foldername of the source_file
    being passed in.
    """
    source_dirname = os.path.dirname(source_file)
    if getattr(sys, 'frozen', False):
        # sys.frozen is True when we're in either a py2exe or pyinstaller
        # build.
        # sys._MEIPASS exists, we're in a Pyinstaller build.
        if not getattr(sys, '_MEIPASS', False):
            # only one os.path.dirname() results in the path being relative to
            # the natcap.invest package, when I actually want natcap/invest to
            # be in the filepath.

            # relpath would be something like <modelname>/<data_file>
            relpath = os.path.relpath(source_file, os.path.dirname(__file__))
            pkg_path = os.path.join('natcap', 'invest', relpath)
            return os.path.join(
                os.path.dirname(sys.executable), os.path.dirname(pkg_path))
        else:
            # assume that if we're in a frozen build, we're in py2exe.  When in
            # py2exe, the directory structure is maintained, so we just return
            # the source_dirname.
            pass
    return source_dirname
