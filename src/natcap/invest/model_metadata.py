import dataclasses

from natcap.invest import gettext


import importlib
import pkgutil

import natcap.invest

def iter_namespace(ns_pkg):
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return

discovered_plugins = {
    name: importlib.import_module(name)
    for finder, name, ispkg in pkgutil.iter_modules(
        natcap.invest.__path__, natcap.invest.__name__ + ".")
}


@dataclasses.dataclass
class _MODELMETA:
    """Dataclass to store frequently used model metadata."""
    model_title: str  # display name for the model
    pyname: str       # importable python module name for the model
    gui: str          # importable python class for the corresponding Qt UI
    userguide: str    # name of the corresponding built userguide file
    aliases: tuple    # alternate names for the model, if any


MODEL_METADATA = {

'annual_water_yield'

    'carbon':
    'coastal_vulnerability':
    'crop_production_percentile':
    'crop_production_regression':
    'forest_carbon_edge_effect':
    'habitat_quality':
    'habitat_risk_assessment'
    'pollination'

    'routedem'
    'scenario_generator_proximity'
    'seasonal_water_yield'
    'stormwater'
    'wave_energy'
    'wind_energy'
    'urban_flood_risk_mitigation'
    'urban_cooling_model'
    'urban_nature_access'

}
