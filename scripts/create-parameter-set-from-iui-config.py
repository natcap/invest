
import sys
import argparse
import os
import json
import itertools

from natcap.invest import datastack

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'invest-data'))

IUI_SCENARIOS = {
    'carbon': 'carbon/carbon_willamette',
    'coastal_blue_carbon': 'CoastalBlueCarbon/cbc_galveston_bay',
    'coastal_blue_carbon_preprocessor': 'CoastalBlueCarbon/cbc_pre_galveston_bay',
    'coastal_vulnerability': 'CoastalProtection/coastal_vuln_wcvi',
    'crop_production': 'CropProduction/crop_production_willamette',
    'delineateit': 'Base_Data/Freshwater/delineateit_willamette',
    'finfish_aquaculture': 'Aquaculture/atlantic_salmon_british_columbia',
    'fisheries': 'Fisheries/blue_crab_galveston_bay',
    'fisheries_hst': 'Fisheries/fisheries_hst_demo',
    'forest_carbon_edge_effect': 'forest_carbon_edge_effect/forest_carbon_amazonia',
    'globio': 'globio/globio_demo',
    'habitat_quality': 'HabitatQuality/habitat_quality_willamette',
    'hra': 'HabitatRiskAssess/hra_wcvi',
    'hra_preprocessor': 'HabitatRiskAssess/hra_pre_wcvi',
    'hydropower_water_yield': 'Hydropower/annual_water_yield_willamette',
    'marine_water_quality_biophysical': 'MarineWaterQuality/marine_floathomes_wcvi',
    'ndr': 'Base_Data/Freshwater/ndr_n_p_willamette',
    'pollination': 'Pollination/pollination_willamette',
    'recreation': 'recreation/recreation_andros',
    'routedem': 'Base_Data/Freshwater/routedem_willamette',
    'scenario_gen_proximity': 'scenario_proximity/scenario_proximity_amazonia',
    'scenario_generator': 'ScenarioGenerator/scenario_generator_demo',
    'scenic_quality': 'ScenicQuality/wind_turbines_wcvi',
    'sdr': 'Base_Data/Freshwater/sdr_willamette',
    'seasonal_water_yield': 'seasonal_water_yield/swy_willamette',
    'wind_energy': 'WindEnergy/new_england'
}

# Extra fisheries args
# Extra wave energy args

def extract_parameters(iui_config_path, relative_to):
    # arguments for all InVEST model UIs are in a flat dictionary.
    return_args = {}

    def _recurse(args_param):
        if isinstance(args_param, dict):
            default_value = None
            try:
                default_value = args_param['defaultValue']
            except KeyError:
                try:
                    if args_param['type'].lower() == 'container':
                        default_value = False
                except KeyError:
                    pass
            finally:
                if default_value is not None and 'args_id' in args_param:
                    if not isinstance(default_value, bool):
                        for type_ in (float, int):
                            try:
                                int_value = int(default_value)
                                float_value = float(default_value)
                                if int_value == float_value:
                                    default_value = int_value
                                else:
                                    default_value = float_value
                            except (TypeError, ValueError):
                                pass

                    try:
                        possible_path = os.path.join(
                            DATA_DIR, default_value.replace('../', ''))
                    except AttributeError:
                        # When we try to call 'replace', on an int or float.
                        possible_path = None
                    if (isinstance(default_value, basestring) and
                            os.path.exists(possible_path) and
                            len(default_value) > 0):
                        # it's a path!
                        return_value = os.path.relpath(
                            path=possible_path,
                            start=relative_to)
                    else:
                        return_value = default_value

                    return_args[args_param['args_id']] = return_value

            if 'elements' in args_param:
                _recurse(args_param['elements'])
        elif isinstance(args_param, list):
            for config in args_param:
                _recurse(config)
        else:
            return

    json_config = json.load(open(iui_config_path))
    _recurse(json_config)
    return json_config['modelName'], return_args


def main(userargs=None):
    if not userargs:
        userargs = sys.argv[1:]

    parser = argparse.ArgumentParser(description=(
        'Create a json parameter set from an IUI configuration file'),
        prog=os.path.basename(__file__))
    parser.add_argument('--convert-all', action='store_true',
                        help='Run for every known IUI file.')
    parser.add_argument('iui_config', nargs='?', default=[],
                        help='The IUI configuration file to use')
    parser.add_argument('datastack_path', nargs='?', default=[],
                        help='Where to save the parameter set')

    args = parser.parse_args(userargs)

    # Convert params to lists if needed.
    # There's probably a better argparse way to do this, but I can't seem to
    # make it happen immediately and this works just fine.
    for attrname in ('iui_config', 'datastack_path'):
        if not isinstance(getattr(args, attrname), list):
            setattr(args, attrname, [getattr(args, attrname)])

    if args.convert_all:
        for model_key, datastack_name in IUI_SCENARIOS.iteritems():
            print model_key, datastack_name
            iui_json_path = os.path.join('src', 'natcap', 'invest', 'iui',
                                         '%s.json' % model_key)
            args.iui_config.append(iui_json_path)
            assert os.path.exists(iui_json_path), 'File not found: %s' % iui_json_path
            datastack_path = os.path.join(DATA_DIR, datastack_name + '.invs.json')
            args.datastack_path.append(datastack_path)

    for iui_config, datastack_path in itertools.izip(args.iui_config,
                                                     args.datastack_path):

        modelname, new_params = extract_parameters(
            iui_config_path=iui_config,
            relative_to=os.path.dirname(os.path.abspath(datastack_path)))
        try:
            del new_params['workspace_dir']
        except KeyError:
            pass

        datastack.write_parameter_set(filepath=datastack_path,
                                       args=new_params,
                                       name=modelname,
                                       relative=True)


if __name__ == '__main__':
    main(sys.argv[1:])
