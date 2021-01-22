import { getSpec } from '../src/server_requests';
const uiSpec = require('../src/ui_config');


describe('Check validity of the UI spec file', async () => {

    const models = [
        "carbon", 
        "coastal_blue_carbon",           
        "coastal_blue_carbon_preprocessor",
        "coastal_vulnerability",
        "crop_production_percentile",     
        "crop_production_regression",      
        "delineateit",          
        "finfish_aquaculture",         
        "fisheries",               
        "fisheries_hst",              
        "forest_carbon_edge_effect",        
        "globio",                           
        "habitat_quality",                 
        "habitat_risk_assessment",          
        "hydropower_water_yield",           
        "ndr",                              
        "pollination",                      
        "recreation",                       
        "routedem",                         
        "scenario_generator_proximity",     
        "scenic_quality",                   
        "sdr",                             
        "seasonal_water_yield",             
        "urban_cooling_model",              
        "urban_flood_risk_mitigation",      
        "wave_energy",                      
        "wind_energy"                      
    ]

    test('Each arg in the UI spec exists in the args spec', async () => {
        await models.forEach(async model => {
            const argsSpec = await getSpec(model);
            for (const property in uiSpec[argsSpec.model_name]) {
                if (property === 'order') {
                    // 'order' is a 2D array of arg names
                    uiSpec[argsSpec.model_name].order.flat().forEach(arg => {
                        expect(argsSpec[arg]).toBeDefined();
                    });
                } else {
                    // for other properties, each key is an arg
                    uiSpec[argsSpec.model_name][property].forEach(arg => {
                        expect(argsSpec[arg]).toBeDefined();
                    });
                }
            }          
        });
    });
    test('No duplicate args in the order', async () => {
        await models.forEach(async mode => {
            const argsSpec = await getSpec(model);
            const order_array = uiSpec[argsSpec.model_name].order.flat();
            const order_set = new Set(order_array);
            expect(order_array.length).toEqual(order_set.size);
        });
    });
});