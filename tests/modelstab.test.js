import React from 'react';
// import ReactTestUtils from 'react-dom/test-utils';
import renderer from 'react-test-renderer';
import { ModelsTab } from '../src/components/ModelsTab';

const investList = {"Carbon Storage and Sequestration": {"internal_name": "carbon", "aliases": []}, "Coastal Blue Carbon": {"internal_name": "coastal_blue_carbon", "aliases": ["cbc"]}, "Coastal Blue Carbon: Preprocessor": {"internal_name": "coastal_blue_carbon_preprocessor", "aliases": ["cbc_pre"]}, "Coastal Vulnerability": {"internal_name": "coastal_vulnerability", "aliases": ["cv"]}, "Crop Production: Percentile Model": {"internal_name": "crop_production_percentile", "aliases": ["cpp"]}, "Crop Production: Regression Model": {"internal_name": "crop_production_regression", "aliases": ["cpr"]}, "DelineateIt": {"internal_name": "delineateit", "aliases": []}, "Marine Finfish Aquaculture Production": {"internal_name": "finfish_aquaculture", "aliases": []}, "Fisheries": {"internal_name": "fisheries", "aliases": []}, "Fisheries: Habitat Scenario Tool": {"internal_name": "fisheries_hst", "aliases": []}, "Forest Carbon Edge Effect": {"internal_name": "forest_carbon_edge_effect", "aliases": ["fc"]}, "GLOBIO": {"internal_name": "globio", "aliases": []}, "Habitat Quality": {"internal_name": "habitat_quality", "aliases": ["hq"]}, "Habitat Risk Assessment": {"internal_name": "habitat_risk_assessment", "aliases": ["hra"]}, "Annual Water Yield": {"internal_name": "hydropower_water_yield", "aliases": ["hwy"]}, "Nutrient Delivery Ratio": {"internal_name": "ndr", "aliases": []}, "Pollinator Abundance: Crop Pollination": {"internal_name": "pollination", "aliases": []}, "Visitation: Recreation and Tourism": {"internal_name": "recreation", "aliases": []}, "RouteDEM": {"internal_name": "routedem", "aliases": []}, "Scenario Generator: Proximity Based": {"internal_name": "scenario_generator_proximity", "aliases": ["sgp"]}, "Unobstructed Views: Scenic Quality Provision": {"internal_name": "scenic_quality", "aliases": ["sq"]}, "Sediment Delivery Ratio": {"internal_name": "sdr", "aliases": []}, "Seasonal Water Yield": {"internal_name": "seasonal_water_yield", "aliases": ["swy"]}, "Offshore Wind Energy Production": {"internal_name": "wind_energy", "aliases": []}, "Wave Energy Production": {"internal_name": "wave_energy", "aliases": []}, "Urban Flood Risk Mitigation": {"internal_name": "urban_flood_risk_mitigation", "aliases": ["ufrm"]}}

test('ModelsTab snapshot', () => {
  const component = renderer.create(
    <ModelsTab
      loadModelSpec={() => {}}
    />
  );
  component.setState({models:investList});
  let tree = component.toJSON();
  expect(tree).toMatchSnapshot();
})
