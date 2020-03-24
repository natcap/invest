import React from 'react';
import PropTypes from 'prop-types';
import Electron from 'electron';

const UG_ROOT = 'http://releases.naturalcapitalproject.org/invest-userguide/latest/'
const FORUM_ROOT = 'https://community.naturalcapitalproject.org/'

// map model names to forum tags:
const FORUM_TAGS = {
  sdr: 'sdr',
  ndr: 'ndr',
  habitat_quality: 'habitat-quality',
  seasonal_water_yield: 'seasonal-water-yield',
  carbon: 'carbon',
  hydropower_water_yield: 'annual-water-yield',
  habitat_risk_assessment: 'hra',
  recreation: 'recreation',
  coastal_vulnerability: 'coastal-vulnerability',
  coastal_blue_carbon: 'blue-carbon',
  crop_production_percentile: 'crop-production',
  crop_production_regression: 'crop-production',
  pollination: 'pollination',
  forest_carbon_edge_effect: 'carbon-edge-effects',
  delineateit: 'delineateit',
  fisheries: 'fisheries',
  fisheries_hst: 'fisheries',
  urban_flood_risk_mitigation: 'urban-flood',
  wind_energy: 'wind-energy',
  scenario_generator_proximity: 'scenario-generator',
  wave_energy: 'wave-energy',
}

function handleClick(event) {
  /** Open the target href in the default web browser */
  event.preventDefault();
  Electron.shell.openExternal(event.target.href);
}

export class ResourcesTab extends React.Component {
  /** Render model-relevant links to the User's Guide and Forum.
  *
  * This should be a link to the model's User's Guide chapter and
  * and a link to list of topics with the model's tag on the forum,
  * e.g. https://community.naturalcapitalproject.org/tag/carbon
  */

  constructor(props) {
    super(props);
  }

  render () {
    let userGuideURL;
    let forumURL;
    let name;

    if (this.props.docs && this.props.modelName) {
      userGuideURL = UG_ROOT + this.props.docs
      forumURL = FORUM_ROOT + 'tags/' + FORUM_TAGS[this.props.modelName]
      name = this.props.modelName
    } else {
      userGuideURL = UG_ROOT
      forumURL = FORUM_ROOT
      name = 'InVEST'
    }
    return(

        <div>
          <h2>
          <a href={userGuideURL} onClick={handleClick}>
            {"User's Guide: " + name}
          </a>
          </h2>
          <br></br>
          <h2>
          <a href={forumURL} onClick={handleClick}>
            {"FAQ: " + name}
          </a>
          </h2>
        </div>
      );
  }
}

ResourcesTab.propTypes = {
  modelName: PropTypes.string,
  docs: PropTypes.string
}