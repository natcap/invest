import React from 'react';

import PropTypes from 'prop-types';
import { MdOpenInNew } from 'react-icons/md';

import UG_ROOT from '../../userguideURL';
import { ipcMainChannels } from '../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;

const FORUM_ROOT = 'https://community.naturalcapitalproject.org';

// map model names to forum tags:
const FORUM_TAGS = {
  annual_water_yield: 'annual-water-yield',
  carbon: 'carbon',
  coastal_vulnerability: 'coastal-vulnerability',
  coastal_blue_carbon: 'blue-carbon',
  coastal_blue_carbon_preprocessor: 'blue-carbon',
  crop_production_percentile: 'crop-production',
  crop_production_regression: 'crop-production',
  delineateit: 'delineateit',
  forest_carbon_edge_effect: 'carbon-edge-effects',
  globio: 'globio',
  habitat_quality: 'habitat-quality',
  habitat_risk_assessment: 'hra',
  ndr: 'ndr',
  pollination: 'pollination',
  recreation: 'recreation',
  routedem: 'routedem',
  scenario_generator_proximity: 'scenario-generator',
  scenic_quality: 'scenic-quality',
  sdr: 'sdr',
  seasonal_water_yield: 'seasonal-water-yield',
  stormwater: 'urban-stormwater',
  urban_cooling_model: 'urban-cooling',
  urban_flood_risk_mitigation: 'urban-flood',
  wave_energy: 'wave-energy',
  wind_energy: 'wind-energy',
};

/**
 * Open the target href in the default web browser.
 */
function handleClick(event) {
  event.preventDefault();
  ipcRenderer.send(
    ipcMainChannels.OPEN_EXTERNAL_URL, event.currentTarget.href
  );
}

/** Render model-relevant links to the User's Guide and Forum.
 *
 * This should be a link to the model's User's Guide chapter and
 * and a link to list of topics with the model's tag on the forum,
 * e.g. https://community.naturalcapitalproject.org/tag/carbon
 */
export default function ResourcesTab(props) {
  const { docs, moduleName } = props;

  let forumURL = FORUM_ROOT;
  const tagName = FORUM_TAGS[moduleName];
  if (tagName) {
    forumURL = `${FORUM_ROOT}/tag/${tagName}`;
  }

  const userGuideURL = `${UG_ROOT}/${docs}#data-needs`;

  return (
    <React.Fragment>
      <a
        href={userGuideURL}
        title={userGuideURL}
        aria-label="go to user's guide in web browser"
        onClick={handleClick}
      >
        <MdOpenInNew className="mr-1" />
        {_("User's Guide")}
      </a>
      <a
        href={forumURL}
        title={forumURL}
        aria-label="go to frequently asked questions in web browser"
        onClick={handleClick}
      >
        <MdOpenInNew className="mr-1" />
        {_("Frequently Asked Questions")}
      </a>
    </React.Fragment>
  );
}

ResourcesTab.propTypes = {
  moduleName: PropTypes.string,
  docs: PropTypes.string,
};
ResourcesTab.defaultProps = {
  moduleName: undefined,
  docs: '',
};
