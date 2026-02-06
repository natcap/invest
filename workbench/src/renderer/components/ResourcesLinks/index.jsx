import React from 'react';

import PropTypes from 'prop-types';
import { MdOpenInNew } from 'react-icons/md';
import { useTranslation } from 'react-i18next';

import { openLinkInBrowser } from '../../utils';
import { ipcMainChannels } from '../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;

const FORUM_ROOT = 'https://community.naturalcapitalalliance.org';

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
  urban_nature_access: 'urban-nature-access',
  wave_energy: 'wave-energy',
  wind_energy: 'wind-energy',
};

/** Render model-relevant links to the User's Guide and Forum.
 *
 * This should be a link to the model's User's Guide chapter and
 * and a link to list of topics with the model's tag on the forum,
 * e.g. https://community.naturalcapitalalliance.org/tag/carbon
 */
export default function ResourcesTab(props) {
  const { docs, isCoreModel, modelID } = props;

  let forumURL = FORUM_ROOT;
  const tagName = FORUM_TAGS[modelID];
  if (tagName) {
    forumURL = `${FORUM_ROOT}/tag/${tagName}`;
  }

  const { t } = useTranslation();

  const userGuideURL = (
    isCoreModel
    ? `${window.Workbench.USERGUIDE_PATH}/${window.Workbench.LANGUAGE}/${docs}`
    : docs
  );
  const userGuideDisplayText = isCoreModel ? "User's Guide" : "Plugin Documentation";
  const userGuideAddlInfo = isCoreModel ? '(opens in new window)' : '(opens in web browser)';
  const userGuideAriaLabel = `${userGuideDisplayText} ${userGuideAddlInfo}`;

  /**
   * Open the target href in an electron window.
   */
  const handleUGClick = (event) => {
    event.preventDefault();
    if (isCoreModel) {
      ipcRenderer.send(
        ipcMainChannels.OPEN_LOCAL_HTML, event.currentTarget.href
      );
    } else {
      ipcRenderer.send(
        ipcMainChannels.OPEN_EXTERNAL_URL, event.currentTarget.href
      );
    }
  }

  return (
    <React.Fragment>
      {
        userGuideURL
        &&
        <a
          href={userGuideURL}
          title={userGuideURL}
          aria-label={t(userGuideAriaLabel)}
          onClick={handleUGClick}
        >
          <MdOpenInNew className="mr-1" />
          {t(userGuideDisplayText)}
        </a>
      }
      <a
        href={forumURL}
        title={forumURL}
        aria-label={t('Frequently Asked Questions (opens in web browser)')}
        onClick={openLinkInBrowser}
      >
        <MdOpenInNew className="mr-1" />
        {t('Frequently Asked Questions')}
      </a>
    </React.Fragment>
  );
}

ResourcesTab.propTypes = {
  modelID: PropTypes.string,
  isCoreModel: PropTypes.bool.isRequired,
  docs: PropTypes.string,
};
ResourcesTab.defaultProps = {
  modelID: undefined,
  docs: '',
};
