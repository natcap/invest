import React from 'react';

import PropTypes from 'prop-types';
import { MdOpenInNew } from 'react-icons/md';
import { useTranslation } from 'react-i18next';

import { openLinkInBrowser } from '../../utils';
import { ipcMainChannels } from '../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;

const FORUM_ROOT = 'https://community.naturalcapitalalliance.org';

// map model names to forum tags
// forum tag paths include both slug and ID
const FORUM_TAGS = {
  annual_water_yield: 'annual-water-yield/19',
  carbon: 'carbon/1',
  coastal_vulnerability: 'coastal-vulnerability/8',
  coastal_blue_carbon: 'blue-carbon/3',
  coastal_blue_carbon_preprocessor: 'blue-carbon/3',
  crop_production_percentile: 'crop-production/15',
  crop_production_regression: 'crop-production/15',
  delineateit: 'delineateit/14',
  forest_carbon_edge_effect: 'carbon-edge-effects/13',
  habitat_quality: 'habitat-quality/4',
  habitat_risk_assessment: 'hra/9',
  ndr: 'ndr/7',
  pollination: 'pollination/18',
  recreation: 'recreation/10',
  routedem: 'routedem/38',
  scenario_generator_proximity: 'scenario-generator/25',
  scenic_quality: 'scenic-quality/30',
  sdr: 'sdr/2',
  seasonal_water_yield: 'seasonal-water-yield/5',
  stormwater: 'urban-stormwater/37',
  urban_cooling_model: 'urban-cooling/27',
  urban_flood_risk_mitigation: 'urban-flood/12',
  urban_mental_health: 'urban-mental-health', //TODO - add ID when forum tag is created
  urban_nature_access: 'urban-nature-access/41',
  wave_energy: 'wave-energy/21',
  wind_energy: 'wind-energy/23',
};

/** Render model-relevant links to the User's Guide and Forum.
 *
 * This should be a link to the model's User's Guide chapter and
 * and a link to list of topics with the model's tag on the forum,
 * e.g. https://community.naturalcapitalalliance.org/tag/carbon/1
 */
export default function ResourcesTab(props) {
  const { docs, isCoreModel, modelID } = props;

  let forumURL = FORUM_ROOT;
  const tagPath = FORUM_TAGS[modelID];
  if (tagPath) {
    forumURL = `${FORUM_ROOT}/tag/${tagPath}`;
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
