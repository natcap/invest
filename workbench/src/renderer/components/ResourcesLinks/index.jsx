import React from 'react';

import PropTypes from 'prop-types';
import { MdOpenInNew } from 'react-icons/md';
import { useTranslation } from 'react-i18next';

import { ipcMainChannels } from '../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;

const FORUM_ROOT = 'https://community.naturalcapitalproject.org';

/**
 * Open the target href in the default web browser.
 */
function handleForumClick(event) {
  event.preventDefault();
  ipcRenderer.send(
    ipcMainChannels.OPEN_EXTERNAL_URL, event.currentTarget.href
  );
}

/**
 * Open the target href in an electron window.
 */
function handleUGClick(event) {
  event.preventDefault();
  ipcRenderer.send(
    ipcMainChannels.OPEN_LOCAL_HTML, event.currentTarget.href
  );
}

/** Render model-relevant links to the User's Guide and Forum.
 *
 * This should be a link to the model's User's Guide chapter and
 * and a link to list of topics with the model's tag on the forum,
 * e.g. https://community.naturalcapitalproject.org/tag/carbon
 */
export default function ResourcesTab(props) {
  const { docs, moduleName, forumTagName } = props;

  let forumURL = FORUM_ROOT;
  if (forumTagName) {
    forumURL = `${FORUM_ROOT}/tag/${forumTagName}`;
  }

  const { t, i18n } = useTranslation();
  const userGuideURL = `${window.Workbench.USERGUIDE_PATH}/${window.Workbench.LANGUAGE}/${docs}`;

  return (
    <React.Fragment>
      <a
        href={userGuideURL}
        title={userGuideURL}
        aria-label="go to user's guide in web browser"
        onClick={handleUGClick}
      >
        <MdOpenInNew className="mr-1" />
        {t("User's Guide")}
      </a>
      <a
        href={forumURL}
        title={forumURL}
        aria-label="go to frequently asked questions in web browser"
        onClick={handleForumClick}
      >
        <MdOpenInNew className="mr-1" />
        {t("Frequently Asked Questions")}
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
