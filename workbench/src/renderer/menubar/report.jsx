import React from 'react';
import ReactDom from 'react-dom';
import {createRoot} from 'react-dom/client';
import { Translation } from 'react-i18next';

import i18n from '../i18n/i18n';
import {
  handleClickExternalURL,
  handleClickFindLogfiles
} from './handlers';
import investLogo from '../static/invest-logo.png';
import natcapLogo from '../static/NatCapLogo.jpg';

await i18n.changeLanguage(window.Workbench.LANGUAGE);
const root = createRoot(document.getElementById('content'));
root.render(
  <Translation>
    {(t, { i18n }) => (
      <React.Fragment>
        <h1 class="header">
          {t('Please help us by reporting problems.')}
        </h1>
        <p>
          <strong>{t('If the problem is related to a specific InVEST model, ')}</strong>
          {t('please see the guidelines here for reporting problems: ')}
          <a
            href="https://community.naturalcapitalalliance.org/t/guidelines-for-posting-software-support-questions/24"
            onClick={handleClickExternalURL}
          >
            {t('Guidelines for posting software support questions')}
          </a>
        </p>
        <p>
          <strong>{t('If the problem is related to this User Interface, ')}</strong>
          {t('rather than with a specific InVEST model,')}
        </p>
        <ol>
          <li>{t('Consider taking a screenshot of the problem.')}</li>
          <li>
            {t('Find the log files using the button below. ' +
                'There may be multiple files with a ".log" extension; ' +
                'please include them all.')}
            <button type="button" onClick={handleClickFindLogfiles}>
              {t('Find My Logs')}
            </button>
          </li>
          <li>
            {t('Create a post on our forum and upload these items, along ' +
                'with a brief description of the problem. ')}
            <a
              href="https://community.naturalcapitalalliance.org/"
              onClick={handleClickExternalURL}
            >
              https://community.naturalcapitalalliance.org
            </a>
          </li>
        </ol>
        <div class="footer">
          <img
            src={investLogo}
            width="143"
            height="119"
            alt="invest logo"
          />
          <img
            src={natcapLogo}
            width="119"
            height="119"
            alt="Natural Capital Alliance logo"
          />
        </div>
      </React.Fragment>
    )}
  </Translation>
);
