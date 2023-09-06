import React from 'react';
import ReactDom from 'react-dom';
import { Translation } from 'react-i18next';

import i18n from '../i18n/i18n';
import {
  handleClickExternalURL,
  handleClickFindLogfiles
} from './handlers';
import investLogo from '../static/invest-logo.png';
import natcapLogo from '../static/NatCapLogo.jpg';

await i18n.changeLanguage(window.Workbench.LANGUAGE);
ReactDom.render(
  <Translation>
    {(t, { i18n }) => (
      <React.Fragment>
        <p id="header">
          {t('Please help us by reporting problems.')}
        </p>
        <p>
          <b>{t('If the problem is related to a specific InVEST model, ')}</b>
          {t('please see the guidelines here for reporting problems: ')}
          <a
            href="https://community.naturalcapitalproject.org/t/guidelines-for-posting-software-support-questions/24"
          >
            {t('Guidelines for posting software support questions')}
          </a>
        </p>
        <p>
          <b>{t('If the problem is related to this User Interface, ')}</b>
          {t('rather than with a specific InVEST model,')}
          <ol>
            <li>{t('Consider taking a screenshot of the problem.')}</li>
            <li>
              {t('Find the log files using the button below. ' +
                 'There may be multiple files with a ".log" extension; ' +
                 'please include them all.')}
              <button type="button">
                {t('Find My Logs')}
              </button>
            </li>
            <li>
              {t('Create a post on our forum and upload these items, along ' +
                 'with a brief description of the problem.')}
              <a
                href="https://community.naturalcapitalproject.org/"
              >
                https://community.naturalcapitalproject.org
              </a>
            </li>
          </ol>
        </p>
        <div id="footer">
          <img
            src={investLogo}
            width="143"
            height="119"
            alt="invest logo"
          />
          <img
            src={natcapLogo}
            width="143"
            height="119"
            alt="Natural Capital Project logo"
          />
        </div>
      </React.Fragment>
    )}
  </Translation>,
  document.getElementById('content')
);
document.querySelector('button').addEventListener('click', handleClickFindLogfiles);
document.querySelectorAll('a').forEach(
  (element) => {
    element.addEventListener('click', handleClickExternalURL);
  }
);
