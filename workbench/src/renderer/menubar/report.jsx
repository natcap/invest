import React from 'react';
import ReactDom from 'react-dom';

import i18n from '../../shared/i18n';
import { Translation } from 'react-i18next';
import {
  handleClickExternalURL,
  handleClickFindLogfiles
} from './handlers';
import { getSettingsValue } from '../components/SettingsModal/SettingsStorage';


await getSettingsValue('language')
  .then((ll) => i18n.changeLanguage(ll))
  .then(() => {
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
              src="/src/renderer/static/invest-logo.png"
              width="143"
              height="125"
            />
            <img
              src="/src/renderer/static/NatCapLogo.jpg"
              width="143"
              height="125"
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
  }
);

