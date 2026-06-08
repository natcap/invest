import React from 'react';
import ReactDom from 'react-dom';
import {createRoot} from 'react-dom/client';
import { Translation } from 'react-i18next';

import i18n from '../i18n/i18n';

import { handleClickExternalURL } from './handlers';
import { ipcMainChannels } from '../../main/ipcMainChannels';
import investLogo from '../static/invest-logo.png';

const { ipcRenderer } = window.Workbench.electron;

async function getInvestVersion() {
  const investVersion = await ipcRenderer.invoke(ipcMainChannels.INVEST_VERSION);
  return investVersion;
}

await i18n.changeLanguage(window.Workbench.LANGUAGE);
const investVersion = await getInvestVersion();

const ariaLabelSuffix = '(opens in web browser)';
const linkDefs = [
  {
    name: 'InVEST User Guide',
    href: 'http://releases.naturalcapitalproject.org/invest-userguide/latest/en/index.html',
  },
  {
    name: 'InVEST Data Collection Notice',
    href: 'https://naturalcapitalalliance.stanford.edu/software/invest/invest-downloads-data#invest-data-collection-notice',
  },
  {
    name: 'InVEST Attribution Guidelines',
    href: 'https://naturalcapitalalliance.stanford.edu/software/invest/invest-downloads-data#invest-attribution-guidelines',
  },
  {
    name: 'InVEST Trademark and Logo Use Policy',
    href: 'https://naturalcapitalalliance.stanford.edu/software/invest/invest-trademark-and-logo-use-policy',
  },
  {
    name: 'InVEST License',
    href: 'https://github.com/natcap/invest/blob/main/LICENSE.txt',
  },
  {
    name: 'InVEST on GitHub',
    href: 'https://github.com/natcap/invest',
  },
  {
    name: 'InVEST Developer Documentation',
    href: `https://invest.readthedocs.io/en/${investVersion}/`,
  },
  {
    name: 'Natural Capital Alliance',
    href: 'https://naturalcapitalalliance.stanford.edu/',
  },
];
const linkListItems = linkDefs.map(({name, href}) => (
  <Translation key={name}>
    {(t, { i18n }) => (
      <li>
        <a
          href={href}
          title={href}
          aria-label={`${t(name)} ${t(ariaLabelSuffix)}`}
          onClick={handleClickExternalURL}
        >{t(name)}</a>
      </li>
    )}
  </Translation>
));

const root = createRoot(document.getElementById('content'));
root.render(
  <Translation>
    {(t, { i18n }) => (
      <React.Fragment>
        <h1 className="visually-hidden">About InVEST</h1>
        <div className="header">
          <img
            src={investLogo}
            width="191"
            height="159"
            alt="InVEST logo"
          />
          <div className="version-and-copyright">
            <span>
              {t('Version:')} <span className="version-string">{ investVersion }</span>
            </span>
            <span>
              {t('Copyright 2026, Natural Capital Alliance')}
            </span>
          </div>
        </div>
        <section>
          <p>{t('InVEST® is free and open-source software used to map and value the goods and services from nature that sustain and fulfill human life.')}</p>
        </section>
        <section>
          <h2 className="section-heading" id="link-list-heading">{t('Resources')}</h2>
          <ul role="list" className="links" aria-describedby="link-list-heading">
            {linkListItems}
          </ul>
        </section>
        <section>
          <h2 className="section-heading">{t('Data Collection')}</h2>
          <p>{t('The Natural Capital Alliance software team collects certain non-personal data each time an InVEST model or InVEST plugin is run via the InVEST Workbench. These data help inform future work on InVEST and support our mission to maintain InVEST as free and open-source software.')}</p>
        </section>
      </React.Fragment>
    )}
  </Translation>
);
