import React from 'react';
import ReactDom from 'react-dom';
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
ReactDom.render(
  <Translation>
    {(t, { i18n }) => (
      <React.Fragment>
        <div id="header">
          <img
            src={investLogo}
            width="191"
            height="159"
            alt="InVEST logo"
          />
          <div id="invest-version">
            <p>{t('version:')}</p>
            <span id="version-string" />
          </div>
          <p id="invest-copyright">
            {t('Copyright 2023, The Natural Capital Project')}
          </p>
        </div>
        <br />
        <div id="links">
          <table>
            <tbody>
              <tr>
                <td>{t('Documentation')}</td>
                <td>
                  <a
                    href="http://releases.naturalcapitalproject.org/invest-userguide/latest/"
                  >
                    http://releases.naturalcapitalproject.org/invest-userguide/latest/
                  </a>
                </td>
              </tr>
              <tr>
                <td>{t('Homepage')}</td>
                <td>
                  <a
                    href="https://naturalcapitalproject.stanford.edu/"
                  >
                    https://naturalcapitalproject.stanford.edu/
                  </a>
                </td>
              </tr>
              <tr>
                <td>{t('Project page')}</td>
                <td>
                  <a
                    href="https://github.com/natcap/invest"
                  >
                    https://github.com/natcap/invest
                  </a>
                </td>
              </tr>
              <tr>
                <td>{t('License')}</td>
                <td>
                  <a
                    href="https://github.com/natcap/invest/blob/main/LICENSE.txt"
                  >
                    Apache 2.0
                  </a>
                </td>
              </tr>
              <tr>
                <td>{t('InVEST Trademark and Logo Use Policy')}</td>
                <td>
                  <a
                    href="https://naturalcapitalproject.stanford.edu/invest-trademark-and-logo-use-policy"
                  >
                    Trademark and Logo Policy
                  </a>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div id="licenses">
          <h4>{t('Open-Source Licenses:')}</h4>
          <table>
            <tbody>
              <tr>
                <td>PyInstaller</td>
                <td>GPL</td>
                <td>
                  <a
                    href="http://pyinstaller.org"
                  >
                    http://pyinstaller.org
                  </a>
                </td>
              </tr>
              <tr>
                <td>GDAL</td>
                <td>{t('MIT and others')}</td>
                <td>
                  <a
                    href="http://gdal.org"
                  >
                    http://gdal.org
                  </a>
                </td>
              </tr>
              <tr>
                <td>numpy</td>
                <td>BSD</td>
                <td>
                  <a
                    href="http://numpy.org"
                  >
                    http://numpy.org
                  </a>
                </td>
              </tr>
              <tr>
                <td>pygeoprocessing</td>
                <td>BSD</td>
                <td>
                  <a
                    href="https://github.com/natcap/pygeoprocessing"
                  >
                    https://github.com/natcap/pygeoprocessing
                  </a>
                </td>
              </tr>
              <tr>
                <td>rtree</td>
                <td>MIT</td>
                <td>
                  <a
                    href="https://github.com/Toblerity/rtree"
                  >
                    https://github.com/Toblerity/rtree
                  </a>
                </td>
              </tr>
              <tr>
                <td>scipy</td>
                <td>BSD</td>
                <td>
                  <a
                    href="http://www.scipy.org/"
                  >
                    http://www.scipy.org/
                  </a>
                </td>
              </tr>
              <tr>
                <td>shapely</td>
                <td>BSD</td>
                <td>
                  <a
                    href="http://github.com/Toblerity/Shapely"
                  >
                    http://github.com/Toblerity/Shapely
                  </a>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </React.Fragment>
    )}
  </Translation>,
  document.getElementById('content')
);
document.querySelectorAll('a').forEach(
  (element) => {
    element.addEventListener('click', handleClickExternalURL);
  }
);
const node = document.getElementById('version-string');
const text = document.createTextNode(investVersion);
node.appendChild(text);
