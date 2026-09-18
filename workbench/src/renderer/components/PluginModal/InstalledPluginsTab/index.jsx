import React from 'react';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Spinner from 'react-bootstrap/Spinner';
import { useTranslation } from 'react-i18next';
import { IconContext } from "react-icons";
import { BsCheckCircle } from "react-icons/bs";

import { ipcMainChannels } from '../../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;

export default function InstalledPluginsTab(props) {
  const {
    plugins,
    removePlugin,
    uninstallLoading,   // (str) ID of plugin being uninstalled
    uninstallErr,       // (str) ID of plugin with an error
    uninstallErrMsg,    // (str) error message
    removalSuccess,     // (bool)
  } = props;

  const { t } = useTranslation();

  let removePluginTab;
  if (uninstallErr) {
    removePluginTab = (
      <>
        <h5>{t('Error removing plugin:')}</h5>
        <div className="plugin-error plugin-install-remove-error">{uninstallErr}</div>
        <Button
          onClick={() => ipcRenderer.send(
            ipcMainChannels.SHOW_ITEM_IN_FOLDER,
            window.Workbench.ELECTRON_LOG_PATH,
          )}
        >
          {t('Find workbench logs')}
        </Button>
      </>
    );
  }

  return (
    <>
      <div>
        <h5 id="installed-plugin-list-title" className="mb-3">{t('Installed Plugins')}</h5>
        {removalSuccess && (
          <>
            <div aria-live="polite" className="pt-3 pb-3 plugin-success-message">
              <IconContext.Provider value={{ className: 'react-icons react-icons-white' }}>
                <BsCheckCircle />
              </IconContext.Provider>
              <span>{t('Plugin successfully removed!')}</span>
            </div>
          </>
        )}
        {Object.keys(plugins).length
          ? (
            Object.keys(plugins).map((pluginID =>
              <InstalledPluginDetailItem
                key={pluginID}
                pluginID={pluginID}
                pluginDetails={plugins[pluginID]}
                removePlugin={removePlugin}
                uninstallLoading={uninstallLoading === pluginID}
                uninstallErr={uninstallErr === pluginID}
                uninstallErrMsg={uninstallErrMsg}
                uninstallDisabled={uninstallLoading && uninstallLoading !== pluginID}
              />
            ))
          )
          : (
            <h6>{t('No plugins are currently installed.')}</h6>
          )
        }
      </div>
    </>
  );
}

function InstalledPluginDetailItem(props) {
  const {
    pluginID,
    pluginDetails,
    removePlugin,
    uninstallLoading,
    uninstallErr,
    uninstallErrMsg,
    uninstallDisabled,
  } = props;

  const { t } = useTranslation();

  const handleRemovePluginClick = () => {
      removePlugin(pluginID);
  };

  return (
    <Row className="pt-2 pb-2 installed-plugin-row">
      <Col sm={9}>
        <h6>{pluginDetails.modelTitle} ({pluginDetails.version})</h6>
        <ul className="list-unstyled plugin-small-text">
          {pluginDetails.sourceType && (
            <li>
              <b>{t('Installed via: ')}</b>
              {pluginDetails.sourceType}
            </li>
          )}
          <li>
            <b>{t('Source: ')}</b>
            {pluginDetails.source}
          </li>
        </ul>
        {uninstallErr &&
          (
            <>
              <h5>{t('Error removing plugin:')}</h5>
              <div className="plugin-error plugin-install-remove-error">{uninstallErrMsg}</div>
              <Button
                onClick={() => ipcRenderer.send(
                  ipcMainChannels.SHOW_ITEM_IN_FOLDER,
                  window.Workbench.ELECTRON_LOG_PATH,
                )}
              >
                {t('Find workbench logs')}
              </Button>
            </>
          )
        }
      </Col>
      <Col sm={3}>
        <Button
          disabled={uninstallLoading || uninstallDisabled}
          onClick={handleRemovePluginClick}
        >
          {uninstallLoading
            ? (
              <div className="adding-button">
                <Spinner animation="border" role="status" size="sm" className="plugin-spinner">
                  <span className="visually-hidden">{t('Removing...')}</span>
                </Spinner>
                {t('Removing...')}
              </div>
            )
            : t('Uninstall')
          }
        </Button>
      </Col>
    </Row>
  );
}