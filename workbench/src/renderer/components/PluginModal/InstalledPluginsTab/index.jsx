import React from 'react';

import Button from 'react-bootstrap/Button';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Spinner from 'react-bootstrap/Spinner';
import { useTranslation } from 'react-i18next';
import { BsCheckCircle } from "react-icons/bs";

import { ipcMainChannels } from '../../../../main/ipcMainChannels';
import { handleClickFindLogfiles } from '../../../menubar/handlers';

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

  return (
    <>
      <div>
        <h5 id="installed-plugin-list-title" className="mb-3">{t('Installed Plugins')}</h5>
        {removalSuccess && (
          <>
            <div aria-live="polite" className="pt-3 pb-3 plugin-success-message">
              <BsCheckCircle className="plugin-modal-icons plugin-modal-icons-white" />
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
        <dl className="plugin-dl">
          {pluginDetails.sourceType && (
            <>
              <dt className="bold-text">{t('Installed via: ')}</dt>
              <dd>{pluginDetails.sourceType}</dd>
            </>
          )}
          <dt className="bold-text">{t('Source: ')}</dt>
          <dd>{pluginDetails.source}</dd>
        </dl>
        {uninstallErr &&
          (
            <>
              <h5>{t('Error removing plugin:')}</h5>
              <div className="plugin-error plugin-install-remove-error">{uninstallErrMsg}</div>
              <Button
                onClick={handleClickFindLogfiles}
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
                <Spinner animation="border" role="status" size="sm" className="plugin-spinner" />
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
