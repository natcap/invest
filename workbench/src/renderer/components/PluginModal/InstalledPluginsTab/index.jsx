import React, { useEffect, useState } from 'react';

import Button from 'react-bootstrap/Button';
import Col from 'react-bootstrap/Col';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Row from 'react-bootstrap/Row';
import Spinner from 'react-bootstrap/Spinner';
import Tooltip from 'react-bootstrap/Tooltip';
import { useTranslation } from 'react-i18next';
import { BsCheckCircle } from "react-icons/bs";

import { ipcMainChannels } from '../../../../main/ipcMainChannels';
import { handleClickFindLogfiles } from '../../../menubar/handlers';

import {
  pluginUninstall,
  addRemoveLoading,
  addRemoveSuccess,
  addRemoveError,
} from '../../PluginModal';

export default function InstalledPluginsTab(props) {
  const {
    plugins,
    removePlugin,
    addRemoveState,
  } = props;

  const { t } = useTranslation();

  return (
    <>
      <div>
        <h5 id="installed-plugin-list-title" className="mb-3">{t('Installed Plugins')}</h5>
        {(
          addRemoveState.opType === pluginUninstall &&
          addRemoveState.opStatus === addRemoveSuccess
        ) && (
          <>
            <div aria-live="polite" className="pt-3 pb-3 plugin-success-message">
              <BsCheckCircle className="plugin-modal-icons plugin-modal-icons-white" />
              <span>{t('Plugin successfully uninstalled!')}</span>
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
                addRemoveState={addRemoveState}
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
    addRemoveState,
  } = props;
  const [uninstallStatus, setUninstallStatus] = useState('None');

  const { t } = useTranslation();

  useEffect(() => {
    if (
      addRemoveState.opPluginID === pluginID &&
      addRemoveState.opType === pluginUninstall
    ) {
      setUninstallStatus(addRemoveState.opStatus);
    } else {
      setUninstallStatus('None');
    }
  }, [addRemoveState]);

  const handleRemovePluginClick = () => {
      if (addRemoveState.opStatus !== addRemoveLoading) {
        removePlugin(pluginID);
      }
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
        {uninstallStatus === addRemoveError &&
          (
            <>
              <h5>{t('Error removing plugin:')}</h5>
              <div className="plugin-error plugin-install-remove-error">{addRemoveState.opErrorMsg}</div>
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
        <OverlayTrigger
          trigger={(addRemoveState.opStatus === addRemoveLoading) ? ['hover', 'focus'] : []}
          rootClose
          placement="top"
          overlay={
            <Tooltip>
              {t("An installation or uninstallation is in progress.")}
            </Tooltip>
          }
        >
          <Button
            className="plugin-submit-btn"
            aria-disabled={addRemoveState.opStatus === addRemoveLoading}
            onClick={handleRemovePluginClick}
          >
            {uninstallStatus === addRemoveLoading
              ? (
                <div className="adding-button">
                  <Spinner animation="border" role="status" size="sm" className="plugin-spinner" />
                  {t('Removing...')}
                </div>
              )
              : t('Uninstall')
            }
          </Button>
        </OverlayTrigger>
      </Col>
    </Row>
  );
}
