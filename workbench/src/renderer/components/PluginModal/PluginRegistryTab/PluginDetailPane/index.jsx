import React, { useEffect, useState } from 'react';

import { useTranslation } from 'react-i18next';

import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Spinner from 'react-bootstrap/Spinner';
import Tooltip from 'react-bootstrap/Tooltip';
import { BsExclamationCircle } from "react-icons/bs";
import { BsCheckCircle } from "react-icons/bs";
import { MdOpenInNew } from "react-icons/md";

import { openLinkInBrowser } from '../../../../utils';
import { ipcMainChannels } from '../../../../../main/ipcMainChannels';
import { handleClickFindLogfiles } from '../../../../menubar/handlers';

import InstallButton from '../../InstallButton';
import NeedsMSVC from '../../NeedsMSVC';
import {
  thisVersionInstalled,
  anotherVersionInstalled,
  notInstalled,
  opTypeInstall,
  opStatusLoading,
  opStatusSuccess,
  opStatusFailure,
  sourceTypeRegistry,
} from '../../constants';

export default function PluginDetailPane(props) {
  const {
    pluginID,
    plugin,
    priorInstallationStatus,
    addPlugin,
    addRemoveState,
    statusMessage,
    needsMSVC,
    downloadMSVC,
  } = props;
  const [userAcknowledgment, setUserAcknowledgment] = useState(false);
  const [userAcknowledgmentError, setUserAcknowledgmentError] = useState(false);
  const [installStatus, setInstallStatus] = useState("None");

  useEffect(() => {
    if (
      addRemoveState.opPluginID === pluginID &&
      addRemoveState.opType === opTypeInstall
    ) {
      setInstallStatus(addRemoveState.opStatus);
    } else {
      setInstallStatus('None');
    }
  }, [addRemoveState]);

  const pluginTypes = {
    preprocessing: "Preprocessing",
    postprocessing: "Postprocessing",
    workflow: "Workflow",
    invest_model_variant: "InVEST Model Variant",
    new_model: "New Model",
    other: "Other"
  }

  const clearFormErrors = () => {
    setUserAcknowledgmentError(false);
  };

  useEffect(() => {
      clearFormErrors();
  }, []);

  useEffect(() => {
    if (userAcknowledgment) {
      setUserAcknowledgmentError(false);
    }
  }, [userAcknowledgment]);

  const handleAddPluginClick = () => {
    if (addRemoveState.opStatus !== opStatusLoading) {
      clearFormErrors();
      if (validateAddPluginForm()) {
        addPlugin(
          pluginID,
          plugin.repository_url, // url
          plugin.version,        // revision
          undefined,             // path, used for manual install
          sourceTypeRegistry     // sourceType
        );
      }
    }
  };

  const validateAddPluginForm = () => {
    let formValid = true;
    if (!userAcknowledgment) {
      formValid = false;
      setUserAcknowledgmentError(true);
    }
    return formValid;
  };

  const pluginType = pluginTypes[plugin.plugin_type];
  const keywords = [pluginType].concat(plugin.keywords).join(", ");

  const { t } = useTranslation();

  let installPane = (
    <>
      {(priorInstallationStatus === anotherVersionInstalled) &&
        <div>
          <BsExclamationCircle className="plugin-modal-icons" />
            <span className="bold-text">{t('Note: ')}</span>
            {t('A different version of this plugin is already installed.')}
        </div>
      }
      <Form aria-labelledby="add-plugin-form-title">
        <Form.Group>
          <Form.Group>
            <Form.Text
              as="span"
              id="plugin-installation-risk-statement"
              className="plugin-form-text"
            >
              {t('As with any third-party software, installing a plugin for use with InVEST '
                + 'may pose a risk to your data, computer, and/or network. Please make sure '
                + 'you trust the authors of the plugin you are installing. If you are '
                + 'installing from a git URL, you are encouraged to review the source code, '
                + 'which can change over time.')}
            </Form.Text>
          </Form.Group>
          <Form.Group>
            <Form.Check
              id={`${pluginID}-user-acknowledgment-checkbox`}
              label={t('I acknowledge and accept the risks associated with installing this plugin.')}
              value={userAcknowledgment}
              onChange={(event) => setUserAcknowledgment(event.target.checked)}
              aria-describedby={`plugin-installation-risk-statement${userAcknowledgmentError ? ' user-acknowledgment-error' : ''}`}
            />
          </Form.Group>
          {userAcknowledgmentError &&
            <Form.Text
              as="p"
              id={`${pluginID}-user-acknowledgment-error`}
              className="plugin-error plugin-user-acknowledgment-error mb-1"
            >
              {t('Error: Before installing a plugin, you must agree to the terms by selecting the checkbox.')}
            </Form.Text>
          }
          <InstallButton
            handleAddPluginClick={handleAddPluginClick}
            pluginID={pluginID}
            installLoading={installStatus === opStatusLoading}
            installDisabled={addRemoveState.opStatus === opStatusLoading}
            statusMessage={statusMessage}
          />
        </Form.Group>
      </Form>
    </>
  );

  if (needsMSVC) {
    installPane = (
      <NeedsMSVC
        downloadMSVC={downloadMSVC}
      />
    );
  } else if (installStatus === opStatusFailure) {
    installPane = (
      <>
        <h5>{t('Error installing plugin:')}</h5>
        <div className="plugin-error plugin-install-remove-error">
          {addRemoveState.opErrorMsg}
        </div>
        <Button
          onClick={handleClickFindLogfiles}
        >
          {t('Find workbench logs')}
        </Button>
      </>
    );
  };

  let alreadyInstalledPane = (
    <>
      <div className="pt-3 pb-3 plugin-version-note">
        <BsCheckCircle className="plugin-modal-icons" />
        <span>{t('This plugin is installed!')}</span>
      </div>
    </>
  );

  return (
    <>
      <div className="plugin-pane">
        <h5>{plugin.plugin_name}</h5>
        <p>
          {plugin.description}
        </p>
        <dl className="plugin-dl">
          <dt>{t("Downloads:")}</dt>
          <dd>{plugin.download_count}</dd>
          {(plugin.authors.length > 0) &&
          <>
            <dt>{t("Authors:")}</dt>
            <dd>{plugin.authors.join("; ")}</dd>
          </>
          }
          {(plugin.maintainers.length > 0) &&
          <>
            <dt>{t("Maintainers:")}</dt>
            <dd>{plugin.maintainers.join("; ")}</dd>
          </>
          }
          <dt>{t("Version:")}</dt>
          <dd>{plugin.version}, updated {plugin.date_last_updated}</dd>
          <dt>{t("License:")}</dt>
          <dd>{plugin.license}</dd>
          <dt>{t("More Info:")}</dt>
          <dd>
            <a
              href={plugin.registry_url}
              title={plugin.registry_url}
              onClick={openLinkInBrowser}
            >
              {t("View on Registry ")}
              <MdOpenInNew
                aria-label={t("(opens in web browser)")}
              />
            </a> | <a
              href={plugin.repository_url}
              title={plugin.repository_url}
              onClick={openLinkInBrowser}
            >{
              t("Source Code ")}
              <MdOpenInNew
                aria-label={t("(opens in web browser)")}
              />
            </a> | <a
              href={plugin.documentation_url}
              title={plugin.documentation_url}
              onClick={openLinkInBrowser}
            >
              {t("Documentation ")}
              <MdOpenInNew
                aria-label={t("(opens in web browser)")}
              />
            </a> | <a
              href={plugin.issues_url}
              title={plugin.issues_url}
              onClick={openLinkInBrowser}
            >
              {t("Issue Tracker ")}
              <MdOpenInNew
                aria-label={t("(opens in web browser)")}
              />
            </a>
          </dd>
          <dt>{t("Tags:")}</dt>
          <dd>{keywords}</dd>
        </dl>
      </div>
      <div className="install-pane registry-install-form">
        {(priorInstallationStatus === thisVersionInstalled)
          ? alreadyInstalledPane
          : installPane
        }
      </div>
    </>
  );
}
