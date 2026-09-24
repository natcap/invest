import React, { useEffect, useState } from 'react';

import { useTranslation } from 'react-i18next';

import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import Spinner from 'react-bootstrap/Spinner';
import Table from 'react-bootstrap/Table';
import { BsExclamationCircle } from "react-icons/bs";
import { BsCheckCircle } from "react-icons/bs";
import { MdOpenInNew } from "react-icons/md";

import { openLinkInBrowser } from '../../../../utils';
import { ipcMainChannels } from '../../../../../main/ipcMainChannels';
import { handleClickFindLogfiles } from '../../../../menubar/handlers';
import {
  thisVersionInstalled,
  anotherVersionInstalled,
  notInstalled
} from '../../PluginRegistryTab';
import { sourceTypeRegistry } from '../../../PluginModal';

const { ipcRenderer } = window.Workbench.electron;

export default function PluginDetailPane(props) {
  const {
    pluginID,
    plugin,
    installStatus,
    addPlugin,
    installLoading,     // true if parent installLoading val === pluginID
    installErr,         // true if parent installErr val === pluginID
    installErrMsg,
    installSuccess,     // true if parent installSuccess val === pluginID
    installDisabled,    // true if parent installLoading val && val !== pluginID
    statusMessage,
    needsMSVC,
    downloadMSVC,
  } = props;
  const [userAcknowledgment, setUserAcknowledgment] = useState(false);
  const [userAcknowledgmentError, setUserAcknowledgmentError] = useState(false);

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
    clearFormErrors();
    if (validateAddPluginForm()) {
      addPlugin(
        pluginID,
        plugin.repository_url, // url
        plugin.version,        // revision
        undefined,             // path, used for manual install
        'registry'             // sourceType
      );
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

  const handleDownloadMSVCClick = () => {
    downloadMSVC();
  };

  const pluginType = pluginTypes[plugin.plugin_type];
  const keywords = [pluginType].concat(plugin.keywords).join(", ");

  const { t } = useTranslation();

  let installPane = (
    <>
      {(installStatus === anotherVersionInstalled) &&
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
          <Button
            disabled={installLoading || installDisabled}
            onClick={handleAddPluginClick}
            aria-describedby="plugin-installation-duration-notice"
          >
            {installLoading
              ? (
                <div className="adding-button">
                  <Spinner animation="border" role="status" size="sm" className="plugin-spinner">
                    <span className="visually-hidden">{t('Adding plugin')}</span>
                  </Spinner>
                  {t(statusMessage)}
                </div>
              )
              : t('Install')
            }
          </Button>
          {installDisabled
            ? (
              <Form.Text
                as="span"
                muted
                id={`${pluginID}-plugin-installation-disabled-notice`}
                className="plugin-form-text"
              >
                {t('An installation is currently in progress. Please wait for it to complete '
                  + 'before installing another plugin.')}
              </Form.Text>
            )
            : (
              <Form.Text
                as="span"
                muted
                id={`${pluginID}-plugin-installation-duration-notice`}
                className="plugin-form-text"
              >
                {t('This may take several minutes.')}
              </Form.Text>
          )}
        </Form.Group>
      </Form>
    </>
  );

  if (needsMSVC) {
    installPane = (
      <>
        <h5>
          {t('Microsoft Visual C++ Redistributable must be installed!')}
        </h5>
        <p>
          {t('Plugin features require the ')}
          <a href="https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist">
            {t('Microsoft Visual C++ Redistributable')}
          </a>
          {t('. You must download and install the redistributable before you can install a plugin.')}
        </p>
        <Button
          className="mt-3"
          onClick={handleDownloadMSVCClick}
        >
          {t('Continue to download and install')}
        </Button>
      </>
    );
  } else if (installErr) {
    installPane = (
      <>
        <h5>{t('Error installing plugin:')}</h5>
        <div className="plugin-error plugin-install-remove-error">{installErrMsg}</div>
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
        {(installStatus === thisVersionInstalled)
          ? alreadyInstalledPane
          : installPane
        }
      </div>
    </>
  );
}
