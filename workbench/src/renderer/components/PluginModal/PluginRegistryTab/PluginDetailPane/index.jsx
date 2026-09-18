import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';

import { useTranslation } from 'react-i18next';

import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import Spinner from 'react-bootstrap/Spinner';
import Table from 'react-bootstrap/Table';
import { IconContext } from "react-icons";
import { BsExclamationCircle } from "react-icons/bs";
import { BsCheckCircle } from "react-icons/bs";

import { openLinkInBrowser } from '../../../../utils';
import { ipcMainChannels } from '../../../../../main/ipcMainChannels';

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
    "preprocessing": "Preprocessing",
    "postprocessing": "Postprocessing",
    "workflow": "Workflow",
    "invest_model_variant": "InVEST Model Variant",
    "new_model": "New Model",
    "other": "Other"
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
      {(installStatus === "anotherVersionInstalled") &&
        <div className="plugin-version-note">
          <IconContext.Provider value={{ className: 'react-icons' }}>
            <BsExclamationCircle />
          </IconContext.Provider>
          <span>
            <b>{t('Note: ')}</b>
            {t('A different version of this plugin is already installed.')}
          </span>
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
          onClick={() => ipcRenderer.send(
            ipcMainChannels.SHOW_ITEM_IN_FOLDER,
            window.Workbench.ELECTRON_LOG_PATH,
          )}
        >
          {t('Find workbench logs')}
        </Button>
      </>
    );
  };

  let alreadyInstalledPane = (
    <>
      <div className="pt-3 pb-3 plugin-version-note">
        <IconContext.Provider value={{ className: 'react-icons' }}>
          <BsCheckCircle />
        </IconContext.Provider>
        <span>{t('This plugin is installed!')}</span>
      </div>
    </>
  );

  return (
    <>
      <div className="plugin-pane">
        <h5>{plugin.plugin_name}</h5>
        <p className="plugin-small-text">
          {plugin.description}
        </p>
        <Table borderless size="sm" className="plugin-small-text plugin-table">
          <tbody>
            <tr>
              <td className="text-end"><b>Downloads:</b></td>
              <td>
                {plugin.download_count}
              </td>
            </tr>
            {(plugin.authors.length > 0) &&
            <tr>
              <td className="text-end"><b>Authors:</b></td>
              <td>
                {plugin.authors.join("; ")}
              </td>
            </tr>
            }
            {(plugin.maintainers.length > 0) &&
            <tr>
              <td className="text-end"><b>Maintainers:</b></td>
              <td>
                {plugin.maintainers.join("; ")}
              </td>
            </tr>
            }
            <tr>
              <td className="text-end"><b>Version:</b></td>
              <td>
                {plugin.version}, updated {plugin.date_last_updated}
              </td>
            </tr>
            <tr>
              <td className="text-end"><b>License:</b></td>
              <td>
                {plugin.license}
              </td>
            </tr>
            <tr>
              <td className="text-end"><b>More Info:</b></td>
              <td>
                <a
                  href={plugin.registry_url}
                  title={plugin.registry_url}
                  aria-label={t("View on Plugin Registry (opens in web browser)")}
                  onClick={openLinkInBrowser}
                >View on Registry</a> | <a
                  href={plugin.repository_url}
                  title={plugin.repository_url}
                  aria-label={t("Plugin source code (opens in web browser)")}
                  onClick={openLinkInBrowser}
                >Source Code</a> | <a
                  href={plugin.documentation_url}
                  title={plugin.documentation_url}
                  aria-label={t("Plugin documentation (opens in web browser)")}
                  onClick={openLinkInBrowser}
                >Documentation</a> | <a
                  href={plugin.issues_url}
                  title={plugin.issues_url}
                  aria-label={t("Plugin issue tracker (opens in web browser)")}
                  onClick={openLinkInBrowser}
                >Issue Tracker</a>
              </td>
            </tr>
            <tr>
              <td className="text-end"><b>Tags:</b></td>
              <td>
                {keywords}
              </td>
            </tr>
          </tbody>
        </Table>
      </div>
      <div className="install-pane registry-install-form">
        {(installStatus === "thisVersionInstalled")
          ? alreadyInstalledPane
          : installPane
        }
      </div>
    </>
  );
}