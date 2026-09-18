import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form';
import Row from 'react-bootstrap/Row';
import Spinner from 'react-bootstrap/Spinner';
import { useTranslation } from 'react-i18next';
import { IconContext } from "react-icons";
import { BsCheckCircle } from "react-icons/bs";
import { MdFolderOpen } from 'react-icons/md';

import { openLinkInBrowser } from '../../../utils';
import { ipcMainChannels } from '../../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;

export default function ManualInstallTab(props) {
  const {
    addPlugin,
    resetManualInstallFormStatus,
    installLoading,
    installErr,
    installErrMsg,
    installSuccess,
    statusMessage,
    needsMSVC,
    downloadMSVC,
    dragOverHandler,
    dragEnterHandler,
    dragLeavingHandler,
    selectDirectory,
    getDroppedFilePath,
    rejectDropHandler,
  } = props;
  const [url, setURL] = useState('');
  const [revision, setRevision] = useState('');
  const [path, setPath] = useState('');
  const [installFrom, setInstallFrom] = useState('url');

  const [userAcknowledgment, setUserAcknowledgment] = useState(false);
  const [userAcknowledgmentError, setUserAcknowledgmentError] = useState(false);
  const [pluginSourceMissingError, setPluginSourceMissingError] = useState(false);

  const manualInstallID = "manualInstall";
  const pluginDocsURL = "https://invest.readthedocs.io/en/latest/plugins.html";

  const clearFormErrors = () => {
    setUserAcknowledgmentError(false);
    setPluginSourceMissingError(false);
  };

  useEffect(() => {
    clearFormErrors();
  }, [installFrom]);

  useEffect(() => {
    if (pluginSourceMissingError) {
      setPluginSourceMissingError(false);
    }
  }, [url, path]);

  useEffect(() => {
    if (userAcknowledgment) {
      setUserAcknowledgmentError(false);
    }
  }, [userAcknowledgment]);

  useEffect(() => {
    if (installSuccess === manualInstallID) {
      setURL('');
      setRevision('');
      setPath('');
    }
  }, [installSuccess]);

  const handleAddPluginClick = () => {
    clearFormErrors();
    if (validateAddPluginForm()) {
      addPlugin(
        manualInstallID,
        installFrom === 'url' ? url : undefined,
        installFrom === 'url' ? revision : undefined,
        installFrom === 'path' ? path : undefined,
        installFrom === 'path' ? 'local_path' : 'git_url'
      );
    }
  };

  const validateAddPluginForm = () => {
    let formValid = true;
    if ((installFrom === 'url' && !url)
        || (installFrom === 'path' && !path)
    ) {
      formValid = false;
      setPluginSourceMissingError(true);
    }
    if (!userAcknowledgment) {
      formValid = false;
      setUserAcknowledgmentError(true);
    }
    return formValid;
  };

  const handleResetForm = () => {
    resetManualInstallFormStatus();
    setUserAcknowledgment(false);
  }

  const { t } = useTranslation();

  let pluginFields;
  if (installFrom === 'url') {
    pluginFields = (
      <Row>
        <Form.Group as={Col} xs={7}>
          <Form.Label htmlFor="url">{t('Git URL')}</Form.Label>
          <Form.Control
            id="url"
            type="text"
            placeholder="https://github.com/owner/repo.git"
            value={url}
            onChange={(event) => setURL(event.currentTarget.value)}
            onDragOver={rejectDropHandler}
            onDrop={rejectDropHandler}
            aria-describedby={`about-git-url${pluginSourceMissingError ? ' url-error' : ''}`}
          />
          <Form.Text
            as="span"
            muted
            id="about-git-url"
            className="plugin-form-text text-italic"
          >
            {t('Default branch used unless otherwise specified.')}
          </Form.Text>
          {pluginSourceMissingError &&
            <Form.Text
              as="span"
              id="url-error"
              className="plugin-error plugin-source-missing-error"
            >
              {t('Error: URL is required.')}
            </Form.Text>
          }
        </Form.Group>
        <Form.Group as={Col}>
          <Form.Label htmlFor="branch">{t('Branch, tag, or commit')}</Form.Label>
          <Form.Control
            id="branch"
            type="text"
            value={revision}
            onChange={(event) => setRevision(event.currentTarget.value)}
            aria-describedby="about-branch-tag-commit"
          />
          <Form.Text
            as="span"
            muted
            id="about-branch-tag-commit"
            className="plugin-form-text text-italic"
          >
            {t('Optional')}
          </Form.Text>
        </Form.Group>
      </Row>
    );
  } else {
    pluginFields = (
      <Form.Group>
        <Form.Label htmlFor="path">{t('Local absolute path')}</Form.Label>
        <div className="d-flex flex-nowrap w-100">
          <Form.Control
            id="path"
            type="text"
            placeholder={window.Workbench.OS === 'darwin'
              ? '/Users/username/path/to/plugin/'
              : 'C:\\Documents\\path\\to\\plugin\\'}
            value={path}
            onChange={(event) => setPath(event.currentTarget.value)}
            onDragOver={dragOverHandler}
            onDragEnter={dragEnterHandler}
            onDragLeave={dragLeavingHandler}
            onDrop={(event) => {
              const droppedPath = getDroppedFilePath(event);
              if (droppedPath) {
                setPath(droppedPath);
              }
            }}
            aria-describedby={pluginSourceMissingError ? 'path-error' : ''}
          />
          <Button
            aria-label="browse for plugin directory"
            className="browse-button ms-2"
            variant="outline-dark"
            onClick={async (event) => setPath(await selectDirectory(event) || path)}
          >
            <MdFolderOpen />
          </Button>
        </div>
        {pluginSourceMissingError &&
          <Form.Text
            as="span"
            id="path-error"
            className="plugin-error plugin-source-missing-error"
          >
            {t('Error: Path is required.')}
          </Form.Text>
        }
      </Form.Group>
    );
  }

  let manualInstallTab = (
    <>
      <div>
        <h5 id="add-plugin-form-title" className="mb-3">{t('Manually Install a Plugin')}</h5>
        <p>
          {t('For more information about creating a plugin, read our ')}
          <a
            href={pluginDocsURL}
            title={pluginDocsURL}
            aria-label={t("Plugins Developer's Guide (opens in web browser)")}
            onClick={openLinkInBrowser}
          >{t("Developer's Guide")}</a>.
        </p>
      </div>
      <hr />
      <Form aria-labelledby="add-plugin-form-title">
        <Form.Group>
          <Form.Label htmlFor="installFrom">{t('Install from')}</Form.Label>
          <Form.Check
            type="radio"
            id="installFromURL"
            name="installFrom"
            label={t('git URL')}
            checked={installFrom === "url"}
            onChange={(event) => setInstallFrom("url")}
          />
          <Form.Check
            type="radio"
            id="installFromLocal"
            name="installFrom"
            label={t('local path')}
            checked={installFrom === "path"}
            onChange={(event) => setInstallFrom("path")}
          />
        </Form.Group>
        {pluginFields}
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
            id="user-acknowledgment-checkbox"
            label={t('I acknowledge and accept the risks associated with installing this plugin.')}
            value={userAcknowledgment}
            onChange={(event) => setUserAcknowledgment(event.target.checked)}
            aria-describedby={`plugin-installation-risk-statement${userAcknowledgmentError ? ' user-acknowledgment-error' : ''}`}
          />
        </Form.Group>
        {userAcknowledgmentError &&
          <Form.Text
            as="p"
            id="user-acknowledgment-error"
            className="plugin-error plugin-user-acknowledgment-error"
          >
            {t('Error: Before installing a plugin, you must agree to the terms by selecting the checkbox.')}
          </Form.Text>
        }
        <Button
          disabled={installLoading}
          onClick={handleAddPluginClick}
          aria-describedby="plugin-installation-duration-notice"
        >
          {(installLoading === manualInstallID)
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
        {installLoading && installLoading !== manualInstallID
          ? (
            <Form.Text
              as="span"
              muted
              id={`plugin-installation-disabled-notice`}
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
              id={`plugin-installation-duration-notice`}
              className="plugin-form-text"
            >
              {t('This may take several minutes.')}
            </Form.Text>
        )}
      </Form>
      {(installSuccess === manualInstallID) &&
        <>
          <div aria-live="polite" className="mt-3 pt-3 pb-3 plugin-success-message">
            <IconContext.Provider value={{ className: 'react-icons react-icons-white' }}>
              <BsCheckCircle />
            </IconContext.Provider>
            <span>{t('Successfully installed plugin!')}</span>
          </div>
        </>
      }
    </>
  );
  if (installErr === manualInstallID) {
    manualInstallTab = (
      <>
        <h5>{t('Error installing plugin:')}</h5>
        <div className="plugin-error plugin-install-remove-error">{installErrMsg}</div>
        <Button
          className="me-2"
          onClick={handleResetForm}
        >
          {t('Return to form')}
        </Button>
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
  if (needsMSVC) {
    manualInstallTab = (
      <>
        <h5>
          {t('Microsoft Visual C++ Redistributable must be installed!')}
        </h5>
        <p>
          {t('Plugin features require the ')}
          <a href="https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist">
            {t('Microsoft Visual C++ Redistributable')}
          </a>
          {t('. You must download and install the redistributable before continuing.')}
        </p>
        <Button
          className="mt-3"
          onClick={downloadMSVC}
        >
          {t('Continue to download and install')}
        </Button>
      </>
    );
  }

  return (
    <>
      {manualInstallTab}
    </>
  );
}