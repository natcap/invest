import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form';
import Modal from 'react-bootstrap/Modal';
import Spinner from 'react-bootstrap/Spinner';
import { useTranslation } from 'react-i18next';
import { MdClose } from 'react-icons/md';

import { ipcMainChannels } from '../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;

export default function PluginModal(props) {
  const {
    updateInvestList,
    closeInvestModel,
    openJobs,
    show,
    closeModal,
    openModal,
  } = props;
  const [url, setURL] = useState('');
  const [revision, setRevision] = useState('');
  const [path, setPath] = useState('');
  const [installErr, setInstallErr] = useState('');
  const [uninstallErr, setUninstallErr] = useState('');
  const [pluginToRemove, setPluginToRemove] = useState('');
  const [installLoading, setInstallLoading] = useState(false);
  const [uninstallLoading, setUninstallLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [needsMSVC, setNeedsMSVC] = useState(false);
  const [plugins, setPlugins] = useState({});
  const [installFrom, setInstallFrom] = useState('url');
  const [userAcknowledgment, setUserAcknowledgment] = useState(false);
  const [userAcknowledgmentError, setUserAcknowledgmentError] = useState(false);
  const [pluginSourceMissingError, setPluginSourceMissingError] = useState(false);

  const handleModalClose = () => {
    setURL('');
    setRevision('');
    setInstallErr('');
    setUninstallErr('');
    clearFormErrors();
    closeModal();
  };

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

  const handleAddPluginClick = () => {
    clearFormErrors();
    if (validateAddPluginForm()) {
      addPlugin();
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

  const addPlugin = () => {
    setInstallLoading(true);
    ipcRenderer.invoke(
      ipcMainChannels.ADD_PLUGIN,
      installFrom === 'url' ? url : undefined, // url
      installFrom === 'url' ? revision : undefined, // revision
      installFrom === 'path' ? path : undefined // path
    ).then(() => {
      setInstallLoading(false);
      updateInvestList();
      // clear the input fields
      setURL('');
      setRevision('');
      setPath('');
    }).catch((err) => {
      setInstallErr(err.toString());
    });
  };

  const removePlugin = () => {
    setUninstallLoading(true);
    openJobs.forEach((job, tabID) => {
      if (job.modelID === pluginToRemove) {
        closeInvestModel(tabID);
      }
    });
    ipcRenderer.invoke(
      ipcMainChannels.REMOVE_PLUGIN, pluginToRemove
    ).then(() => {
      updateInvestList();
      setUninstallLoading(false);
    }).catch((err) => {
      setUninstallErr(err.toString());
    });
  };

  const downloadMSVC = () => {
    closeModal();
    ipcRenderer.invoke(ipcMainChannels.DOWNLOAD_MSVC).then(
      openModal()
    );
  };

  useEffect(() => {
    ipcRenderer.on('plugin-install-status', (msg) => { setStatusMessage(msg); });
    if (show) {
      if (window.Workbench.OS === 'win32') {
        ipcRenderer.invoke(ipcMainChannels.HAS_MSVC).then((hasMSVC) => {
          setNeedsMSVC(!hasMSVC);
        });
      }
    }
    return () => { ipcRenderer.removeAllListeners('plugin-install-status'); };
  }, [show]);

  useEffect(() => {
    ipcRenderer.invoke(ipcMainChannels.GET_SETTING, 'plugins').then(
      (data) => {
        if (data) {
          setPlugins(data);
          setPluginToRemove(Object.keys(data)[0]);
        }
      }
    );
  }, [installLoading, uninstallLoading]);

  const { t } = useTranslation();

  let pluginFields;
  if (installFrom === 'url') {
    pluginFields = (
      <Form.Row>
        <Form.Group as={Col} xs={7}>
          <Form.Label htmlFor="url">{t('Git URL')}</Form.Label>
          <Form.Control
            id="url"
            type="text"
            placeholder="https://github.com/owner/repo.git"
            value={url}
            onChange={(event) => setURL(event.currentTarget.value)}
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
          {
            pluginSourceMissingError
            &&
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
      </Form.Row>
    );
  } else {
    pluginFields = (
      <Form.Group>
        <Form.Label htmlFor="path">{t('Local absolute path')}</Form.Label>
        <Form.Control
          id="path"
          type="text"
          placeholder={window.Workbench.OS === 'darwin'
            ? '/Users/username/path/to/plugin/'
            : 'C:\\Documents\\path\\to\\plugin\\'}
          value={path}
          onChange={(event) => setPath(event.currentTarget.value)}
          aria-describedby={pluginSourceMissingError ? 'path-error' : ''}
        />
        {
          pluginSourceMissingError
          &&
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

  let modalBody = (
    <Modal.Body>
      <Form aria-labelledby="add-plugin-form-title">
        <Form.Group>
          <h5 id="add-plugin-form-title" className="mb-3">{t('Add a plugin')}</h5>
          <Form.Group>
            <Form.Label htmlFor="installFrom">{t('Install from')}</Form.Label>
            <Form.Control
              id="installFrom"
              as="select"
              onChange={(event) => setInstallFrom(event.target.value)}
              className="w-auto"
            >
              <option value="url">{t('git URL')}</option>
              <option value="path">{t('local path')}</option>
            </Form.Control>
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
          {
            userAcknowledgmentError
            &&
            <Form.Text
              as="span"
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
            {
              installLoading ? (
                <div className="adding-button">
                  <Spinner animation="border" role="status" size="sm" className="plugin-spinner">
                    <span className="sr-only">{t('Adding plugin')}</span>
                  </Spinner>
                  {t(statusMessage)}
                </div>
              ) : t('Add')
            }
          </Button>
          <Form.Text
            as="span"
            muted
            id="plugin-installation-duration-notice"
            className="plugin-form-text"
          >
            {t('This may take several minutes.')}
          </Form.Text>
        </Form.Group>
      </Form>
      <hr />
      <Form aria-labelledby="remove-plugin-form-title">
        <Form.Group>
          <h5 id="remove-plugin-form-title" className="mb-3">{t('Remove a plugin')}</h5>
          <Form.Label htmlFor="selectPluginToRemove">{t('Plugin name')}</Form.Label>
          <Form.Control
            id="selectPluginToRemove"
            as="select"
            value={pluginToRemove}
            onChange={(event) => setPluginToRemove(event.currentTarget.value)}
          >
            {
              Object.keys(plugins).map(
                (pluginID) => (
                  <option
                    value={pluginID}
                    key={pluginID}
                  >
                    {`${plugins[pluginID].modelTitle} (${plugins[pluginID].version})`}
                  </option>
                )
              )
            }
          </Form.Control>
          <Button
            disabled={uninstallLoading || !Object.keys(plugins).length}
            className="mt-3"
            onClick={removePlugin}
          >
            {
              uninstallLoading ? (
                <div className="adding-button">
                  <Spinner animation="border" role="status" size="sm" className="plugin-spinner">
                    <span className="sr-only">{t('Removing...')}</span>
                  </Spinner>
                  {t('Removing...')}
                </div>
              ) : t('Remove')
            }
          </Button>
        </Form.Group>
      </Form>
    </Modal.Body>
  );
  if (installErr) {
    modalBody = (
      <Modal.Body>
        <h5>{t('Error installing plugin:')}</h5>
        <div className="plugin-error plugin-install-remove-error">{installErr}</div>
        <Button
          onClick={() => ipcRenderer.send(
            ipcMainChannels.SHOW_ITEM_IN_FOLDER,
            window.Workbench.ELECTRON_LOG_PATH,
          )}
        >
          {t('Find workbench logs')}
        </Button>
      </Modal.Body>
    );
  } else if (uninstallErr) {
    modalBody = (
      <Modal.Body>
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
      </Modal.Body>
    );
  }
  if (needsMSVC) {
    modalBody = (
      <Modal.Body>
        <h5>
          {t('Microsoft Visual C++ Redistributable must be installed!')}
        </h5>

        {t('Plugin features require the ')}
        <a href="https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist">
          {t('Microsoft Visual C++ Redistributable')}
        </a>
        {t('. You must download and install the redistributable before continuing.')}

        <Button
          className="mt-3"
          onClick={downloadMSVC}
        >
          {t('Continue to download and install')}
        </Button>
      </Modal.Body>
    );
  }

  return (
    <Modal show={show} onHide={handleModalClose} contentClassName="plugin-modal">
      <Modal.Header>
        <Modal.Title>{t('Manage plugins')}</Modal.Title>
        <Button
          variant="secondary-outline"
          onClick={handleModalClose}
          className="float-right"
          aria-label={t('Close modal')}
        >
          <MdClose />
        </Button>
      </Modal.Header>
      {modalBody}
    </Modal>
  );
}

PluginModal.propTypes = {
  show: PropTypes.bool.isRequired,
  closeModal: PropTypes.func.isRequired,
  openModal: PropTypes.func.isRequired,
  updateInvestList: PropTypes.func.isRequired,
  closeInvestModel: PropTypes.func.isRequired,
  openJobs: PropTypes.shape({
    modelID: PropTypes.string,
  }).isRequired,
};
