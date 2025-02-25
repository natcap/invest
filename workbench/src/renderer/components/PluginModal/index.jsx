import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form';
import Modal from 'react-bootstrap/Modal';
import Spinner from 'react-bootstrap/Spinner';
import { useTranslation } from 'react-i18next';

import { ipcMainChannels } from '../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;

export default function PluginModal(props) {
  const { updateInvestList, closeInvestModel, openJobs } = props;
  const [showPluginModal, setShowPluginModal] = useState(false);
  const [url, setURL] = useState('');
  const [revision, setRevision] = useState('');
  const [path, setPath] = useState('');
  const [installErr, setInstallErr] = useState('');
  const [uninstallErr, setUninstallErr] = useState('');
  const [pluginToRemove, setPluginToRemove] = useState('');
  const [installLoading, setInstallLoading] = useState(false);
  const [uninstallLoading, setUninstallLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [plugins, setPlugins] = useState({});
  const [installFrom, setInstallFrom] = useState('url');

  const handleModalClose = () => {
    setURL('');
    setRevision('');
    setInstallErr('');
    setUninstallErr('');
    setShowPluginModal(false);
  };
  const handleModalOpen = () => setShowPluginModal(true);

  const addPlugin = () => {
    setInstallLoading(true);
    ipcRenderer.on(`plugin-install-status`, (msg) => { setStatusMessage(msg); });
    ipcRenderer.invoke(
      ipcMainChannels.ADD_PLUGIN,
      installFrom === 'url' ? url : undefined, // url
      installFrom === 'url' ? revision : undefined, // revision
      installFrom === 'path' ? path : undefined // path
    ).then((addPluginErr) => {
      setInstallLoading(false);
      updateInvestList();
      if (addPluginErr) {
        setInstallErr(addPluginErr);
      } else {
        // clear the input fields
        setURL('');
        setRevision('');
        setPath('');
      }
    });
  };

  const removePlugin = () => {
    setUninstallLoading(true);
    Object.keys(openJobs).forEach((tabID) => {
      if (openJobs[tabID].modelID === pluginToRemove) {
        closeInvestModel(tabID);
      }
    });
    ipcRenderer.invoke(ipcMainChannels.REMOVE_PLUGIN, pluginToRemove).then((err) => {
      if (err) {
        setUninstallErr(err)
      } else {
        updateInvestList();
        setUninstallLoading(false);
      }
    });
  };

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
          />
          <Form.Text className="text-muted">
            <i>{t('Default branch used unless otherwise specified')}</i>
          </Form.Text>
        </Form.Group>
        <Form.Group as={Col}>
          <Form.Label htmlFor="branch">{t('Branch, tag, or commit')}</Form.Label>
          <Form.Control
            id="branch"
            type="text"
            value={revision}
            onChange={(event) => setRevision(event.currentTarget.value)}
          />
          <Form.Text className="text-muted">
            <i>{t('Optional')}</i>
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
        />
      </Form.Group>
    );
  }

  let modalBody = (
    <Modal.Body>
      <Form>
        <Form.Group>
          <h5 className="mb-3">{t('Add a plugin')}</h5>
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
          <Button
            disabled={installLoading}
            onClick={addPlugin}
          >
            {
              installLoading ? (
                <div className="adding-button">
                  <Spinner animation="border" role="status" size="sm" className="plugin-spinner">
                    <span className="sr-only">{t('Adding...')}</span>
                  </Spinner>
                  {t(statusMessage)}
                </div>
              ) : t('Add')
            }
          </Button>
          <Form.Text className="text-muted">
            {t('This may take several minutes')}
          </Form.Text>
        </Form.Group>
        <hr />
        <Form.Group>
          <h5 className="mb-3">{t('Remove a plugin')}</h5>
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
                    {plugins[pluginID].modelTitle}
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
        <div className="plugin-error">{installErr}</div>
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
        <div className="plugin-error">{uninstallErr}</div>
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

  return (
    <React.Fragment>
      <Button onClick={handleModalOpen} variant="outline-dark">
        {t('Manage plugins')}
      </Button>

      <Modal show={showPluginModal} onHide={handleModalClose} contentClassName="plugin-modal">
        <Modal.Header>
          <Modal.Title>{t('Manage plugins')}</Modal.Title>
        </Modal.Header>
        {modalBody}
      </Modal>
    </React.Fragment>
  );
}

PluginModal.propTypes = {
  updateInvestList: PropTypes.func.isRequired,
  closeInvestModel: PropTypes.func.isRequired,
  openJobs: PropTypes.shape({
    modelID: PropTypes.string,
  }).isRequired,
};
