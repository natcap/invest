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
  const { updateInvestList } = props;
  const [showPluginModal, setShowPluginModal] = useState(false);
  const [url, setURL] = useState(undefined);
  const [revision, setRevision] = useState(undefined);
  const [path, setPath] = useState(undefined);
  const [err, setErr] = useState(undefined);
  const [pluginToRemove, setPluginToRemove] = useState(undefined);
  const [loading, setLoading] = useState(false);
  const [plugins, setPlugins] = useState({});
  const [installFrom, setInstallFrom] = useState('url');

  const handleModalClose = () => {
    setURL(undefined);
    setRevision(undefined);
    setErr(false);
    setShowPluginModal(false);
  };
  const handleModalOpen = () => setShowPluginModal(true);

  const addPlugin = () => {
    setLoading(true);
    ipcRenderer.invoke(
      ipcMainChannels.ADD_PLUGIN,
      installFrom === 'url' ? url : undefined, // url
      installFrom === 'url' ? revision : undefined, // revision
      installFrom === 'path' ? path : undefined // path
    ).then((addPluginErr) => {
      setLoading(false);
      updateInvestList();
      if (addPluginErr) {
        setErr(true);
      } else {
        setShowPluginModal(false);
      }
    });
  };

  const removePlugin = () => {
    setLoading(true);
    ipcRenderer.invoke(ipcMainChannels.REMOVE_PLUGIN, pluginToRemove).then(() => {
      updateInvestList();
      setLoading(false);
      setShowPluginModal(false);
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
  }, [loading]);

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
            disabled={loading}
            onClick={addPlugin}
          >
            {t('Add')}
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
                    {plugins[pluginID].model_name}
                  </option>
                )
              )
            }
          </Form.Control>
          <Button
            disabled={loading || !Object.keys(plugins).length}
            className="mt-3"
            onClick={removePlugin}
          >
            {t('Remove')}
          </Button>
        </Form.Group>
      </Form>
    </Modal.Body>
  );
  if (err) {
    modalBody = (
      <Modal.Body>
        {t('Plugin installation failed. Check the workbench log for details.')}
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
          {loading && (
            <Form.Group>
              <Spinner animation="border" role="status" className="m-2">
                <span className="sr-only">{t('Loading...')}</span>
              </Spinner>
            </Form.Group>
          )}
        </Modal.Header>
        {modalBody}
      </Modal>
    </React.Fragment>
  );
}

PluginModal.propTypes = {
  updateInvestList: PropTypes.func.isRequired,
};
