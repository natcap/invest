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
      <>
        <Form.Row>
          <Form.Group as={Col} xs={7} className="mb-1">
            <Form.Label>Git URL</Form.Label>
            <Form.Control
              id="url"
              type="text"
              placeholder={t('https://github.com/foo/bar.git')}
              onChange={(event) => setURL(event.currentTarget.value)}
            />
          </Form.Group>
          <Form.Group as={Col} className="mb-1">
            <Form.Label>Branch, tag, or commit</Form.Label>
            <Form.Control
              id="branch"
              type="text"
              placeholder={t('default')}
              onChange={(event) => setRevision(event.currentTarget.value)}
            />
          </Form.Group>
        </Form.Row>
        <Form.Text className="text-muted mt-0">
          {t('Default branch is used unless otherwise specified')}
        </Form.Text>
      </>
    );
  } else {
    pluginFields = (
      <Form.Group className="px-0 mb-1">
        <Form.Label>Local path</Form.Label>
        <Form.Control
          id="path"
          type="text"
          onChange={(event) => setPath(event.currentTarget.value)}
        />
        <Form.Text className="text-muted mt-0">
          {t('Must be an absolute path')}
        </Form.Text>
      </Form.Group>
    );
  }

  let modalBody = (
    <Modal.Body>
      <Form>
        <Form.Group className="mb-3">
          <Form.Label className="col-form-label-lg" htmlFor="url">{t('Add a plugin')}</Form.Label>
          <Form.Group>
            <Form.Row>
              <Col>
                <Form.Control
                  as="select"
                  onChange={(event) => setInstallFrom(event.target.value)}
                  className="w-auto"
                >
                  <option value="URL">Install from git URL</option>
                  <option value="path">Install from local path</option>
                </Form.Control>
              </Col>
            </Form.Row>
          </Form.Group>
          {pluginFields}
          <Button
            disabled={loading}
            className="mt-2"
            onClick={addPlugin}
          >
            {t('Add')}
          </Button>
        </Form.Group>
        <hr />
        <Form.Group className="mb-3">
          <Form.Label className="col-form-label-lg" htmlFor="plugin-select">{t('Remove a plugin')}</Form.Label>
          <Form.Control
            id="plugin-select"
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
            className="mt-2"
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

      <Modal show={showPluginModal} onHide={handleModalClose} >
        <Modal.Header>
          <Modal.Title>{t('Manage plugins')}</Modal.Title>
          {loading && (
            <Spinner animation="border" role="status" className="m-2">
              <span className="sr-only">{t('Loading...')}</span>
              <Form.Text className="text-muted">
                {t('This may take several minutes')}
              </Form.Text>
            </Spinner>
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
