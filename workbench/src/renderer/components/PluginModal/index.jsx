import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';
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
  const [err, setErr] = useState(undefined);
  const [pluginToRemove, setPluginToRemove] = useState(undefined);
  const [loading, setLoading] = useState(false);
  const [plugins, setPlugins] = useState({});

  const handleModalClose = () => {
    setURL(undefined);
    setErr(false);
    setShowPluginModal(false);
  };
  const handleModalOpen = () => setShowPluginModal(true);
  const addPlugin = () => {
    setLoading(true);
    ipcRenderer.invoke(ipcMainChannels.ADD_PLUGIN, url).then((addPluginErr) => {
      setLoading(false);
      updateInvestList();
      if (addPluginErr) {
        setErr(addPluginErr);
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

  let modalBody = (
    <Modal.Body>
      <Form>
        <Form.Group className="mb-3">
          <Form.Label htmlFor="url">{t('Add a plugin')}</Form.Label>
          <Form.Control
            id="url"
            type="text"
            placeholder={t('Enter Git URL')}
            onChange={(event) => setURL(event.currentTarget.value)}
          />
          <Form.Text className="text-muted">
            {t('This may take several minutes')}
          </Form.Text>
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
          <Form.Label htmlFor="plugin-select">{t('Remove a plugin')}</Form.Label>
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
        <span>{t('Plugin installation failed')}</span>
        <br />
        <br />
        <span>{err.toString()}</span>
      </Modal.Body>
    );
  }

  return (
    <React.Fragment>
      <Button onClick={handleModalOpen} variant="outline-dark">
        {t('Manage plugins')}
      </Button>

      <Modal show={showPluginModal} onHide={handleModalClose}>
        <Modal.Header>
          <Modal.Title>{t('Manage plugins')}</Modal.Title>
          {loading && (
            <Spinner animation="border" role="status" className="m-2">
              <span className="sr-only">Loading...</span>
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
