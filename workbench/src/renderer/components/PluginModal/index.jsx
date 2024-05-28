import React, { useState } from 'react';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import Modal from 'react-bootstrap/Modal';
import Spinner from 'react-bootstrap/Spinner';
import { MdOutlineAdd } from 'react-icons/md';
import { useTranslation } from 'react-i18next';

import { ipcMainChannels } from '../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;

export default function PluginModal(props) {
  const { updateInvestList } = props;
  const [showAddPluginModal, setShowAddPluginModal] = useState(false);
  const [url, setURL] = useState(undefined);
  const [err, setErr] = useState(undefined);
  const [loading, setLoading] = useState(false);

  const handleModalClose = () => setShowAddPluginModal(false);
  const handleModalOpen = () => setShowAddPluginModal(true);
  const handleSubmit = () => {
    setLoading(true);
    ipcRenderer.invoke(ipcMainChannels.ADD_PLUGIN, url).then((addPluginErr) => {
      setLoading(false);
      updateInvestList();
      if (addPluginErr) {
        setErr(addPluginErr);
      } else {
        setShowAddPluginModal(false);
      }
    });
  };
  const handleChange = (event) => {
    setURL(event.currentTarget.value);
  };

  const { t } = useTranslation();

  let modalBody = (
    <Modal.Body>
      <Form>
        <Form.Group className="mb-3">
          <Form.Label htmlFor="url">Git URL</Form.Label>
          <Form.Control
            id="url"
            name="url"
            type="text"
            placeholder={t('Enter Git URL')}
            onChange={handleChange}
          />
        </Form.Group>

        <Button
          name="submit"
          onClick={handleSubmit}
        >
          {t('Add')}
        </Button>
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
      <Button onClick={handleModalOpen}>
        <MdOutlineAdd className="mr-1" />
        {t('Add a plugin')}
      </Button>

      <Modal show={showAddPluginModal} onHide={handleModalClose}>
        <Modal.Header>
          <Modal.Title>{t('Add a plugin')}</Modal.Title>
        </Modal.Header>
        {loading && (
          <Spinner animation="border" role="status">
            <span className="sr-only">Loading...</span>
          </Spinner>
        )}
        {modalBody}
      </Modal>
    </React.Fragment>
  );
}

PluginModal.propTypes = {
  updateInvestList: PropTypes.func.isRequired,
};
