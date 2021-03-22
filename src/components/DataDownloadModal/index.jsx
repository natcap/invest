import path from 'path';
import React from 'react';
import PropTypes from 'prop-types';
import { remote, ipcRenderer } from 'electron';

import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';

import SaveFileButton from '../SaveFileButton';
import pkg from '../../../package.json';

/** Render a dialog with a form for configuring global invest settings */
export default class DataDownloadModal extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      all: true,
    };

    this.handleClose = this.handleClose.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  componentDidMount() {
    const prefix = encodeURIComponent(`invest/${pkg.invest.version}/data`);
    const queryURL = `https://www.googleapis.com/storage/v1/b/${pkg.invest.bucket}/o?prefix=${prefix}`;
    fetch(queryURL)
      .then((response) => {
        if (response.status === 200) {
          return response.json();
        }
        console.log(response.status);
      })
      .then((data) => {
        console.log(data);
      });
  }

  handleClose() {
    // storing something sends the signal that the user declined
    // and doesn't need to be asked again on app startup.
    this.props.storeDownloadDir('');
  }

  async handleSubmit(event) {
    event.preventDefault();
    // need two things here
    // 1. list of files to download
    // 2. downloadDir to save them
    // downloads in background? or keep modal open?
    // progress is important.
    const allDataURL = path.join(
      this.props.releaseDataURL, 'InVEST_3.9.0.post235+g296690d7_sample_data.zip'
    );
    ipcRenderer.send('download-url', allDataURL);
  }

  render() {
    return (
      <Modal show={this.props.show} onHide={this.handleClose}>
        <Form>
          <Modal.Header>
            <Modal.Title>Download InVEST sample data</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <Form.Group>
              <Form.Label>
                Download to:
              </Form.Label>
              <Form.Control
                name="downloadDir"
                type="text"
                value={''} // empty string is handled better than `undefined`
                onChange={this.handleDirChange}
              />
            </Form.Group>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={this.handleClose}>
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={this.handleSubmit}
            >
              Download All
            </Button>
          </Modal.Footer>
        </Form>
      </Modal>
    );
  }
}

// DataDownladModal.propTypes = {
//   saveSettings: PropTypes.func,
//   investSettings: PropTypes.shape({
//     nWorkers: PropTypes.string,
//     loggingLevel: PropTypes.string,
//   })
// };
