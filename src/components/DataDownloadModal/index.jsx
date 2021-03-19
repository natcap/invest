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

function validateDir(dir) {
  return [true, ''];
}

/** Render a dialog with a form for configuring global invest settings */
export default class DataDownloadModal extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      defaultTargetPath: '',
      // show: true,
      // downloadDir: null,
      // dirIsValid: null,
      // validationMessage: null,
    };

    // this.handleShow = this.handleShow.bind(this);
    this.handleClose = this.handleClose.bind(this);
    this.handleDirChange = this.handleDirChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  componentDidMount() {
    ipcRenderer.on('variable-reply', (event, arg) => {
      this.setState({
        defaultTargetPath: arg.userDataPath,
      });
    });
    ipcRenderer.send('variable-request', 'ping');
  }

  handleClose() {
    // this.setState({
    //   show: false,
    // });
    // storing something sends the signal that the user declined
    // and doesn't need to be asked again on app startup.
    this.props.storeDownloadDir('');
  }

  // handleShow() {
  //   this.setState({
  //     show: true,
  //   });
  // }

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
    // const data = await remote.dialog.showSaveDialog(
    //   { defaultPath: this.state.defaultTargetPath }
    // );
    // if (data.filePath) {
      // this.props.storeDownloadDir(path.dirname(data.filePath));
      // ipcRenderer.send('download-url', allDataURL);
    // }
  }

  handleDirChange(event) {
    // const { value } = event.target;
    // const [isValid, message] = validateDir(value);
    // this.setState({
    //   downloadDir: value,
    //   dirIsValid: isValid,
    //   validationMessage: message,
    // });
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
