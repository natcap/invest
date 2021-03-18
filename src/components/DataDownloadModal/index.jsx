import React from 'react';
import PropTypes from 'prop-types';

import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';

function validateDir(dir) {
  return [true, ''];
}

/** Render a dialog with a form for configuring global invest settings */
export default class DataDownloadModal extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      // show: true,
      downloadDir: null,
      dirIsValid: null,
      validationMessage: null,
    };

    // this.handleShow = this.handleShow.bind(this);
    this.handleClose = this.handleClose.bind(this);
    this.handleDirChange = this.handleDirChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  componentDidUpdate(prevProps) {

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

  handleSubmit(event) {
    event.preventDefault();
    // need two things here
    // 1. list of files to download
    // 2. downloadDir to save them
    // downloadSampleData(fileList);
    // downloads in background? or keep modal open?
    // progress is important.
    this.props.storeDownloadDir(this.state.downloadDir);
    // this.setState({
    //   show: false,
    // });
  }

  handleDirChange(event) {
    const { value } = event.target;
    const [isValid, message] = validateDir(value);
    this.setState({
      downloadDir: value,
      dirIsValid: isValid,
      validationMessage: message,
    });
  }

  render() {
    const {
      downloadDir,
      dirIsValid,
      validationMessage,
      // show,
    } = this.state;

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
                name={'downloadDir'}
                type="text"
                value={downloadDir || ''} // empty string is handled better than `undefined`
                onChange={this.handleDirChange}
                isValid={dirIsValid}
                isInvalid={validationMessage}
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
              type="submit"
            >
              Download
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
