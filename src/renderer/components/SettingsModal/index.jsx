import React from 'react';
import PropTypes from 'prop-types';

import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import { MdSettings } from 'react-icons/md';

import { getDefaultSettings } from './SettingsStorage';

/** Render a dialog with a form for configuring global invest settings */
export default class SettingsModal extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      show: false,
      nWorkersOptions: [],
      logLevelOptions: ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    };

    this.handleShow = this.handleShow.bind(this);
    this.handleClose = this.handleClose.bind(this);
    this.handleChange = this.handleChange.bind(this);
    this.handleReset = this.handleReset.bind(this);
    this.switchToDownloadModal = this.switchToDownloadModal.bind(this);
  }

  componentDidMount() {
    // const nCPU = ipcRenderer.invoke(ipcMainChannels.GET_N_CPUS, 'ping');
    const nCPU = 12;
    const nWorkersOptions = [];
    for (let i = -1; i <= nCPU; i++) {
      nWorkersOptions.push(i);
    }
    this.setState({
      nWorkersOptions: nWorkersOptions
    });
  }

  handleClose() {
    this.setState({
      show: false,
    });
  }

  handleShow() {
    this.setState({ show: true });
  }

  handleReset(event) {
    event.preventDefault();
    const resetSettings = getDefaultSettings();
    this.props.saveSettings(resetSettings);
  }

  handleChange(event) {
    const newSettings = { ...this.props.investSettings };
    const { name, value } = event.currentTarget;
    newSettings[name] = value;
    this.props.saveSettings(newSettings);
  }

  switchToDownloadModal() {
    this.props.showDownloadModal();
    this.handleClose();
  }

  render() {
    return (
      <React.Fragment>
        <Button
          aria-label="settings"
          className="settings-icon-btn"
          onClick={this.handleShow}
        >
          <MdSettings
            className="settings-icon"
          />
        </Button>

        <Modal show={this.state.show} onHide={this.handleClose}>
          <Modal.Header>
            <Modal.Title>InVEST Settings</Modal.Title>
            <Button
              variant="secondary"
              onClick={this.handleClose}
              className="float-right"
            >
              Cancel
            </Button>
          </Modal.Header>
          <Modal.Body>
            <Form.Group as={Row}>
              <Form.Label column sm="8" htmlFor="logging-select">Logging threshold</Form.Label>
              <Col sm="4">
                <Form.Control
                  id="logging-select"
                  as="select"
                  name="loggingLevel"
                  value={this.props.investSettings.loggingLevel}
                  onChange={this.handleChange}
                >
                  {this.state.logLevelOptions.map(
                    (opt) => <option value={opt} key={opt}>{opt}</option>
                  )}
                </Form.Control>
              </Col>
            </Form.Group>
            <Form.Group as={Row}>
              <Form.Label column sm="8" htmlFor="nworkers-select">
                Taskgraph n_workers parameter
                <br />
                (must be an integer &gt;= -1)
              </Form.Label>
              <Col sm="4">
                <Form.Control
                  id="nworkers-select"
                  as="select"
                  name="nWorkers"
                  type="text"
                  value={this.props.investSettings.nWorkers}
                  onChange={this.handleChange}
                >
                  {this.state.nWorkersOptions.map(
                    (opt) => <option value={opt} key={opt}>{opt}</option>
                  )}
                </Form.Control>
              </Col>
            </Form.Group>
            <Button
              variant="secondary"
              onClick={this.handleReset}
              type="button"
            >
              Reset to Defaults
            </Button>
            <hr />
            <Button
              variant="primary"
              onClick={this.switchToDownloadModal}
            >
              Download Sample Data
            </Button>
            <hr />
            <Button
              variant="secondary"
              onClick={this.props.clearJobsStorage}
            >
              Clear Recent Jobs
            </Button>
            <div>(no invest workspaces will be deleted)</div>
          </Modal.Body>
        </Modal>
      </React.Fragment>
    );
  }
}

SettingsModal.propTypes = {
  saveSettings: PropTypes.func,
  investSettings: PropTypes.shape({
    nWorkers: PropTypes.string,
    loggingLevel: PropTypes.string,
    sampleDataDir: PropTypes.string,
  }),
  showDownloadModal: PropTypes.func,
};
