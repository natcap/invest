import React from 'react';
import PropTypes from 'prop-types';

import Accordion from 'react-bootstrap/Accordion';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import {
  MdSettings,
  MdClose,
} from 'react-icons/md';
import { BsChevronExpand } from 'react-icons/bs';

import { getDefaultSettings } from './SettingsStorage';

/** Render a dialog with a form for configuring global invest settings */
export default class SettingsModal extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      show: false,
      nWorkersOptions: null,
      logLevelOptions: ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    };

    this.handleShow = this.handleShow.bind(this);
    this.handleClose = this.handleClose.bind(this);
    this.handleChange = this.handleChange.bind(this);
    this.handleReset = this.handleReset.bind(this);
    this.switchToDownloadModal = this.switchToDownloadModal.bind(this);
  }

  async componentDidMount() {
    const nWorkersOptions = [];
    for (let i = -1; i <= this.props.nCPU; i++) {
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
              variant="secondary-outline"
              onClick={this.handleClose}
              className="float-right"
              aria-label="close settings"
            >
              <MdClose />
            </Button>
          </Modal.Header>
          <Modal.Body>
            <Form.Group as={Row}>
              <Form.Label column sm="7" htmlFor="logging-select">Logging threshold</Form.Label>
              <Col sm="5">
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
            {
              (this.state.nWorkersOptions)
                ? (
                  <Form.Group as={Row}>
                    <Col sm="7">
                      <Form.Label htmlFor="nworkers-select">
                        Taskgraph n_workers parameter
                      </Form.Label>
                    </Col>
                    <Col sm="5">
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
                    <Accordion>
                      <Accordion.Toggle
                        as={Button}
                        variant="secondary-outline"
                        eventKey="0"
                        className="pt-0"
                      >
                        <BsChevronExpand className="mx-1" />
                        <span className="small"><u>more info</u></span>
                      </Accordion.Toggle>
                      <Accordion.Collapse eventKey="0" className="pr-1">
                        <ul>
                          <li>-1: (recommended) synchronous mode</li>
                          <li>0: single process with threaded task management</li>
                          <li>
                            n: depending on the InVEST model, tasks may execute
                            in parallel using up to this many processes.
                          </li>
                        </ul>
                      </Accordion.Collapse>
                    </Accordion>
                  </Form.Group>
                )
                : <div />
            }
            <Row className="justify-content-end">
              <Col sm="5">
                <Button
                  variant="secondary"
                  onClick={this.handleReset}
                  type="button"
                  className="w-100"
                >
                  Reset to Defaults
                </Button>
              </Col>
            </Row>
            <hr />
            <Button
              variant="primary"
              onClick={this.switchToDownloadModal}
              className="w-50"
            >
              Download Sample Data
            </Button>
            <hr />
            <Button
              variant="secondary"
              onClick={this.props.clearJobsStorage}
              className="mr-2 w-50"
            >
              Clear Recent Jobs
            </Button>
            <span><small>no invest workspaces will be deleted</small></span>
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
  nCPU: PropTypes.number
};
