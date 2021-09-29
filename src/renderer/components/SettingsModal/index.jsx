import React from 'react';
import PropTypes from 'prop-types';

import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';

import { getDefaultSettings } from './SettingsStorage';

/** Validate that n_workers is an acceptable value for Taskgraph.
 *
 * @param  {string} value - value for Taskgraph n_workers parameter.
 * @returns {boolean} - true if a valid value or false otherwise.
 *
 */
function validateNWorkers(value) {
  const nInt = parseInt(value);
  return Number.isInteger(nInt) && nInt >= -1;
}

/** Render a dialog with a form for configuring global invest settings */
export default class SettingsModal extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      show: false,
      localSettings: {
        nWorkers: '',
        loggingLevel: '',
        sampleDataDir: null,
        language: 'en'
      },
    };

    this.handleShow = this.handleShow.bind(this);
    this.handleClose = this.handleClose.bind(this);
    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.handleReset = this.handleReset.bind(this);
    this.switchToDownloadModal = this.switchToDownloadModal.bind(this);
  }

  componentDidMount() {
    this.setState({
      localSettings: this.props.investSettings
    });
  }

  componentDidUpdate(prevProps) {
    const { investSettings } = this.props;
    if (JSON.stringify(investSettings) !== JSON.stringify(prevProps.investSettings)) {
      this.setState({ localSettings: investSettings });
    }
  }

  handleClose() {
    /** reset the local settings from the app's state because we closed w/o save */
    const appSettings = Object.assign({}, this.props.investSettings)
    this.setState({
      show: false,
      localSettings: appSettings,
    });
  }

  handleShow() {
    this.setState({ show: true });
  }

  handleSubmit(event) {
    /** Handle a click on the "Save" button.
     *
     *  Updates the parent's state and persistent store.
     */
    event.preventDefault();
    this.props.saveSettings(this.state.localSettings);
    this.setState({ show: false });
  }

  handleReset(event) {
    event.preventDefault();
    const resetSettings = getDefaultSettings();
    this.setState({
      localSettings: resetSettings,
    });
  }

  handleChange(event) {
    const newSettings = Object.assign({}, this.state.localSettings);
    newSettings[event.target.name] = event.target.value;
    this.setState({
      localSettings: newSettings,
    });
  }

  switchToDownloadModal() {
    this.props.showDownloadModal();
    this.handleClose();
  }

  render() {
    const logLevelOptions = [
      'DEBUG', 'INFO', 'WARNING', 'ERROR'];

    // map display names to standard language codes
    const languageOptions = {
      'English': 'en',
      'Español': 'es'
    }

    const nWorkersIsValid = validateNWorkers(
      this.state.localSettings.nWorkers
    );

    // define a custom button component to have a gear icon and no background
    const CustomButton = React.forwardRef(({ children, onClick }, ref) => (
      <a
        href=""
        ref={ref}
        onClick={e => {
          e.preventDefault();
          onClick(e);
        }}
      >
        <i className="material-icons mdc-button__icon settings-icon"
          title="settings">
          settings
        </i>
        {children}
      </a>
    ));

    return (
      <React.Fragment>
        <Button
          as={CustomButton}
          onClick={this.handleShow}
        />

        <Modal show={this.state.show} onHide={this.handleClose}>
          <Modal.Header>
            <Modal.Title>InVEST Settings</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <Form.Group as={Row}>
              <Form.Label column sm="8" htmlFor="logging-select">Logging threshold</Form.Label>
              <Col sm="4">
                <Form.Control
                  id="logging-select"
                  as="select"
                  name="loggingLevel"
                  value={this.state.localSettings.loggingLevel}
                  onChange={this.handleChange}
                >
                  {logLevelOptions.map(opt =>
                    <option value={opt} key={opt}>{opt}</option>
                  )}
                </Form.Control>
              </Col>
            </Form.Group>
            <Form.Group as={Row}>
              <Form.Label column sm="8" htmlFor="nworkers-text">
                Taskgraph n_workers parameter
                <br />
                (must be an integer &gt;= -1)
              </Form.Label>
              <Col sm="4">
                <Form.Control
                  id="nworkers-text"
                  name="nWorkers"
                  type="text"
                  value={this.state.localSettings.nWorkers}
                  onChange={this.handleChange}
                  isInvalid={!nWorkersIsValid}
                />
              </Col>
            </Form.Group>
            <Form.Group as={Row}>
              <Form.Label column sm="8" htmlFor="logging-select">Language</Form.Label>
              <Col sm="4">
                <Form.Control
                  id="language-select"
                  as="select"
                  name="language"
                  value={this.state.localSettings.language}
                  onChange={this.handleChange}
                >
                  {languageOptions.keys().map(opt =>
                    <option value={opt} key={opt}>{opt}</option>
                  )}
                </Form.Control>
              </Col>
            </Form.Group>
            <Form.Group as={Row}>
              <Form.Label column sm="8">
                Reset to Defaults
              </Form.Label>
              <Col sm="4">
                <Button
                  variant="secondary"
                  onClick={this.handleReset}
                  type="button"
                  className="float-right"
                >
                  Reset
                </Button>
              </Col>
            </Form.Group>
            <hr />
            <Button
              variant="primary"
              onClick={this.switchToDownloadModal}
            >
              Download Sample Data
            </Button>
            <hr />
            <Form.Group as={Row}>
              <Form.Label column sm="8">
                Clear Recent Jobs Shortcuts
                <br />
                (no invest workspaces will be deleted)
              </Form.Label>
              <Col sm="4">
                <Button
                  variant="secondary"
                  onClick={this.props.clearJobsStorage}
                  className="float-right"
                >
                  Clear
                </Button>
              </Col>
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
              disabled={!nWorkersIsValid}
            >
              Save Changes
            </Button>
          </Modal.Footer>
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
