import React from 'react';
import PropTypes from 'prop-types';

import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';

/** Render a dialog with a form for configuring global invest settings */
export default class SettingsModal extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      show: false,
      localSettings: {}
    };

    this.handleShow = this.handleShow.bind(this);
    this.handleClose = this.handleClose.bind(this);
    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  componentDidUpdate(prevProps) {
    /** Any time the parent's state of investSettings changes, 
    * this component should reflect that.
    */
    if (this.props.investSettings !== prevProps.investSettings) {
      const globalSettings = Object.assign({}, this.props.investSettings)
      this.setState({localSettings: globalSettings})
    }
  }

  handleClose() {
    /** reset the local settings from the app's state because we closed w/o save */
    const appSettings = Object.assign({}, this.props.investSettings)
    this.setState({
      show: false,
      localSettings: appSettings
    });
  }

  handleShow() {
    this.setState({show: true});
  }

  handleSubmit(event) {
    /** Handle a click on the "Save" button, which updates the parent's state */
    event.preventDefault();
    this.props.saveSettings(this.state.localSettings);
    this.setState({show: false});
  }

  handleChange(event) {
    /** Handle changes to inputs by reflecting them back immediately 
    * via localSettings object. But do not update the values stored 
    * in the parent's state.
    */
    let newSettings = Object.assign({}, this.state.localSettings);
    newSettings[event.target.name] = event.target.value
    this.setState({
      localSettings: newSettings
    });
  }

  render() {
    const logLevelOptions = [
      'DEBUG', 'INFO', 'WARNING', 'ERROR'];

    const nWorkersIsValid = validateNWorkers(this.state.localSettings.nWorkers)

    return (
      <React.Fragment>
        <Button className="mx-3" variant="primary" onClick={this.handleShow}>
          Settings
        </Button>

        <Modal show={this.state.show} onHide={this.handleClose}>
          <Form>
            <Modal.Header>
              <Modal.Title>InVEST Settings</Modal.Title>
            </Modal.Header>
            <Modal.Body>
              <Form.Group as={Row}>
                <Form.Label column sm="8" htmlFor="logging-select">Logging threshold</Form.Label>
                <Col sm="3">
                  <Form.Control
                    id="logging-select"
                    as="select"
                    name="loggingLevel"
                    value={this.state.localSettings.loggingLevel}
                    onChange={this.handleChange}>
                    {logLevelOptions.map(opt =>
                      <option value={opt} key={opt}>{opt}</option>
                    )}
                  </Form.Control>
                </Col>
              </Form.Group>
              <Form.Group as={Row}>
                <Form.Label column sm="8" htmlFor="nworkers-text">
                  Taskgraph n_workers parameter (must be an integer &gt;= -1)
                </Form.Label>
                <Col sm="3">
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
            </Modal.Body>
            <Modal.Footer>
              <Button variant="secondary" onClick={this.handleClose}>
                Cancel
              </Button>
              <Button 
                variant="primary"
                onClick={this.handleSubmit}
                type="submit"
                disabled={!nWorkersIsValid}>
                Save Changes
              </Button>
            </Modal.Footer>
          </Form>
        </Modal>
      </React.Fragment>
    )
  }
}

SettingsModal.propTypes = {
  saveSettings: PropTypes.func,
  investSettings: PropTypes.shape({
    nWorkers: PropTypes.string,
    loggingLevel: PropTypes.string,
  })
};

/** Validate that n_wokers is an acceptable value for Taskgraph. */
function validateNWorkers(value) {
  const nInt = parseInt(value);
  return Number.isInteger(nInt) && nInt >= -1;
}
