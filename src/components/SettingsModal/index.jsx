import React from 'react';

import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';

export class SettingsModal extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
    	show: false,
      localSettings: {}
    }

    this.handleClose = this.handleClose.bind(this);
    this.handleShow = this.handleShow.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.handleChange = this.handleChange.bind(this);
  }

  componentDidUpdate(prevProps) {
    if (this.props.investSettings !== prevProps.investSettings) {
      const globalSettings = Object.assign({}, this.props.investSettings)
      this.setState({localSettings: globalSettings})
    }
  }

  handleClose() {
    // reset the local settings from the app's state because we closed w/o save
    const appSettings = Object.assign({}, this.props.investSettings)
  	this.setState({
      show: false,
      localSettings: appSettings
    });
  };

  handleShow() {
  	this.setState({show: true});
  };

  handleSubmit(event) {
    event.preventDefault();
    this.props.saveSettings(this.state.localSettings);
    this.setState({show: false});
  }

  handleChange(event) {
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
        <Button variant="primary" onClick={this.handleShow}>
          Settings
        </Button>

        <Modal show={this.state.show} onHide={this.handleClose}>
          <Form>
            <Modal.Header>
              <Modal.Title>InVEST Settings</Modal.Title>
            </Modal.Header>
            <Modal.Body>
              <Form.Group as={Row}>
                <Form.Label column sm="8">Logging threshold</Form.Label>
                <Col sm="3">
                  <Form.Control
                    as='select'
                    name='loggingLevel'
                    value={this.state.localSettings.loggingLevel}
                    onChange={this.handleChange}>
                    {logLevelOptions.map(opt =>
                      <option value={opt} key={opt}>{opt}</option>
                    )}
                  </Form.Control>
                </Col>
              </Form.Group>
              <Form.Group as={Row}>
                <Form.Label column sm="8">
                  Taskgraph n_workers parameter (must be an integer >= -1)
                </Form.Label>
                <Col sm="3">
                  <Form.Control
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

function validateNWorkers(value) {
  const nInt = parseInt(value)
  return Number.isInteger(nInt) && nInt >= -1
}