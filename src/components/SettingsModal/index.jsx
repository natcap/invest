import React from 'react';

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
      console.log(this.props.investSettings);
      const globalSettings = Object.assign({}, this.props.investSettings)
      this.setState({localSettings: globalSettings})
    }
  }

  handleClose() {
    // reset the local settings from the app's state
    const localSettings = Object.assign({}, this.props.investSettings)
    console.log(localSettings);
  	this.setState({
      show: false,
      localSettings: localSettings
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
    console.log(event.target.name);
    console.log(event.target.value);
    this.setState({
      localSettings: newSettings
    });
  }

  render() {
    const logLevelOptions = [
      'NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];

    return (
      <>
        <Button variant="primary" onClick={this.handleShow}>
          settings
        </Button>

        <Modal show={this.state.show} onHide={this.handleClose}>
          <Form>
            <Modal.Header>
              <Modal.Title>InVEST Settings</Modal.Title>
            </Modal.Header>
            <Modal.Body>
              <Form.Group>
                <Form.Label column sm="3">Logging threshold</Form.Label>
                <Form.Control
                  as='select'
                  name='loggingLevel'
                  value={this.state.localSettings.loggingLevel}
                  onChange={this.handleChange}>
                  {logLevelOptions.map(opt =>
                    <option value={opt} key={opt}>{opt}</option>
                  )}
                </Form.Control>
              </Form.Group>
            </Modal.Body>
            <Modal.Footer>
              <Button variant="secondary" onClick={this.handleClose}>
                Cancel
              </Button>
              <Button 
                variant="primary"
                onClick={this.handleSubmit}
                type="submit">
                Save Changes
              </Button>
            </Modal.Footer>
          </Form>
        </Modal>
      </>
    )
  }
}