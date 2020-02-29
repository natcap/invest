import React from 'react';
import Electron from 'electron'

import Row from 'react-bootstrap/Row';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import Dropdown from 'react-bootstrap/Dropdown';


export class SaveSessionButtonModal extends React.Component {

  constructor(props) {
    super(props);
    this.state = { show: false }

    this.handleClose = this.handleClose.bind(this);
    this.handleShow = this.handleShow.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  handleClose() {
    this.setState({ show: false });
  }

  handleShow() {
    this.setState({show: true});
  }

  handleSubmit(event) {
    event.preventDefault();
    this.props.saveState();
    this.setState({show: false});
  }

  render() {
    
    return (
      <React.Fragment>
        
        <Button onClick={this.handleShow} variant="link">Save session</Button>

        <Modal show={this.state.show} onHide={this.handleClose}>
          <Form onSubmit={this.handleSubmit}>
            <Modal.Header>
              <Modal.Title>Save Session</Modal.Title>
            </Modal.Header>
            <Modal.Body>
              <Form.Group as={Row} key="1">
                <Form.Label className="mx-3">Title</Form.Label>
                <Form.Control
                  type="text"
                  placeholder={this.props.sessionID}
                  value={this.props.sessionID}
                  onChange={this.props.setSessionID}
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
                disabled={false}>
                Save Session
              </Button>
            </Modal.Footer>
          </Form>
        </Modal>
      </React.Fragment>
    )
  }
}


export class SaveParametersButton extends React.Component {

  constructor(props) {
    super(props);
    this.browseFile = this.browseFile.bind(this);
  }

  browseFile(event) {
    Electron.remote.dialog.showSaveDialog(
      { defaultPath: 'invest_args.json' }, (filepath) => {
      this.props.argsToJsonFile(filepath);
    });
  }

  render() {
    // disabled when there's no modelSpec, i.e. before a model is selected
    return(
      <Button 
        onClick={this.browseFile}
        disabled={this.props.disabled}
        variant="link">
        Save parameters to JSON
      </Button>
    );
  }
}

export class SavePythonButton extends React.Component {
  
  constructor(props) {
    super(props);
    this.browseFile = this.browseFile.bind(this);
  }

  browseFile(event) {
    Electron.remote.dialog.showSaveDialog(
      { defaultPath: 'execute_invest.py' }, (filepath) => {
      this.props.savePythonScript(filepath)
    });
  }

  render() {
    // disabled when there's no modelSpec, i.e. before a model is selected
    return(
      <Button 
        onClick={this.browseFile}
        disabled={this.props.disabled}
        variant="link">
        Save to Python script
      </Button>
    );
  }
}