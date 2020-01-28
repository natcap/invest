import React from 'react';
import Electron from 'electron'

import Row from 'react-bootstrap/Row';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import Dropdown from 'react-bootstrap/Dropdown';

class DropdownItemModal extends React.Component {
  constructor(props) {
    super(props);
    this.state = { show: false }

    this.handleClose = this.handleClose.bind(this);
    this.handleShow = this.handleShow.bind(this);
  }

  handleClose() {
    this.setState({ show: false });
  }

  handleShow() {
    this.setState({show: true});
  }; 
}

export class SaveSessionDropdownItem extends DropdownItemModal {

  constructor(props) {
    super(props);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  handleSubmit(event) {
    event.preventDefault();
    this.props.saveState();
    this.setState({show: false});
  }

  render() {
    
    return (
      <React.Fragment>
        
        <Dropdown.Item onClick={this.handleShow}>Save session</Dropdown.Item>

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


export class SaveParametersDropdownItem extends React.Component {

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
    return(
      <Dropdown.Item onClick={this.browseFile}>Save parameters to JSON</Dropdown.Item>
    );
  }
}

export class SavePythonDropdownItem extends React.Component {
  
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
    return(
      <Dropdown.Item onClick={this.browseFile}>Save to Python script</Dropdown.Item>
    );
  }
}