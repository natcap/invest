import React from 'react';
import Electron from 'electron'

import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import Dropdown from 'react-bootstrap/Dropdown';
import InputGroup from 'react-bootstrap/InputGroup';

class DropdownItemModal extends React.Component {
  constructor(props) {
    super(props);
    this.state = { show: false }

    this.handleClose = this.handleClose.bind(this);
    this.handleShow = this.handleShow.bind(this);
  }

  handleClose() {
    this.setState({ show: false });
  };

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
          <Form>
            <Modal.Header>
              <Modal.Title>Save Session</Modal.Title>
            </Modal.Header>
            <Modal.Body>
              <Form inline onSubmit={this.props.saveState} className="mx-3">
                <Form.Label className="mx-3">Title</Form.Label>
                <Form.Control
                  type="text"
                  placeholder={this.props.sessionID}
                  value={this.props.sessionID}
                  onChange={this.props.setSessionID}
                />
              </Form>
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


export class SaveParametersDropdownItem extends DropdownItemModal {

  constructor(props) {
    super(props);
    this.state['filepath'] = '';

    this.handleSubmit = this.handleSubmit.bind(this);
    this.handleChange = this.handleChange.bind(this);
    this.browseFile = this.browseFile.bind(this);
  }

  handleSubmit(event) {
    event.preventDefault();
    this.props.argsToJsonFile(this.state.filepath);
    this.setState({show: false});
  }

  handleChange(event) {
    this.setState({ filepath: event.target.value })
  }

  browseFile(event) {
    // Handle clicks on form browse-button inputs
    Electron.remote.dialog.showSaveDialog((filepath) => {
      console.log(filepath); // 0 is safe since we only allow 1 selection
      this.setState({ filepath: filepath })
    });
  }

  render() {
    
    return (
      <React.Fragment>
        
        <Dropdown.Item onClick={this.handleShow}>Save parameters</Dropdown.Item>

        <Modal show={this.state.show} onHide={this.handleClose}>
          <Form>
            <Modal.Header>
              <Modal.Title>Save Parameters</Modal.Title>
            </Modal.Header>
            <Modal.Body>
              <Form.Group as={Row} key="saveParameters">
                <Form.Label column sm="3">Save as</Form.Label>
                <Col sm="8">
                  <InputGroup>
                    <Form.Control
                      name="saveParameters"
                      type="text" 
                      value={this.state.filepath || ''}
                      onChange={this.handleChange}
                    />
                    <InputGroup.Append>
                      <Button 
                        variant="outline-secondary"
                        name="browse"
                        onClick={this.browseFile}>
                        Browse
                      </Button>
                    </InputGroup.Append>
                  </InputGroup>
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

export class SavePythonDropdownItem extends React.Component {
  constructor(props) {
    super(props);

    this.browseFile = this.browseFile.bind(this);
  }

  browseFile(event) {
    // Handle clicks on form browse-button inputs
    Electron.remote.dialog.showSaveDialog(
      { defaultPath: 'python_script.py' }, (filepath) => {
      console.log(filepath); // 0 is safe since we only allow 1 selection
      this.props.savePythonScript(filepath)
    });
  }

  render() {

    return(
      <Dropdown.Item onClick={this.browseFile}>Save to Python script</Dropdown.Item>
    );
  }
}