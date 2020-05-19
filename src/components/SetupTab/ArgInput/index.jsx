import React from 'react';
import PropTypes from 'prop-types';

import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import InputGroup from 'react-bootstrap/InputGroup';
import Modal from 'react-bootstrap/Modal';


export class ArgInput extends React.PureComponent {
  constructor(props) {
    super(props)
  }

  render() {
    const argkey = this.props.argkey
    const argSpec = this.props.argSpec
    let ArgInput;

    // These types need a text input, and some also need a file browse button
    if (['csv', 'vector', 'raster', 'directory', 'freestyle_string', 'number'].includes(argSpec.type)) {
      const typeLabel = argSpec.type !== 'freestyle_string' ? argSpec.type : 'string'
      const labelText = <span>{argSpec.name} <em> ({typeLabel})</em></span>
      ArgInput = 
        <Form.Group as={Row} key={argkey} className={'arg-' + this.props.ui_option} data-testid={'group-' + argkey}>
          <Form.Label column sm="3"  htmlFor={argkey}>
            {labelText}
          </Form.Label>
          <Col sm="8">
            <InputGroup>
              <AboutModal argument={argSpec}/>
              <Form.Control
                id={argkey}
                name={argkey}
                type="text" 
                value={this.props.value || ''} // empty string is handled better than `undefined`
                onChange={this.props.handleChange}
                isValid={this.props.touched && this.props.isValid}
                isInvalid={this.props.touched && this.props.validationMessage}
                disabled={this.props.ui_option === 'disable' || false}
              />
              {
                ['csv', 'vector', 'raster', 'directory'].includes(argSpec.type) ?
                <InputGroup.Append>
                  <Button
                    id={argkey}
                    variant="outline-secondary"
                    value={argSpec.type}  // dialog will limit options to files or dirs accordingly
                    name={argkey}
                    onClick={this.props.selectFile}>
                    Browse
                  </Button>
                </InputGroup.Append> : <React.Fragment/>
              }
              <Form.Control.Feedback type='invalid' id={argkey + '-feedback'}>
                {argSpec.type + ' : ' + (this.props.validationMessage || '')}
              </Form.Control.Feedback>
            </InputGroup>
          </Col>
        </Form.Group>
    
    // Radio select for boolean args
    } else if (argSpec.type === 'boolean') {
      // The `checked` property does not treat 'undefined' the same as false,
      // instead React avoids setting the property altogether. Hence, !! to
      // cast undefined to false.
      ArgInput = 
        <Form.Group as={Row} key={argkey} data-testid={'group-' + argkey}>
          <Form.Label column sm="3" htmlFor={argkey}>{argSpec.name}</Form.Label>
          <Col sm="8">
            <AboutModal argument={argSpec}/>
            <Form.Check
              id={argkey}
              inline
              type="radio"
              label="Yes"
              value={"true"}
              checked={!!this.props.value}  // double bang casts undefined to false
              onChange={this.props.handleBoolChange}
              name={argkey}
            />
            <Form.Check
              id={argkey}
              inline
              type="radio"
              label="No"
              value={"false"}
              checked={!this.props.value}  // undefined becomes true, that's okay
              onChange={this.props.handleBoolChange}
              name={argkey}
            />
          </Col>
        </Form.Group>

    // Dropdown menus for args with options
    } else if (argSpec.type === 'option_string') {
      ArgInput = 
        <Form.Group as={Row} key={argkey} className={'arg-' + this.props.ui_option} data-testid={'group-' + argkey}>
          <Form.Label column sm="3" htmlFor={argkey}>{argSpec.name}</Form.Label>
          <Col sm="4">
            <InputGroup>
              <AboutModal argument={argSpec}/>
              <Form.Control 
                id={argkey}
                as='select'
                name={argkey}
                value={this.props.value}
                onChange={this.props.handleChange}
                disabled={this.props.ui_option === 'disable' || false}>
                {argSpec.validation_options.options.map(opt =>
                  <option value={opt} key={opt}>{opt}</option>
                )}
              </Form.Control>
              <Form.Control.Feedback type='invalid' id={argkey + '-feedback'}>
                {argSpec.type + ' : ' + (this.props.validationMessage || '')}
              </Form.Control.Feedback>
            </InputGroup>
          </Col>
        </Form.Group>
    }
    return(ArgInput)
  }
}

ArgInput.propTypes = {
  argkey: PropTypes.string,
  argSpec: PropTypes.object,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.bool]),
  touched: PropTypes.bool,
  ui_option: PropTypes.string,
  isValid: PropTypes.bool,
  validationMessage: PropTypes.string,
  handleChange: PropTypes.func,
  handleBoolChange: PropTypes.func,
  selectFile: PropTypes.func
}

class AboutModal extends React.PureComponent {
  constructor(props) {
    super(props)
    this.state = {
      aboutShow: false
    }
    this.handleAboutOpen = this.handleAboutOpen.bind(this);
    this.handleAboutClose = this.handleAboutClose.bind(this);
  }

  handleAboutClose() {
    this.setState({aboutShow: false});
  }

  handleAboutOpen() {
    this.setState({aboutShow: true});
  }

  render() {
    return(
      <React.Fragment>
        <Button  className="mr-3"
          onClick={this.handleAboutOpen}
          variant="outline-info">
          i
        </Button>
        <Modal show={this.state.aboutShow} onHide={this.handleAboutClose}>
          <Modal.Header>
            <Modal.Title>{this.props.argument.name}</Modal.Title>
            </Modal.Header>
          <Modal.Body>{this.props.argument.about}</Modal.Body>
        </Modal>
      </React.Fragment>
    );
  }
}

AboutModal.propTypes = {
  argument: PropTypes.shape({
    name: PropTypes.string,
    about: PropTypes.string
  })
}
