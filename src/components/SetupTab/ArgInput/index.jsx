import React from 'react';
import PropTypes from 'prop-types';

import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import InputGroup from 'react-bootstrap/InputGroup';
import Modal from 'react-bootstrap/Modal';

function FormLabel(props) {
  return (
    <Form.Label column sm="3" htmlFor={props.argkey}>
      {props.children}
    </Form.Label>
  );
}
FormLabel.propTypes = {
  argkey: PropTypes.string.isRequired,
  children: PropTypes.element.isRequired,
};

function Feedback(props) {
  return (
    <Form.Control.Feedback type="invalid" id={`${props.argkey}-feedback`}>
      {`${props.argtype} : ${(props.message)}`}
    </Form.Control.Feedback>
  );
}
Feedback.propTypes = {
  argkey: PropTypes.string.isRequired,
  argtype: PropTypes.string.isRequired,
  message: PropTypes.string,
};
Feedback.defaultProps = {
  message: '',
};

export default class ArgInput extends React.Component {
  render() {
    const {
      argkey,
      argSpec,
      handleBoolChange,
      handleChange,
      isValid,
      selectFile,
      touched,
      ui_option,
      validationMessage,
      value,
    } = this.props;
    let Input;

    // These types need a text input, and some also need a file browse button
    if (['csv', 'vector', 'raster', 'directory', 'freestyle_string', 'number'].includes(argSpec.type)) {
      const typeLabel = argSpec.type !== 'freestyle_string' ? argSpec.type : 'string';
      Input = (
        <Form.Group
          as={Row}
          key={argkey}
          className={`arg-${ui_option}`}
          data-testid={`group-${argkey}`}
        >
          <FormLabel argkey={argkey}>
            <span>
              {argSpec.name} <em> ({typeLabel})</em>
            </span>
          </FormLabel>
          <Col sm="8">
            <InputGroup>
              <AboutModal argument={argSpec} />
              <Form.Control
                id={argkey}
                name={argkey}
                type="text"
                value={value || ''} // empty string is handled better than `undefined`
                onChange={handleChange}
                isValid={touched && isValid}
                isInvalid={touched && validationMessage}
                disabled={ui_option === 'disable'}
              />
              {
                ['csv', 'vector', 'raster', 'directory'].includes(argSpec.type)
                  ? (
                    <InputGroup.Append>
                      <Button
                        id={argkey}
                        variant="outline-secondary"
                        value={argSpec.type} // dialog will limit options accordingly
                        name={argkey}
                        onClick={selectFile}
                      >
                        Browse
                      </Button>
                    </InputGroup.Append>
                  )
                  : <React.Fragment />
              }
              <Feedback
                argkey={argkey}
                argtype={argSpec.type}
                message={validationMessage}
              />
            </InputGroup>
          </Col>
        </Form.Group>
      );

    // Radio select for boolean args
    } else if (argSpec.type === 'boolean') {
      // The `checked` property does not treat 'undefined' the same as false,
      // instead React avoids setting the property altogether. Hence, !! to
      // cast undefined to false.
      Input = (
        <Form.Group as={Row} key={argkey} data-testid={`group-${argkey}`}>
          <FormLabel argkey={argkey}>
            <span>{argSpec.name}</span>
          </FormLabel>
          <Col sm="8">
            <AboutModal argument={argSpec} />
            <Form.Check
              id={argkey}
              inline
              type="radio"
              label="Yes"
              value="true"
              checked={!!value} // double bang casts undefined to false
              onChange={handleBoolChange}
              name={argkey}
            />
            <Form.Check
              id={argkey}
              inline
              type="radio"
              label="No"
              value="false"
              checked={!value} // undefined becomes true, that's okay
              onChange={handleBoolChange}
              name={argkey}
            />
          </Col>
        </Form.Group>
      );

    // Dropdown menus for args with options
    } else if (argSpec.type === 'option_string') {
      Input = (
        <Form.Group as={Row} key={argkey} className={`arg-${ui_option}`} data-testid={`group-${argkey}`}>
          <FormLabel argkey={argkey}>
            <span>{argSpec.name}</span>
          </FormLabel>
          <Col sm="4">
            <InputGroup>
              <AboutModal argument={argSpec} />
              <Form.Control
                id={argkey}
                as="select"
                name={argkey}
                value={value}
                onChange={handleChange}
                disabled={ui_option === 'disable'}
              >
                {argSpec.validation_options.options.map((opt) =>
                  <option value={opt} key={opt}>{opt}</option>
                )}
              </Form.Control>
              <Feedback
                argkey={argkey}
                argtype={argSpec.type}
                message={validationMessage}
              />
            </InputGroup>
          </Col>
        </Form.Group>
      );
    }
    return Input;
  }
}

ArgInput.propTypes = {
  argkey: PropTypes.string.isRequired,
  argSpec: PropTypes.object.isRequired,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.bool]),
  touched: PropTypes.bool,
  ui_option: PropTypes.string,
  isValid: PropTypes.bool,
  validationMessage: PropTypes.string,
  handleChange: PropTypes.func.isRequired,
  handleBoolChange: PropTypes.func.isRequired,
  selectFile: PropTypes.func.isRequired,
};
ArgInput.defaultProps = {
  value: undefined,
  touched: false,
  ui_option: undefined,
  isValid: undefined,
  validationMessage: '',
};

class AboutModal extends React.PureComponent {
  constructor(props) {
    super(props);
    this.state = {
      aboutShow: false
    };
    this.handleAboutOpen = this.handleAboutOpen.bind(this);
    this.handleAboutClose = this.handleAboutClose.bind(this);
  }

  handleAboutClose() {
    this.setState({ aboutShow: false });
  }

  handleAboutOpen() {
    this.setState({ aboutShow: true });
  }

  render() {
    return (
      <React.Fragment>
        <Button
          className="mr-3"
          onClick={this.handleAboutOpen}
          variant="outline-info"
        >
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
    about: PropTypes.string,
  }).isRequired,
};
