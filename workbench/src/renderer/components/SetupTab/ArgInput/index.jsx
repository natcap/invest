import os from 'os';
import React from 'react';
import PropTypes from 'prop-types';

import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import InputGroup from 'react-bootstrap/InputGroup';
import Modal from 'react-bootstrap/Modal';
import { MdFolderOpen, MdInfo, MdOpenInNew } from 'react-icons/md';
import { shell } from 'electron';

const baseUserguideURL = 'https://storage.googleapis.com/natcap-dev-build-artifacts/invest/emlys/3.9.1.post3250+g130227a76/userguide';

/**
 * Filter a message that refers to many spatial inputs' bounding boxes.
 *
 * Messages are oneline strings with `|` separating each
 * filename & bounding-box pair. E.g:
 *
 *   `Bounding boxes do not intersect:
 *   ./a.shp: [-84.9, 19.1, -69.15, 29.5] |
 *   ./b.shp: [-79.0198, 26.481, -78.3717, 27.268] | ...`
 *
 * @param {string} message - a string that contains `filepath` e.g:
 * @param {string} filepath - string preceding the relevant part of `message`
 * @returns {string} - the filtered and formatted part of the message
 */
function filterSpatialOverlapFeedback(message, filepath) {
  const newPrefix = _('Bounding box does not intersect at least one other:');
  const bbox = message.split(`${filepath}:`).pop().split('|')[0];
  const bboxFormatted = bbox.split(' ').map(
    (str) => str.padEnd(22, ' ')
  ).join('').trim();
  return `${newPrefix}${os.EOL}${bboxFormatted}`;
}

function FormLabel(props) {
  const { argkey, argname, required } = props;
  return (
    <Form.Label column sm="3" htmlFor={argkey}>
      <span>
        {argname}
        <em>
          {
            (typeof required === 'boolean' && !required)
              ? ' (optional)' : ''
          }
        </em>
      </span>
    </Form.Label>
  );
}
FormLabel.propTypes = {
  argkey: PropTypes.string.isRequired,
  argname: PropTypes.string.isRequired,
  required: PropTypes.oneOfType(
    [PropTypes.string, PropTypes.bool]
  ),
};

function Feedback(props) {
  const { argkey, argtype, message } = props;
  return (
    // d-block class is needed because of a bootstrap bug
    // https://github.com/twbs/bootstrap/issues/29439
    <Form.Control.Feedback
      className="d-block"
      type="invalid"
      id={`${argkey}-feedback`}
    >
      {`${argtype} : ${(message)}`}
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

/** Prevent the default case for onDragOver so onDrop event will be fired. */
function dragOverHandler(event) {
  event.preventDefault();
  event.stopPropagation();
  if (event.currentTarget.disabled) {
    event.dataTransfer.dropEffect = 'none';
  } else {
    event.dataTransfer.dropEffect = 'copy';
  }
}

function dragEnterHandler(event) {
  event.preventDefault();
  event.stopPropagation();
  if (event.currentTarget.disabled) {
    event.dataTransfer.dropeffect = 'none';
  } else {
    event.dataTransfer.dropEffect = 'copy';
    event.currentTarget.classList.add('input-dragging');
  }
}

function dragLeavingHandler(event) {
  event.preventDefault();
  event.stopPropagation();
  event.dataTransfer.dropEffect = 'copy';
  event.currentTarget.classList.remove('input-dragging');
}

export default class ArgInput extends React.PureComponent {
  render() {
    const {
      argkey,
      argSpec,
      userguide,
      enabled,
      handleBoolChange,
      handleChange,
      handleFocus,
      inputDropHandler,
      isValid,
      selectFile,
      touched,
      dropdownOptions,
      value,
    } = this.props;
    let { validationMessage } = this.props;
    // Messages with this pattern include validation feedback about
    // multiple inputs, but the whole message is repeated for each input.
    // It's more readable if filtered on the individual input.
    const pattern = 'Bounding boxes do not intersect';
    if (validationMessage.startsWith(pattern)) {
      validationMessage = filterSpatialOverlapFeedback(
        validationMessage, value
      );
    }

    const className = enabled ? null : 'arg-disable';

    let feedback = <React.Fragment />;
    if (validationMessage && touched && argSpec.type !== 'boolean') {
      feedback = (
        <Feedback
          argkey={argkey}
          argtype={argSpec.type}
          message={validationMessage}
        />
      );
    }

    let fileSelector = <React.Fragment />;
    if (['csv', 'vector', 'raster', 'directory', 'file'].includes(argSpec.type)) {
      fileSelector = (
        <InputGroup.Append>
          <Button
            aria-label={`browse for ${argSpec.name}`}
            className="ml-2"
            id={argkey}
            variant="outline-dark"
            value={argSpec.type} // dialog will limit options accordingly
            name={argkey}
            onClick={selectFile}
            disabled={!enabled}
          >
            <MdFolderOpen />
          </Button>
        </InputGroup.Append>
      );
    }

    // These types benefit from more descriptive placeholder text.
    let placeholderText;
    switch (argSpec.type) {
      case 'freestyle_string':
        placeholderText = _('text');
        break;
      case 'percent':
        placeholderText = _('percent: a number from 0 - 100');
        break;
      case 'ratio':
        placeholderText = _('ratio: a decimal from 0 - 1');
        break;
      default:
        placeholderText = _(argSpec.type);
    }

    let form;
    if (argSpec.type === 'boolean') {
      form = (
        <React.Fragment>
          <Form.Check
            id={argkey}
            inline
            type="radio"
            label="Yes"
            value="true"
            checked={!!value} // double bang casts undefined to false
            onChange={handleBoolChange}
            name={argkey}
            disabled={!enabled}
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
            disabled={!enabled}
          />
        </React.Fragment>
      );
    } else if (argSpec.type === 'option_string') {
      form = (
        <Form.Control
          id={argkey}
          as="select"
          name={argkey}
          value={value}
          onChange={handleChange}
          onFocus={handleChange}
          disabled={!enabled}
        >
          {dropdownOptions.map(
            (opt) => <option value={opt} key={opt}>{opt}</option>
          )}
        </Form.Control>
      );
    } else {
      form = (
        <React.Fragment>
          <Form.Control
            id={argkey}
            name={argkey}
            type="text"
            placeholder={placeholderText}
            value={value || ''} // empty string is handled better than `undefined`
            onChange={handleChange}
            onFocus={handleFocus}
            isValid={enabled && touched && isValid}
            isInvalid={enabled && validationMessage}
            disabled={!enabled}
            onDrop={inputDropHandler}
            onDragOver={dragOverHandler}
            onDragEnter={dragEnterHandler}
            onDragLeave={dragLeavingHandler}
          />
          {fileSelector}
        </React.Fragment>
      );
    }

    return (
      <Form.Group
        as={Row}
        key={argkey}
        data-testid={`group-${argkey}`}
        className={className} // this grays out the label but doesn't actually disable the field
      >
        <FormLabel
          argkey={argkey}
          argname={argSpec.name}
          required={argSpec.required}
        />
        <Col>
          <InputGroup>
            <div className="d-flex flex-nowrap w-auto">
              <AboutModal arg={argSpec} userguide={userguide} argkey={argkey} />
              {form}
            </div>
            {feedback}
          </InputGroup>
        </Col>
      </Form.Group>
    );
  }
}

ArgInput.propTypes = {
  argkey: PropTypes.string.isRequired,
  argSpec: PropTypes.shape({
    name: PropTypes.string.isRequired,
    type: PropTypes.string.isRequired,
    required: PropTypes.oneOfType([PropTypes.string, PropTypes.bool]),
  }).isRequired,
  userguide: PropTypes.string.isRequired,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.bool]),
  touched: PropTypes.bool,
  isValid: PropTypes.bool,
  validationMessage: PropTypes.string,
  handleFocus: PropTypes.func.isRequired,
  handleChange: PropTypes.func.isRequired,
  handleBoolChange: PropTypes.func.isRequired,
  selectFile: PropTypes.func.isRequired,
  enabled: PropTypes.bool.isRequired,
  dropdownOptions: PropTypes.arrayOf(PropTypes.string),
  inputDropHandler: PropTypes.func.isRequired,
};
ArgInput.defaultProps = {
  value: undefined,
  touched: false,
  isValid: undefined,
  validationMessage: '',
  dropdownOptions: undefined,
};

/**
 * Open the target href in the default web browser.
 */
function handleMoreInfoClick(event) {
  event.preventDefault();
  shell.openExternal(event.currentTarget.href);
}

class AboutModal extends React.PureComponent {
  constructor(props) {
    super(props);
    this.state = {
      aboutShow: false,
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
    const { userguide, arg, argkey } = this.props;
    const { aboutShow } = this.state;
    // create link to users guide entry for this arg
    // anchor name is the arg name, with underscores replaced with hyphens
    const userguideURL = `${baseUserguideURL}/${userguide}#${argkey.replace(/_/g, '-')}`;
    return (
      <React.Fragment>
        <Button
          aria-label={`info about ${arg.name}`}
          className="mr-2"
          onClick={this.handleAboutOpen}
          variant="outline-info"
        >
          <MdInfo />
        </Button>
        <Modal show={aboutShow} onHide={this.handleAboutClose}>
          <Modal.Header>
            <Modal.Title>{arg.name}</Modal.Title>
          </Modal.Header>
          <Modal.Body>{arg.about}</Modal.Body>
          <a
            href={userguideURL}
            title={userguideURL}
            aria-label="open user guide section for this input in web browser"
            onClick={handleMoreInfoClick}
          >
            <MdOpenInNew className="mr-1" />
            {_('More info')}
          </a>
        </Modal>
      </React.Fragment>
    );
  }
}

AboutModal.propTypes = {
  arg: PropTypes.shape({
    name: PropTypes.string,
    about: PropTypes.string,
  }).isRequired,
  userguide: PropTypes.string.isRequired,
  argkey: PropTypes.string.isRequired,
};
