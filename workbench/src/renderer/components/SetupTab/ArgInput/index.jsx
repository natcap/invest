import React, { useEffect, useRef, useState } from 'react';
import PropTypes from 'prop-types';
import { useTranslation } from 'react-i18next';

import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import InputGroup from 'react-bootstrap/InputGroup';
import Modal from 'react-bootstrap/Modal';
import Table from 'react-bootstrap/Table';
import { MdFolderOpen, MdInfo, MdOpenInNew } from 'react-icons/md';

import { ipcMainChannels } from '../../../../main/ipcMainChannels';
import i18n from '../../../i18n/i18n';

const { ipcRenderer } = window.Workbench.electron;

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
  const newPrefix = i18n.t(
    'Not all of the spatial layers overlap each other. Bounding box:');
  const bbox = message.split(`${filepath}:`).pop().split('|')[0];
  const bboxFormatted = bbox.split(' ').map(
    (str) => str.padEnd(22, ' ')
  ).join('').trim();
  return `${newPrefix}\n${bboxFormatted}`;
}

function FormLabel(props) {
  const {
    argkey, argname, argtype, required, units,
  } = props;

  const userFriendlyArgType = parseArgType(argtype);
  const optional = typeof required === 'boolean' && !required;
  const includeComma = userFriendlyArgType && optional;

  return (
    <Form.Label column sm="3" htmlFor={argkey}>
      <span className="argname">{argname} </span>
      {
       (userFriendlyArgType || optional) && 
        <span>
          (
            {userFriendlyArgType}
            {includeComma && ', '}
            {optional && <em>{i18n.t('optional')}</em>}
          )
        </span>
      }
      {/* display units at the end of the arg name, if applicable */}
      { (units && units !== 'unitless') ? <span> ({units})</span> : <React.Fragment /> }
    </Form.Label>
  );
}

FormLabel.propTypes = {
  argkey: PropTypes.string.isRequired,
  argname: PropTypes.string.isRequired,
  argtype: PropTypes.string.isRequired,
  required: PropTypes.oneOfType(
    [PropTypes.string, PropTypes.bool]
  ),
  units: PropTypes.string,
};

function Feedback(props) {
  const { argkey, message } = props;
  return (
    // d-block class is needed because of a bootstrap bug
    // https://github.com/twbs/bootstrap/issues/29439
    <Form.Control.Feedback
      className="d-block"
      type="invalid"
      id={`${argkey}-feedback`}
    >
      {message}
    </Form.Control.Feedback>
  );
}
Feedback.propTypes = {
  argkey: PropTypes.string.isRequired,
  message: PropTypes.string,
};
Feedback.defaultProps = {
  message: '',
};

/**
 * Prevent the default case for onDragOver so onDrop event will be fired.
 *
 * @param {Event} event - dragover event
 */
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

function parseArgType(argtype) {
  const { t, i18n } = useTranslation();
  // These types benefit from more descriptive placeholder text.
  let userFriendlyArgType;
  switch (argtype) {
    case 'freestyle_string':
      userFriendlyArgType = t('text');
      break;
    case 'percent':
      userFriendlyArgType = t('percent: a number from 0 - 100');
      break;
    case 'ratio':
      userFriendlyArgType = t('ratio: a decimal from 0 - 1');
      break;
    case 'boolean':
    case 'option_string':
      userFriendlyArgType = '';
      break;
    default:
      userFriendlyArgType = t(argtype);
  }
  return userFriendlyArgType;
}

export function ArgInput(props) {
  const inputRef = useRef();

  const {
    argkey,
    argSpec,
    formTable,
    userguide,
    enabled,
    updateArgValues,
    handleFocus,
    inputDropHandler,
    isValid,
    selectFile,
    touched,
    dropdownOptions,
    value,
    scrollEventCount,
  } = props;
  let { validationMessage } = props;

  const { t } = useTranslation();

  const initFormTableValues = {};
  if (formTable) {
    Object.keys(argSpec.columns).forEach((colName) => {
      initFormTableValues[colName] = '';
    });
  }

  const [formTableValues, setFormTableValues] = useState(initFormTableValues);

  // Occasionaly we want to force a scroll to the end of input fields
  // so that the most important part of a filepath is visible.
  // scrollEventCount changes on drop events and on use of the browse button.
  // Also depend on [value] so we don't scroll before the value updated.
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.scrollLeft = inputRef.current.scrollWidth;
    }
  }, [scrollEventCount, value]);

  function handleChange(event) {
    /** Pass input value up to SetupTab for storage & validation. */
    const { name, value } = event.currentTarget;
    updateArgValues(name, value);
  }

  // Messages with this pattern include validation feedback about
  // multiple inputs, but the whole message is repeated for each input.
  // It's more readable if filtered on the individual input.
  const pattern = 'Not all of the spatial layers overlap each other';
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
    );
  }

  // These types benefit from more descriptive placeholder text.
  const placeholderText = parseArgType(argSpec.type);

  let form;
  if (argSpec.type === 'boolean') {
    form = (
      <Form.Check
        inline
        type="switch"
        id={argkey}
        name={argkey}
        checked={value}
        onChange={() => updateArgValues(argkey, !value)}
        disabled={!enabled}
        bsCustomPrefix="form-switch"
      />
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
        isValid={enabled && isValid}
        custom
      >
        {
          Array.isArray(dropdownOptions) ?
          dropdownOptions.map(
            (opt) => <option value={opt} key={opt}>{opt}</option>
          ) :
          Object.entries(dropdownOptions).map(
            ([opt, info]) => <option value={opt} key={opt}>{info.display_name}</option>
          )
        }
      </Form.Control>
    );
  } else if (argSpec.type === 'csv' && formTable) {
    const rows = [];
    Object.entries(argSpec.columns).forEach(([key, spec]) => {
      rows.push(
        <tr>
          <td>
            <AboutModal arg={spec} userguide={userguide} argkey={key} />
          </td>
          <td>{key}</td>
          <td>
            <Form.Control
              ref={inputRef}
              id={argkey}
              name={key}
              type="text"
              placeholder={t(spec.type)}
              value={formTableValues[key] || ''} // empty string is handled better than `undefined`
              onChange={handleChangeFormTable}
              onFocus={() => handleFocus({ currentTarget: { name: argkey } })}
              onBlur={(e) => e.target.scrollLeft = e.target.scrollWidth}
              isValid={enabled && touched && isValid}
              isInvalid={enabled && validationMessage}
              disabled={!enabled}
              onDrop={inputDropHandler}
              onDragOver={dragOverHandler}
              onDragEnter={dragEnterHandler}
              onDragLeave={dragLeavingHandler}
            />
          </td>
        </tr>
      );
    });
    form = (
      <Table striped bordered hover>
        <tbody
          isValid={enabled && touched && isValid}
          isInvalid={enabled && validationMessage}>
          {rows}
        </tbody>
      </Table>
    );
  } else {
    form = (
      <React.Fragment>
        <Form.Control
          ref={inputRef}
          id={argkey}
          name={argkey}
          type="text"
          placeholder={placeholderText}
          value={value || ''} // empty string is handled better than `undefined`
          onChange={handleChange}
          onFocus={handleFocus}
          onBlur={(e) => e.target.scrollLeft = e.target.scrollWidth}
          isValid={enabled && touched && isValid}
          isInvalid={enabled && validationMessage}
          disabled={!enabled}
          onDrop={inputDropHandler}
          onDragOver={dragOverHandler}
          onDragEnter={dragEnterHandler}
          onDragLeave={dragLeavingHandler}
          aria-describedby={`${argkey}-feedback`}
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
      style={{ display: (enabled ? 'flex' : 'none') }}
    >
      <FormLabel
        argkey={argkey}
        argname={argSpec.name}
        argtype={argSpec.type}
        required={argSpec.required}
        units={argSpec.units} // undefined for all types except number
      />
      <Col>
        <InputGroup>
          <div className="d-flex flex-nowrap w-100">
            <AboutModal arg={argSpec} userguide={userguide} argkey={argkey} />
            {form}
          </div>
          {feedback}
        </InputGroup>
      </Col>
    </Form.Group>
  );
}

ArgInput.propTypes = {
  argkey: PropTypes.string.isRequired,
  argSpec: PropTypes.shape({
    name: PropTypes.string.isRequired,
    type: PropTypes.string.isRequired,
    required: PropTypes.oneOfType([PropTypes.string, PropTypes.bool]),
    units: PropTypes.string, // for numbers only
  }).isRequired,
  userguide: PropTypes.string.isRequired,
  value: PropTypes.oneOfType(
    [PropTypes.string, PropTypes.bool, PropTypes.number]),
  touched: PropTypes.bool,
  isValid: PropTypes.bool,
  validationMessage: PropTypes.string,
  updateArgValues: PropTypes.func.isRequired,
  handleFocus: PropTypes.func.isRequired,
  selectFile: PropTypes.func.isRequired,
  enabled: PropTypes.bool.isRequired,
  dropdownOptions: PropTypes.oneOfType([PropTypes.arrayOf(PropTypes.string), PropTypes.object]),
  inputDropHandler: PropTypes.func.isRequired,
  scrollEventCount: PropTypes.number,
};
ArgInput.defaultProps = {
  value: undefined,
  touched: false,
  isValid: undefined,
  validationMessage: '',
  dropdownOptions: undefined,
  scrollEventCount: 0,
};

/**
 * Open the target href in the default web browser.
 *
 * @param {Event} event - event triggered by a click on the user's guide link
 */
function handleClickUsersGuideLink(event) {
  event.preventDefault();
  ipcRenderer.send(
    ipcMainChannels.OPEN_LOCAL_HTML, event.currentTarget.href
  );
}

function AboutModal(props) {
  const [aboutShow, setAboutShow] = useState(false);
  const handleAboutClose = () => setAboutShow(false);
  const handleAboutOpen = () => setAboutShow(true);

  const { userguide, arg, argkey } = props;
  const { t, i18n } = useTranslation();

  // create link to users guide entry for this arg
  // anchor name is the arg name, with underscores replaced with hyphens
  const userguideURL = `
    ${window.Workbench.USERGUIDE_PATH}/${window.Workbench.LANGUAGE}/${userguide}#${argkey.replace(/_/g, '-')}`;
  return (
    <React.Fragment>
      <Button
        aria-label={`info about ${arg.name}`}
        className="mr-2"
        onClick={handleAboutOpen}
        variant="outline-info"
      >
        <MdInfo />
      </Button>
      <Modal show={aboutShow} onHide={handleAboutClose}>
        <Modal.Header>
          <Modal.Title>{arg.name}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {arg.about}
          <br />
          <a
            href={userguideURL}
            title={userguideURL}
            aria-label="open user guide section for this input in web browser"
            onClick={handleClickUsersGuideLink}
          >
            {t("User's guide entry")}
            <MdOpenInNew className="mr-1" />
          </a>
        </Modal.Body>
      </Modal>
    </React.Fragment>
  );
}

AboutModal.propTypes = {
  arg: PropTypes.shape({
    name: PropTypes.string,
    about: PropTypes.string,
  }).isRequired,
  userguide: PropTypes.string.isRequired,
  argkey: PropTypes.string.isRequired,
};

export function FormTableInput(props) {
  const {
    argkeys,
    argsSpec,
    groupName,
    argsEnabled,
    formTable,
    userguide,
    updateArgValues,
    handleFocus,
    inputDropHandler,
    argsValidation,
    selectFile,
    touched,
    dropdownOptions,
    argsValues,
    scrollEventCount,
  } = props;
  let { validationMessage } = props;

  console.log('render formtable', argkeys);

  const { t } = useTranslation();

  const initFormTableValues = {};
  if (formTable) {
    Object.keys(argSpec.columns).forEach((colName) => {
      initFormTableValues[colName] = '';
    });
  }

  const [formTableValues, setFormTableValues] = useState(initFormTableValues);

  function handleChange(event) {
    /** Pass input value up to SetupTab for storage & validation. */
    const { name, value } = event.currentTarget;
    updateArgValues(name, value);
  }

  function handleChangeFormTable(event, argkey) {
    const { name, value } = event.currentTarget;
    const newValues = { ...formTableValues };
    newValues[name] = value;
    setFormTableValues(newValues);
    ipcRenderer.invoke(ipcMainChannels.WRITE_CSV, newValues).then(
      (path) => handleChange({ currentTarget: { name: argkey, value: path } })
    );
  }

  const rows = [];
  let anyEnabled = false;
  argkeys.forEach((key) => {
    const spec = argsSpec[key];

    let feedback = <React.Fragment />;
    if (argsValidation[key].validationMessage && argsValues[key].touched && spec.type !== 'boolean') {
      feedback = (
        <Feedback
          argkey={key}
          argtype={spec.type}
          message={argsValidation[key].validationMessage}
        />
      );
    }

    anyEnabled = anyEnabled || argsEnabled[key];

    rows.push(
      <tr style={argsEnabled[key] ? {} : { display: 'none' }}>
        <td>
          <AboutModal arg={spec} userguide={userguide} argkey={key} />
        </td>
        <td>
          <FormLabel
            argkey={key}
            argname={argsSpec[key].name}
            required={argsSpec[key].required}
            units={argsSpec[key].units} // undefined for all types except number
          />
        </td>
        <td className="is-valid">
          <Form.Control
            id={key}
            name={key}
            type="text"
            placeholder={t(spec.type)}
            value={argsValues[key].value || ''} // empty string is handled better than `undefined`
            onChange={handleChange}
            onFocus={() => handleFocus({ currentTarget: { name: key } })}
            onBlur={(e) => e.target.scrollLeft = e.target.scrollWidth}
            isValid={argsEnabled[key] && argsValues[key].touched && argsValidation[key].valid}
            isInvalid={argsEnabled[key] && argsValidation[key].validationMessage}
            disabled={!argsEnabled[key]}
            onDrop={inputDropHandler}
            onDragOver={dragOverHandler}
            onDragEnter={dragEnterHandler}
            onDragLeave={dragLeavingHandler}
          />
          {feedback}
        </td>
      </tr>
    );
  });
  const form = (
    <Table striped bordered hover>
      <tbody>
        {rows}
      </tbody>
    </Table>
  );

  return (
    <Form.Group
      as={Row}
      key={groupName}
      data-testid={`group-${groupName}`}
      style={anyEnabled ? {} : { display: 'none' }}
    >
      <FormLabel
        argkey={groupName}
        argname={groupName}
      />
      <Col>
        <InputGroup>
          <div className="d-flex flex-nowrap w-100">
            {/*<AboutModal arg={argSpec} userguide={userguide} argkey={argkey} />*/}
            {form}
          </div>
        </InputGroup>
      </Col>
    </Form.Group>
  );
}
