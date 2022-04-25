import React from 'react';
import PropTypes from 'prop-types';

import Form from 'react-bootstrap/Form';

import ArgInput from '../ArgInput';
import { boolStringToBoolean } from '../../../utils';
import { ipcMainChannels } from '../../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;

/** Prevent the default case for onDragOver so onDrop event will be fired. */
function dragOverHandler(event) {
  event.preventDefault();
  event.stopPropagation();
  event.dataTransfer.dropEffect = 'copy';
}

/** Renders a form with a list of input components. */
export default class ArgsForm extends React.Component {
  constructor(props) {
    super(props);
    this.handleFocus = this.handleFocus.bind(this);
    this.handleChange = this.handleChange.bind(this);
    this.handleBoolChange = this.handleBoolChange.bind(this);
    this.selectFile = this.selectFile.bind(this);
    this.inputDropHandler = this.inputDropHandler.bind(this);
    this.onArchiveDragDrop = this.onArchiveDragDrop.bind(this);
    this.dragEnterHandler = this.dragEnterHandler.bind(this);
    this.dragLeaveHandler = this.dragLeaveHandler.bind(this);
    this.formRef = React.createRef(); // For dragging CSS
    this.dragDepth = 0; // To determine Form dragging CSS
  }

  async onArchiveDragDrop(event) {
    /** Handle drag-drop of datastack JSON files and InVEST logfiles */
    event.preventDefault();
    event.stopPropagation();
    // No longer dragging so reset dragging depth and remove CSS
    this.dragDepth = 0;
    const formElement = this.formRef.current;
    formElement.classList.remove('dragging');

    const fileList = event.dataTransfer.files;
    if (fileList.length !== 1) {
      alert(_('Only drop one file at a time.')); // eslint-disable-line no-alert
      return;
    }
    this.props.loadParametersFromFile(fileList[0].path);
  }

  /** Handle drag enter events for the Form elements. */
  dragEnterHandler(event) {
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = 'copy';
    this.dragDepth += 1;
    const formElement = this.formRef.current;
    if (!formElement.classList.contains('dragging')) {
      formElement.classList.add('dragging');
    }
  }

  /** Handle drag leave events for the Form elements. */
  dragLeaveHandler(event) {
    event.preventDefault();
    event.stopPropagation();
    this.dragDepth -= 1;
    const formElement = this.formRef.current;
    if (this.dragDepth <= 0) {
      formElement.classList.remove('dragging');
    }
  }

  /** Handle drop events for input elements from the ArgInput components. */
  inputDropHandler(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.remove('input-dragging');
    // Don't take any action on disabled elements
    if (event.currentTarget.disabled) {
      return;
    }
    const { name } = event.currentTarget; // the arg's key and type
    // TODO: could add more filters based on argType (e.g. only show .csv)
    const fileList = event.dataTransfer.files;
    if (fileList.length !== 1) {
      alert(_('Only drop one file at a time.')); // eslint-disable-line no-alert
    } else if (fileList.length === 1) {
      this.props.updateArgValues(name, fileList[0].path);
    } else {
      throw new Error('Error handling input file drop');
    }
  }

  handleFocus(event) {
    const { name } = event.currentTarget;
    this.props.updateArgTouched(name);
  }

  handleChange(event) {
    /** Pass input value up to SetupTab for storage & validation. */
    const { name, value } = event.currentTarget;
    this.props.updateArgValues(name, value);
  }

  handleBoolChange(event) {
    /** Handle changes from boolean inputs that submit strings */
    const { name, value } = event.currentTarget;
    const boolVal = boolStringToBoolean(value);
    this.props.updateArgValues(name, boolVal);
  }

  async selectFile(event) {
    /** Handle clicks on browse-button inputs */
    const { name, value } = event.currentTarget; // the arg's key and type
    const prop = (value === 'directory') ? 'openDirectory' : 'openFile';
    // TODO: could add more filters based on argType (e.g. only show .csv)
    const data = await ipcRenderer.invoke(
      ipcMainChannels.SHOW_OPEN_DIALOG, { properties: [prop] }
    );
    if (data.filePaths.length) {
      // dialog defaults allow only 1 selection
      this.props.updateArgValues(name, data.filePaths[0]);
    }
  }

  render() {
    const {
      argsOrder,
      argsSpec,
      argsValues,
      argsValidation,
      argsEnabled,
      argsDropdownOptions,
      userguide,
    } = this.props;
    const formItems = [];
    let k = 0;
    argsOrder.forEach((groupArray) => {
      k += 1;
      const groupItems = [];
      groupArray.forEach((argkey) => {
        groupItems.push(
          <ArgInput
            argkey={argkey}
            argSpec={argsSpec[argkey]}
            userguide={userguide}
            dropdownOptions={argsDropdownOptions[argkey]}
            enabled={argsEnabled[argkey]}
            handleBoolChange={this.handleBoolChange}
            handleChange={this.handleChange}
            handleFocus={this.handleFocus}
            inputDropHandler={this.inputDropHandler}
            isValid={argsValidation[argkey].valid}
            key={argkey}
            selectFile={this.selectFile}
            touched={argsValues[argkey].touched}
            validationMessage={argsValidation[argkey].validationMessage}
            value={argsValues[argkey].value}
          />
        );
      });
      formItems.push(
        <div className="arg-group" key={k}>
          {groupItems}
        </div>
      );
    });

    return (
      <Form
        ref={this.formRef}
        data-testid="setup-form"
        className="args-form"
        validated={false}
        onDrop={this.onArchiveDragDrop}
        onDragOver={dragOverHandler}
        onDragEnter={this.dragEnterHandler}
        onDragLeave={this.dragLeaveHandler}
      >
        {formItems}
      </Form>
    );
  }
}

ArgsForm.propTypes = {
  argsValues: PropTypes.objectOf(
    PropTypes.shape({
      value: PropTypes.oneOfType([PropTypes.string, PropTypes.bool]),
      touched: PropTypes.bool,
    })
  ).isRequired,
  argsValidation: PropTypes.objectOf(
    PropTypes.shape({
      validationMessage: PropTypes.string,
      valid: PropTypes.bool,
    })
  ).isRequired,
  argsSpec: PropTypes.objectOf(
    PropTypes.shape({
      name: PropTypes.string,
      type: PropTypes.string,
    })
  ).isRequired,
  argsOrder: PropTypes.arrayOf(
    PropTypes.arrayOf(PropTypes.string)
  ).isRequired,
  userguide: PropTypes.string.isRequired,
  updateArgValues: PropTypes.func.isRequired,
  loadParametersFromFile: PropTypes.func.isRequired,
};
