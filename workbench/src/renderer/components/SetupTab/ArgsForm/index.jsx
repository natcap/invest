import React from 'react';
import PropTypes from 'prop-types';

import Form from 'react-bootstrap/Form';

import ArgInput from '../ArgInput';
import { ipcMainChannels } from '../../../../main/ipcMainChannels';
import { withTranslation } from 'react-i18next';

const { ipcRenderer } = window.Workbench.electron;

/** Prevent the default case for onDragOver so onDrop event will be fired. */
function dragOverHandler(event) {
  event.preventDefault();
  event.stopPropagation();
  event.dataTransfer.dropEffect = 'copy';
}

/** Renders a form with a list of input components. */
function  ArgsForm(props) {
  const formRef = React.useRef(null);
  let dragDepth = 0;

  async function onArchiveDragDrop(event) {
    /** Handle drag-drop of datastack JSON files and InVEST logfiles */
    event.preventDefault();
    event.stopPropagation();
    // No longer dragging so reset dragging depth and remove CSS
    dragDepth = 0;
    const formElement = formRef.current;
    formElement.classList.remove('dragging');

    const { loadParametersFromFile, t } = props;
    const fileList = event.dataTransfer.files;
    if (fileList.length !== 1) {
      alert(t('Only drop one file at a time.')); // eslint-disable-line no-alert
      return;
    }
    loadParametersFromFile(fileList[0].path);
  }

  /** Handle drag enter events for the Form elements. */
  function dragEnterHandler(event) {
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = 'copy';
    dragDepth += 1;
    const formElement = formRef.current;
    if (!formElement.classList.contains('dragging')) {
      formElement.classList.add('dragging');
    }
  }

  /** Handle drag leave events for the Form elements. */
  function dragLeaveHandler(event) {
    event.preventDefault();
    event.stopPropagation();
    dragDepth -= 1;
    const formElement = formRef.current;
    if (dragDepth <= 0) {
      formElement.classList.remove('dragging');
    }
  }

  /** Handle drop events for input elements from the ArgInput components. */
  function inputDropHandler(event) {
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
    const { triggerScrollEvent, updateArgValues, t } = props;
    if (fileList.length !== 1) {
      alert(t('Only drop one file at a time.')); // eslint-disable-line no-alert
    } else if (fileList.length === 1) {
      updateArgValues(name, fileList[0].path);
    } else {
      throw new Error('Error handling input file drop');
    }
    event.currentTarget.focus();
    triggerScrollEvent();
  }

  function handleFocus(event) {
    const { name } = event.currentTarget;
    props.updateArgTouched(name);
  }

  async function selectFile(event) {
    /** Handle clicks on browse-button inputs */
    const { name, value } = event.currentTarget; // the arg's key and type
    const prop = (value === 'directory') ? 'openDirectory' : 'openFile';
    // TODO: could add more filters based on argType (e.g. only show .csv)
    const data = await ipcRenderer.invoke(
      ipcMainChannels.SHOW_OPEN_DIALOG, { properties: [prop] }
    );
    if (data.filePaths.length) {
      // dialog defaults allow only 1 selection
      props.updateArgValues(name, data.filePaths[0]);
      props.triggerScrollEvent();
    }
  }

  const {
    argsOrder,
    argsSpec,
    argsValues,
    argsValidation,
    argsEnabled,
    argsDropdownOptions,
    userguide,
    scrollEventCount,
  } = props;
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
          updateArgValues={props.updateArgValues}
          handleFocus={handleFocus}
          inputDropHandler={inputDropHandler}
          isValid={argsValidation[argkey].valid}
          key={argkey}
          selectFile={selectFile}
          touched={argsValues[argkey].touched}
          validationMessage={argsValidation[argkey].validationMessage}
          value={argsValues[argkey].value}
          scrollEventCount={scrollEventCount}
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
      ref={formRef}
      data-testid="setup-form"
      className="args-form"
      validated={false}
      onDrop={onArchiveDragDrop}
      onDragOver={dragOverHandler}
      onDragEnter={dragEnterHandler}
      onDragLeave={dragLeaveHandler}
    >
      {formItems}
    </Form>
  );
}

ArgsForm.propTypes = {
  argsValues: PropTypes.objectOf(
    PropTypes.shape({
      value: PropTypes.oneOfType(
        [PropTypes.string, PropTypes.bool, PropTypes.number]),
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
  scrollEventCount: PropTypes.number,
  triggerScrollEvent: PropTypes.func.isRequired,
};

ArgsForm.defaultProps = {
  scrollEventCount: 0,
};

export default withTranslation()(ArgsForm);
