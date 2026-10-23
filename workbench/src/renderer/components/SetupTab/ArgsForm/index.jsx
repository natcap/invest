import React from 'react';
import PropTypes from 'prop-types';

import Form from 'react-bootstrap/Form';

import ArgInput from '../ArgInput';
import { ipcMainChannels } from '../../../../main/ipcMainChannels';
import { withTranslation } from 'react-i18next';
import { getVectorBoundingBox } from '../../../server_requests';
import { DataHubSearchSiblingType } from '../../DataHubSearchModal/models';

const { getFilePath, ipcRenderer } = window.Workbench.electron;

/** Prevent the default case for onDragOver so onDrop event will be fired. */
function dragOverHandler(event) {
  event.preventDefault();
  event.stopPropagation();
  event.dataTransfer.dropEffect = 'copy';
}

/** Renders a form with a list of input components. */
class ArgsForm extends React.Component {
  constructor(props) {
    super(props);
    this.handleFocus = this.handleFocus.bind(this);
    this.selectFile = this.selectFile.bind(this);
    this.inputDropHandler = this.inputDropHandler.bind(this);
    this.onArchiveDragDrop = this.onArchiveDragDrop.bind(this);
    this.dragEnterHandler = this.dragEnterHandler.bind(this);
    this.dragLeaveHandler = this.dragLeaveHandler.bind(this);
    this.formRef = React.createRef(); // For dragging CSS
    this.dragDepth = 0; // To determine Form dragging CSS
    this.state = {
      focusOnAoiRequested: false,
      readyToFocusOnAoi: false,
      searchExtent: [],
      searchExtentUpdateComplete: false,
      searchCollections: {
        [DataHubSearchSiblingType.LULC]: [],
        [DataHubSearchSiblingType.BIOPHYSICAL_TABLE]: [],
      },
    };
  }

  async onArchiveDragDrop(event) {
    /** Handle drag-drop of datastack JSON files and InVEST logfiles */
    event.preventDefault();
    event.stopPropagation();
    // No longer dragging so reset dragging depth and remove CSS
    this.dragDepth = 0;
    const formElement = this.formRef.current;
    formElement.classList.remove('dragging');

    const { loadParametersFromFile, t } = this.props;
    const fileList = event.dataTransfer.files;
    if (fileList.length !== 1) {
      alert(t('Only drop one file at a time.')); // eslint-disable-line no-alert
      return;
    }
    loadParametersFromFile(getFilePath(fileList[0]));
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
    const { triggerScrollEvent, updateArgValues, t } = this.props;
    if (fileList.length !== 1) {
      alert(t('Only drop one file at a time.')); // eslint-disable-line no-alert
    } else if (fileList.length === 1) {
      updateArgValues(name, getFilePath(fileList[0]));
    } else {
      throw new Error('Error handling input file drop');
    }
    event.currentTarget.focus();
    triggerScrollEvent();
  }

  handleFocus(event) {
    const { name } = event.currentTarget;
    this.props.updateArgTouched(name);
  }

  async selectFile(event) {
    /** Handle clicks on browse-button inputs */
    const { name, value } = event.currentTarget; // the arg's key and type
    const prop = (value === 'workspace') ? 'openDirectory' : 'openFile';
    // TODO: could add more filters based on argType (e.g. only show .csv)
    const data = await ipcRenderer.invoke(
      ipcMainChannels.SHOW_OPEN_DIALOG, { properties: [prop] }
    );
    if (data.filePaths.length) {
      // dialog defaults allow only 1 selection
      this.props.updateArgValues(name, data.filePaths[0]);
      this.props.updateArgTouched(name);
      this.props.triggerScrollEvent();
    }
  }

  /* Some local state to support handling of calls to updateSearchExtent. */
  currentBoundingBox = [];
  pendingAoiPath = '';
  aoiBboxCallStack = [];

  /**
   * @function updateSearchExtentState
   * If the provided timestamp is at the top of the call stack,
   * update search extent-related state and clear call stack.
   * Otherwise, do nothing.
   *
   * @param {number} timestamp
   * @param {array[number]} bbox
   *
   * @returns {void}
   */
  updateSearchExtentState = (timestamp, bbox) => {
    const stackSize = this.aoiBboxCallStack.length;
    if (stackSize && this.aoiBboxCallStack[stackSize - 1] === timestamp) {
      console.log(`Updating state and clearing call stack…`);
      this.aoiBboxCallStack = [];
      this.currentBoundingBox = bbox;
      this.setState({
        searchExtent: this.currentBoundingBox,
        searchExtentUpdateComplete: true,
      });
    // @TODO: Remove else block when done testing.
    } else {
      console.log('Ignoring outdated call.');
    }
  };

  /**
   * @function updateSearchExtent
   * Handle request to update the extent parameter in a Data Hub search query.
   * Depending on context, either set extent to [] and update state,
   * fetch the current AOI's bounding box and update state, or do nothing.
   *
   * @returns {void}
   */
  updateSearchExtent = () => {
    const { aoiInputId, argsValidation, argsValidationComplete, argsValues } = this.props;
    const aoiIsValid = argsValidation[aoiInputId]?.valid || false;
    const aoiPath = argsValues[aoiInputId]?.value || '';

    // If awaiting validation
    // OR if awaiting a bounding box calculation for this aoiPath
    // OR if AOI is invalid and bounding box is already [],
    // do nothing.
    if (
      !argsValidationComplete
      || aoiPath === this.pendingAoiPath
      || (!aoiIsValid && !this.currentBoundingBox.length)
    ) {
      console.log('Bbox is unchanged, or bbox calculation for this aoiPath is in progress. Goodbye.');
      return;
    }

    // Signal that an update is in progress.
    this.setState({
      searchExtentUpdateComplete: false,
    });

    // Store timestamp in call stack.
    // State will be updated only if this is the latest call.
    const timestamp = Date.now();
    this.aoiBboxCallStack.push(timestamp);
    console.log(this.aoiBboxCallStack);

    if (!aoiIsValid && this.currentBoundingBox.length) {
      this.updateSearchExtentState(timestamp, []);
    } else if (aoiPath !== this.pendingAoiPath) {
      this.pendingAoiPath = aoiPath;
      console.log(`Fetching bounding box for ${aoiPath}…`);
      getVectorBoundingBox(
        { vector_path: aoiPath }
      ).then(({ vector_bbox }) => {
        this.updateSearchExtentState(timestamp, vector_bbox);
        this.pendingAoiPath = '';
      });
      // Note: state is updated only when handling the response to the latest
      // call to getVectorBoundingBox. Other (outdated) responses are ignored.
      // It could be useful to cache outdated responses in case they're needed
      // later. But since that adds significant complexity and could have
      // unwanted side effects, we decided it's not worth implementing
      // until/unless there's an obvious need.
    }
  };

  selectSearchResult = (argkey, url, collections, siblingType) => {
    this.props.updateArgValues(argkey, url);
    this.props.updateArgTouched(argkey);
    this.props.triggerScrollEvent();
    if (
      siblingType === DataHubSearchSiblingType.LULC
      || siblingType === DataHubSearchSiblingType.BIOPHYSICAL_TABLE
    ) {
      this.setState({
        ...this.state,
        searchCollections: {
          ...this.state.searchCollections,
          [siblingType]: collections,
        },
      });
    }
  };

  requestFocusOnAoiInput = () => {
    this.setState({
      ...this.state,
      focusOnAoiRequested: true,
    });
  };

  setReadyToFocusOnAoi = (newState) => {
    this.setState({
      ...this.state,
      readyToFocusOnAoi: newState,
    });
  };

  resetAoiFocusState = () => {
    this.setState({
      ...this.state,
      focusOnAoiRequested: false,
      readyToFocusOnAoi: false,
    });
  };

  getSiblingType = (argkey) => {
    // These hard-coded values support AWY, NDR, SDR, and UFRM.
    // @TODO: add support for all search-enabled models (via model spec, probably).
    if (argkey === 'lulc_path') {
      return DataHubSearchSiblingType.LULC;
    } else if (argkey === 'biophysical_table_path') {
      return DataHubSearchSiblingType.BIOPHYSICAL_TABLE;
    }
    return DataHubSearchSiblingType.NONE;
  };

  getSiblingCollections = (siblingType) => {
    if (siblingType === DataHubSearchSiblingType.LULC) {
      return this.state.searchCollections[DataHubSearchSiblingType.BIOPHYSICAL_TABLE];
    } else if (siblingType === DataHubSearchSiblingType.BIOPHYSICAL_TABLE) {
      return this.state.searchCollections[DataHubSearchSiblingType.LULC];
    }
    return [];
  };

  clearSearchCollections = (siblingType) => {
    if (siblingType === DataHubSearchSiblingType.LULC || siblingType === DataHubSearchSiblingType.BIOPHYSICAL_TABLE) {
      this.setState({
        ...this.state,
        searchCollections: {
          ...this.state.searchCollections,
          [siblingType]: [],
        },
      });
    }
  };

  render() {
    const {
      argsOrder,
      argsSpec,
      argsValues,
      argsValidation,
      argsValidationComplete,
      argsEnabled,
      argsDropdownOptions,
      userguide,
      isCoreModel,
      scrollEventCount,
      aoiInputId,
    } = this.props;
    const formItems = [];
    let k = 0;
    argsOrder.forEach((groupArray) => {
      k += 1;
      const groupItems = [];
      groupArray.forEach((argkey) => {
        const siblingType = this.getSiblingType(argkey);
        const siblingCollections = this.getSiblingCollections(siblingType);
        groupItems.push(
          <ArgInput
            argkey={argkey}
            argSpec={argsSpec[argkey]}
            userguide={userguide}
            isCoreModel={isCoreModel}
            dropdownOptions={argsDropdownOptions[argkey]}
            enabled={argsEnabled[argkey]}
            updateArgValues={this.props.updateArgValues}
            handleFocus={this.handleFocus}
            inputDropHandler={this.inputDropHandler}
            isValid={argsValidation[argkey].valid}
            key={argkey}
            selectFile={this.selectFile}
            touched={argsValues[argkey].touched}
            validationMessage={argsValidation[argkey].validationMessage}
            value={argsValues[argkey].value}
            scrollEventCount={scrollEventCount}
            aoiInputName={argsSpec[aoiInputId]?.name || ''}
            argsValidationComplete={argsValidationComplete}
            aoiIsValid={argsValidation[aoiInputId]?.valid || false}
            searchExtent={this.state.searchExtent}
            searchExtentUpdateComplete={this.state.searchExtentUpdateComplete}
            updateSearchExtent={this.updateSearchExtent}
            searchCollections={siblingCollections}
            clearSearchCollections={this.clearSearchCollections}
            searchSiblingType={siblingType}
            selectSearchResult={this.selectSearchResult}
            requestFocusOnAoiInput={this.requestFocusOnAoiInput}
            setReadyToFocusOnAoi={this.setReadyToFocusOnAoi}
            resetAoiFocusState={this.resetAoiFocusState}
            autoFocus={(argkey === aoiInputId) && this.state.readyToFocusOnAoi && this.state.focusOnAoiRequested}
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
  argsEnabled: PropTypes.objectOf(PropTypes.bool),
  userguide: PropTypes.string.isRequired,
  isCoreModel: PropTypes.bool.isRequired,
  updateArgValues: PropTypes.func.isRequired,
  loadParametersFromFile: PropTypes.func.isRequired,
  scrollEventCount: PropTypes.number,
  triggerScrollEvent: PropTypes.func.isRequired,
};

ArgsForm.defaultProps = {
  scrollEventCount: 0,
};

export default withTranslation()(ArgsForm);
