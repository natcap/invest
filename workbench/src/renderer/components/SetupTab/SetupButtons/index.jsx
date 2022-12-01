import React from 'react';

import Button from 'react-bootstrap/Button';
import ToggleButton from 'react-bootstrap/ToggleButton';
import ButtonGroup from 'react-bootstrap/ButtonGroup';
import ToggleButtonGroup from 'react-bootstrap/ToggleButtonGroup';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Form from 'react-bootstrap/Form';
import Modal from 'react-bootstrap/Modal';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Tooltip from 'react-bootstrap/Tooltip';

import SaveFileButton from '../../SaveFileButton';
import { MdSave, MdSettings, MdClose } from 'react-icons/md';

function HoverText(props) {
  return (
    <OverlayTrigger
      placement="right"
      delay={{ show: 250, hide: 400 }}
      overlay={(
        <Tooltip>
          {props.hoverText}
        </Tooltip>
      )}
    >
      {/* the first child must be a DOM element, not a custom component,
      so that event handlers from OverlayTrigger make it to DOM
      https://github.com/react-bootstrap/react-bootstrap/issues/2208 */}
      <div>
        {props.children}
      </div>
    </OverlayTrigger>
  );
}

/** Render a dialog with a form for configuring global invest settings */
export class SaveAsButton extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      show: false,
      datastackType: 'JSON file',
      relativePaths: false,
    };
    this.handleShow = this.handleShow.bind(this);
    this.handleClose = this.handleClose.bind(this);
    this.handleChange = this.handleChange.bind(this);
    this.handleSave = this.handleSave.bind(this);
    this.handleRelativePathsCheckbox = this.handleRelativePathsCheckbox.bind(this);
  }

  handleClose() {
    this.setState({ show: false });
  }

  handleShow() {
    this.setState({ show: true });
  }

  handleChange(event) {
    const { name, value } = event.currentTarget;
    const newState = { ...this.state };
    newState[name] = value;
    this.setState(newState);
  }

  handleRelativePathsCheckbox(event) {
    const newState = { ...this.state };
    newState.relativePaths = event.target.checked;
    this.setState(newState);
  }

  handleSave(event) {
    const { saveJsonFile, savePythonScript, saveDatastack } = this.props;
    switch (this.state.datastackType) {
      case "json":
        saveJsonFile(relativePaths);
      case "tgz":
        saveDatastack();
      case "py":
        savePythonScript();
    }
  }

  render() {
    const { show, datastackType, relativePaths } = this.state;

    return (
      <React.Fragment>
        <Button
          aria-label="settings"
          className="settings-icon-btn"
          onClick={this.handleShow}
        >
          <MdSave className="mr-1" />
          {_("Save as...")}
        </Button>

        <Modal
          className="settings-modal"
          show={show}
          onHide={this.handleClose}
        >
          <Modal.Header>
            <Modal.Title>{_('Datastack options')}</Modal.Title>
            <Button
              variant="secondary-outline"
              onClick={this.handleClose}
              className="float-right"
              aria-label="close settings"
            >
              <MdClose />
            </Button>
          </Modal.Header>
          <Modal.Body>
            <ButtonGroup vertical>
              <ToggleButton
                type="radio"
                value="json"
                checked={datastackType === "json"}
                name="datastackType"
                className="text-left"
                variant="light"
                onChange={this.handleChange}
              >
              <span className="ml-2">Parameters only</span>
              <Form.Text muted>
                {_(`Save your parameters in a JSON file.
                This includes the paths to your input data, but not the data itself.
                Open this file in InVEST to restore your parameters.`)}
              </Form.Text>
              <Form.Check
                id="relativePaths"
                label="Use relative paths"
                name="relativePaths"
                disabled={datastackType !== "json"}
                onChange={this.handleRelativePathsCheckbox}
              />
              </ToggleButton>
              <ToggleButton
                type="radio"
                value="tgz"
                checked={datastackType === "tgz"}
                name="datastackType"
                className="text-left"
                variant="light"
                onChange={this.handleChange}
              >
              <span className="ml-2">Parameters and data</span>
              <Form.Text muted>
                {_(`Save your parameters and input data in a compressed archive.
                This archive contains the same JSON file produced by the "Parameters
                only" option, plus the data. You can open this file in InVEST to restore your
                parameters. This option is useful to copy all the necessary data to a
                different location.`)}
              </Form.Text>
              </ToggleButton>
              <ToggleButton
                type="radio"
                value="py"
                checked={datastackType === "py"}
                name="datastackType"
                className="text-left"
                variant="light"
                onChange={this.handleChange}
              >
              <span className="ml-2">Python script</span>
              <Form.Text muted>
                {_(`Save your parameters in a python script. This includes the
                paths to your input data, but not the data itself. Running the python script
                will programmatically run the model with your parameters. Use this as a
                starting point for batch scripts.`)}
              </Form.Text>
              </ToggleButton>
            </ButtonGroup>
            <Button
              onClick={this.handleSave}
              className="w-50"
            >
              <MdSave className="mr-1" />
              {_('Save')}
            </Button>
          </Modal.Body>
        </Modal>
      </React.Fragment>
    );
  }
}

export function SaveParametersButtons(props) {
  return (
    <React.Fragment>
      <HoverText
        hoverText={_("Save model setup to a JSON file")}
      >
        <SaveFileButton
          title={_("Save to JSON")}
          defaultTargetPath="invest_args.json"
          func={props.saveJsonFile}
        />
      </HoverText>
      <HoverText
        hoverText={_("Save model setup to a Python script")}
      >
        <SaveFileButton
          title={_("Save to Python script")}
          defaultTargetPath="execute_invest.py"
          func={props.savePythonScript}
        />
      </HoverText>
      <HoverText
        hoverText={_("Export all input data to a compressed archive")}
      >
        <SaveFileButton
          title={_("Save datastack")}
          defaultTargetPath="invest_datastack.tgz"
          func={props.saveDatastack}
        />
      </HoverText>
    </React.Fragment>
  );
}

export function RunButton(props) {
  return (
    <Button
      block
      variant="primary"
      size="lg"
      onClick={props.wrapInvestExecute}
      disabled={props.disabled}
    >
      {props.buttonText}
    </Button>
  );
}
