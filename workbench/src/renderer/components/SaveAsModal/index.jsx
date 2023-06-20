import React from 'react';

import Button from 'react-bootstrap/Button';
import ToggleButton from 'react-bootstrap/ToggleButton';
import ButtonGroup from 'react-bootstrap/ButtonGroup';
import Form from 'react-bootstrap/Form';
import Modal from 'react-bootstrap/Modal';
import { MdSave, MdClose } from 'react-icons/md';
import { withTranslation } from 'react-i18next';

import { ipcMainChannels } from '../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;


/** Render a dialog with a form for configuring global invest settings */
class SaveAsModal extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      show: false,
      datastackType: 'json', // default to the JSON option
      relativePaths: false,
    };
    this.handleShow = this.handleShow.bind(this);
    this.handleClose = this.handleClose.bind(this);
    this.handleChange = this.handleChange.bind(this);
    this.handleRelativePathsCheckbox = this.handleRelativePathsCheckbox.bind(this);
    this.browseSaveFile = this.browseSaveFile.bind(this);
  }

  async browseSaveFile(event) {
    const {
      modelName,
      saveJsonFile,
      saveDatastack,
      savePythonScript
    } = this.props;
    const { datastackType, relativePaths } = this.state;
    const defaultTargetPaths = {
      json: `invest_${modelName}_args.json`,
      tgz: `invest_${modelName}_datastack.tgz`,
      py: `execute_invest_${modelName}.py`,
    };

    const data = await ipcRenderer.invoke(
      ipcMainChannels.SHOW_SAVE_DIALOG,
      { defaultPath: defaultTargetPaths[datastackType] }
    );
    if (data.filePath) {
      switch (datastackType) {
        case "json":
          saveJsonFile(data.filePath, relativePaths);
          break;
        case "tgz":
          saveDatastack(data.filePath);
          break;
        case "py":
          savePythonScript(data.filePath);
      }
    }
    this.handleClose();
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

  render() {
    const { show, datastackType } = this.state;
    const { t } = this.props;

    return (
      <React.Fragment>
        <Button
          aria-label="save-as"
          variant="link"
          onClick={this.handleShow}
        >
          <MdSave className="mr-1" />
          {t("Save as...")}
        </Button>

        <Modal
          className="save-as-modal"
          show={show}
          onHide={this.handleClose}
        >
          <Modal.Header>
            <Modal.Title>{t('Datastack options')}</Modal.Title>
            <Button
              variant="secondary-outline"
              onClick={this.handleClose}
              className="float-right"
              aria-label="close save-as dialog"
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
                  {t('Save your parameters in a JSON file. This includes the ' +
                     'paths to your input data, but not the data itself. ' +
                     'Open this file in InVEST to restore your parameters.')}
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
                  {t('Save your parameters and input data in a compressed archive. ' +
                     'This archive contains the same JSON file produced by the ' +
                     '"Parameters only" option, plus the data. You can open this ' +
                     'file in InVEST to restore your parameters. This option is ' +
                     'useful to copy all the necessary data to a different location.')}
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
                  {t('Save your parameters in a python script. This includes the ' +
                     'paths to your input data, but not the data itself. Running ' +
                     'the python script will programmatically run the model with ' +
                     'your parameters. Use this as a starting point for batch scripts.')}
                </Form.Text>
              </ToggleButton>
            </ButtonGroup>
            <Button onClick={this.browseSaveFile}>
              <MdSave className="mr-1" />
              {t('Save')}
            </Button>
          </Modal.Body>
        </Modal>
      </React.Fragment>
    );
  }
}

export default withTranslation()(SaveAsModal);
