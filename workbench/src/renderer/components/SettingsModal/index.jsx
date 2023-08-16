import React from 'react';
import PropTypes from 'prop-types';

import Accordion from 'react-bootstrap/Accordion';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import {
  MdSettings,
  MdClose,
  MdTranslate,
  MdWarningAmber,
} from 'react-icons/md';
import { BsChevronExpand } from 'react-icons/bs';
import { withTranslation } from 'react-i18next';

import { ipcMainChannels } from '../../../main/ipcMainChannels';
import { getSupportedLanguages } from '../../server_requests';

const { ipcRenderer } = window.Workbench.electron;

/** Render a dialog with a form for configuring global invest settings */
class SettingsModal extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      show: false,
      languageOptions: null,
      loggingLevel: null,
      taskgraphLoggingLevel: null,
      nWorkers: null,
      language: window.Workbench.LANGUAGE,
      showConfirmLanguageChange: false,
    };
    this.handleShow = this.handleShow.bind(this);
    this.handleClose = this.handleClose.bind(this);
    this.handleChange = this.handleChange.bind(this);
    this.handleChangeNumber = this.handleChangeNumber.bind(this);
    this.loadSettings = this.loadSettings.bind(this);
    this.handleChangeLanguage = this.handleChangeLanguage.bind(this);
    this.switchToDownloadModal = this.switchToDownloadModal.bind(this);
  }

  async componentDidMount() {
    const languageOptions = await getSupportedLanguages();
    this.setState({
      languageOptions: languageOptions,
    });
    this.loadSettings();
  }

  handleClose() {
    this.setState({
      show: false,
    });
  }

  handleShow() {
    this.setState({ show: true });
  }

  handleChange(event) {
    const { name, value } = event.currentTarget;
    this.setState({ [name]: value });
    ipcRenderer.send(ipcMainChannels.SET_SETTING, name, value);
  }

  handleChangeNumber(event) {
    const { name, value } = event.currentTarget;
    const numeral = Number(value);
    this.setState({ [name]: numeral });
    ipcRenderer.send(ipcMainChannels.SET_SETTING, name, numeral);
  }

  async loadSettings() {
    const loggingLevel = await ipcRenderer
      .invoke(ipcMainChannels.GET_SETTING, 'loggingLevel');
    const taskgraphLoggingLevel = await ipcRenderer
      .invoke(ipcMainChannels.GET_SETTING, 'taskgraphLoggingLevel');
    const nWorkers = await ipcRenderer
      .invoke(ipcMainChannels.GET_SETTING, 'nWorkers');
    this.setState({
      loggingLevel: loggingLevel,
      taskgraphLoggingLevel: taskgraphLoggingLevel,
      nWorkers: nWorkers
    });
  }

  handleChangeLanguage() {
    // if language has changed, refresh the app
    if (this.state.language !== window.Workbench.LANGUAGE) {
      // tell the main process to update the language setting in storage
      // and then relaunch the app
      ipcRenderer.invoke(ipcMainChannels.CHANGE_LANGUAGE, this.state.language);
    }
  }

  switchToDownloadModal() {
    this.props.showDownloadModal();
    this.handleClose();
  }

  render() {
    const {
      show,
      languageOptions,
      language,
      loggingLevel,
      taskgraphLoggingLevel,
      nWorkers,
      showConfirmLanguageChange,
    } = this.state;
    const { clearJobsStorage, nCPU, t } = this.props;

    const nWorkersOptions = [
      [-1, `${t('Synchronous')} (-1)`],
      [0, `${t('Threaded task management')} (0)`],
    ];
    for (let i = 1; i <= nCPU; i += 1) {
      nWorkersOptions.push([i, `${i} ${t('CPUs')}`]);
    }
    const logLevelOptions = { // map value to display name
      DEBUG: t('DEBUG'),
      INFO: t('INFO'),
      WARNING: t('WARNING'),
      ERROR: t('ERROR'),
    };
    return (
      <React.Fragment>
        <Button
          aria-label="settings"
          className="settings-icon-btn"
          onClick={this.handleShow}
        >
          <MdSettings
            className="settings-icon"
          />
        </Button>

        <Modal
          className="settings-modal"
          show={show}
          onHide={this.handleClose}
        >
          <Modal.Header>
            <Modal.Title>{t('InVEST Settings')}</Modal.Title>
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
            {
              (languageOptions) ? (
                <Form.Group as={Row}>
                  <Form.Label column sm="8" htmlFor="language-select">
                    <MdTranslate className="language-icon" />
                    {t('Language')}
                  </Form.Label>
                  <Col sm="4">
                    <Form.Control
                      id="language-select"
                      as="select"
                      name="language"
                      value={window.Workbench.LANGUAGE}
                      onChange={
                        (event) => this.setState({
                          showConfirmLanguageChange: true,
                          language: event.target.value
                        })}
                    >
                      {Object.entries(languageOptions).map((entry) => {
                        const [value, displayName] = entry;
                        return <option value={value} key={value}>{displayName}</option>;
                      })}
                    </Form.Control>
                  </Col>
                </Form.Group>
              ) : <React.Fragment />
            }
            <Form.Group as={Row}>
              <Form.Label column sm="6" htmlFor="logging-select">
                {t('Logging threshold')}
              </Form.Label>
              <Col sm="6">
                <Form.Control
                  id="logging-select"
                  as="select"
                  name="loggingLevel"
                  value={loggingLevel}
                  onChange={this.handleChange}
                >
                  {Object.entries(logLevelOptions).map(
                    ([opt, displayName]) => <option value={opt} key={opt}>{displayName}</option>
                  )}
                </Form.Control>
              </Col>
            </Form.Group>
            <Form.Group as={Row}>
              <Form.Label column sm="6" htmlFor="taskgraph-logging-select">
                {t('Taskgraph logging threshold')}
              </Form.Label>
              <Col sm="6">
                <Form.Control
                  id="taskgraph-logging-select"
                  as="select"
                  name="taskgraphLoggingLevel"
                  value={taskgraphLoggingLevel}
                  onChange={this.handleChange}
                >
                  {Object.entries(logLevelOptions).map(
                    ([opt, displayName]) => <option value={opt} key={opt}>{displayName}</option>
                  )}
                </Form.Control>
              </Col>
            </Form.Group>
            {
              (nWorkersOptions)
                ? (
                  <Form.Group as={Row}>
                    <Col sm="6">
                      <Form.Label htmlFor="nworkers-select">
                        {t('Taskgraph n_workers parameter')}
                      </Form.Label>
                    </Col>
                    <Col sm="6">
                      <Form.Control
                        id="nworkers-select"
                        as="select"
                        name="nWorkers"
                        type="text"
                        value={nWorkers}
                        onChange={this.handleChangeNumber}
                      >
                        {nWorkersOptions.map(
                          (opt) => <option value={opt[0]} key={opt[0]}>{opt[1]}</option>
                        )}
                      </Form.Control>
                    </Col>
                    <Accordion>
                      <Accordion.Toggle
                        as={Button}
                        variant="secondary-outline"
                        eventKey="0"
                        className="pt-0"
                      >
                        <BsChevronExpand className="mx-1" />
                        <span className="small"><u>{t('more info')}</u></span>
                      </Accordion.Toggle>
                      <Accordion.Collapse eventKey="0" className="pr-1">
                        <ul>
                          <li>{t('synchronous task execution is most reliable')}</li>
                          <li>
                            {t('threaded task management: tasks execute only ' +
                               'in the main process, using multiple threads.')}
                          </li>
                          <li>
                            {t('n CPUs: depending on the InVEST model, tasks ' +
                               'may execute in parallel using up to this many processes.')}
                          </li>
                        </ul>
                      </Accordion.Collapse>
                    </Accordion>
                  </Form.Group>
                )
                : <div />
            }
            <hr />
            <Button
              variant="primary"
              onClick={this.switchToDownloadModal}
              className="w-50"
            >
              {t('Download Sample Data')}
            </Button>
            <hr />
            <Button
              variant="secondary"
              onClick={clearJobsStorage}
              className="mr-2 w-50"
            >
              {t('Clear Recent Jobs')}
            </Button>
            <span>{t('no invest workspaces will be deleted')}</span>
          </Modal.Body>
        </Modal>
        {
          (languageOptions) ? (
            <Modal show={showConfirmLanguageChange} className="confirm-modal" >
              <Modal.Header>
                <Modal.Title as="h5" >{t('Warning')}</Modal.Title>
              </Modal.Header>
              <Modal.Body>
                <p>
                  {t('Changing this setting will close your tabs and relaunch the app.')}
                </p>
              </Modal.Body>
              <Modal.Footer>
                <Button
                  variant="secondary"
                  onClick={() => this.setState({ showConfirmLanguageChange: false })}
                >{t('Cancel')}</Button>
                <Button
                  variant="primary"
                  onClick={this.handleChangeLanguage}
                >{t('Change to ') + languageOptions[language]}</Button>
              </Modal.Footer>
            </Modal>
          ) : <React.Fragment />
        }
      </React.Fragment>
    );
  }
}

SettingsModal.propTypes = {
  clearJobsStorage: PropTypes.func.isRequired,
  showDownloadModal: PropTypes.func.isRequired,
  nCPU: PropTypes.number.isRequired,
};

export default withTranslation()(SettingsModal);
