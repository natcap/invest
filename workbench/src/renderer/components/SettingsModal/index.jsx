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

import { getDefaultSettings } from './SettingsStorage';
import { ipcMainChannels } from '../../../main/ipcMainChannels';
import { getSupportedLanguages } from '../../server_requests';

import { Translation } from 'react-i18next';
import { Trans } from 'react-i18next';
import { withTranslation } from 'react-i18next';

const { ipcRenderer } = window.Workbench.electron;

const logLevelOptions = ['DEBUG', 'INFO', 'WARNING', 'ERROR'];

/** Render a dialog with a form for configuring global invest settings */
class SettingsModal extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      show: false,
      nWorkersOptions: null,
      languageOptions: null,
    };
    this.handleShow = this.handleShow.bind(this);
    this.handleClose = this.handleClose.bind(this);
    this.handleChange = this.handleChange.bind(this);
    this.handleReset = this.handleReset.bind(this);
    this.switchToDownloadModal = this.switchToDownloadModal.bind(this);
  }

  async componentDidMount() {
    const nWorkersOptions = [];
    nWorkersOptions.push([-1, 'Synchronous (-1)']);
    nWorkersOptions.push([0, 'Threaded task management (0)']);
    for (let i = 1; i <= this.props.nCPU; i += 1) {
      nWorkersOptions.push([i, `${i} CPUs`]);
    }
    const languageOptions = await getSupportedLanguages();
    this.setState({
      nWorkersOptions: nWorkersOptions,
      languageOptions: languageOptions,
    });
  }

  handleClose() {
    this.setState({
      show: false,
    });
  }

  handleShow() {
    this.setState({ show: true });
  }

  handleReset(event) {
    event.preventDefault();
    const resetSettings = getDefaultSettings();
    this.props.saveSettings(resetSettings);
  }

  handleChange(event) {
    const newSettings = { ...this.props.investSettings };
    const { name, value } = event.currentTarget;
    newSettings[name] = value;
    this.props.saveSettings(newSettings);
  }

  switchToDownloadModal() {
    this.props.showDownloadModal();
    this.handleClose();
  }

  render() {
    const { show, nWorkersOptions, languageOptions } = this.state;
    const { investSettings, clearJobsStorage, t } = this.props;
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
                    <Form.Text className="text-nowrap" muted>
                      <MdWarningAmber className="align-text-bottom ml-3" />
                      {t('Changing this setting will refresh the app and close all tabs')}
                    </Form.Text>
                  </Form.Label>
                  <Col sm="4">
                    <Form.Control
                      id="language-select"
                      as="select"
                      name="language"
                      value={investSettings.language}
                      onChange={this.handleChange}
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
                  value={investSettings.loggingLevel}
                  onChange={this.handleChange}
                >
                  {logLevelOptions.map(
                    (opt) => <option value={opt} key={opt}>{t(opt)}</option>
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
                  value={investSettings.taskgraphLoggingLevel}
                  onChange={this.handleChange}
                >
                  {logLevelOptions.map(
                    (opt) => <option value={opt} key={opt}>{t(opt)}</option>
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
                        value={investSettings.nWorkers}
                        onChange={this.handleChange}
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
                        <span className="small"><u>more info</u></span>
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
            <Row className="justify-content-end">
              <Col sm="5">
                <Button
                  variant="secondary"
                  onClick={this.handleReset}
                  type="button"
                  className="w-100"
                >
                  {t('Reset to Defaults')}
                </Button>
              </Col>
            </Row>
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
      </React.Fragment>
    );
  }
}

export default withTranslation()(SettingsModal);

SettingsModal.propTypes = {
  saveSettings: PropTypes.func.isRequired,
  clearJobsStorage: PropTypes.func.isRequired,
  investSettings: PropTypes.shape({
    nWorkers: PropTypes.string,
    taskgraphLoggingLevel: PropTypes.string,
    loggingLevel: PropTypes.string,
    sampleDataDir: PropTypes.string,
    language: PropTypes.string,
  }).isRequired,
  showDownloadModal: PropTypes.func.isRequired,
  nCPU: PropTypes.number.isRequired,
};
