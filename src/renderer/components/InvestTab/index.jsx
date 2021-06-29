import path from 'path';
import React from 'react';
import PropTypes from 'prop-types';
import { ipcRenderer } from 'electron';

import TabPane from 'react-bootstrap/TabPane';
import TabContent from 'react-bootstrap/TabContent';
import TabContainer from 'react-bootstrap/TabContainer';
import Nav from 'react-bootstrap/Nav';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Spinner from 'react-bootstrap/Spinner';

import SetupTab from '../SetupTab';
import LogTab from '../LogTab';
import ResourcesLinks from '../ResourcesLinks';
import { getSpec } from '../../server_requests';
import { dragOverHandlerNone } from '../../utils';

const logger = window.Workbench.getLogger(__filename.split('/').slice(-1)[0]);

/** Get an invest model's ARGS_SPEC when a model button is clicked.
 *
 * @param {string} modelName - as in a model name appearing in `invest list`
 * @returns {object} destructures to:
 *   { modelSpec, argsSpec, uiSpec }
 */
async function investGetSpec(modelName) {
  const spec = await getSpec(modelName);
  if (spec) {
    const { args, ...modelSpec } = spec;
    const uiSpecs = require('../../ui_config');
    const uiSpec = uiSpecs[modelSpec.model_name];
    if (uiSpec) {
      return { modelSpec: modelSpec, argsSpec: args, uiSpec: uiSpec };
    }
    logger.error(`no UI spec found for ${modelName}`);
  } else {
    logger.error(`no args spec found for ${modelName}`);
  }
  return undefined;
}

/**
 * Render an invest model setup form, log display, etc.
 * Manage launching of an invest model in a child process.
 * And manage saves of executed jobs to a persistent store.
 */
export default class InvestTab extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      activeTab: 'setup',
      modelSpec: null, // ARGS_SPEC dict with all keys except ARGS_SPEC.args
      argsSpec: null, // ARGS_SPEC.args, the immutable args stuff
      logStdErr: null, // stderr data from the invest subprocess
      jobStatus: null, // 'running', 'error', 'success'
      workspaceDir: null,
      logfile: null,
    };

    this.ipcSuffix = window.crypto.getRandomValues(new Uint16Array(1))[0];
    this.investExecute = this.investExecute.bind(this);
    this.switchTabs = this.switchTabs.bind(this);
    this.terminateInvestProcess = this.terminateInvestProcess.bind(this);
  }

  async componentDidMount() {
    const { job } = this.props;
    const {
      modelSpec, argsSpec, uiSpec
    } = await investGetSpec(job.metadata.modelRunName);
    this.setState({
      modelSpec: modelSpec,
      argsSpec: argsSpec,
      uiSpec: uiSpec,
      jobStatus: job.metadata.status,
      logfile: job.metadata.logfile
    }, () => { this.switchTabs('setup'); });
  }

  componentWillUnmount() {
    ipcRenderer.removeAllListeners(`invest-logging-${this.ipcSuffix}`);
    ipcRenderer.removeAllListeners(`invest-stderr-${this.ipcSuffix}`);
    ipcRenderer.removeAllListeners(`invest-exit-${this.ipcSuffix}`);
  }

  /** Spawn a child process to run an invest model via the invest CLI.
   *
   * e.g. `invest -vvv run <model> --headless -d <datastack path>`
   *
   * When the process starts (on first stdout callback), job metadata is saved
   * and local state is updated to display the invest log.
   * When the process exits, job metadata is updated with final status of run.
   *
   * @param {object} argsValues - the invest "args dictionary"
   *   as a javascript object
   */
  async investExecute(argsValues) {
    const {
      job,
      investSettings,
      saveJob,
    } = this.props;
    const args = { ...argsValues };
    // Not strictly necessary, but resolving to a complete path
    // here to be extra certain we avoid unexpected collisions
    // of workspaceHash, which uniquely ids a job in the database
    // in part by it's workspace directory.
    args.workspace_dir = path.resolve(argsValues.workspace_dir);

    job.setProperty('argsValues', args);
    job.setProperty('status', 'running');

    // Setting this very early in the click handler so the Execute button
    // can display an appropriate visual cue when it's clicked
    this.setState({
      jobStatus: job.metadata.status,
      workspaceDir: args.workspace_dir
    });
    ipcRenderer.on(`invest-logging-${this.ipcSuffix}`, (event, logfile) => {
      this.setState({
        logfile: logfile
      }, () => {
        this.switchTabs('log');
      });
      job.setProperty('logfile', logfile);
      saveJob(job);
    });
    ipcRenderer.on(`invest-stderr-${this.ipcSuffix}`, (event, data) => {
      let stderr = Object.assign('', this.state.logStdErr);
      stderr += data;
      this.setState({
        logStdErr: stderr,
      });
    });
    ipcRenderer.on(`invest-exit-${this.ipcSuffix}`, (event, code) => {
      // Invest CLI exits w/ code 1 when it catches errors,
      // Models exit w/ code 255 (on all OS?) when errors raise from execute()
      // Windows taskkill yields exit code 1
      // Non-windows process.kill yields exit code null
      const status = (code === 0) ? 'success' : 'error';
      this.setState({
        jobStatus: status
      });
      job.setProperty('status', status);
      saveJob(job);
    });

    ipcRenderer.send(
      'invest-run',
      job.metadata.modelRunName,
      this.state.modelSpec.module,
      args,
      investSettings.loggingLevel,
      this.ipcSuffix
    );
  }

  terminateInvestProcess() {
    ipcRenderer.send('invest-kill', this.state.workspaceDir);
    this.setState({
      logStdErr: 'Run Canceled'
    });
  }

  /** Change the tab that is currently visible.
   *
   * @param {string} key - the value of one of the Nav.Link eventKey.
   */
  switchTabs(key) {
    this.setState(
      { activeTab: key }
    );
  }

  render() {
    const {
      activeTab,
      modelSpec,
      argsSpec,
      uiSpec,
      jobStatus,
      logStdErr,
      logfile,
    } = this.state;
    const {
      navID,
      modelRunName,
      argsValues,
    } = this.props.job.metadata;

    // Don't render the model setup & log until data has been fetched.
    if (!modelSpec) {
      return (<div />);
    }

    const isRunning = jobStatus === 'running';
    const logDisabled = !logfile;
    const sidebarSetupElementId = `sidebar-setup-${navID}`;
    const sidebarFooterElementId = `sidebar-footer-${navID}`;

    return (
      <TabContainer activeKey={activeTab} id="invest-tab">
        <Row className="flex-nowrap">
          <Col sm={3} className="invest-sidebar-col" onDragOver={dragOverHandlerNone}>
            <Nav
              className="flex-column"
              id="vertical tabs"
              variant="pills"
              activeKey={activeTab}
              onSelect={this.switchTabs}
            >
              <Nav.Item>
                <Nav.Link eventKey="setup">
                  Setup
                </Nav.Link>
              </Nav.Item>
              <div
                className="sidebar-setup"
                id={sidebarSetupElementId}
              />
              <Nav.Item>
                <Nav.Link eventKey="log" disabled={logDisabled}>
                  Log
                  { isRunning
                  && (
                    <Spinner
                      animation="border"
                      size="sm"
                      role="status"
                      aria-hidden="true"
                    />
                  )}
                </Nav.Link>
              </Nav.Item>
            </Nav>
            <div className="sidebar-row">
              <ResourcesLinks
                moduleName={modelRunName}
                docs={modelSpec.userguide_html}
              />
            </div>
            <div
              className="sidebar-row sidebar-footer"
              id={sidebarFooterElementId}
            />
          </Col>
          <Col sm={9} className="invest-main-col">
            <TabContent>
              <TabPane eventKey="setup" title="Setup">
                <SetupTab
                  pyModuleName={modelSpec.module}
                  modelName={modelSpec.model_name}
                  argsSpec={argsSpec}
                  uiSpec={uiSpec}
                  argsInitValues={argsValues}
                  investExecute={this.investExecute}
                  nWorkers={this.props.investSettings.nWorkers}
                  sidebarSetupElementId={sidebarSetupElementId}
                  sidebarFooterElementId={sidebarFooterElementId}
                  isRunning={isRunning}
                />
              </TabPane>
              <TabPane eventKey="log" title="Log">
                <LogTab
                  jobStatus={jobStatus}
                  logfile={logfile}
                  logStdErr={logStdErr}
                  terminateInvestProcess={this.terminateInvestProcess}
                  pyModuleName={modelSpec.module}
                  sidebarFooterElementId={sidebarFooterElementId}
                />
              </TabPane>
            </TabContent>
          </Col>
        </Row>
      </TabContainer>
    );
  }
}

InvestTab.propTypes = {
  job: PropTypes.shape({
    metadata: PropTypes.shape({
      modelRunName: PropTypes.string.isRequired,
      modelHumanName: PropTypes.string.isRequired,
      navID: PropTypes.string.isRequired,
      argsValues: PropTypes.object,
      logfile: PropTypes.string,
      status: PropTypes.string,
    }),
    save: PropTypes.func.isRequired,
    setProperty: PropTypes.func.isRequired,
  }).isRequired,
  investSettings: PropTypes.shape({
    nWorkers: PropTypes.string,
    loggingLevel: PropTypes.string,
  }).isRequired,
  saveJob: PropTypes.func.isRequired,
};
