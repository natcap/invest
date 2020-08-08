import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { spawn } from 'child_process';
import React from 'react';
import PropTypes from 'prop-types';

import TabPane from 'react-bootstrap/TabPane';
import TabContent from 'react-bootstrap/TabContent';
import TabContainer from 'react-bootstrap/TabContainer';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';
import Spinner from 'react-bootstrap/Spinner';

import HomeTab from './components/HomeTab';
import SetupTab from './components/SetupTab';
import LogTab from './components/LogTab';
import ResourcesTab from './components/ResourcesTab';
import LoadButton from './components/LoadButton';
import { SettingsModal } from './components/SettingsModal';
import {
  getSpec, fetchDatastackFromFile, writeParametersToFile
} from './server_requests';
import {
  findMostRecentLogfile, cleanupDir
} from './utils';
import { fileRegistry } from './constants';
import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

// to translate to the invest CLI's verbosity flag:
const LOGLEVELMAP = {
  DEBUG: '--debug',
  INFO: '-vvv',
  WARNING: '-vv',
  ERROR: '-v',
};

function changeSetupKey(int) {
  // Used for generating a new unique `key` for SetupTab component.
  return (int + 1);
}

/** This component and it's children render all the visible parts of the app.
 *
 * This component's state includes all the data needed to represent one invest
 * job.
 */
export default class InvestJob extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      setupKey: 0,
      sessionID: null,                 // hash of modelName + workspace generated at model execution
      modelName: '',                   // as appearing in `invest list`
      modelSpec: {},                   // ARGS_SPEC dict with all keys except ARGS_SPEC.args
      argsSpec: null,                  // ARGS_SPEC.args, the immutable args stuff
      argsInitDict: null,              // 
      workspace: { 
        directory: null, suffix: null
      },                               // only set values when execute starts the subprocess
      logfile: null,                   // path to the invest logfile associated with invest job
      logStdErr: null,                 // stderr data from the invest subprocess
      sessionProgress: 'home',         // 'home', 'setup', 'log' - used on loadState to decide which tab to activate
      jobStatus: null,                 // 'running', 'error', 'success'
      activeTab: 'home',               // controls which tab is currently visible
    };

    this.argsToJsonFile = this.argsToJsonFile.bind(this);
    this.investGetSpec = this.investGetSpec.bind(this);
    this.investExecute = this.investExecute.bind(this);
    this.switchTabs = this.switchTabs.bind(this);
    this.saveState = this.saveState.bind(this);
    this.loadState = this.loadState.bind(this);
  }

  componentDidMount() {
    // If these dirs already exist, this will err and pass
    fs.mkdir(fileRegistry.CACHE_DIR, (err) => {});
    fs.mkdir(fileRegistry.TEMP_DIR, (err) => {});
  }

  /** Save the state of this component (1) and the current InVEST job (2).
   * 1. Save the state object of this component to a JSON file .
   * 2. Append metadata of the invest job to a persistent database/file.
   * This triggers automatically when the invest subprocess starts and again
   * when it exits.
   */
  saveState() {
    const {
      sessionID,
      modelName,
      workspace,
      jobStatus,
    } = this.state;
    const jsonContent = JSON.stringify(this.state, null, 2);
    const filepath = path.join(fileRegistry.CACHE_DIR, `${sessionID}.json`);
    fs.writeFile(filepath, jsonContent, 'utf8', (err) => {
      if (err) {
        logger.error('An error occured while writing JSON Object to File.');
        return logger.error(err.stack);
      }
    });
    const jobMetadata = {};
    jobMetadata[sessionID] = {
      model: modelName,
      workspace: workspace,
      statefile: filepath,
      status: jobStatus,
      humanTime: new Date().toLocaleString(),
      systemTime: new Date().getTime(),
    };
    this.props.updateRecentSessions(jobMetadata, this.props.jobDatabase);
  }

  /** Set this component's state to the object parsed from a JSON file.
   *
   * @param {string} sessionFilename - path to a JSON file.
   */
  async loadState(sessionFilename) {
    if (fs.existsSync(sessionFilename)) {
      const loadedState = JSON.parse(fs.readFileSync(sessionFilename, 'utf8'));
      // saveState is only called w/in investExecute and only after invest
      // has created a logfile, which means an invest logfile should always
      // exist and can be used to get args values and initialize SetupTab.
      const datastack = await fetchDatastackFromFile(loadedState.logfile);
      if (datastack) {
        loadedState.argsInitDict = datastack.args;
        Object.assign(
          loadedState, { setupKey: changeSetupKey(this.state.setupKey) }
        );
        this.setState(loadedState,
          () => {
            this.switchTabs(loadedState.sessionProgress);
          });
      } else {
        alert('Cannot load this session because data is missing')
      }
    } else {
      logger.error(`state file not found: ${sessionFilename}`);
    }
  }

  /** Write an invest args JSON file for passing to invest cli.
   *
   * Outsourcing this to natcap.invest.datastack via flask ensures
   * a compliant json with an invest version string.
   *
   * @param {string} datastackPath - path to a JSON file.
   * @param {object} argsValues - the invest "args dictionary"
   *   as a javascript object
   */
  async argsToJsonFile(datastackPath, argsValues) {
    // The n_workers value always needs to be inserted into args
    const argsValuesCopy = {
      ...argsValues, n_workers: this.props.investSettings.nWorkers
    };

    const payload = {
      parameterSetPath: datastackPath,
      moduleName: this.state.modelSpec.module,
      relativePaths: false,
      args: argsValuesCopy,
    };
    await writeParametersToFile(payload);
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
    const { investExe, investSettings } = this.props;

    const workspace = {
      directory: path.resolve(argsValues.workspace_dir),
      suffix: argsValues.results_suffix,
    };
    // If the same model, workspace, and suffix are executed, invest
    // will overwrite the previous outputs. So our recent session
    // catalogue should overwite as well, and that's assured by this
    // non-unique sessionID.
    const sessionID = crypto.createHash('sha1').update(
      `${this.state.modelName}${JSON.stringify(workspace)}`
    ).digest('hex');

    // Write a temporary datastack json for passing to invest CLI
    const tempDir = fs.mkdtempSync(path.join(
      fileRegistry.TEMP_DIR, 'data-'
    ));
    const datastackPath = path.join(tempDir, 'datastack.json');
    await this.argsToJsonFile(datastackPath, argsValues);

    const verbosity = LOGLEVELMAP[investSettings.loggingLevel];
    const cmdArgs = [
      verbosity,
      'run',
      this.state.modelName,
      '--headless',
      `-d ${datastackPath}`,
    ];
    const investRun = spawn(path.basename(investExe), cmdArgs, {
      env: { PATH: path.dirname(investExe) },
      shell: true, // without true, IOError when datastack.py loads json
    });

    // There's no general way to know that a spawned process started,
    // so this logic when listening for stdout seems like the way.
    let logfilename = '';
    investRun.stdout.on('data', async () => {
      if (!logfilename) {
        logfilename = await findMostRecentLogfile(workspace.directory);
        // TODO: handle case when logfilename is undefined? It seems like
        // sometimes there is some stdout emitted before a logfile exists.
        this.setState(
          {
            logfile: logfilename,
            sessionID: sessionID,
            sessionProgress: 'log',
            workspace: workspace,
            jobStatus: 'running',
          }, () => {
            this.switchTabs('log');
            this.saveState();
          }
        );
      }
    });

    // Capture stderr to a string separate from the invest log
    // so that it can be displayed separately when invest exits.
    // And because it could actually be stderr emitted from the
    // invest CLI or even the shell, rather than the invest model,
    // in which case it's useful to logger.debug too.
    let stderr = Object.assign('', this.state.logStdErr);
    investRun.stderr.on('data', (data) => {
      stderr += `${data}`;
      this.setState({
        logStdErr: stderr,
      });
    });

    // Set some state when the invest process exits and update the app's
    // persistent database by calling saveState.
    investRun.on('close', (code) => {
      // TODO: there are non-zero exit cases that should be handled
      // differently from one-another, but right now they are all exit code 1.
      // E.g. this callback is designed with a model crash in mind, but not a fail to 
      // launch, in which case the saveState call will probably crash.
      const status = (code === 0 ? 'success' : 'error');
      this.setState({
        jobStatus: status,
      }, () => {
        this.saveState();
        cleanupDir(tempDir);
      });
    });
  }

  /** Get an invest model's ARGS_SPEC when a model button is clicked.
   *
   * Also get the model's UI spec if it exists.
   * Then reset much of this component's state in case a prior job's
   * state exists. This includes setting a new setupKey, which is passed
   * as a key to the SetupTab component, triggering it to re-mount
   * rather than just re-render, allowing one-time initilization of
   * arg grouping and ordering.
   *
   * @param {string} modelName - as in a model name appearing in `invest list`
   * @param {object} argsInitDict - empty, or a JSON representation of an
   *   invest model's agruments dictionary
   */
  async investGetSpec(modelName, argsInitDict = {}) {
    const spec = await getSpec(modelName);
    if (spec) {
      const { args, ...modelSpec } = spec;
      // A model's UI Spec is optional and may not exist
      let uiSpec = {};
      try {
        uiSpec = JSON.parse(fs.readFileSync(
          path.join(fileRegistry.INVEST_UI_DATA, `${spec.module}.json`)
        ));
      } catch (err) {
        if (err.code === 'ENOENT') {
          logger.warn(err);
          logger.warn(`No UI spec exists for ${spec.module}`);
        } else {
          logger.error(err.stack);
        }
      }
      // extend the args spec with the UI spec
      Object.keys(args).forEach((key) => {
        Object.assign(args[key], uiSpec[key]);
      });
      // This event represents a user selecting a model,
      // and so some existing state should be reset.
      this.setState({
        modelName: modelName,
        modelSpec: modelSpec,
        argsSpec: args,
        argsInitDict: argsInitDict,
        sessionProgress: 'setup',
        jobStatus: null,
        logStdErr: '',
        logStdOut: '',
        sessionID: null,
        workspace: null,
        setupKey: changeSetupKey(this.state.setupKey),
        activeTab: 'setup',
      });
    } else {
      logger.error(`no spec found for ${modelName}`);
    }
  }

  /** Change the tab that is currently visible.
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
      modelName,
      setupKey,
      argsSpec,
      argsInitDict,
      jobStatus,
      logfile,
      logStdErr,
    } = this.state;
    const {
      saveSettings,
      investSettings,
      investList,
      recentSessions,
    } = this.props;
    const setupDisabled = !(this.state.argsSpec); // enable once modelSpec has loaded
    const logDisabled = (this.state.jobStatus == null); // enable during and after execution

    return (
      <TabContainer activeKey={activeTab}>
        <Navbar bg="light" expand="lg">
          <Nav
            variant="tabs"
            id="controlled-tab-example"
            className="mr-auto"
            activeKey={activeTab}
            onSelect={this.switchTabs}
          >
            <Nav.Item>
              <Nav.Link eventKey="home">
                Home
              </Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="setup" disabled={setupDisabled}>
                Setup
              </Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="log" disabled={logDisabled}>
                { this.state.jobStatus === 'running'
                && (
                  <Spinner
                    animation="border"
                    size="sm"
                    role="status"
                    aria-hidden="true"
                  />
                )}
                Log
              </Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="resources">
                Resources
              </Nav.Link>
            </Nav.Item>
          </Nav>
          <Navbar.Brand>{modelSpec.model_name}</Navbar.Brand>
          <LoadButton
            investGetSpec={this.investGetSpec}
            batchUpdateArgs={this.batchUpdateArgs}
          />
          <SettingsModal
            className="mx-3"
            saveSettings={saveSettings}
            investSettings={investSettings}
          />
        </Navbar>
        <TabContent className="mt-3">
          <TabPane eventKey="home" title="Home">
            <HomeTab
              investList={investList}
              investGetSpec={this.investGetSpec}
              saveState={this.saveState}
              loadState={this.loadState}
              recentSessions={recentSessions}
            />
          </TabPane>
          <TabPane eventKey="setup" title="Setup">
            <SetupTab key={setupKey}
              pyModuleName={modelSpec.module}
              modelName={modelName}
              argsSpec={argsSpec}
              argsInitValues={argsInitDict}
              investExecute={this.investExecute}
              argsToJsonFile={this.argsToJsonFile}
            />
          </TabPane>
          <TabPane eventKey="log" title="Log">
            <LogTab
              jobStatus={jobStatus}
              logfile={logfile}
              logStdErr={logStdErr}
            />
          </TabPane>
          <TabPane eventKey="resources" title="Resources">
            <ResourcesTab 
              modelName={modelSpec.model_name}
              docs={modelSpec.userguide_html}
            />
          </TabPane>
        </TabContent>
      </TabContainer>
    );
  }
}

InvestJob.propTypes = {
  investExe: PropTypes.string,
  investList: PropTypes.object,
  investSettings: PropTypes.shape({
    nWorkers: PropTypes.string,
    loggingLevel: PropTypes.string,
  }),
  recentSessions: PropTypes.array,
  jobDatabase: PropTypes.string,
  directoryConstants: PropTypes.shape({
    CACHE_DIR: PropTypes.string,
    TEMP_DIR: PropTypes.string,
    INVEST_UI_DATA: PropTypes.string
  }),
  updateRecentSessions: PropTypes.func,
  saveSettings: PropTypes.func
}


