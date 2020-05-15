import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import React from 'react';
import PropTypes from 'prop-types';

import TabPane from 'react-bootstrap/TabPane';
import TabContent from 'react-bootstrap/TabContent';
import TabContainer from 'react-bootstrap/TabContainer';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';
import Spinner from 'react-bootstrap/Spinner';

import { HomeTab } from './components/HomeTab';
import { SetupTab } from './components/SetupTab';
import { LogTab } from './components/LogTab';
import { ResultsTab } from './components/ResultsTab'
import { ResourcesTab } from './components/ResourcesTab';
import { LoadButton } from './components/LoadButton';
import { SettingsModal } from './components/SettingsModal';
import { getSpec, fetchDatastackFromFile,
         writeParametersToFile } from './server_requests';
import { argsDictFromObject, findMostRecentLogfile } from './utils';

// TODO see issue #12
import { createStore } from 'redux';
import { Provider } from 'react-redux';
import rootReducer from './components/ResultsTab/Visualization/habitat_risk_assessment/reducers';
const store = createStore(rootReducer)

let INVEST_EXE = 'invest'
if (process.env.INVEST) {  // if it was set, override
  INVEST_EXE = process.env.INVEST
}

let gdalEnv = null;
if (process.env.GDAL_DATA) {
  gdalEnv = { GDAL_DATA: process.env.GDAL_DATA }
}

// TODO: some of these 'global' vars are defined in multiple files
// const CACHE_DIR = 'cache' //  for storing state snapshot files
// const TEMP_DIR = 'tmp'  // for saving datastack json files prior to investExecute
// const INVEST_UI_DATA = 'ui_data'

// to translate to the invest CLI's verbosity flag:
const LOGLEVELMAP = {
  'DEBUG':   '--debug',
  'INFO':    '-vvv',
  'WARNING': '-vv',
  'ERROR':   '-v',
}

function getSetupHash() {
  // TODO: are we always calling this to generate a new key for SetupTab
  // whenever we update argsInitDict state? If so maybe set SetupTab's
  // key as a hash of that state so we don't have to remember to reset setupHash?
  return(crypto.randomBytes(10).toString('hex'))
}

export class InvestJob extends React.Component {
  /** This component and it's children render all the visible parts of the app.
  *
  * This component's state includes all the data needed to represent one invest
  * job.
  */

  constructor(props) {
    super(props);

    this.state = {
      setupHash: getSetupHash(),
      sessionID: null,                 // modelName + workspace.directory + workspace.suffix
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
    this.setSessionID = this.setSessionID.bind(this);
  }

  saveState() {
    /** Save the state of this component (1) and the current InVEST job (2).
    * 1. Save the state object of this component to a JSON file .
    * 2. Append metadata of the invest job to a persistent database/file.
    * This triggers automatically when the invest subprocess starts and again
    * when it exits.
    */
    const jobName = this.state.sessionID;
    const jsonContent = JSON.stringify(this.state, null, 2);
    const filepath = path.join(this.props.directoryConstants.CACHE_DIR, jobName + '.json');
    fs.writeFile(filepath, jsonContent, 'utf8', function (err) {
      if (err) {
        console.log("An error occured while writing JSON Object to File.");
        return console.log(err);
      }
      console.log("saved: " + jobName);
    });
    let job = {};
    job[jobName] = {
      model: this.state.modelName,
      workspace: this.state.workspace,
      statefile: filepath,
      status: this.state.jobStatus,
      humanTime: new Date().toLocaleString(),
      systemTime: new Date().getTime(),
    }
    this.props.updateRecentSessions(job, this.props.appdata);
  }
  
  setSessionID(event) {
    // TODO: this functionality might be deprecated - probably no need to set custom
    // session names. But the same function could be repurposed for a job description.
    event.preventDefault();
    const value = event.target.value;
    this.setState(
      {sessionID: value});
  }

  async loadState(sessionFilename) {
    /** Set this component's state to the object parsed from a JSON file.
    *
    * @params {string} sessionFilename - path to a JSON file.
    */

    if (fs.existsSync(sessionFilename)) {
      const loadedState = JSON.parse(fs.readFileSync(sessionFilename, 'utf8'));
      
      // Right now we saveState is only called w/in investExecute and only
      // after invest has created a logfile, which means an invest logfile
      // should always exist and can be used to get args values and initialize SetupTab.
      const datastack = await fetchDatastackFromFile(
        { datastack_path: loadedState.logfile })
      loadedState['argsInitDict'] = datastack['args']
      Object.assign(loadedState, {setupHash: getSetupHash()})
      this.setState(loadedState,
        () => {
          this.switchTabs(loadedState.sessionProgress);
        });
    } else {
      console.log('state file not found: ' + sessionFilename);
    }
  }

  async argsToJsonFile(datastackPath, argsValues) {
    /** Write an invest args JSON file for passing to invest cli.
    *
    * Outsourcing this to natcap.invest.datastack via flask ensures
    * a compliant json with an invest version string.
    *
    * @params {string} datastackPath - path to a JSON file.
    */

    // The n_workers value always needs to be inserted into args
    argsValues['n_workers'] = this.props.investSettings.nWorkers;
    
    const payload = {
      parameterSetPath: datastackPath, 
      moduleName: this.state.modelSpec.module,
      relativePaths: false,
      args: argsDictFromObject(argsValues)
    }
    await writeParametersToFile(payload);
  }

  async investExecute(argsValues) {
    /** Spawn a child process to run an invest model via the invest CLI:
    * `invest -vvv run <model> --headless -d <datastack path>`
    *
    * When the process starts (on first stdout callback), job metadata is saved
    * and local state is updated to display the log.

    * When the process exits, job metadata is saved again (overwriting previous)
    * with the final status of the invest run.
    */
    const workspace = {
      directory: argsValues.workspace_dir.value,
      suffix: argsValues.results_suffix.value
    }
    // model name, workspace, and suffix are suitable for a unique job identifier
    const sessionName = [
      this.state.modelName, workspace.directory, workspace.suffix].join('-')

    // Write a temporary datastack json for passing as a command-line arg
    const temp_dir = fs.mkdtempSync(path.join(
      process.cwd(), this.props.directoryConstants.TEMP_DIR, 'data-'))
    const datastackPath = path.join(temp_dir, 'datastack.json')
    const _ = await this.argsToJsonFile(datastackPath, argsValues);

    // Get verbosity level from the app's settings
    const verbosity = LOGLEVELMAP[this.props.investSettings.loggingLevel]
    
    const cmdArgs = [verbosity, 'run', this.state.modelName, '--headless', '-d ' + datastackPath]
    const investRun = spawn(INVEST_EXE, cmdArgs, {
        cwd: process.cwd(),
        shell: true, // without true, IOError when datastack.py loads json
        env: gdalEnv
      });

    
    // There's no general way to know that a spawned process started,
    // so this logic when listening for stdout seems like the way.
    let logfilename = ''
    investRun.stdout.on('data', async (data) => {
      if (!logfilename) {
        logfilename = await findMostRecentLogfile(workspace.directory)
        console.log(logfilename)
        // TODO: handle case when logfilename is undefined? It seems like
        // sometimes there is some stdout emitted before a logfile exists.
        this.setState(
          {
            logfile: logfilename,
            sessionID: sessionName,
            sessionProgress: 'log',
            workspace: workspace,
            jobStatus: 'running'
          }, () => {
            this.switchTabs('log')
            this.saveState()
          }
        );
      }
    });

    // Capture stderr to a string separate from the invest log
    // so that it can be displayed separately when invest exits,
    // and because it could actually be stderr emitted from the 
    // invest CLI or even the shell, rather than the invest model,
    // in which case it's useful to console.log too.
    let stderr = Object.assign('', this.state.logStdErr);
    investRun.stderr.on('data', (data) => {
      console.log(`${data}`)
      stderr += `${data}`
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
      const status = (code === 0 ? 'success' : 'error')
      this.setState({
        jobStatus: status,
      }, () => {
        this.saveState();
      });
    });
  }

  async investGetSpec(modelName, argsInitDict={}) {
    /** Get an invest model's ARGS_SPEC when a model button is clicked.
    *  
    * Also get the model's UI spec if it exists.
    * Then reset much of this component's state in case a prior job's 
    * state exists. This includes setting a new setupHash, which is passed
    * as a key to the SetupTab component, triggering it to re-mount
    * rather than just re-render, allowing one-time initilization of
    * arg grouping and ordering.
    *
    * @param {string} - as in a model name appearing in `invest list`
    * @param {object} - empty, or a JSON representation of an invest model's
    *                   agruments dictionary
    */

    const payload = { 
        model: modelName
    };
    const spec = await getSpec(payload);
    if (spec) {
      // This "destructuring" captures spec.args into args and leaves 
      // the rest of spec in modelSpec.
      const {args, ...modelSpec} = spec;
      
      // Even if UI spec doesn't exist for a model, a minimum viable input
      // form can still be generated, so don't crash here.
      let uiSpec = {};
      try {
        uiSpec = JSON.parse(fs.readFileSync(
          path.join(this.props.directoryConstants.INVEST_UI_DATA, spec.module + '.json')))
      } catch (err) {
        if (err.code !== 'ENOENT') {
          throw err
        }
      }
      
      // extend the args spec with the UI spec
      for (const key in args) {
        Object.assign(args[key], uiSpec[key])
      }
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
        setupHash: getSetupHash(),
        activeTab: 'setup'
      });
    } else {
      console.log('no spec found')
      return new Promise((resolve) => resolve(false))
    }
    return new Promise((resolve) => resolve(true))
  }

  switchTabs(key) {
    /** Change the tab that is currently visible.
    * @param {string} key - the value of one of the Nav.Link eventKey.
    */
    this.setState(
      {activeTab: key}
    );
  }

  render () {
    const activeTab = this.state.activeTab;
    const setupDisabled = !(this.state.argsSpec); // enable once modelSpec has loaded
    const logDisabled = (this.state.jobStatus == null);  // enable during and after execution
    const resultsDisabled = (this.state.jobStatus !== 'success');  // enable only on complete execute with no errors
    
    return(
      <TabContainer activeKey={activeTab}>
        <Navbar bg="light" expand="lg">
          <Nav variant="tabs" id="controlled-tab-example" className="mr-auto"
            activeKey={activeTab}
            onSelect={this.switchTabs}>
            <Nav.Item>
              <Nav.Link eventKey="home">Home</Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="setup" disabled={setupDisabled}>Setup</Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="log" disabled={logDisabled}>
                {this.state.jobStatus === 'running' && 
                 <Spinner animation='border' size='sm' role='status' aria-hidden='true'/>
                } Log
              </Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="results" disabled={resultsDisabled}>Results</Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="resources">Resources</Nav.Link>
            </Nav.Item>
          </Nav>
          <Navbar.Brand>{this.state.modelSpec.model_name}</Navbar.Brand>
          <LoadButton
            investGetSpec={this.investGetSpec}
            batchUpdateArgs={this.batchUpdateArgs}
          />
          <SettingsModal className="mx-3"
            saveSettings={this.props.saveSettings}
            investSettings={this.props.investSettings}
          />
        </Navbar>
        <TabContent className="mt-3">
          <TabPane eventKey="home" title="Home">
            <HomeTab
              investList={this.props.investList}
              investGetSpec={this.investGetSpec}
              saveState={this.saveState}
              loadState={this.loadState}
              recentSessions={this.props.recentSessions}
            />
          </TabPane>
          <TabPane eventKey="setup" title="Setup">
            <SetupTab key={this.state.setupHash}
              pyModuleName={this.state.modelSpec.module}
              modelName={this.state.modelName}
              argsSpec={this.state.argsSpec}
              argsInitValues={this.state.argsInitDict}
              investExecute={this.investExecute}
              argsToJsonFile={this.argsToJsonFile}
            />
          </TabPane>
          <TabPane eventKey="log" title="Log">
            <LogTab
              jobStatus={this.state.jobStatus}
              logfile={this.state.logfile}
              logStdErr={this.state.logStdErr}
            />
          </TabPane>
          <TabPane eventKey="results" title="Results">
          <Provider store={store}>
            <ResultsTab
              model={this.state.modelName}
              workspace={this.state.workspace}
              sessionID={this.state.sessionID}
              activeTab={activeTab}/> 
          </Provider>
          </TabPane>
          <TabPane eventKey="resources" title="Resources">
            <ResourcesTab 
              modelName={this.state.modelSpec.model_name}
              docs={this.state.modelSpec.userguide_html}
            />
          </TabPane>
        </TabContent>
      </TabContainer>
    );
  }
}

InvestJob.propTypes = {
  investList: PropTypes.object,
  investSettings: PropTypes.object,
  recentSessions: PropTypes.array,
  appdata: PropTypes.string,
  updateRecentSessions: PropTypes.func,
  saveSettings: PropTypes.func
}


