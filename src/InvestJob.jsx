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

import { HomeTab } from './components/HomeTab';
import { SetupTab } from './components/SetupTab';
import { LogTab } from './components/LogTab';
import { ResultsTab } from './components/ResultsTab'
import { ResourcesTab } from './components/ResourcesTab';
import { LoadButton } from './components/LoadButton';
import { SettingsModal } from './components/SettingsModal';
import { getSpec, fetchDatastackFromFile,
         writeParametersToFile } from './server_requests';
import { argsDictFromObject, findMostRecentLogfile,
         cleanupDir } from './utils';
import { fileRegistry } from './constants';

// TODO see issue #12
import { createStore } from 'redux';
import { Provider } from 'react-redux';
import rootReducer from './components/ResultsTab/Visualization/habitat_risk_assessment/reducers';
const store = createStore(rootReducer)

// to translate to the invest CLI's verbosity flag:
const LOGLEVELMAP = {
  'DEBUG':   '--debug',
  'INFO':    '-vvv',
  'WARNING': '-vv',
  'ERROR':   '-v',
}

function changeSetupKey(int) {
  /* Return a value different from the input value.
  *
  * Used for generating a new unique `key` for SetupTab component.
  */
  return(int + 1)
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
    this.setSessionID = this.setSessionID.bind(this);
  }

  componentDidMount() {
    // TODO: clear out tmp dir on quit?
    fs.mkdir(fileRegistry.CACHE_DIR, (err) => {})
    fs.mkdir(fileRegistry.TEMP_DIR, (err) => {})
  }

  saveState() {
    /** Save the state of this component (1) and the current InVEST job (2).
    * 1. Save the state object of this component to a JSON file .
    * 2. Append metadata of the invest job to a persistent database/file.
    * This triggers automatically when the invest subprocess starts and again
    * when it exits.
    */
    const jsonContent = JSON.stringify(this.state, null, 2);
    const filepath = path.join(
      fileRegistry.CACHE_DIR, this.state.sessionID + '.json');
    fs.writeFile(filepath, jsonContent, 'utf8', function (err) {
      if (err) {
        console.log("An error occured while writing JSON Object to File.");
        return console.log(err);
      }
    });
    let job = {};
    job[this.state.sessionID] = {
      model: this.state.modelName,
      workspace: this.state.workspace,
      statefile: filepath,
      status: this.state.jobStatus,
      humanTime: new Date().toLocaleString(),
      systemTime: new Date().getTime(),
    }
    this.props.updateRecentSessions(job, this.props.jobDatabase);
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
      
      // Right now saveState is only called w/in investExecute and only
      // after invest has created a logfile, which means an invest logfile
      // should always exist and can be used to get args values and initialize SetupTab.
      const datastack = await fetchDatastackFromFile(
        { datastack_path: loadedState.logfile })
      if (datastack) {
        loadedState['argsInitDict'] = datastack['args']
        Object.assign(loadedState, {setupKey: changeSetupKey(this.state.setupKey)})
        this.setState(loadedState,
          () => {
            this.switchTabs(loadedState.sessionProgress);
          });
      } else {
        alert('Cannot load this session because data is missing')
      }
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
      directory: path.resolve(argsValues.workspace_dir.value),
      suffix: argsValues.results_suffix.value
    }
    // If the same model, workspace, and suffix are executed, invest
    // will overwrite the previous outputs. So our recent session
    // catalogue should overwite as well, and that's assured by this
    // non-unique sessionID.
    const sessionID = crypto.createHash('sha1').update(
      this.state.modelName + JSON.stringify(workspace)).digest('hex')

    // Write a temporary datastack json for passing as a command-line arg
    const temp_dir = fs.mkdtempSync(path.join(
      fileRegistry.TEMP_DIR, 'data-'))
    const datastackPath = path.join(temp_dir, 'datastack.json')
    const _ = await this.argsToJsonFile(datastackPath, argsValues);

    // Get verbosity level from the app's settings
    const verbosity = LOGLEVELMAP[this.props.investSettings.loggingLevel]
    
    const cmdArgs = [verbosity, 'run', this.state.modelName, '--headless', '-d ' + datastackPath]
    const investRun = spawn(this.props.investExe, cmdArgs, {
        cwd: process.cwd(),
        shell: true // without true, IOError when datastack.py loads json
      });

    
    // There's no general way to know that a spawned process started,
    // so this logic when listening for stdout seems like the way.
    let logfilename = ''
    investRun.stdout.on('data', async (data) => {
      if (!logfilename) {
        logfilename = await findMostRecentLogfile(workspace.directory)
        // TODO: handle case when logfilename is undefined? It seems like
        // sometimes there is some stdout emitted before a logfile exists.
        this.setState(
          {
            logfile: logfilename,
            sessionID: sessionID,
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
        cleanupDir(temp_dir)
      });
    });
  }

  async investGetSpec(modelName, argsInitDict={}) {
    /** Get an invest model's ARGS_SPEC when a model button is clicked.
    *  
    * Also get the model's UI spec if it exists.
    * Then reset much of this component's state in case a prior job's 
    * state exists. This includes setting a new setupKey, which is passed
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
          path.join(fileRegistry.INVEST_UI_DATA, spec.module + '.json')))
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
        setupKey: changeSetupKey(this.state.setupKey),
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
            <SetupTab key={this.state.setupKey}
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


