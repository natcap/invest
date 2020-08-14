import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { spawn } from 'child_process';
import React from 'react';
import PropTypes from 'prop-types';

import TabPane from 'react-bootstrap/TabPane';
import TabContent from 'react-bootstrap/TabContent';
import TabContainer from 'react-bootstrap/TabContainer';
import Nav from 'react-bootstrap/Nav';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Spinner from 'react-bootstrap/Spinner';

import SetupTab from './components/SetupTab';
import LogTab from './components/LogTab';
import ResourcesTab from './components/ResourcesTab';
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

/** This component and it's children render all the visible parts of the app.
 *
 * This component's state includes all the data needed to represent one invest
 * job.
 */
export default class InvestJob extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      activeTab: 'setup',
      // sessionID: null, // hash of modelName + workspace generated at model execution
      modelSpec: null, // ARGS_SPEC dict with all keys except ARGS_SPEC.args
      argsSpec: null, // ARGS_SPEC.args, the immutable args stuff
      logfile: null, // path to the invest logfile associated with invest job
      logStdErr: null, // stderr data from the invest subprocess
      jobStatus: null, // 'running', 'error', 'success'
    };

    this.argsToJsonFile = this.argsToJsonFile.bind(this);
    this.investGetSpec = this.investGetSpec.bind(this);
    this.investExecute = this.investExecute.bind(this);
    this.switchTabs = this.switchTabs.bind(this);
  }

  async componentDidMount() {
    console.log('investjob didmoutn')
    // If these dirs already exist, this will err and pass
    fs.mkdir(fileRegistry.CACHE_DIR, (err) => {});
    fs.mkdir(fileRegistry.TEMP_DIR, (err) => {});

    const { modelSpec, argsSpec } = await this.investGetSpec(
      this.props.modelRunName
    );
    this.setState({
      modelSpec: modelSpec,
      argsSpec: argsSpec,
      logfile: this.props.logfile,
    }, () => { this.switchTabs('setup'); });
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
      args: JSON.stringify(argsValuesCopy),
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
      `${this.props.modelRunName}${JSON.stringify(workspace)}`
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
      this.props.modelRunName,
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
            jobStatus: 'running',
          }, () => {
            this.switchTabs('log');
            this.props.saveJob(
              sessionID, this.props.modelRunName, argsValues, logfilename, workspace
            );
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
        // this.saveState();
        cleanupDir(tempDir);
      });
    });
  }

  /** Get an invest model's ARGS_SPEC when a model button is clicked.
   *
   * Also get the model's UI spec if it exists.
   *
   * @param {string} modelName - as in a model name appearing in `invest list`
   * @param {object} argsInitDict - empty, or a JSON representation of an
   *   invest model's agruments dictionary
   */
  async investGetSpec(modelName) {
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
      return { modelSpec: modelSpec, argsSpec: args };
    }
    logger.error(`no spec found for ${modelName}`);
    return;
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
      argsSpec,
      jobStatus,
      logfile,
      logStdErr,
    } = this.state;

    const logDisabled = (!logfile);

    // Don't render the model setup & log until data has been fetched.
    if (!modelSpec) {
      return (<div />);
    }

    return (
      <TabContainer activeKey={activeTab}>
        <Row>
          <Col sm={3}>
            <Nav
              variant="pills"
              id="vertical tabs"
              className="flex-column"
              activeKey={activeTab}
              onSelect={this.switchTabs}
            >
              <Nav.Item>
                <Nav.Link eventKey="setup">
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
          </Col>
          <Col sm={9}>
            <TabContent className="mt-3">
              <TabPane eventKey="setup" title="Setup">
                <SetupTab
                  pyModuleName={modelSpec.module}
                  modelName={modelName}
                  argsSpec={argsSpec}
                  argsInitValues={this.props.argsInitValues}
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
          </Col>
        </Row>
      </TabContainer>
    );
  }
}

InvestJob.propTypes = {
  investExe: PropTypes.string.isRequired,
  investSettings: PropTypes.shape({
    nWorkers: PropTypes.string,
    loggingLevel: PropTypes.string,
  }).isRequired,
};
