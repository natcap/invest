import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { spawn, execFile } from 'child_process';
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
  getSpec, writeParametersToFile
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

/** Get an invest model's ARGS_SPEC when a model button is clicked.
 *
 * Also get the model's UI spec if it exists.
 *
 * @param {string} modelName - as in a model name appearing in `invest list`
 * @returns {object} destructures to:
 *   { modelSpec, argsSpec, uiSpec }
 */
async function investGetSpec(modelName) {
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
    return { modelSpec: modelSpec, argsSpec: args, uiSpec: uiSpec };
  }
  logger.error(`no spec found for ${modelName}`);
  return undefined;
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
      activeTab: 'setup',
      modelSpec: null, // ARGS_SPEC dict with all keys except ARGS_SPEC.args
      argsSpec: null, // ARGS_SPEC.args, the immutable args stuff
      logfile: null, // path to the invest logfile associated with invest job
      logStdErr: null, // stderr data from the invest subprocess
      jobStatus: null, // 'running', 'error', 'success', 'canceled'
    };

    this.argsToJsonFile = this.argsToJsonFile.bind(this);
    this.investExecute = this.investExecute.bind(this);
    this.switchTabs = this.switchTabs.bind(this);
    this.terminateInvestProcess = this.terminateInvestProcess.bind(this);

    this.investRun = undefined;
  }

  async componentDidMount() {
    // If these dirs already exist, this will err and pass
    fs.mkdir(fileRegistry.CACHE_DIR, (err) => {});
    fs.mkdir(fileRegistry.TEMP_DIR, (err) => {});
    // logfile and jobStatus are undefined unless this is a pre-existing job.
    const { modelRunName, logfile, jobStatus } = this.props;
    const {
      modelSpec, argsSpec, uiSpec
    } = await investGetSpec(modelRunName);
    this.setState({
      modelSpec: modelSpec,
      argsSpec: argsSpec,
      uiSpec: uiSpec,
      logfile: logfile,
      jobStatus: jobStatus,
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
    const {
      investExe,
      investSettings,
      modelRunName,
      saveJob,
    } = this.props;

    // If the same model, workspace, and suffix are executed, invest
    // will overwrite the previous outputs. So our recent jobs
    // catalogue should overwite as well, and that's assured by this
    // non-unique jobID.
    const workspace = {
      directory: path.resolve(argsValues.workspace_dir),
      suffix: argsValues.results_suffix,
    };
    const jobID = crypto.createHash('sha1').update(
      `${modelRunName}${JSON.stringify(workspace)}`
    ).digest('hex');

    const job = {
      jobID: jobID,
      modelRunName: modelRunName,
      argsValues: argsValues,
      workspace: workspace,
      logfile: undefined,
      status: undefined,
    };

    // Write a temporary datastack json for passing to invest CLI
    const tempDir = fs.mkdtempSync(path.join(
      fileRegistry.TEMP_DIR, 'data-'
    ));
    const datastackPath = path.join(tempDir, 'datastack.json');
    await this.argsToJsonFile(datastackPath, job.argsValues);

    const verbosity = LOGLEVELMAP[investSettings.loggingLevel];
    const cmdArgs = [
      verbosity,
      'run',
      job.modelRunName,
      '--headless',
      `-d ${datastackPath}`,
    ];
    // let investRun;
    if (process.platform !== 'foo') {
      this.investRun = spawn(path.basename(investExe), cmdArgs, {
        env: { PATH: path.dirname(investExe) },
        shell: true, // without shell, IOError when datastack.py loads json
        detached: true, // we want invest to terminate when this shell terminates
      });
      this.investRun.terminate = () => {
        if (this.state.jobStatus === 'running') {
          process.kill(-this.investRun.pid, 'SIGTERM');
        }
      };
    } else {
      this.investRun = execFile(path.basename(investExe), cmdArgs, {
        env: { PATH: path.dirname(investExe) },
      });
    }

    // There's no general way to know that a spawned process started,
    // so this logic when listening for stdout seems like the way.
    this.investRun.stdout.on('data', async () => {
      if (!job.logfile) {
        job.logfile = await findMostRecentLogfile(job.workspace.directory);
        // TODO: handle case when job.logfile is still undefined?
        // Could be if some stdout is emitted before a logfile exists.
        job.status = 'running';
        this.setState(
          {
            logfile: job.logfile,
            jobStatus: job.status,
          }, () => {
            this.switchTabs('log');
            saveJob(job);
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
    this.investRun.stderr.on('data', (data) => {
      stderr += `${data}`;
      this.setState({
        logStdErr: stderr,
      });
    });

    // Set some state when the invest process exits and update the app's
    // persistent database by calling saveJob.
    this.investRun.on('exit', (code) => {
      // TODO: there are non-zero exit cases that should be handled
      // differently from one-another, but right now they are all exit code 1.
      // E.g. this state update is designed with a model crash in mind,
      // not a fail to launch
      if (code === 0) {
        job.status = 'success';
      } else if (code === 1) {
        job.status = 'error';
      } else {
        // code is null if the process was killed
        // ideally we could send a special code for that instead.
        job.status = 'canceled';
      }
      this.setState({
        jobStatus: job.status,
      }, () => {
        saveJob(job);
        cleanupDir(tempDir);
      });
    });
  }

  terminateInvestProcess() {
    this.investRun.terminate();
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
      modelName,
      argsSpec,
      uiSpec,
      jobStatus,
      logfile,
      logStdErr,
      subprocessPID,
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
                  uiSpec={uiSpec}
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
                  terminateInvestProcess={this.terminateInvestProcess}
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
  modelRunName: PropTypes.string.isRequired,
  logfile: PropTypes.string,
  argsInitValues: PropTypes.object,
  jobStatus: PropTypes.string,
  investSettings: PropTypes.shape({
    nWorkers: PropTypes.string,
    loggingLevel: PropTypes.string,
  }).isRequired,
  saveJob: PropTypes.func.isRequired,
};
