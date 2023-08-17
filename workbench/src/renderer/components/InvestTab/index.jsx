import React from 'react';
import PropTypes from 'prop-types';

import TabPane from 'react-bootstrap/TabPane';
import TabContent from 'react-bootstrap/TabContent';
import TabContainer from 'react-bootstrap/TabContainer';
import Nav from 'react-bootstrap/Nav';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import {
  MdKeyboardArrowRight,
} from 'react-icons/md';
import { withTranslation } from 'react-i18next';

import ModelStatusAlert from './ModelStatusAlert';
import SetupTab from '../SetupTab';
import LogTab from '../LogTab';
import ResourcesLinks from '../ResourcesLinks';
import { getSpec } from '../../server_requests';
import { UI_SPEC } from '../../ui_config';
import { ipcMainChannels } from '../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;
const { logger } = window.Workbench;

/** Get an invest model's MODEL_SPEC when a model button is clicked.

 *
 * @param {string} modelName - as in a model name appearing in `invest list`
 * @returns {object} destructures to:
 *   { modelSpec, argsSpec, uiSpec }
 */
async function investGetSpec(modelName) {
  const spec = await getSpec(modelName);
  if (spec) {
    const { args, ...modelSpec } = spec;
    const uiSpec = UI_SPEC[modelName];
    if (uiSpec) {
      return { modelSpec: modelSpec, argsSpec: args, uiSpec: uiSpec };
    }
    logger.error(`no UI spec found for ${modelName}`);
  } else {
    logger.error(`no args spec found for ${modelName}`);
  }
  return undefined;
}

function handleOpenWorkspace(logfile) {
  ipcRenderer.send(ipcMainChannels.SHOW_ITEM_IN_FOLDER, logfile);
}

/**
 * Render an invest model setup form, log display, etc.
 * Manage launching of an invest model in a child process.
 * And manage saves of executed jobs to a persistent store.
 */
class InvestTab extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      activeTab: 'setup',
      modelSpec: null, // MODEL_SPEC dict with all keys except MODEL_SPEC.args
      argsSpec: null, // MODEL_SPEC.args, the immutable args stuff
      uiSpec: null,
      userTerminated: false,
      executeClicked: false,
    };

    this.investExecute = this.investExecute.bind(this);
    this.switchTabs = this.switchTabs.bind(this);
    this.terminateInvestProcess = this.terminateInvestProcess.bind(this);
    this.investLogfileCallback = this.investLogfileCallback.bind(this);
    this.investExitCallback = this.investExitCallback.bind(this);
  }

  async componentDidMount() {
    const { job } = this.props;
    const {
      modelSpec, argsSpec, uiSpec,
    } = await investGetSpec(job.modelRunName);
    this.setState({
      modelSpec: modelSpec,
      argsSpec: argsSpec,
      uiSpec: uiSpec,
    }, () => { this.switchTabs('setup'); });
    const { tabID } = this.props;
    ipcRenderer.on(`invest-logging-${tabID}`, this.investLogfileCallback);
    ipcRenderer.on(`invest-exit-${tabID}`, this.investExitCallback);
  }

  componentWillUnmount() {
    ipcRenderer.removeListener(
      `invest-logging-${this.props.tabID}`, this.investLogfileCallback
    );
    ipcRenderer.removeListener(
      `invest-exit-${this.props.tabID}`, this.investExitCallback
    );
  }

  investLogfileCallback(logfile) {
    // Only now do we know for sure the process is running
    this.props.updateJobProperties(this.props.tabID, {
      logfile: logfile,
      status: 'running',
    });
  }

  /** Receive data about the exit status of the invest process.
   *
   * @param {object} data - of shape { code: number }
   */
  investExitCallback(data) {
    const {
      tabID,
      updateJobProperties,
      saveJob,
    } = this.props;
    let status = (data.code === 0) ? 'success' : 'error';
    if (this.state.userTerminated) {
      status = 'canceled';
    }
    updateJobProperties(tabID, {
      status: status,
    });
    saveJob(tabID);
    this.setState({
      executeClicked: false,
      userTerminated: false,
    });
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
    this.setState({
      executeClicked: true, // disable the button until invest exits
    });
    const {
      job,
      tabID,
      updateJobProperties,
    } = this.props;
    const args = { ...argsValues };

    updateJobProperties(tabID, {
      argsValues: args,
      status: undefined, // in case of re-run, clear an old status
    });

    ipcRenderer.send(
      ipcMainChannels.INVEST_RUN,
      job.modelRunName,
      this.state.modelSpec.pyname,
      args,
      tabID
    );
    this.switchTabs('log');
  }

  terminateInvestProcess() {
    this.setState({
      userTerminated: true,
    }, () => {
      ipcRenderer.send(
        ipcMainChannels.INVEST_KILL, this.props.tabID
      );
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
      executeClicked,
    } = this.state;
    const {
      status,
      modelRunName,
      argsValues,
      logfile,
    } = this.props.job;

    const { tabID, t } = this.props;

    // Don't render the model setup & log until data has been fetched.
    if (!modelSpec) {
      return (<div />);
    }

    const logDisabled = !logfile;
    const sidebarSetupElementId = `sidebar-setup-${tabID}`;
    const sidebarFooterElementId = `sidebar-footer-${tabID}`;

    return (
      <TabContainer activeKey={activeTab} id="invest-tab">
        <Row className="flex-nowrap">
          <Col
            className="invest-sidebar-col"
          >
            <Nav
              className="flex-column"
              id="vertical tabs"
              variant="pills"
              activeKey={activeTab}
              onSelect={this.switchTabs}
            >
              <Nav.Link eventKey="setup">
                {t('Setup')}
                <MdKeyboardArrowRight />
              </Nav.Link>
              <Nav.Link eventKey="log" disabled={logDisabled}>
                {t('Log')}
                <MdKeyboardArrowRight />
              </Nav.Link>
            </Nav>
            <div
              className="sidebar-row sidebar-buttons"
              id={sidebarSetupElementId}
            />
            <div className="sidebar-row sidebar-links">
              <ResourcesLinks
                moduleName={modelRunName}
                docs={modelSpec.userguide}
              />
            </div>
            <div
              className="sidebar-row sidebar-footer"
              id={sidebarFooterElementId}
            >
              {
                status
                  ? (
                    <ModelStatusAlert
                      status={status}
                      handleOpenWorkspace={() => handleOpenWorkspace(logfile)}
                      terminateInvestProcess={this.terminateInvestProcess}
                    />
                  )
                  : null
              }
            </div>
          </Col>
          <Col className="invest-main-col">
            <TabContent>
              <TabPane
                eventKey="setup"
                aria-label="model setup tab"
              >
                <SetupTab
                  pyModuleName={modelSpec.pyname}
                  userguide={modelSpec.userguide}
                  modelName={modelRunName}
                  argsSpec={argsSpec}
                  uiSpec={uiSpec}
                  argsInitValues={argsValues}
                  investExecute={this.investExecute}
                  sidebarSetupElementId={sidebarSetupElementId}
                  sidebarFooterElementId={sidebarFooterElementId}
                  executeClicked={executeClicked}
                  switchTabs={this.switchTabs}
                />
              </TabPane>
              <TabPane
                eventKey="log"
                aria-label="model log tab"
              >
                <LogTab
                  logfile={logfile}
                  executeClicked={executeClicked}
                  tabID={tabID}
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
    modelRunName: PropTypes.string.isRequired,
    modelHumanName: PropTypes.string.isRequired,
    argsValues: PropTypes.object,
    logfile: PropTypes.string,
    status: PropTypes.string,
  }).isRequired,
  tabID: PropTypes.string.isRequired,
  saveJob: PropTypes.func.isRequired,
  updateJobProperties: PropTypes.func.isRequired,
};

export default withTranslation()(InvestTab);
