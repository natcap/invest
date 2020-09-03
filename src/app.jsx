import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import React from 'react';
import PropTypes from 'prop-types';

import TabPane from 'react-bootstrap/TabPane';
import TabContent from 'react-bootstrap/TabContent';
import TabContainer from 'react-bootstrap/TabContainer';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';
import Button from 'react-bootstrap/Button';

import HomeTab from './components/HomeTab';
import InvestJob from './InvestJob';
import LoadButton from './components/LoadButton';
import { SettingsModal } from './components/SettingsModal';
import { getInvestList } from './server_requests';
import { updateRecentJobs, loadRecentJobs } from './utils';
import { fileRegistry } from './constants';
import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

/** This component manages any application state that should persist
 * and be independent from properties of a single invest job.
 */
export default class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      activeTab: 'home',
      openJobs: [],
      investList: {},
      recentJobs: [],
      investSettings: {},
    };
    this.setRecentJobs = this.setRecentJobs.bind(this);
    this.saveSettings = this.saveSettings.bind(this);
    this.switchTabs = this.switchTabs.bind(this);
    this.openInvestModel = this.openInvestModel.bind(this);
    this.closeInvestModel = this.closeInvestModel.bind(this);
    this.saveJob = this.saveJob.bind(this);
  }

  /** Initialize the list of available invest models and recent invest jobs. */
  async componentDidMount() {
    const { jobDatabase } = this.props;
    const investList = await getInvestList();
    let recentJobs = [];
    if (fs.existsSync(jobDatabase)) {
      recentJobs = await loadRecentJobs(jobDatabase);
    }
    // TODO: also load and set investSettings from a cached state, instead
    // of always re-setting to these hardcoded values on first launch?

    this.setState({
      investList: investList,
      recentJobs: recentJobs,
      investSettings: {
        nWorkers: '-1',
        loggingLevel: 'INFO',
      },
    });
  }

  /** Update the recent jobs list when a new invest job was saved.
   * This triggers on InvestJob.saveState().
   *
   * @param {object} jobdata - the metadata describing an invest job.
   */
  async setRecentJobs(jobdata) {
    const recentJobs = await updateRecentJobs(
      jobdata, this.props.jobDatabase
    );
    this.setState({
      recentJobs: recentJobs,
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

  saveSettings(settings) {
    this.setState({
      investSettings: settings,
    });
  }

  /**
   * Push data for a new InvestJob component to a new array.
   * When this is called to load a "recent job", optional argsValues, logfile,
   * and jobStatus parameters will be defined, otherwise they can be undefined.
   *
   * @param  {string} modelRunName - invest model name as appears in `invest list`
   * @param  {object} argsValues - an invest "args dictionary" with initial values
   * @param  {string} logfile - path to an existing invest logfile
   * @param  {string} status - indicates how the job exited, if it's a recent job.
   */
  openInvestModel(modelRunName, argsValues, logfile, status) {
    const navID = crypto.randomBytes(16).toString('hex');
    this.setState((state) => ({
      openJobs: [
        ...state.openJobs,
        {
          modelRunName: modelRunName,
          argsValues: argsValues,
          logfile: logfile,
          status: status,
          navID: navID,
        },
      ],
    }), () => this.switchTabs(navID));
  }

  /**
   * Click handler for the close-tab button on an Invest model tab.
   *
   * @param  {string} navID - the eventKey of the tab containing the
   *   InvestJob component that will be removed.
   */
  closeInvestModel(navID) {
    let index;
    const { openJobs } = this.state;
    openJobs.forEach((job) => {
      if (job.navID === navID) {
        index = openJobs.indexOf(job);
        openJobs.splice(index, 1);
      }
    });
    // Switch to the next tab if there is one, or the previous, or home.
    let switchTo = 'home';
    if (openJobs[index]) {
      switchTo = openJobs[index].navID;
    } else if (openJobs[index - 1]) {
      switchTo = openJobs[index - 1].navID;
    }
    this.setState({
      openJobs: openJobs
    }, () => this.switchTabs(switchTo));
  }

  /** Save the state of this component (1) and the current InVEST job (2).
   * 1. Save the state object of this component to a JSON file .
   * 2. Append metadata of the invest job to a persistent database/file.
   * This triggers automatically when the invest subprocess starts and again
   * when it exits.
   */
  saveJob(jobData) {
    const jsonContent = JSON.stringify(jobData);
    const filepath = path.join(
      fileRegistry.CACHE_DIR, `${jobData.jobID}.json`
    );
    fs.writeFile(filepath, jsonContent, 'utf8', (err) => {
      if (err) {
        logger.error('An error occured while writing JSON Object to File.');
        return logger.error(err.stack);
      }
    });
    const jobMetadata = {};
    jobMetadata[jobData.jobID] = {
      model: jobData.modelRunName,
      workspace: jobData.workspace,
      humanTime: new Date().toLocaleString(),
      systemTime: new Date().getTime(),
      jobDataPath: filepath,
    };
    this.setRecentJobs(jobMetadata, this.props.jobDatabase);
  }

  render() {
    const { investExe, jobDatabase } = this.props;
    const {
      investList,
      investSettings,
      recentJobs,
      openJobs,
      activeTab,
    } = this.state;

    const investNavItems = [];
    const investTabPanes = [];
    openJobs.forEach((job) => {
      investNavItems.push(
        <Nav.Item key={job.navID}>
          <Nav.Link eventKey={job.navID}>
            <React.Fragment>
              {job.modelRunName}
              <Button
                className="close-tab"
                variant="outline-secondary"
                onClick={() => this.closeInvestModel(job.navID)}
              >
                x
              </Button>
            </React.Fragment>
          </Nav.Link>
        </Nav.Item>
      );
      investTabPanes.push(
        <TabPane
          key={job.navID}
          eventKey={job.navID}
          title={job.modelRunName}
        >
          <InvestJob
            investExe={investExe}
            modelRunName={job.modelRunName}
            argsInitValues={job.argsValues}
            logfile={job.logfile}
            jobStatus={job.status}
            investSettings={investSettings}
            saveJob={this.saveJob}
          />
        </TabPane>
      );
    });
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
            {investNavItems}
          </Nav>
          <Navbar.Brand>InVEST</Navbar.Brand>
          <LoadButton
            openInvestModel={this.openInvestModel}
            batchUpdateArgs={this.batchUpdateArgs}
          />
          <SettingsModal
            className="mx-3"
            saveSettings={this.saveSettings}
            investSettings={investSettings}
          />
        </Navbar>
        <TabContent className="mt-3">
          <TabPane eventKey="home" title="Home">
            <HomeTab
              investList={investList}
              openInvestModel={this.openInvestModel}
              recentJobs={recentJobs}
            />
          </TabPane>
          {investTabPanes}
        </TabContent>
      </TabContainer>
    );
  }
}

App.propTypes = {
  investExe: PropTypes.string.isRequired,
  jobDatabase: PropTypes.string,
};
