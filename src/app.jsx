import fs from 'fs';
import path from 'path';
import React from 'react';
import PropTypes from 'prop-types';

import TabPane from 'react-bootstrap/TabPane';
import TabContent from 'react-bootstrap/TabContent';
import TabContainer from 'react-bootstrap/TabContainer';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';

import HomeTab from './components/HomeTab';
import InvestJob from './InvestJob';
import LoadButton from './components/LoadButton';
import { SettingsModal } from './components/SettingsModal';
import { getInvestList } from './server_requests';
import { updateRecentSessions, loadRecentSessions } from './utils';
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
      recentSessions: [],
      investSettings: {},
    };
    this.setRecentSessions = this.setRecentSessions.bind(this);
    this.saveSettings = this.saveSettings.bind(this);
    this.switchTabs = this.switchTabs.bind(this);
    this.openInvestModel = this.openInvestModel.bind(this);
    this.saveJob = this.saveJob.bind(this);
  }

  /** Initialize the list of available invest models and recent invest jobs. */
  async componentDidMount() {
    const { jobDatabase } = this.props;
    const investList = await getInvestList();
    let recentSessions = [];
    if (fs.existsSync(jobDatabase)) {
      recentSessions = await loadRecentSessions(jobDatabase);
    }
    // TODO: also load and set investSettings from a cached state, instead
    // of always re-setting to these hardcoded values on first launch?

    this.setState({
      investList: investList,
      recentSessions: recentSessions,
      investSettings: {
        nWorkers: '-1',
        loggingLevel: 'INFO',
      },
    });
  }

  /** Update the recent sessions list when a new invest job was saved.
   * This triggers on InvestJob.saveState().
   *
   * @param {object} jobdata - the metadata describing an invest job.
   */
  async setRecentSessions(jobdata) {
    const recentSessions = await updateRecentSessions(
      jobdata, this.props.jobDatabase
    );
    this.setState({
      recentSessions: recentSessions,
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

  openInvestModel(modelRunName, argsValues, logfile, sessionID) {
    const navID = sessionID || modelRunName;
    this.setState((state) => ({
      openJobs: [
        ...state.openJobs,
        {
          modelRunName: modelRunName,
          argsValues: argsValues,
          logfile: logfile,
          navID: navID,
        },
      ],
    }), () => this.switchTabs(navID));
  }

  /** Save the state of this component (1) and the current InVEST job (2).
   * 1. Save the state object of this component to a JSON file .
   * 2. Append metadata of the invest job to a persistent database/file.
   * This triggers automatically when the invest subprocess starts and again
   * when it exits.
   */
  saveJob(sessionID, modelRunName, argsValues, logfile, workspace) {
    const jsonContent = JSON.stringify({
      modelRunName: modelRunName,
      argsValues: argsValues,
      logfile: logfile,
      workspace: workspace,
    });
    const filepath = path.join(fileRegistry.CACHE_DIR, `${sessionID}.json`);
    fs.writeFile(filepath, jsonContent, 'utf8', (err) => {
      if (err) {
        logger.error('An error occured while writing JSON Object to File.');
        return logger.error(err.stack);
      }
    });
    const jobMetadata = {};
    jobMetadata[sessionID] = {
      model: modelRunName,
      workspace: workspace,
      humanTime: new Date().toLocaleString(),
      systemTime: new Date().getTime(),
      sessionDataPath: filepath,
    };
    this.setRecentSessions(jobMetadata, this.props.jobDatabase);
  }

  render() {
    const { investExe, jobDatabase } = this.props;
    const {
      investList,
      investSettings,
      recentSessions,
      openJobs,
      activeTab,
    } = this.state;

    const investNavItems = [];
    const investTabPanes = [];
    openJobs.forEach((job) => {
      investNavItems.push(
        <Nav.Item key={job.navID}>
          <Nav.Link eventKey={job.navID}>
            {job.modelRunName}
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
              recentSessions={recentSessions}
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
