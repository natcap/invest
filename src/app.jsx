import crypto from 'crypto';
import React from 'react';
import PropTypes from 'prop-types';
import localforage from 'localforage';

import TabPane from 'react-bootstrap/TabPane';
import TabContent from 'react-bootstrap/TabContent';
import TabContainer from 'react-bootstrap/TabContainer';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';
import Button from 'react-bootstrap/Button';

import HomeTab from './components/HomeTab';
import InvestTab from './components/InvestTab';
import LoadButton from './components/LoadButton';
import SettingsModal from './components/SettingsModal';
import { getInvestModelNames } from './server_requests';
import { getLogger } from './logger';
import InvestJob from './InvestJob';
import { dragOverHandlerNone } from './utils';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

const investSettingsStore = localforage.createInstance({
  name: 'InvestSettings',
});

/** Getter function for global default settings.
 *
 * @returns {object} to destructure into two args:
 *     {String} nWorkers - TaskGraph number of workers
 *     {String} logggingLevel - InVEST model logging level
 */
function  getDefaultSettings() {
  const defaultSettings = {
    nWorkers: '-1',
    loggingLevel: 'INFO',
  };
  return defaultSettings;
}

/** Helper function for testing purposes
 *
 * @returns {object} localforage store for invest settings
 */
function getSettingsStore() {
  const postSettings = { ...investSettingsStore };
  return postSettings;
}

/** Helper function for testing purposes */
function clearSettingsStore() {
  investSettingsStore.clear();
}

export const testables = {
  getSettingsStore: getSettingsStore,
  clearSettingsStore: clearSettingsStore,
};

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
    this.saveSettings = this.saveSettings.bind(this);
    this.setDefaultSettings = this.setDefaultSettings.bind(this);
    this.switchTabs = this.switchTabs.bind(this);
    this.openInvestModel = this.openInvestModel.bind(this);
    this.closeInvestModel = this.closeInvestModel.bind(this);
    this.saveJob = this.saveJob.bind(this);
    this.clearRecentJobs = this.clearRecentJobs.bind(this);
  }

  /** Initialize the list of available invest models and recent invest jobs. */
  async componentDidMount() {
    const investList = await getInvestModelNames();
    const recentJobs = await InvestJob.getJobStore();
    // Placeholder for instantiating global settings.
    let investSettings = {};
    const globalDefaultSettings = getDefaultSettings();

    try {
      for (const [setting, _val] of Object.entries(globalDefaultSettings)) {
        const value = await investSettingsStore.getItem(setting);
        if (!value) {
          throw new Error('Value not defined or null, use defaults.');
        }
        investSettings[setting] = value;
      }
    } catch (err) {
      // This code runs if there were any errors.
      investSettings = globalDefaultSettings;
    }

    this.setState({
      investList: investList,
      recentJobs: recentJobs,
      investSettings: investSettings,
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

    // Using ``settings`` instead of ``this.state.investSettings`` because
    // setState can be asynchronous.
    try {
      for (const [setting, value] of Object.entries(settings)) {
        investSettingsStore.setItem(setting, value);
      }
    } catch (err) {
      logger.error(`Error saving settings: ${err}`);
    }
  }

  /** Reset global settings to defaults. */
  setDefaultSettings() {
    const defaultSettings = getDefaultSettings();
    this.setState({
      investSettings: defaultSettings,
    });
  }

  /** Push data for a new InvestTab component to an array.
   *
   * @param {InvestJob} job - as constructed by new InvestJob()
   */
  openInvestModel(job) {
    const navID = crypto.randomBytes(16).toString('hex');
    job.setProperty('navID', navID);
    this.setState((state) => ({
      openJobs: [...state.openJobs, job],
    }), () => this.switchTabs(navID));
  }

  /**
   * Click handler for the close-tab button on an Invest model tab.
   *
   * @param  {string} navID - the eventKey of the tab containing the
   *   InvestTab component that will be removed.
   */
  closeInvestModel(navID) {
    let index;
    const { openJobs } = this.state;
    openJobs.forEach((job) => {
      if (job.metadata.navID === navID) {
        index = openJobs.indexOf(job);
        openJobs.splice(index, 1);
      }
    });
    // Switch to the next tab if there is one, or the previous, or home.
    let switchTo = 'home';
    if (openJobs[index]) {
      switchTo = openJobs[index].metadata.navID;
    } else if (openJobs[index - 1]) {
      switchTo = openJobs[index - 1].metadata.navID;
    }
    this.switchTabs(switchTo);
    this.setState({
      openJobs: openJobs,
    });
  }

  /** Save data describing an invest job to a persistent JSON file.
   *
   * @param {object} job - as constructed by new InvestJob()
   */
  async saveJob(job) {
    const recentJobs = await job.save();
    this.setState({
      recentJobs: recentJobs,
    });
  }

  async clearRecentJobs() {
    const recentJobs = await InvestJob.clearStore();
    this.setState({
      recentJobs: recentJobs,
    });
  }

  render() {
    const { investExe } = this.props;
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
        <Nav.Item key={job.metadata.navID}>
          <Nav.Link eventKey={job.metadata.navID}>
            {job.metadata.modelHumanName}
            <Button
              className="close-tab"
              variant="outline-dark"
              onClick={(event) => {
                event.stopPropagation();
                this.closeInvestModel(job.metadata.navID);
              }}
              onDragOver={dragOverHandlerNone}
            >
              x
            </Button>
          </Nav.Link>
        </Nav.Item>
      );
      investTabPanes.push(
        <TabPane
          key={job.metadata.navID}
          eventKey={job.metadata.navID}
          title={job.metadata.modelHumanName}
        >
          <InvestTab
            job={job}
            investExe={investExe}
            investSettings={investSettings}
            saveJob={this.saveJob}
          />
        </TabPane>
      );
    });

    return (
      <TabContainer activeKey={activeTab}>
        <Navbar onDragOver={dragOverHandlerNone}>
          <Navbar.Brand onDragOver={dragOverHandlerNone}>
            <Nav.Link
              onSelect={this.switchTabs}
              eventKey="home"
              onDragOver={dragOverHandlerNone}
            >
              InVEST
            </Nav.Link>
          </Navbar.Brand>
          <Nav
            variant="pills"
            className="mr-auto horizontal-scroll"
            activeKey={activeTab}
            onSelect={this.switchTabs}
            onDragOver={dragOverHandlerNone}
          >
            {investNavItems}
          </Nav>
          <LoadButton
            openInvestModel={this.openInvestModel}
            batchUpdateArgs={this.batchUpdateArgs}
          />
          <SettingsModal
            className="mx-3"
            saveSettings={this.saveSettings}
            setDefaultSettings={this.setDefaultSettings}
            investSettings={investSettings}
            clearStorage={this.clearRecentJobs}
          />
        </Navbar>

        <TabContent id="top-tab-content">
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
};
