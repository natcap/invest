import crypto from 'crypto';
import React from 'react';
import PropTypes from 'prop-types';
import { ipcRenderer } from 'electron';

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
import {
  DataDownloadModal, DownloadProgressBar
} from './components/DataDownloadModal';
import {
  saveSettingsStore, getAllSettings, getDefaultSettings,
} from './components/SettingsModal/SettingsStorage';
import { getInvestModelNames } from './server_requests';
import { getLogger } from './logger';
import InvestJob from './InvestJob';
import { dragOverHandlerNone } from './utils';

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
      didAskForSampleData: true, // for the pre-DidMount render
      downloadedNofN: null,
    };
    this.saveSettings = this.saveSettings.bind(this);
    this.switchTabs = this.switchTabs.bind(this);
    this.openInvestModel = this.openInvestModel.bind(this);
    this.closeInvestModel = this.closeInvestModel.bind(this);
    this.saveJob = this.saveJob.bind(this);
    this.clearRecentJobs = this.clearRecentJobs.bind(this);
    this.storeDownloadDir = this.storeDownloadDir.bind(this);
    this.clearDownloadDirPath = this.clearDownloadDirPath.bind(this);
  }

  /** Initialize the list of invest models, recent invest jobs, etc. */
  async componentDidMount() {
    const investList = await getInvestModelNames();
    const recentJobs = await InvestJob.getJobStore();
    const investSettings = await getAllSettings();

    let didAskForSampleData = false;
    if (investSettings.sampleDataDir) {
      didAskForSampleData = true;
    }
    logger.debug(`Already asked for sample data? ${didAskForSampleData}`)

    this.setState({
      investList: investList,
      recentJobs: recentJobs,
      investSettings: investSettings,
      didAskForSampleData: didAskForSampleData,
    });

    ipcRenderer.on('download-status', (event, downloadedNofN) => {
      this.setState({
        downloadedNofN: downloadedNofN
      });
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

    saveSettingsStore(settings);
  }

  /** Store a sampledata filepath in localforage.
   *
   * @param {String} dir - the path to the user-selected dir
   */
  storeDownloadDir(dir) {
    const { investSettings } = this.state;
    investSettings.sampleDataDir = dir;
    this.setState({
      didAskForSampleData: true,
    });
    this.saveSettings(investSettings);
  }

  /** Clear the stored sampledata filepath. Prompt Modal to show. */
  clearDownloadDirPath() {
    const { investSettings } = this.state;
    investSettings.sampleDataDir = getDefaultSettings()['sampleDataDir'];
    this.setState({
      didAskForSampleData: false,
    });
    this.saveSettings(investSettings);
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
      didAskForSampleData,
      downloadedNofN,
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
      <React.Fragment>
        <DataDownloadModal
          show={!didAskForSampleData}
          storeDownloadDir={this.storeDownloadDir}
        />
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
            {
              (downloadedNofN)
                ? (
                  <DownloadProgressBar
                    downloadedNofN={downloadedNofN}
                    expireAfter={5000} // milliseconds
                  />
                )
                : <div />
            }
            <LoadButton
              openInvestModel={this.openInvestModel}
              batchUpdateArgs={this.batchUpdateArgs}
            />
            <SettingsModal
              className="mx-3"
              saveSettings={this.saveSettings}
              investSettings={investSettings}
              clearJobsStorage={this.clearRecentJobs}
              clearDownloadDirPath={this.clearDownloadDirPath}
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
      </React.Fragment>
    );
  }
}

App.propTypes = {
  investExe: PropTypes.string.isRequired,
};
