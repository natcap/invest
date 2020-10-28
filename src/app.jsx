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
import SettingsModal from './components/SettingsModal';
import { getInvestList } from './server_requests';
import { updateRecentJobs, loadRecentJobs } from './utils';
import { fileRegistry } from './constants';
import { getLogger } from './logger';
import Job from './Job';

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
    // let recentJobs = [];
    // if (fs.existsSync(jobDatabase)) {
    //   recentJobs = await loadRecentJobs(jobDatabase);
    // }
    await Job.init();
    const recentJobs = await Job.getJobStore(); // why does this return a Promise?
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

  /** Push data for a new InvestJob component to an array.
   *
   */
  openInvestModel(job) {
    const navID = crypto.randomBytes(16).toString('hex');
    const { metadata } = job;
    Object.assign(metadata, { navID: navID });
    this.setState((state) => ({
      openJobs: [...state.openJobs, metadata],
    }), () => this.switchTabs(metadata.navID));
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

  /** Save data describing an invest job to a persistent JSON file.
   *
   * @param {object} job - data that can be passed to openInvestModel
   */
  async saveJob(job) {
    // const jobMetadata = job.save();
    const recentJobs = await job.save();
    console.log(recentJobs);
    // const recentJobs = await updateRecentJobs(
    //   jobMetadata, this.props.jobDatabase
    // );
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
      console.log(job);
      investNavItems.push(
        <Nav.Item key={job.navID}>
          <Nav.Link eventKey={job.navID}>
            <React.Fragment>
              {job.modelHumanName}
              <Button
                className="close-tab"
                variant="outline-dark"
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
          title={job.modelHumanName}
        >
          <InvestJob
            navID={job.navID}
            investExe={investExe}
            modelRunName={job.modelRunName}
            modelHumanName={job.modelHumanName}
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
        <Navbar expand="lg">
          <Nav
            variant="pills"
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
  jobDatabase: PropTypes.string,
};
