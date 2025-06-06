import React from 'react';
import PropTypes from 'prop-types';
import i18n from 'i18next';

import Badge from 'react-bootstrap/Badge';
import TabPane from 'react-bootstrap/TabPane';
import TabContent from 'react-bootstrap/TabContent';
import TabContainer from 'react-bootstrap/TabContainer';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';
import Button from 'react-bootstrap/Button';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Spinner from 'react-bootstrap/Spinner';
import Tooltip from 'react-bootstrap/Tooltip';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import { MdClose, MdHome } from 'react-icons/md';
import { AiOutlineTrademarkCircle } from 'react-icons/ai';

import HomeTab from './components/HomeTab';
import InvestTab from './components/InvestTab';
import AppMenu from './components/AppMenu';
import SettingsModal from './components/SettingsModal';
import DataDownloadModal from './components/DataDownloadModal';
import DownloadProgressBar from './components/DownloadProgressBar';
import PluginModal from './components/PluginModal';
import MetadataModal from './components/MetadataModal';
import InvestJob from './InvestJob';
import { dragOverHandlerNone } from './utils';
import { ipcMainChannels } from '../main/ipcMainChannels';
import { getInvestModelIDs } from './server_requests';
import Changelog from './components/Changelog';

const { ipcRenderer } = window.Workbench.electron;

/** This component manages any application state that should persist
 * and be independent from properties of a single invest job.
 */
export default class App extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      activeTab: 'home',
      openTabIDs: [],
      openJobs: {},
      investList: null,
      recentJobs: [],
      showDownloadModal: false,
      showPluginModal: false,
      downloadedNofN: null,
      showChangelog: false,
      showSettingsModal: false,
      showMetadataModal: false,
      changelogDismissed: false,
    };
    this.switchTabs = this.switchTabs.bind(this);
    this.openInvestModel = this.openInvestModel.bind(this);
    this.closeInvestModel = this.closeInvestModel.bind(this);
    this.updateJobProperties = this.updateJobProperties.bind(this);
    this.saveJob = this.saveJob.bind(this);
    this.deleteJob = this.deleteJob.bind(this);
    this.clearRecentJobs = this.clearRecentJobs.bind(this);
    this.toggleDownloadModal = this.toggleDownloadModal.bind(this);
    this.toggleSettingsModal = this.toggleSettingsModal.bind(this);
    this.toggleMetadataModal = this.toggleMetadataModal.bind(this);
    this.togglePluginModal = this.togglePluginModal.bind(this);
    this.updateInvestList = this.updateInvestList.bind(this);
  }

  /** Initialize the list of invest models, recent invest jobs, etc. */
  async componentDidMount() {
    const investList = await this.updateInvestList();
    const recentJobs = await InvestJob.getJobStore();
    this.setState({
      // filter out models that do not exist in current version of invest
      recentJobs: recentJobs.filter((job) => (
        Object.keys(investList)
          .includes(job.modelID)
      )),
      showDownloadModal: this.props.isFirstRun,
      // Show changelog if this is a new version,
      // but if it's the first run ever, wait until after download modal closes.
      showChangelog: this.props.isNewVersion && !this.props.isFirstRun,
    });
    await i18n.changeLanguage(window.Workbench.LANGUAGE);
    ipcRenderer.on('download-status', (downloadedNofN) => {
      this.setState({
        downloadedNofN: downloadedNofN,
      });
    });
  }

  componentWillUnmount() {
    ipcRenderer.removeAllListeners('download-status');
  }

  /**
   * Change the tab that is currently visible.
   * @param {string} key - the value of one of the Nav.Link eventKey.
   */
  switchTabs(key) {
    this.setState(
      { activeTab: key }
    );
  }

  toggleDownloadModal(shouldShow) {
    this.setState({
      showDownloadModal: shouldShow,
    });
    // After close, show changelog if new version and app has just launched
    // (i.e., show changelog only once, after the first time the download modal closes).
    if (!shouldShow && this.props.isNewVersion && !this.state.changelogDismissed) {
      this.setState({
        showChangelog: true,
      });
    }
  }

  closeChangelogModal() {
    this.setState({
      showChangelog: false,
      changelogDismissed: true,
    });
  }

  togglePluginModal(show) {
    this.setState({
      showPluginModal: show
    });
  }

  toggleMetadataModal(show) {
    this.setState({
      showMetadataModal: show
    });
  }

  toggleSettingsModal(show) {
    this.setState({
      showSettingsModal: show
    });
  }

  /**
   * Push data for a new InvestTab component to an array.
   * @param {InvestJob} job - as constructed by new InvestJob()
   */
  openInvestModel(job) {
    const tabID = window.crypto.getRandomValues(
      new Uint32Array(1)
    ).toString();
    const { openJobs, openTabIDs } = this.state;
    openTabIDs.push(tabID);
    openJobs[tabID] = job;
    this.setState({
      openTabIDs: openTabIDs,
      openJobs: openJobs,
    }, () => this.switchTabs(tabID));
  }

  /**
   * Click handler for the close-tab button on an Invest model tab.
   * @param  {string} tabID - the eventKey of the tab containing the
   *   InvestTab component that will be removed.
   */
  closeInvestModel(tabID) {
    let index;
    const { openTabIDs, openJobs } = this.state;
    delete openJobs[tabID];
    openTabIDs.forEach((id) => {
      if (id === tabID) {
        index = openTabIDs.indexOf(tabID);
        openTabIDs.splice(index, 1);
      }
    });
    // Switch to the next tab if there is one, or the previous, or home.
    let switchTo = 'home';
    if (openTabIDs[index]) {
      switchTo = openTabIDs[index];
    } else if (openTabIDs[index - 1]) {
      switchTo = openTabIDs[index - 1];
    }
    this.switchTabs(switchTo);
    this.setState({
      openTabIDs: openTabIDs,
      openJobs: openJobs,
    });
  }

  /**
   * Update properties of an open InvestTab.
   * @param {string} tabID - the unique identifier of an open tab
   * @param {obj} jobObj - key-value pairs of any job properties to be updated
   */
  updateJobProperties(tabID, jobObj) {
    const { openJobs } = this.state;
    openJobs[tabID] = { ...openJobs[tabID], ...jobObj };
    this.setState({
      openJobs: openJobs
    });
  }

  /**
   * Save data describing an invest job to a persistent store.
   * @param {string} tabID - the unique identifier of an open InvestTab.
   */
  async saveJob(tabID) {
    const job = this.state.openJobs[tabID];
    const recentJobs = await InvestJob.saveJob(job);
    this.setState({
      recentJobs: recentJobs,
    });
  }

  /**
   * Delete the job record from the store.
   * @param {string} jobHash - the unique identifier of a saved Job.
   */
  async deleteJob(jobHash) {
    const recentJobs = await InvestJob.deleteJob(jobHash);
    this.setState({
      recentJobs: recentJobs,
    });
  }

  /**
   * Delete all the jobs from the store.
   */
  async clearRecentJobs() {
    const recentJobs = await InvestJob.clearStore();
    this.setState({
      recentJobs: recentJobs,
    });
  }

  async updateInvestList() {
    const coreModels = {};
    const investList = await getInvestModelIDs();
    Object.keys(investList).forEach((modelID) => {
      coreModels[modelID] = { modelTitle: investList[modelID].model_title, type: 'core' };
    });
    const plugins = await ipcRenderer.invoke(ipcMainChannels.GET_SETTING, 'plugins') || {};
    Object.keys(plugins).forEach((plugin) => {
      plugins[plugin].type = 'plugin';
    });
    this.setState({
      investList: { ...coreModels, ...plugins },
    });
    return { ...coreModels, ...plugins };
  }

  render() {
    const {
      investList,
      recentJobs,
      openJobs,
      openTabIDs,
      activeTab,
      showDownloadModal,
      showPluginModal,
      showChangelog,
      showSettingsModal,
      showMetadataModal,
      downloadedNofN,
    } = this.state;

    const investNavItems = [];
    const investTabPanes = [];
    openTabIDs.forEach((id) => {
      const job = openJobs[id];
      let statusSymbol;
      switch (job.status) {
        case 'success':
          statusSymbol = '\u{2705}'; // green check
          break;
        case 'error':
          statusSymbol = '\u{1F6AB}'; // red do-not-enter
          break;
        case 'running':
          statusSymbol = (
            <Spinner
              className="mb-1"
              animation="border"
              size="sm"
              role="status"
              aria-hidden="true"
            />
          );
          break;
        default:
          statusSymbol = '';
      }
      let badge;
      if (investList) {
        const modelType = investList[job.modelID].type;
        if (modelType === 'plugin') {
          badge = <Badge className="mr-1" variant="secondary">Plugin</Badge>;
        }
      }

      investNavItems.push(
        <OverlayTrigger
          key={`${id}-tooltip`}
          placement="bottom"
          overlay={(
            <Tooltip>
              {job.modelTitle}
            </Tooltip>
          )}
        >
          <Nav.Item
            key={id}
            className={id === activeTab ? 'active' : ''}
          >
            <Nav.Link
              eventKey={id}
              onAuxClick={(event) => {
                event.stopPropagation();
                event.preventDefault();
                if (event.button === 1) {
                  // middle mouse button clicked, close tab
                  this.closeInvestModel(id);
                }
              }}
            >
              {badge}
              {statusSymbol}
              {` ${job.modelTitle}`}
            </Nav.Link>
            <Button
              aria-label={`close ${job.modelTitle} tab`}
              className="close-tab"
              variant="outline-dark"
              onClick={(event) => {
                event.stopPropagation();
                this.closeInvestModel(id);
              }}
              onDragOver={dragOverHandlerNone}
            >
              <MdClose />
            </Button>
          </Nav.Item>
        </OverlayTrigger>
      );
      investTabPanes.push(
        <TabPane
          key={id}
          eventKey={id}
          aria-label={`${job.modelTitle} tab`}
        >
          <InvestTab
            job={job}
            tabID={id}
            saveJob={this.saveJob}
            updateJobProperties={this.updateJobProperties}
            investList={investList}
          />
        </TabPane>
      );
    });

    return (
      <React.Fragment>
        {showDownloadModal && (
          <DataDownloadModal
            show={showDownloadModal}
            closeModal={() => this.toggleDownloadModal(false)}
          />
        )}
        {showPluginModal && (
          <PluginModal
            show={showPluginModal}
            closeModal={() => this.togglePluginModal(false)}
            openModal={() => this.togglePluginModal(true)}
            updateInvestList={this.updateInvestList}
            closeInvestModel={this.closeInvestModel}
            openJobs={openJobs}
          />
        )}
        {showChangelog && (
          <Changelog
            show={showChangelog}
            close={() => this.closeChangelogModal()}
          />
        )}
        {showMetadataModal && (
          <MetadataModal
            show={showMetadataModal}
            close={() => this.toggleMetadataModal(false)}
          />
        )}
        {showSettingsModal && (
          <SettingsModal
            show={showSettingsModal}
            close={() => this.toggleSettingsModal(false)}
            nCPU={this.props.nCPU}
          />
        )}
        <TabContainer activeKey={activeTab}>
          <Navbar
            onDragOver={dragOverHandlerNone}
          >
            <Row
              className="w-100 flex-nowrap"
            >
              <Col sm={3}>
                <Navbar.Brand>
                  <Nav.Link
                    onSelect={this.switchTabs}
                    eventKey="home"
                  >
                    <MdHome />
                    InVEST
                  </Nav.Link>
                </Navbar.Brand>
                <AiOutlineTrademarkCircle className="rtm" />
              </Col>
              <Col className="navbar-middle">
                <Nav
                  justify
                  variant="tabs"
                  className="mr-auto"
                  activeKey={activeTab}
                  onSelect={this.switchTabs}
                >
                  {investNavItems}
                </Nav>
              </Col>
              <Col className="text-right navbar-right">
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
                <AppMenu
                  openDownloadModal={() => this.toggleDownloadModal(true)}
                  openPluginModal={() => this.togglePluginModal(true)}
                  openChangelogModal={() => this.setState({ showChangelog: true })}
                  openSettingsModal={() => this.toggleSettingsModal(true)}
                  openMetadataModal={() => this.toggleMetadataModal(true)}
                />
              </Col>
            </Row>
          </Navbar>

          <TabContent
            id="home-tab-content"
            onDragOver={dragOverHandlerNone}
          >
            <TabPane
              eventKey="home"
              aria-label="home tab"
            >
              {(investList)
                ? (
                  <HomeTab
                    investList={investList}
                    openInvestModel={this.openInvestModel}
                    recentJobs={recentJobs}
                    batchUpdateArgs={this.batchUpdateArgs}
                    deleteJob={this.deleteJob}
                    clearRecentJobs={this.clearRecentJobs}
                  />
                ) : <div />}
            </TabPane>
            {investTabPanes}
          </TabContent>
        </TabContainer>
      </React.Fragment>
    );
  }
}

App.propTypes = {
  isFirstRun: PropTypes.bool,
  isNewVersion: PropTypes.bool,
  nCPU: PropTypes.number,
};

// Setting a default here mainly to make testing easy, so these props
// can be undefined for unrelated tests.
App.defaultProps = {
  isFirstRun: false,
  isNewVersion: false,
  nCPU: 1,
};
