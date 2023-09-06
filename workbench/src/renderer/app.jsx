import React from 'react';
import PropTypes from 'prop-types';
import i18n from 'i18next';

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
import SettingsModal from './components/SettingsModal';
import DataDownloadModal from './components/DataDownloadModal';
import DownloadProgressBar from './components/DownloadProgressBar';
import { getInvestModelNames } from './server_requests';
import InvestJob from './InvestJob';
import { dragOverHandlerNone } from './utils';

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
      downloadedNofN: null,
    };
    this.switchTabs = this.switchTabs.bind(this);
    this.openInvestModel = this.openInvestModel.bind(this);
    this.closeInvestModel = this.closeInvestModel.bind(this);
    this.updateJobProperties = this.updateJobProperties.bind(this);
    this.saveJob = this.saveJob.bind(this);
    this.clearRecentJobs = this.clearRecentJobs.bind(this);
    this.showDownloadModal = this.showDownloadModal.bind(this);
  }

  /** Initialize the list of invest models, recent invest jobs, etc. */
  async componentDidMount() {
    const investList = await getInvestModelNames();
    const recentJobs = await InvestJob.getJobStore();
    this.setState({
      investList: investList,
      // filter out models that do not exist in current version of invest
      recentJobs: recentJobs.filter((job) => (
        Object.values(investList)
          .map((m) => m.model_name)
          .includes(job.modelRunName)
      )),
      showDownloadModal: this.props.isFirstRun,
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

  /** Change the tab that is currently visible.
   *
   * @param {string} key - the value of one of the Nav.Link eventKey.
   */
  switchTabs(key) {
    this.setState(
      { activeTab: key }
    );
  }

  showDownloadModal(shouldShow) {
    this.setState({
      showDownloadModal: shouldShow,
    });
  }

  /** Push data for a new InvestTab component to an array.
   *
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
   *
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

  /** Update properties of an open InvestTab.
   *
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

  /** Save data describing an invest job to a persistent store.
   *
   * And update the app's view of that store.
   *
   * @param {string} tabID - the unique identifier of an open InvestTab.
   */
  async saveJob(tabID) {
    const job = this.state.openJobs[tabID];
    const recentJobs = await InvestJob.saveJob(job);
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
    const {
      investList,
      recentJobs,
      openJobs,
      openTabIDs,
      activeTab,
      showDownloadModal,
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
      investNavItems.push(
        <OverlayTrigger
          key={`${id}-tooltip`}
          placement="bottom"
          overlay={(
            <Tooltip>
              {job.modelHumanName}
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
              {statusSymbol}
              {` ${job.modelHumanName}`}
            </Nav.Link>
            <Button
              aria-label={`close ${job.modelHumanName} tab`}
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
          aria-label={`${job.modelHumanName} tab`}
        >
          <InvestTab
            job={job}
            tabID={id}
            saveJob={this.saveJob}
            updateJobProperties={this.updateJobProperties}
          />
        </TabPane>
      );
    });

    return (
      <React.Fragment>
        <DataDownloadModal
          show={showDownloadModal}
          closeModal={() => this.showDownloadModal(false)}
        />
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
                <SettingsModal
                  className="mx-3"
                  clearJobsStorage={this.clearRecentJobs}
                  showDownloadModal={() => this.showDownloadModal(true)}
                  nCPU={this.props.nCPU}
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
                  />
                )
                : <div />}
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
  nCPU: PropTypes.number,
};

// Setting a default here mainly to make testing easy, so these props
// can be undefined for unrelated tests.
App.defaultProps = {
  isFirstRun: false,
  nCPU: 1,
};
