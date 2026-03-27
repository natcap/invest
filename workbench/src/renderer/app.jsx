import React, { useEffect, useState } from 'react';
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
export default function App(props) {

  const [activeTab, setActiveTab] = useState('home');
  const [openJobs, setOpenJobs] = useState(new Map());
  const [investList, setInvestList] = useState(null);
  const [recentJobs, setRecentJobs] = useState([]);
  const [showDownloadModal, setShowDownloadModal] = useState(false);
  const [showPluginModal, setShowPluginModal] = useState(false);
  const [downloadedNofN, setDownloadedNofN] = useState(null);
  const [showChangelog, setShowChangelog] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [showMetadataModal, setShowMetadataModal] = useState(false);
  const [changelogDismissed, setChangelogDismissed] = useState(false);

  /** Initialize the list of invest models, recent invest jobs, etc. */
  useEffect(() => {
    async function setup() {
      await updateInvestList();
      setShowDownloadModal(props.isFirstRun);
      // Show changelog if this is a new version,
      // but if it's the first run ever, wait until after download modal closes.
      setShowChangelog(props.isNewVersion && !props.isFirstRun);
      await i18n.changeLanguage(window.Workbench.LANGUAGE);
      ipcRenderer.on('download-status', (downloadedNofN) => {
        setDownloadedNofN(downloadedNofN)
      });
    }
    setup();

    // Cleanup function to execute on component unmount
    return () => { ipcRenderer.removeAllListeners('download-status'); }
  }, []);

  // When investList changes, re-filter the recent jobs according to the new list
  useEffect(() => {
    updateRecentJobs();
  }, [investList]);

  function toggleDownloadModal(shouldShow) {
    setShowDownloadModal(shouldShow);
    // After close, show changelog if new version and app has just launched
    // (i.e., show changelog only once, after the first time the download modal closes).
    if (!shouldShow && props.isNewVersion && !changelogDismissed) {
      setShowChangelog(true);
    }
  }

  function closeChangelogModal() {
    setShowChangelog(false);
    setChangelogDismissed(true);
  }

  /**
   * Push data for a new InvestTab component to an array.
   * @param {InvestJob} job - as constructed by new InvestJob()
   */
  function openInvestModel(job) {
    const tabID = window.crypto.getRandomValues(
      new Uint32Array(1)
    ).toString();
    const newOpenJobs = new Map(openJobs);
    newOpenJobs.set(tabID, job);
    setOpenJobs(newOpenJobs);
    setActiveTab(tabID);
  }

  /**
   * Click handler for the close-tab button on an Invest model tab.
   * @param  {string} tabID - the eventKey of the tab containing the
   *   InvestTab component that will be removed.
   */
  function closeInvestModel(tabID) {
    // Find the tab ID to switch to once this tab is closed
    const openTabIDs = Array.from(openJobs.keys());
    const index = openTabIDs.indexOf(tabID);
    let switchTo;
    // Switch to the next tab, if there is one
    if (openTabIDs[index + 1]) {
      switchTo = openTabIDs[index + 1];
    // Otherwise, switch to the previous tab, if there is one
    } else if (openTabIDs[index - 1]) {
      switchTo = openTabIDs[index - 1];
    // Otherwise, there are no tabs left. Switch to home.
    } else {
      switchTo = 'home';
    }
    const newOpenJobs = new Map(openJobs);
    newOpenJobs.delete(tabID);
    setOpenJobs(newOpenJobs);
    setActiveTab(switchTo);
  }

  /**
   * Update properties of an open InvestTab.
   * @param {string} tabID - the unique identifier of an open tab
   * @param {obj} jobObj - key-value pairs of any job properties to be updated
   * @param {boolean} save - if true, save the updated job to persistent store
   */
  async function updateJobProperties(tabID, jobObj, save = false) {
    const newOpenJobs = new Map(openJobs);
    const updatedJob = { ...openJobs.get(tabID), ...jobObj };
    newOpenJobs.set(tabID, updatedJob);
    setOpenJobs(newOpenJobs);
    if (save) {
      await InvestJob.saveJob(updatedJob);
      updateRecentJobs();
    }
  }

  /**
   * Delete the job record from the store.
   * @param {string} jobHash - the unique identifier of a saved Job.
   */
  async function deleteJob(jobHash) {
    await InvestJob.deleteJob(jobHash);
    updateRecentJobs();
  }

  /**
   * Delete all the jobs from the store.
   */
  async function clearRecentJobs() {
    await InvestJob.clearStore();
    updateRecentJobs();
  }

  async function updateInvestList() {
    const coreModels = {};
    let investList = await getInvestModelIDs();
    Object.keys(investList).forEach((modelID) => {
      coreModels[modelID] = { modelTitle: investList[modelID].model_title, type: 'core' };
    });
    const plugins = await ipcRenderer.invoke(ipcMainChannels.GET_SETTING, 'plugins') || {};
    Object.keys(plugins).forEach((plugin) => {
      plugins[plugin].type = 'plugin';
    });
    investList = { ...coreModels, ...plugins };
    setInvestList(investList);
    return { ...coreModels, ...plugins };
  }

  async function updateRecentJobs() {
    if (investList) {
      const recentJobs = await InvestJob.getJobStore();
      // filter out models that do not exist in current version of invest
      setRecentJobs(recentJobs.filter((job) => (
          Object.keys(investList).includes(job.modelID)
        ))
      );
    }
  }

  const investNavItems = [];
  const investTabPanes = [];
  openJobs.forEach((job, id) => {
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
                closeInvestModel(id);
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
              closeInvestModel(id);
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
          updateJobProperties={updateJobProperties}
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
          closeModal={() => toggleDownloadModal(false)}
        />
      )}
      {showPluginModal && (
        <PluginModal
          show={showPluginModal}
          closeModal={() => setShowPluginModal(false)}
          openModal={() => setShowPluginModal(true)}
          updateInvestList={updateInvestList}
          closeInvestModel={closeInvestModel}
          openJobs={openJobs}
        />
      )}
      {showChangelog && (
        <Changelog
          show={showChangelog}
          close={() => closeChangelogModal()}
        />
      )}
      {showMetadataModal && (
        <MetadataModal
          show={showMetadataModal}
          close={() => setShowMetadataModal(false)}
        />
      )}
      {showSettingsModal && (
        <SettingsModal
          show={showSettingsModal}
          close={() => setShowSettingsModal(false)}
          nCPU={props.nCPU}
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
                  onSelect={setActiveTab}
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
                onSelect={setActiveTab}
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
                openDownloadModal={() => toggleDownloadModal(true)}
                openPluginModal={() => setShowPluginModal(true)}
                openChangelogModal={() => setShowChangelog(true)}
                openSettingsModal={() => setShowSettingsModal(true)}
                openMetadataModal={() => setShowMetadataModal(true)}
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
                  openInvestModel={openInvestModel}
                  recentJobs={recentJobs}
                  deleteJob={deleteJob}
                  clearRecentJobs={clearRecentJobs}
                />
              ) : <div />}
          </TabPane>
          {investTabPanes}
        </TabContent>
      </TabContainer>
    </React.Fragment>
  );
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
