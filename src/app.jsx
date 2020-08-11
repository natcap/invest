import fs from 'fs';
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

/** This component manages any application state that should persist
 * and be independent from properties of a single invest job.
 */
export default class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      activeTab: 'home',
      openModels: [],
      argsInitDict: undefined,
      investList: {},
      recentSessions: [],
      investSettings: {},
    };
    this.setRecentSessions = this.setRecentSessions.bind(this);
    this.saveSettings = this.saveSettings.bind(this);
    this.switchTabs = this.switchTabs.bind(this);
    this.openInvestModel = this.openInvestModel.bind(this);
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

  openInvestModel(modelRunName) {
    this.setState((state) => ({
      openModels: [...state.openModels, modelRunName],
    }), () => this.switchTabs('invest'));
  }

  render() {
    const { investExe, jobDatabase } = this.props;
    const {
      investList,
      investSettings,
      recentSessions,
      openModels,
      argsInitDict,
      activeTab,
    } = this.state;

    const investNavItems = [];
    const investTabPanes = [];
    openModels.forEach((modelRunName) => {
      investNavItems.push(
        <Nav.Item>
          <Nav.Link eventKey={modelRunName} disabled={false}>
            {modelRunName}
          </Nav.Link>
        </Nav.Item>
      );
      investTabPanes.push(
        <TabPane eventKey={modelRunName} title={modelRunName}>
          <InvestJob
            investExe={investExe}
            modelRunName={modelRunName}
            argsInitDict={argsInitDict}
            investSettings={investSettings}
            jobDatabase={jobDatabase}
            updateRecentSessions={this.setRecentSessions}
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
            investGetSpec={this.investGetSpec}
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
              saveState={this.saveState}
              loadState={this.loadState}
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
