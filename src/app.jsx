import React from 'react';
import PropTypes from 'prop-types';

import { InvestJob } from './InvestJob';
import { getInvestList, getFlaskIsReady } from './server_requests';
import { updateRecentSessions, loadRecentSessions } from './utils';

const DIRECTORY_CONSTANTS = {
  CACHE_DIR: path.join(__dirname, 'cache'), //  for storing state snapshot files
  TEMP_DIR: path.join(__dirname, 'tmp'),  // for saving datastack json files prior to investExecute
  INVEST_UI_DATA: path.join(__dirname, 'ui_data')
}

export default class App extends React.Component {
  /** This component manages any application state that should persist
  * and be independent from properties of a single invest job.
  */

  constructor(props) {
    super(props);
    this.state = {
      investEXE: undefined,
      investList: {},
      recentSessions: [],
      investSettings: {},
    };
    this.updateRecentSessions = this.updateRecentSessions.bind(this);
    this.saveSettings = this.saveSettings.bind(this);
  }

  async componentDidMount() {
    /** Initialize the list of available invest models and recent invest jobs.*/

    // Inexplicably, when getFlaskIsReady is immediately followed by getInvestList,
    // ~50% of the time getInvestList is called but never returns, or even calls fetch.
    // Adding even a 1ms pause between calls avoids that problem, as does not
    // calling getFlaskIsReady. Unless we observe the app trying getInvestList
    // before flask is ready, we don't need getFlaskIsReady anyway.
    // const readydata = await getFlaskIsReady(); 
    // await new Promise(resolve => setTimeout(resolve, 1));
    const investList = await getInvestList();
    const recentSessions = await loadRecentSessions(this.props.appdata)
    // TODO: also load and set investSettings from a cached state, instead 
    // of always re-setting to these hardcoded values on first launch?

    const version = this.props.investRegistry['active']
    const investEXE = this.props.investRegistry['registry'][version]['invest']
    this.setState(
      {
        investEXE: investEXE,
        investList: investList,
        recentSessions: recentSessions,
        investSettings: {
          nWorkers: '-1',
          loggingLevel: 'INFO',
        }
      });
  }

  async updateRecentSessions(jobdata) {
    /** Update the recent sessions list when a new invest job was saved.
    * This triggers on InvestJob.saveState().
    * 
    * @param {object} jobdata - the metadata describing an invest job.
    */
    const recentSessions = await updateRecentSessions(jobdata, this.props.appdata);
    this.setState({
      recentSessions: recentSessions
    })
  }

  saveSettings(settings) {
    this.setState({
      investSettings: settings
    });
  }

  render() {
    return (
      <InvestJob
        investEXE={this.state.investEXE} 
        investList={this.state.investList}
        investSettings={this.state.investSettings}
        recentSessions={this.state.recentSessions}
        appdata={this.props.appdata}
        directoryConstants={DIRECTORY_CONSTANTS}
        updateRecentSessions={this.updateRecentSessions}
        saveSettings={this.saveSettings}
      />
    );
  }
}

App.propTypes = {
  appdata: PropTypes.string
}
