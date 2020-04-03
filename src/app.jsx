import React from 'react';
import PropTypes from 'prop-types';

import { InvestJob } from './InvestJob';
import { getInvestList, getFlaskIsReady } from './server_requests';
import { updateRecentSessions, loadRecentSessions } from './utils';

export default class App extends React.Component {
  /** This component manages any application state that should persist
  * and be independent from properties of a single invest job.
  */

  constructor(props) {
    super(props);
    this.state = {
      investList: {},
      recentSessions: [],
      investSettings: {},
    };
    this.updateRecentSessions = this.updateRecentSessions.bind(this);
    this.saveSettings = this.saveSettings.bind(this);
  }

  async componentDidMount() {
    /** Initialize the list of available invest models and recent invest jobs.*/

    // TODO: intermittently (10 - 20% of times) we don't get to getInvestList.
    // Hence all the logging here.
    const readydata = await getFlaskIsReady();  // The app's first server calls follow this
    console.log(readydata)
    const investList = await getInvestList();
    console.log('app invest list')
    const recentSessions = await loadRecentSessions(this.props.appdata)
    console.log('app load recents')
    // TODO: also load and set investSettings from a cached state, instead 
    // of always re-setting to these hardcoded values on first launch?
    this.setState(
      {
        investList: investList,
        recentSessions: recentSessions,
        investSettings: {
          nWorkers: '-1',
          loggingLevel: 'INFO',
        }
      }, () => {console.log('app first setstate')});
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
        investList={this.state.investList}
        investSettings={this.state.investSettings}
        recentSessions={this.state.recentSessions}
        appdata={this.props.appdata}
        updateRecentSessions={this.updateRecentSessions}
        saveSettings={this.saveSettings}
      />
    );
  }
}

App.propTypes = {
  appdata: PropTypes.string
}
