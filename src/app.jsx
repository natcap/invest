import React from 'react';
import PropTypes from 'prop-types';

import { InvestJob } from './InvestJob';
import { investList } from './server_requests';
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
    /** Initialize the list of available invest models and recent invest jobs.
    */
    const investList = await getInvestList();
    const recentSessions = await loadRecentSessions(this.props.appdata)
    // TODO: also load and set investSettings from a cached state, instead 
    // of always re-setting to these hardcoded values.
    this.setState(
      {
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

function getInvestList() {
  /** A wrapper around a server call that waits before making the request
  * because the flask server only just launched in a subprocess. 
  * TODO: is there a better way to control that this request never
  * happens before the server is ready?
  */
  return new Promise(function(resolve, reject) {
    setTimeout(() => {
      resolve(investList())
    }, 500)  // wait, the server only just launched in a subprocess.
  });
}