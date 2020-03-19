import React from 'react';
import PropTypes from 'prop-types';

import { InvestJob } from './InvestJob';
import { investList } from './server_requests';
import { updateRecentSessions, loadRecentSessions } from './utils';

export default class App extends React.Component {

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
    // TODO: also load and set investSettings from a cached state
    const investList = await getInvestList();
    const recentSessions = await loadRecentSessions(this.props.appdata)
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
    // This triggers on saveState clicks
    // let recentSessions = Object.assign([], this.state.recentSessions);
    // recentSessions.unshift(sessionID);
    // this.setState({recentSessions: recentSessions});
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
  return new Promise(function(resolve, reject) {
    setTimeout(() => {
      resolve(investList())
    }, 500)  // wait, the server only just launched in a subprocess.
  });
}