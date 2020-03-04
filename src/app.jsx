import React from 'react';

import { InvestJob } from './InvestJob';
import { investList } from './server_requests';
import { findRecentSessions, loadRecentSessions } from './utils';

const CACHE_DIR = 'cache' //  for storing state snapshot files

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
    const recentSessions = await loadRecentSessions()
    console.log(recentSessions);
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

  async updateRecentSessions() {
    // This triggers on saveState clicks
    // let recentSessions = Object.assign([], this.state.recentSessions);
    // recentSessions.unshift(sessionID);
    // this.setState({recentSessions: recentSessions});
    const recentSessions = await loadRecentSessions();
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
        updateRecentSessions={this.updateRecentSessions}
        saveSettings={this.saveSettings}
      />
    );
  }
}

function getInvestList() {
  return new Promise(function(resolve, reject) {
    setTimeout(() => {
      resolve(investList())
    }, 500)  // wait, the server only just launched in a subprocess.
  });
}