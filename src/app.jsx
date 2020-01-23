import fs from 'fs';
import path from 'path';
import React from 'react';

import {InvestJob} from './InvestJob';
import { getInvestList } from './getInvestList'

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
    const recentSessions = await findRecentSessions(CACHE_DIR);
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

  updateRecentSessions(sessionID) {
    // This triggers on saveState clicks
    let recentSessions = Object.assign([], this.state.recentSessions);
    recentSessions.unshift(sessionID);
    this.setState({recentSessions: recentSessions});
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

function findRecentSessions(cache_dir) {
  // Populate recentSessions from list of files in cache dir
  // sorted by modified time.

  // TODO: check that files are actually state config files
  // before putting them on the array
  return new Promise(function(resolve, reject) {
    const files = fs.readdirSync(cache_dir);

    // reverse sort (b - a) based on last-modified time
    const sortedFiles = files.sort(function(a, b) {
      return fs.statSync(path.join(cache_dir, b)).mtimeMs -
           fs.statSync(path.join(cache_dir, a)).mtimeMs
    });
    // trim off extension, since that is how sessions
    // were named orginally
    resolve(sortedFiles
      .map(f => path.parse(f).name)
      .slice(0, 15) // max 15 items returned
    );
  });
}

