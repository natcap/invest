import fs from 'fs';
import React from 'react';
import PropTypes from 'prop-types';

import InvestJob from './InvestJob';
import { getInvestList } from './server_requests';
import { updateRecentSessions, loadRecentSessions } from './utils';

/** This component manages any application state that should persist
 * and be independent from properties of a single invest job.
 */
export default class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      investList: {},
      recentSessions: [],
      investSettings: {},
    };
    this.setRecentSessions = this.setRecentSessions.bind(this);
    this.saveSettings = this.saveSettings.bind(this);
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

  saveSettings(settings) {
    this.setState({
      investSettings: settings,
    });
  }

  render() {
    const { investExe, jobDatabase } = this.props;
    const { investList, investSettings, recentSessions } = this.state;
    return (
      <InvestJob
        investExe={investExe}
        investList={investList}
        investSettings={investSettings}
        recentSessions={recentSessions}
        jobDatabase={jobDatabase}
        updateRecentSessions={this.setRecentSessions}
        saveSettings={this.saveSettings}
      />
    );
  }
}

App.propTypes = {
  investExe: PropTypes.string.isRequired,
  jobDatabase: PropTypes.string,
};
