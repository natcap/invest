import path from 'path';
import { remote } from 'electron'; // eslint-disable-line import/no-extraneous-dependencies

const USER_DATA = remote.app.getPath('userData');

export const fileRegistry = {
  //  for storing state snapshot files
  CACHE_DIR: path.join(USER_DATA, 'state_cache'),
  JOBS_DATABASE: path.join(USER_DATA, 'state_cache', 'jobdb.json'),

  // for saving datastack json files prior to investExecute
  TEMP_DIR: path.join(USER_DATA, 'tmp'),

  // to track local, alternate invest binaries
  INVEST_REGISTRY_PATH: path.join(USER_DATA, 'invest_registry.json'),

  // UI spec data
  INVEST_UI_DATA: path.join(__dirname, 'ui_data'),
};
