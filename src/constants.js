import path from 'path';
import { ipcRenderer } from 'electron';

//const USER_DATA = remote.app.getPath('userData');
const USER_DATA = await ipcRenderer.invoke('user-data');

export const fileRegistry = {
  //  for storing state snapshot files
  CACHE_DIR: path.join(USER_DATA, 'state_cache'),

  // for saving datastack json files prior to investExecute
  TEMP_DIR: path.join(USER_DATA, 'tmp'),

  // to track local, alternate invest binaries
  INVEST_REGISTRY_PATH: path.join(USER_DATA, 'invest_registry.json'),

  // UI spec data
  INVEST_UI_DATA: path.join(__dirname, 'ui_data'),
};
