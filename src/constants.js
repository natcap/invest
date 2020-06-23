import path from 'path';
import { remote } from 'electron';

const USER_DATA = remote.app.getPath('userData')

export const directories = {
	//  for storing state snapshot files
	CACHE_DIR: path.join(USER_DATA, 'state_cache'),
	// for saving datastack json files prior to investExecute
	TEMP_DIR: path.join(USER_DATA, 'tmp'),
	INVEST_UI_DATA: path.join(__dirname, 'ui_data')
}
