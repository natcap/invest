import path from 'path';
import { remote } from 'electron';

export const directories = {
	//  for storing state snapshot files
	CACHE_DIR: path.join(remote.app.getPath('userData'), 'state_cache'),
	// for saving datastack json files prior to investExecute
	TEMP_DIR: path.join(remote.app.getPath('userData'), 'tmp'),
	INVEST_UI_DATA: path.join(__dirname, 'ui_data')
}
