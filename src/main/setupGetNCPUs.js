import os from 'os';

import { ipcMain } from 'electron';

import { ipcMainChannels } from './ipcMainChannels';

export default function setupGetNCPUs() {
  ipcMain.handle(
    ipcMainChannels.GET_N_CPUS, () => {
      return os.cpus().length;
    }
  );
}
