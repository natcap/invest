import {
  ipcMain,
} from 'electron';

import { ipcMainChannels } from './ipcMainChannels';

import baseUrl from './baseUrl';

export function setupBaseUrl() {
  ipcMain.handle(
    ipcMainChannels.BASE_URL, () => baseUrl
  );
}
