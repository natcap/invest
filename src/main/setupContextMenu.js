import {
  BrowserWindow,
  ipcMain,
  Menu
} from 'electron';

import ELECTRON_DEV_MODE from './isDevMode';
import { ipcMainChannels } from './ipcMainChannels';

export default function setupContextMenu() {
  ipcMain.on(ipcMainChannels.SHOW_CONTEXT_MENU, (event, rightClickPos) => {
    const template = [];
    if (ELECTRON_DEV_MODE) {
      template.push({
        label: 'Inspect Element',
        click: () => {
          BrowserWindow.fromWebContents(event.sender)
            .inspectElement(rightClickPos.x, rightClickPos.y);
        }
      });
    }
    const menu = Menu.buildFromTemplate(template);
    menu.popup(BrowserWindow.fromWebContents(event.sender));
  });
}
