import {
  BrowserWindow,
  ipcMain,
  Menu
} from 'electron';

export default function setupContextMenu() {
  ipcMain.on('show-context-menu', (event, rightClickPos) => {
    const template = [
      {
        label: 'Inspect Element',
        click: () => {
          BrowserWindow.fromWebContents(event.sender)
            .inspectElement(rightClickPos.x, rightClickPos.y);
        }
      },
    ];
    const menu = Menu.buildFromTemplate(template);
    menu.popup(BrowserWindow.fromWebContents(event.sender));
  });
}
