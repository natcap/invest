import { Menu } from 'electron';

import ELECTRON_DEV_MODE from './isDevMode';

const selectionArray = [
  { role: 'copy' },
  { type: 'separator' },
  { role: 'selectall' },
];

const inputArray = [
  { role: 'undo' },
  { role: 'redo' },
  { type: 'separator' },
  { role: 'cut' },
  { role: 'copy' },
  { role: 'paste' },
  { type: 'separator' },
  { role: 'selectall' },
];

export default function setupContextMenu(win) {
  win.webContents.on('context-menu', (e, props) => {
    const template = [];
    if (ELECTRON_DEV_MODE) {
      template.push({
        label: 'Inspect Element',
        click: () => {
          win.inspectElement(props.x, props.y);
        }
      });
      template.push({ type: 'separator' });
    }
    const { selectionText, isEditable } = props;
    let menu;
    if (isEditable) {
      menu = Menu.buildFromTemplate(template.concat(inputArray));
    } else if (selectionText && selectionText.trim() !== '') {
      menu = Menu.buildFromTemplate(template.concat(selectionArray));
    } else {
      menu = Menu.buildFromTemplate(template);
    }
    menu.popup(win);
  });
}
