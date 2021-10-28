import { Menu } from 'electron'; // eslint-disable-line import/no-extraneous-dependencies

import ELECTRON_DEV_MODE from './isDevMode';

// context-menu options for selected text
const selectionArray = [
  { role: 'copy' },
  { type: 'separator' },
  { role: 'selectall' },
];

// context-menu options for editable input fields
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

/** Setup listener for right-clicks on a given browser window.
 *
 * The content of the menu is constructed based on whether the target
 * is an editable element or selected text. And whether we're in dev mode.
 *
 * @param {BrowserWindow} win - an instance of an electron BrowserWindow.
 */
export default function setupContextMenu(win) {
  win.webContents.on('context-menu', (event, props) => {
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
