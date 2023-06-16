import { ipcMainChannels } from '../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;

export function handleClickExternalURL(event) {
  event.preventDefault();
  ipcRenderer.send(
    ipcMainChannels.OPEN_EXTERNAL_URL, event.currentTarget.href
  );
}

export function handleClickFindLogfiles() {
  ipcRenderer.send(
    ipcMainChannels.SHOW_ITEM_IN_FOLDER,
    window.Workbench.ELECTRON_LOG_PATH,
  );
}
