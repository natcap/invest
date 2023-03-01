import { ipcMainChannels } from '../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;
const { LOGFILE_PATH } = window.Workbench;

export function handleClickExternalURL(event) {
  event.preventDefault();
  ipcRenderer.send(
    ipcMainChannels.OPEN_EXTERNAL_URL, event.currentTarget.href
  );
}

export function handleClickFindLogfiles() {
  ipcRenderer.send(
    ipcMainChannels.SHOW_ITEM_IN_FOLDER,
    LOGFILE_PATH
  );
}
