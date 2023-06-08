import { ipcMainChannels } from '../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;

export function handleClickExternalURL(event) {
  event.preventDefault();
  ipcRenderer.send(
    ipcMainChannels.OPEN_EXTERNAL_URL, event.currentTarget.href
  );
}

export async function handleClickFindLogfiles() {
  const logfilePath = await ipcRenderer.invoke(
    ipcMainChannels.GET_ELECTRON_LOG_PATH
  );
  ipcRenderer.send(
    ipcMainChannels.SHOW_ITEM_IN_FOLDER,
    logfilePath
  );
}
