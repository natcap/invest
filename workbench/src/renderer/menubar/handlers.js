import { ipcMainChannels } from '../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;

export function handleClick(event) {
  event.preventDefault();
  ipcRenderer.send(
    ipcMainChannels.OPEN_EXTERNAL_URL, event.currentTarget.href
  );
}
