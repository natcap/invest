import { ipcMainChannels } from '../../main/ipcMainChannels';
import { handleClick } from './handlers';

const { ipcRenderer } = window.Workbench.electron;
const { LOGFILE_PATH } = window.Workbench;

function handleButtonClick() {
  ipcRenderer.send(
    ipcMainChannels.SHOW_ITEM_IN_FOLDER,
    LOGFILE_PATH
  );
}

document.querySelector('button').addEventListener('click', handleButtonClick);
document.querySelectorAll('a').forEach(
  (element) => {
    element.addEventListener('click', handleClick);
  }
);
