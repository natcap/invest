import { ipcMainChannels } from '../../main/ipcMainChannels';
import { handleClick } from './handlers';

const { ipcRenderer } = window.Workbench.electron;

async function getInvestVersion() {
  const investVersion = await ipcRenderer.invoke(ipcMainChannels.INVEST_VERSION);
  return investVersion;
}

document.querySelectorAll('a').forEach(
  (element) => {
    element.addEventListener('click', handleClick);
  }
);
const node = document.getElementById('version-string');
const investVersion = await getInvestVersion();
const text = document.createTextNode(investVersion);
node.appendChild(text);
