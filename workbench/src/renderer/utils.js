import i18n from 'i18next';

import { ipcMainChannels } from '../main/ipcMainChannels';
import { fetchDatastackFromFile } from './server_requests';

const { ipcRenderer } = window.Workbench.electron;
const { logger } = window.Workbench;

/**
 * Create a JSON string with invest argument keys and values.
 * @param {object} args - object keyed by invest argument keys and
 *   with each item including a `value` property, among others.
 * @returns {object} - invest argument key: value pairs as expected
 * by invest model `execute` and `validate` functions
 */
export function argsDictFromObject(args) {
  const argsDict = {};
  Object.keys(args).forEach((argname) => {
    argsDict[argname] = args[argname].value;
  });
  return argsDict;
}

/**
 * Prevent the default case for onDragOver and set dropEffect to none.
 * @param {Event} event - an ondragover event.
 */
export function dragOverHandlerNone(event) {
  event.preventDefault();
  event.stopPropagation();
  event.dataTransfer.dropEffect = 'none';
}

/**
 * Open the target href in the default web browser.
 * @param {Event} event - with a currentTarget.href property.
 */
export function openLinkInBrowser(event) {
  event.preventDefault();
  ipcRenderer.send(
    ipcMainChannels.OPEN_EXTERNAL_URL,
    event.currentTarget.href
  );
}

export async function openDatastack(filepath) {
  const { t } = i18n;
  let datastack;
  try {
    if (filepath.endsWith('gz')) { // .tar.gz, .tgz
      const extractLocation = await ipcRenderer.invoke(
        ipcMainChannels.SHOW_OPEN_DIALOG,
        {
          // title is only for Windows, but default 'Open' is misleading
          title: i18n.t('Choose a directory'),
          buttonLabel: t('Extract archive here'),
          properties: ['openDirectory', 'createDirectory'],
        }
      );
      if (extractLocation.filePaths.length) {
        const directoryPath = extractLocation.filePaths[0];
        const writable = await ipcRenderer.invoke(
          ipcMainChannels.CHECK_FILE_PERMISSIONS, directoryPath);
        if (writable) {
          const newDir = filepath.split(/\\+|\/+/).pop()
            .replace(/\.tgz|\.tar\.gz/gi, '');
          datastack = await fetchDatastackFromFile({
            filepath: filepath,
            extractPath: `${directoryPath}/${newDir}`,
          });
        } else {
          throw new Error(`${t('Permission denied extracting files to:')}\n${directoryPath}`);
        }
      }
    } else {
      datastack = await fetchDatastackFromFile({ filepath: filepath });
    }
  } catch (error) {
    logger.error(error);
    throw new Error(`${t('No InVEST model data can be parsed from the file:')}\n${filepath}`);
  }
  return datastack;
}
