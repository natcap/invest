import path from 'path';

// eslint-disable-next-line import/no-extraneous-dependencies
import {
  app,
  BrowserWindow,
  screen,
  nativeTheme,
  Menu,
  ipcMain
} from 'electron';

import {
  createPythonFlaskProcess,
  getFlaskIsReady,
  shutdownPythonProcess
} from './createPythonFlaskProcess';
import findInvestBinaries from './findInvestBinaries';
import setupDownloadHandlers from './setupDownloadHandlers';
import setupDialogs from './setupDialogs';
import setupContextMenu from './setupContextMenu';
import { setupCheckFirstRun } from './setupCheckFirstRun';
import { setupCheckStorageToken } from './setupCheckStorageToken';
import {
  setupInvestRunHandlers,
  setupInvestLogReaderHandler
} from './setupInvestHandlers';
import setupSetLanguage from './setLanguage';
import setupGetNCPUs from './setupGetNCPUs';
import setupOpenExternalUrl from './setupOpenExternalUrl';
import { ipcMainChannels } from './ipcMainChannels';
import menuTemplate from './menubar';
import ELECTRON_DEV_MODE from './isDevMode';
import BASE_URL from './baseUrl';
import { getLogger } from './logger';
import pkg from '../../package.json';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

if (!process.env.PORT) {
  process.env.PORT = '56789';
}

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow;
let splashScreen;
let flaskSubprocess;

export function destroyWindow() {
  mainWindow = null;
}

/** Create an Electron browser window and start the flask application. */
export const createWindow = async () => {
  logger.info(`Running invest-workbench version ${pkg.version}`);
  nativeTheme.themeSource = 'light'; // override OS/browser setting

  splashScreen = new BrowserWindow({
    width: 574, // dims set to match the image in splash.html
    height: 500,
    transparent: true,
    frame: false,
    alwaysOnTop: false,
  });
  splashScreen.loadURL(path.join(BASE_URL, 'splash.html'));

  const investExe = findInvestBinaries(ELECTRON_DEV_MODE);
  flaskSubprocess = createPythonFlaskProcess(investExe);
  setupDialogs();
  setupCheckFirstRun();
  setupCheckStorageToken();
  await getFlaskIsReady();

  // Create the browser window.
  mainWindow = new BrowserWindow({
    minWidth: 800,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, '../preload/preload.js'),
      defaultEncoding: 'UTF-8',
    },
  });
  const menubar = Menu.buildFromTemplate(
    menuTemplate(mainWindow, ELECTRON_DEV_MODE)
  );
  Menu.setApplicationMenu(menubar);
  mainWindow.loadURL(path.join(BASE_URL, 'index.html'));

  mainWindow.once('ready-to-show', () => {
    splashScreen.destroy();
    mainWindow.maximize();
    mainWindow.show();
  });

  // Open the DevTools.
  // The timing of this is fussy due a chromium bug. It seems to only
  // come up if there is an unrelated uncaught exception during page load.
  // https://bugs.chromium.org/p/chromium/issues/detail?id=1085215
  // https://github.com/electron/electron/issues/23662
  mainWindow.webContents.on('did-frame-finish-load', async () => {
    if (ELECTRON_DEV_MODE) {
      mainWindow.webContents.openDevTools();
    }
    // We use this stdout as a signal in a puppeteer test
    process.stdout.write('main window loaded');
  });

  mainWindow.webContents.on('render-process-gone', (event, details) => {
    logger.error('render-process-gone');
    logger.error(details);
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  setupDownloadHandlers(mainWindow);
  setupInvestRunHandlers(investExe);
  setupInvestLogReaderHandler();
  setupContextMenu(mainWindow);
  setupGetNCPUs();
  setupSetLanguage();
  setupOpenExternalUrl();
  return Promise.resolve(); // lets tests await createWindow(), then assert
};

export function removeIpcMainListeners() {
  Object.values(ipcMainChannels).forEach((channel) => {
    ipcMain.removeAllListeners(channel);
  });
}

export function main() {
  // calling requestSingleInstanceLock on mac causes a crash
  if (process.platform.startsWith('win')) {
    logger.info('Windows detected, requesting single instance lock');
    // Single instance lock so subsequent instances of the application redirect to
    // the already-open one.
    // Adapted from https://www.electronjs.org/docs/api/app#apprequestsingleinstancelock
    const gotTheLock = app.requestSingleInstanceLock();
    if (!gotTheLock) {
      // If we don't get the lock, then we assume another instance has the lock.
      logger.info('Another instance already has the application lock; exiting');
      app.exit(1);
    }
  }

  app.on('ready', async () => {
    if (ELECTRON_DEV_MODE) {
      const {
        default: installExtension,
        REACT_DEVELOPER_TOOLS,
      } = require('electron-devtools-installer');
      await installExtension(REACT_DEVELOPER_TOOLS, {
        loadExtensionOptions: { allowFileAccess: true },
      });
    }
    createWindow();
  });
  app.on('activate', () => {
    if (mainWindow === null) {
      createWindow();
    }
  });
  app.on('window-all-closed', async () => {
    // On OS X it is common for applications and their menu bar
    // to stay active until the user quits explicitly with Cmd + Q
    if (process.platform !== 'darwin') {
      app.quit();
    }
  });
  let shuttingDown = false;
  app.on('before-quit', async (event) => {
    // prevent quitting until after we're done with cleanup,
    // then programatically quit
    if (shuttingDown) { return; }
    event.preventDefault();
    shuttingDown = true;
    removeIpcMainListeners();
    await shutdownPythonProcess(flaskSubprocess);
    app.quit();
  });
}

if (typeof require !== 'undefined' && require.main === module) {
  main(process.argv);
}
