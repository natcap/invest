import path from 'path';
// eslint-disable-next-line import/no-extraneous-dependencies
import {
  app,
  BrowserWindow,
  nativeTheme,
  Menu,
  ipcMain
} from 'electron';

import i18n from './i18n/i18n';
import BASE_URL from './baseUrl';
import {
  createCoreServerProcess,
  shutdownPythonProcess
} from './createPythonFlaskProcess';
import { findInvestBinaries, findMicromambaExecutable } from './findBinaries';
import setupDownloadHandlers from './setupDownloadHandlers';
import setupDialogs from './setupDialogs';
import setupContextMenu from './setupContextMenu';
import setupCheckFilePermissions from './setupCheckFilePermissions';
import { setupCheckFirstRun } from './setupCheckFirstRun';
import { setupCheckStorageToken } from './setupCheckStorageToken';
import {
  setupInvestRunHandlers,
  setupLaunchPluginServerHandler,
  setupInvestLogReaderHandler
} from './setupInvestHandlers';
import {
  setupAddPlugin,
  setupRemovePlugin,
  setupWindowsMSVCHandlers
} from './setupAddRemovePlugin';
import { ipcMainChannels } from './ipcMainChannels';
import ELECTRON_DEV_MODE from './isDevMode';
import { getLogger } from './logger';
import menuTemplate from './menubar';
import pkg from '../../package.json';
import { settingsStore, setupSettingsHandlers } from './settingsStore';
import { setupBaseUrl } from './setupBaseUrl';
import setupGetElectronPaths from './setupGetElectronPaths';
import setupGetNCPUs from './setupGetNCPUs';
import { setupIsNewVersion } from './setupIsNewVersion';
import setupOpenExternalUrl from './setupOpenExternalUrl';
import setupOpenLocalHtml from './setupOpenLocalHtml';
import setupRendererLogger from './setupRendererLogger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

process.on('uncaughtException', (err) => {
  logger.error(err);
  process.exit(1);
});
process.on('unhandledRejection', (err, promise) => {
  logger.error(`unhandled rejection at: ${promise}`);
  logger.error(err);
  process.exit(1);
});

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow;
let splashScreen;
let forceQuit = false;

export function destroyWindow() {
  mainWindow = null;
}

/** Create an Electron browser window and start the flask application. */
export const createWindow = async () => {
  logger.info(`Running invest-workbench version ${pkg.version}`);
  nativeTheme.themeSource = 'light'; // override OS/browser setting

  i18n.changeLanguage(settingsStore.get('language'));

  splashScreen = new BrowserWindow({
    width: 574, // dims set to match the image in splash.html
    height: 479,
    transparent: true,
    frame: false,
    alwaysOnTop: false,
  });
  splashScreen.loadURL(path.join(BASE_URL, 'splash.html'));

  settingsStore.set('investExe', findInvestBinaries(ELECTRON_DEV_MODE));
  settingsStore.set('micromamba', findMicromambaExecutable(ELECTRON_DEV_MODE));
  // No plugin server processes should persist between workbench sessions
  // In case any were left behind, remove them
  const plugins = settingsStore.get('plugins');
  if (plugins) {
    Object.keys(plugins).forEach((modelID) => {
      settingsStore.set(`plugins.${modelID}.pid`, '');
      settingsStore.set(`plugins.${modelID}.port`, '');
    });
  }

  await createCoreServerProcess();
  setupDialogs();
  setupCheckFilePermissions();
  setupCheckFirstRun();
  setupIsNewVersion();
  setupCheckStorageToken();
  setupSettingsHandlers();
  setupGetElectronPaths();
  setupGetNCPUs();
  setupInvestLogReaderHandler();
  setupOpenExternalUrl();
  setupRendererLogger();
  setupBaseUrl();

  const devModeArg = ELECTRON_DEV_MODE ? '--devmode' : '';
  // Create the browser window.
  mainWindow = new BrowserWindow({
    minWidth: 800,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, '../preload/preload.js'),
      defaultEncoding: 'UTF-8',
      additionalArguments: [devModeArg],
    },
  });
  Menu.setApplicationMenu(
    Menu.buildFromTemplate(
      menuTemplate(mainWindow, ELECTRON_DEV_MODE, i18n)
    )
  );
  mainWindow.loadURL(path.join(BASE_URL, 'index.html'));

  mainWindow.once('ready-to-show', () => {
    splashScreen.destroy();
    mainWindow.maximize();
    mainWindow.show();
  });

  mainWindow.webContents.on('did-finish-load', () => {
    process.stdout.write('main window loaded');
  });

  mainWindow.webContents.on('render-process-gone', (event, details) => {
    logger.error('render-process-gone');
    logger.error(details);
  });

  mainWindow.on('close', (event) => {
    // 'close' is triggered by the red traffic light button on mac
    // override this behavior and just minimize,
    // unless we're actually quitting the app
    if (process.platform === 'darwin' & !forceQuit) {
      event.preventDefault();
      mainWindow.minimize()
    }
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // register listeners that need a reference to the mainWindow or
  // have callbacks that won't work until the invest server is ready.
  setupContextMenu(mainWindow);
  setupDownloadHandlers(mainWindow);
  setupInvestRunHandlers();
  setupLaunchPluginServerHandler();
  setupAddPlugin(i18n);
  setupRemovePlugin();
  setupWindowsMSVCHandlers();
  setupOpenLocalHtml(mainWindow, ELECTRON_DEV_MODE);
  if (ELECTRON_DEV_MODE) {
    // The timing of this is fussy due a chromium bug. It seems to only
    // come up if there is an unrelated uncaught exception during page load.
    // https://bugs.chromium.org/p/chromium/issues/detail?id=1085215
    // https://github.com/electron/electron/issues/23662
    // Calling this in a 'did-finish-load' listener would make a lot of sense,
    // but most of the time it doesn't work.
    mainWindow.webContents.openDevTools();
  }
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
    createWindow();
  });
  app.on('activate', () => {
    if (mainWindow === null) {
      createWindow();
    }
  });

  let shuttingDown = false;
  app.on('before-quit', async (event) => {
    // prevent quitting until after we're done with cleanup,
    // then programatically quit
    forceQuit = true;
    if (shuttingDown) { return; }
    event.preventDefault();
    shuttingDown = true;
    await shutdownPythonProcess(settingsStore.get('core.pid'));
    settingsStore.set('core.pid', '');
    settingsStore.set('core.port', '');
    const pluginServerPIDs = [];
    const plugins = settingsStore.get('plugins') || {};
    Object.keys(plugins).forEach((pluginID) => {
      const pid = settingsStore.get(`plugins.${pluginID}.pid`);
      if (pid) {
        pluginServerPIDs.push(pid);
      }
      settingsStore.set(`plugins.${pluginID}.pid`, '');
      settingsStore.set(`plugins.${pluginID}.port`, '');
    });
    await Promise.all(pluginServerPIDs.map((pid) => shutdownPythonProcess(pid)));

    removeIpcMainListeners();
    app.quit();
  });
}

if (typeof require !== 'undefined' && require.main === module) {
  main(process.argv);
}
