import path from 'path';

import {
  app,
  BrowserWindow,
  screen,
  nativeTheme,
  Menu,
} from 'electron'; // eslint-disable-line import/no-extraneous-dependencies

import {
  createPythonFlaskProcess,
  getFlaskIsReady,
  shutdownPythonProcess,
} from './createPythonFlaskProcess';
import findInvestBinaries from './findInvestBinaries';
import setupDownloadHandlers from './setupDownloadHandlers';
import setupDialogs from './setupDialogs';
import setupContextMenu from './setupContextMenu';
import { setupCheckFirstRun } from './setupCheckFirstRun';
import { setupInvestRunHandlers } from './setupInvestHandlers';
import { getLogger } from '../logger';
import { menuTemplate } from './menubar';
import pkg from '../../package.json';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

const ELECTRON_DEV_MODE = !!process.defaultApp; // a property added by electron.

// This could be optionally configured already in '.env'
if (!process.env.PORT) {
  process.env.PORT = '56789';
}

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow;
let splashScreen;

export function destroyWindow() {
  mainWindow = null;
}

/** Create an Electron browser window and start the flask application. */
export const createWindow = async () => {
  splashScreen = new BrowserWindow({
    width: 574, // dims set to match the image in splash.html
    height: 500,
    transparent: true,
    frame: false,
    alwaysOnTop: true,
  });
  splashScreen.loadURL(`file://${__dirname}/../static/splash.html`);

  const investExe = findInvestBinaries(ELECTRON_DEV_MODE);
  createPythonFlaskProcess(investExe);
  logger.info(`Running invest-workbench version ${pkg.version}`);
  setupDialogs();
  setupContextMenu();
  setupCheckFirstRun();

  // always use light mode regardless of the OS/browser setting
  nativeTheme.themeSource = 'light';
  // Wait for a response from the server before loading the app
  await getFlaskIsReady();

  // Create the browser window.
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;
  mainWindow = new BrowserWindow({
    width: width,
    height: height,
    useContentSize: true,
    show: true, // see comment in 'ready-to-show' listener
    webPreferences: {
      contextIsolation: false,
      nodeIntegration: true,
      preload: path.join(__dirname, '..', 'preload.js'),
      additionalArguments: [
        ELECTRON_DEV_MODE ? '--dev' : 'packaged'
      ],
      defaultEncoding: 'UTF-8',
    },
  });
  const menubar = Menu.buildFromTemplate(
    menuTemplate(mainWindow, ELECTRON_DEV_MODE)
  );
  Menu.setApplicationMenu(menubar);
  mainWindow.loadURL(`file://${__dirname}/../index.html`);

  mainWindow.once('ready-to-show', () => {
    splashScreen.destroy();
    // We should be able to hide mainWindow until it's ready,
    // but there's a bug where a window initialized with { show: false }
    // will load with invisible elements until it's touched/resized, etc.
    // https://github.com/electron/electron/issues/27353
    // So for now, we have mainWindow showing the whole time, w/ splash on top.
    // If this bug is fixed, we'll need an explicit mainWindow.show() here.
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
  });

  mainWindow.on('closed', async () => {
    mainWindow = null;
    // We shouldn't need to shutdown flask here, as it will be
    // shutdown on the window-all-closed listener, but that one
    // is not triggering in the puppeteer test on linux.
    if (process.platform !== 'darwin') {
      logger.debug('requesting flask shutdown on main window close');
      await shutdownPythonProcess();
    }
  });

  setupDownloadHandlers(mainWindow);
  setupInvestRunHandlers(investExe);
  // TODO: remove listeners on exit
};

export function main(argv) {
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
    } else {
      // TODO: it doesn't make sense to do mac stuff here, we know we're on windows
      // But does it make sense to do the 'second-instance' stuff when !gotTheLock?
      // https://www.electronjs.org/docs/api/app#event-second-instance
      // On mac, it's possible to bypass above single instance constraint by
      // launching the app through the CLI.  If this happens, focus on the main
      // window.
      app.on('second-instance', (event, commandLine, workingDirectory) => {
        if (mainWindow) {
          if (mainWindow.isMinimized()) {
            mainWindow.restore();
          }
          mainWindow.focus();
        }
      });
    }
  }

  app.on('ready', createWindow);
  app.on('activate', () => {
    if (mainWindow === null) {
      createWindow();
    }
  });
  app.on('window-all-closed', async () => {
    // On OS X it is common for applications and their menu bar
    // to stay active until the user quits explicitly with Cmd + Q
    if (process.platform !== 'darwin') {
      await shutdownPythonProcess();
      app.quit();
    }
  });
  app.on('will-quit', async () => {
    if (process.platform === 'darwin') {
      await shutdownPythonProcess();
    }
  });
}

if (typeof require !== 'undefined' && require.main === module) {
  main(process.argv);
}

// module.exports = main;

