const ELECTRON_DEV_MODE = process.argv[2] === '--dev';
if (ELECTRON_DEV_MODE) {
  // in dev mode we can have babel transpile modules on import
  require("@babel/register");
  // load the '.env' file from the project root
  const dotenv = require('dotenv');
  dotenv.config();
}

const fs = require('fs');
const path = require('path');

const {
  app,
  BrowserWindow,
  ipcMain,
  screen,
  nativeTheme,
  Menu,
  dialog,
} = require('electron'); // eslint-disable-line import/no-extraneous-dependencies

const {
  getFlaskIsReady, shutdownPythonProcess
} = require('./server_requests');
const {
  findInvestBinaries,
  createPythonFlaskProcess,
  extractZipInplace,
  checkFirstRun,
} = require('./main_helpers');
const { getLogger } = require('./logger');
const { menuTemplate } = require('./menubar');
const pkg = require('../package.json');

const logger = getLogger(__filename.split('/').slice(-1)[0]);

// This could be optionally configured already in '.env'
if (!process.env.PORT) {
  process.env.PORT = '56789';
}

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow;
let splashScreen;

/** Create an Electron browser window and start the flask application. */
const createWindow = async () => {
  splashScreen = new BrowserWindow({
    width: 574, // dims set to match the image in splash.html
    height: 500,
    transparent: true,
    frame: false,
    alwaysOnTop: true,
  });
  splashScreen.loadURL(`file://${__dirname}/static/splash.html`);

  const [investExe, investVersion] = await findInvestBinaries(
    ELECTRON_DEV_MODE
  );
  createPythonFlaskProcess(investExe);
  logger.info(`Running invest-workbench version ${pkg.version}`);
  const mainProcessVars = {
    investExe: investExe,
    investVersion: investVersion,
    workbenchVersion: pkg.version,
    userDataPath: app.getPath('userData'),
    isFirstRun: checkFirstRun(),
  };
  ipcMain.handle('show-context-menu', (event, rightClickPos) => {
    const template = [
      {
        label: 'Inspect Element',
        click: () => {
          BrowserWindow.fromWebContents(event.sender)
            .inspectElement(rightClickPos.x, rightClickPos.y);
        }
      },
    ];
    const menu = Menu.buildFromTemplate(template);
    menu.popup(BrowserWindow.fromWebContents(event.sender));
  });
  ipcMain.handle('variable-request', async (event) => {
    return mainProcessVars;
  });
  ipcMain.handle('show-open-dialog', async (event, options) => {
    const result = await dialog.showOpenDialog(options);
    return result;
  });
  ipcMain.handle('show-save-dialog', async (event, options) => {
    const result = await dialog.showSaveDialog(options);
    return result;
  });
  ipcMain.handle('is-dev-mode', async (event) => {
    const result = ELECTRON_DEV_MODE;
    return result;
  });
  ipcMain.handle('user-data', async (event) => {
    const result = await mainProcessVars.userDataPath;
    return result;
  });

  // Wait for a response from the server before loading the app
  await getFlaskIsReady();

  // always use light mode regardless of the OS/browser setting
  // in the future we can add a dark theme
  nativeTheme.themeSource = 'light';

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
  mainWindow.loadURL(`file://${__dirname}/index.html`);

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

  // Emitted when the window is closed.
  mainWindow.on('closed', async () => {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    mainWindow = null;
    // We shouldn't need to shutdown flask here, as it will be
    // shutdown on the window-all-closed listener, but that one
    // is not triggering in the puppeteer test on linux.
    if (process.platform !== 'darwin') {
      logger.debug('requesting flask shutdown on main window close');
      await shutdownPythonProcess();
    }
  });

  // Setup handlers & defaults to manage downloads.
  // Specifically, invest sample data downloads
  let downloadDir;
  let downloadLength;
  const downloadQueue = [];
  ipcMain.on('download-url', async (event, urlArray, directory) => {
    logger.debug(`${urlArray}`);
    downloadDir = directory;
    downloadQueue.push(...urlArray);
    downloadLength = downloadQueue.length;
    mainWindow.webContents.send(
      'download-status',
      [(downloadLength - downloadQueue.length), downloadLength]
    );
    urlArray.forEach((url) => mainWindow.webContents.downloadURL(url));
  });

  mainWindow.webContents.session.on('will-download', (event, item) => {
    const filename = item.getFilename();
    item.setSavePath(path.join(downloadDir, filename));
    const itemURL = item.getURL();
    item.on('updated', (event, state) => {
      if (state === 'interrupted') {
        logger.info('download interrupted');
      } else if (state === 'progressing') {
        if (item.isPaused()) {
          logger.info('download paused');
        } else {
          logger.info(`${item.getSavePath()}`);
          logger.info(`Received bytes: ${item.getReceivedBytes()}`);
        }
      }
    });
    item.once('done', async (event, state) => {
      if (state === 'completed') {
        logger.info(`${itemURL} complete`);
        await extractZipInplace(item.savePath);
        fs.unlink(item.savePath, (err) => {
          if (err) { logger.error(err); }
        });
        const idx = downloadQueue.findIndex((item) => item === itemURL);
        downloadQueue.splice(idx, 1);
        mainWindow.webContents.send(
          'download-status',
          [(downloadLength - downloadQueue.length), downloadLength]
        );
      } else {
        logger.info(`download failed: ${state}`);
      }
      if (!downloadQueue.length) {
        logger.info('all downloads complete');
      }
    });
  });
};

function main(argv) {
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

  // This method will be called when Electron has finished
  // initialization and is ready to create browser windows.
  // Some APIs can only be used after this event occurs.
  app.on('ready', createWindow);

  app.on('activate', () => {
    // On OS X it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (mainWindow === null) {
      createWindow();
    }
  });

  // Quit when all windows are closed.
  app.on('window-all-closed', async () => {
    // On OS X it is common for applications and their menu bar
    // to stay active until the user quits explicitly with Cmd + Q
    if (process.platform !== 'darwin') {
      await shutdownPythonProcess();
      app.quit();
    }
  });

  // TODO: I haven't actually tested this yet on MacOS
  app.on('will-quit', async () => {
    if (process.platform === 'darwin') {
      await shutdownPythonProcess();
    }
  });
}

if (typeof require !== 'undefined' && require.main === module) {
  main(process.argv);
}

module.exports = main;
