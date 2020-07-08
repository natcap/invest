const fs = require('fs')
const path = require('path')
const spawn = require('child_process').spawn;
const { app, BrowserWindow, ipcMain, screen } = require('electron')
const fetch = require('node-fetch')
const { getLogger } = require('./logger')

const logger = getLogger('main')

const isDevMode = process.argv[2] == '--dev'
if (isDevMode) {
  // load the '.env' file from the project root
  const dotenv = require('dotenv');
  dotenv.config();
}

function findInvestBinaries() {

  return new Promise(resolve => {
    // Binding to the invest server binary:
    let serverExe;
    let investExe;

    // A) look for a local registry of available invest installations
    const investRegistryPath = path.join(
      app.getPath('userData'), 'invest_registry.json')
    if (fs.existsSync(investRegistryPath)) {
      const investRegistry = JSON.parse(fs.readFileSync(investRegistryPath))
      const activeVersion = investRegistry['active']
      serverExe = investRegistry['registry'][activeVersion]['server']
      investExe = investRegistry['registry'][activeVersion]['invest']

    // B) check for dev mode and an environment variable from dotenv
    } else if (isDevMode) {
      serverExe = process.env.SERVER
      investExe = process.env.INVEST

    // C) point to binaries included in this app's installation.
    } else {
      const ext = (process.platform === 'win32') ? '.exe' : ''
      binaryPath = path.join(
        process.resourcesPath, 'app.asar.unpacked', 'build', 'invest') 
      serverExe = path.join(binaryPath, 'server' + ext)
      investExe = path.join(binaryPath, 'invest' + ext)
      console.log(serverExe)
    }
    resolve({ invest: investExe, server: serverExe })
  })
}


let PORT = (process.env.PORT || '5000').trim();

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow;
const createWindow = async () => {
  /** Much of this is electron app boilerplate, but here is also
  * where we fire up the python flask server.
  */

  // The main process needs to know the location of the invest server binary
  // The renderer process needs the invest cli binary. We can find them
  // together here and pass data to the renderer upon request.
  const binaries = await findInvestBinaries()
  const mainProcessVars = { investExe: binaries.invest }
  ipcMain.on('variable-request', (event, arg) => {
    event.reply('variable-reply', mainProcessVars)
  })

  createPythonFlaskProcess(binaries.server);

  // Create the browser window.
  const { width, height } = screen.getPrimaryDisplay().workAreaSize
  logger.debug(width + ' ' + height)
  mainWindow = new BrowserWindow({
    width: width * 0.75,
    height: height,
    useContentSize: true,
    webPreferences: {
      nodeIntegration: true
    }
  });

  // and load the index.html of the app.
  mainWindow.loadURL(`file://${__dirname}/index.html`);

  // Open the DevTools.
  mainWindow.webContents.on('did-frame-finish-load', async () => {
    if (isDevMode) {
      const { default: installExtension, REACT_DEVELOPER_TOOLS } = require('electron-devtools-installer');
      await installExtension(REACT_DEVELOPER_TOOLS);
      // enableLiveReload({ strategy: 'react-hmr' });
      mainWindow.webContents.openDevTools();
    }
  })

  // Emitted when the window is closed.
  mainWindow.on('closed', () => {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    mainWindow = null;
  });
};

function createPythonFlaskProcess(serverExe) {
  /** Spawn a child process running the Python Flask server.*/
  if (serverExe) {
    // The most reliable, cross-platform way to make sure spawn
    // can find the exe is to pass only the command name while
    // also putting it's location on the PATH:
    const pythonServerProcess = spawn(path.basename(serverExe), {
        env: {PATH: path.dirname(serverExe)}
      });

    logger.debug('Started python process as PID ' + pythonServerProcess.pid);
    logger.debug(serverExe)
    pythonServerProcess.stdout.on('data', (data) => {
      logger.debug(`${data}`);
    });
    pythonServerProcess.stderr.on('data', (data) => {
      logger.debug(`${data}`);
    });
    pythonServerProcess.on('error', (err) => {
      logger.debug('Process failed.');
      logger.debug(err);
    });
    pythonServerProcess.on('close', (code, signal) => {
      logger.debug(code);
      logger.debug('Child process terminated due to signal ' + signal);
    });
  } else {
    logger.debug('no existing invest installations found')
  }
}

function shutdownPythonProcess() {
  return(
    fetch(`http://localhost:${PORT}/shutdown`, {
      method: 'get',
    })
    .then((response) => { return response.text() })
    .then((text) => { logger.debug(text) })
    .catch((error) => { logger.debug(error) })
  )
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
    // It's crucial to await here, otherwise the parent
    // process dies before flask has time to kill its server.
    await shutdownPythonProcess();
    app.quit()
  }
});

// TODO: I haven't actually tested this yet on MacOS
app.on('will-quit', async () => {
  if (process.platform === 'darwin') {
    await shutdownPythonProcess();
  }
});
