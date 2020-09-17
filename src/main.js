const isDevMode = process.argv[2] === '--dev';
if (isDevMode) {
  // in dev mode we can have babel transpile modules on import
  require("@babel/register");
  // load the '.env' file from the project root
  const dotenv = require('dotenv');
  dotenv.config();
}

const {
  app, BrowserWindow, ipcMain, screen
} = require('electron'); // eslint-disable-line import/no-extraneous-dependencies
const {
  getFlaskIsReady, shutdownPythonProcess
} = require('./server_requests');
const {
  findInvestBinaries, createPythonFlaskProcess
} = require('./main_helpers');

const PORT = (process.env.PORT || '5000').trim();

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow;
/** Create an Electron browser window and start the flask application. */
const createWindow = async () => {
  // The main process needs to know the location of the invest server binary.
  // The renderer process needs the invest cli binary. We can find them
  // together here and pass data to the renderer upon request.
  const binaries = await findInvestBinaries(isDevMode);
  const mainProcessVars = { investExe: binaries.invest };
  ipcMain.on('variable-request', (event, arg) => {
    event.reply('variable-reply', mainProcessVars);
  });

  createPythonFlaskProcess(binaries.server, isDevMode);
  // Wait for a response from the server before loading the app
  await getFlaskIsReady();

  // Create the browser window.
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;
  mainWindow = new BrowserWindow({
    width: width * 0.75,
    height: height,
    useContentSize: true,
    webPreferences: {
      nodeIntegration: true,
      enableRemoteModule: true,
    },
  });

  // and load the index.html of the app.
  mainWindow.loadURL(`file://${__dirname}/index.html`);

  // Open the DevTools.
  mainWindow.webContents.on('did-frame-finish-load', async () => {
    if (isDevMode) {
      const {
        default: installExtension, REACT_DEVELOPER_TOOLS
      } = require('electron-devtools-installer');
      await installExtension(REACT_DEVELOPER_TOOLS);
      mainWindow.webContents.openDevTools();
    }
  });

  // Emitted when the window is closed.
  mainWindow.on('closed', () => {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    mainWindow = null;
  });
};

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
    app.quit();
  }
});

// TODO: I haven't actually tested this yet on MacOS
app.on('will-quit', async () => {
  if (process.platform === 'darwin') {
    await shutdownPythonProcess();
  }
});
