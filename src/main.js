const spawn = require('child_process').spawn;
const app = require('electron').app
const BrowserWindow = require('electron').BrowserWindow
const fetch = require('node-fetch')

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow;

const isDevMode = process.execPath.match(/[\\/]electron/);
if (isDevMode) {
  // load the '.env' file from the project root
  const dotenv = require('dotenv');
  dotenv.config();  
}

let PYTHON = 'python';
if (process.env.PYTHON) {  // if it was set, override
  PYTHON = process.env.PYTHON.trim();
}

const createWindow = async () => {
  /** Much of this is electron app boilerplate, but here is also
  * where we fire up the python flask server.
  */
  createPythonFlaskProcess();

  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 1000,
    webPreferences: {
      nodeIntegration: true
    }
  });

  // and load the index.html of the app.
  mainWindow.loadURL(`file://${__dirname}/index.html`);

  // Open the DevTools.
  if (isDevMode) {
    const { default: installExtension, REACT_DEVELOPER_TOOLS } = require('electron-devtools-installer');
    await installExtension(REACT_DEVELOPER_TOOLS);
    // enableLiveReload({ strategy: 'react-hmr' });
    mainWindow.webContents.openDevTools();
  }

  // Emitted when the window is closed.
  mainWindow.on('closed', () => {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    mainWindow = null;
  });
};

let pythonServerProcess;
function createPythonFlaskProcess() {
  /** Spawn a child process running the Python Flask server.*/
  pythonServerProcess = spawn(
    PYTHON, ['-m', 'flask', 'run'], {
      shell: true,
      // stdio: 'ignore',
      detatched: true,
    });

  console.log('Started python process as PID ' + pythonServerProcess.pid);

  pythonServerProcess.stdout.on('data', (data) => {
    console.log(`${data}`);
  });
  pythonServerProcess.stderr.on('data', (data) => {
    console.log(`${data}`);
  });
  pythonServerProcess.on('error', (err) => {
    console.log('Process failed.');
    console.log(err);
  });
  pythonServerProcess.on('close', (code, signal) => {
    console.log(code);
    console.log('Child process terminated due to signal ' + signal);
  });
}

function shutdownPythonProcess() {
  return(
    fetch('http://localhost:5000/shutdown', {
      method: 'get',
    })
    .then((response) => { return response.text() })
    .then((text) => { console.log(text) })
    .catch((error) => { console.log(error) })
  )
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', createWindow);

// Quit when all windows are closed.
app.on('window-all-closed', async () => {
  // On OS X it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') {
    // It's crucial to await here, otherwise the parent
    // process dies before flask has time to kill its server.
    await shutdownPythonProcess();
  }
  // TODO: handle flask shutdown on darwin
});

app.on('activate', () => {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (mainWindow === null) {
    createWindow();
  }
});

// Couldn't get this callback to fire, moved to 'window-all-closed',
// but that doesn't cover OSX
// app.on('will-quit', shutdownPythonProcess);
