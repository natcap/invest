import path from 'path';

import { app, BrowserWindow } from 'electron'; // eslint-disable-line import/no-extraneous-dependencies

import setupContextMenu from './setupContextMenu';
import BASE_URL from './baseUrl';
import { getLogger } from './logger';
import {
  createJupyterProcess,
  shutdownPythonProcess,
  getJupyterIsReady,
} from './createPythonFlaskProcess';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

const isMac = process.platform === 'darwin';

export default function menuTemplate(parentWindow, isDevMode, i18n, jupyterExe) {
  // Much of this template comes straight from the docs
  // https://www.electronjs.org/docs/api/menu
  const template = [
    // { role: 'appMenu' }
    ...(isMac ? [{
      label: app.name,
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideothers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' }
      ]
    }] : []),
    // { role: 'fileMenu' }
    {
      label: i18n.t('File'),
      submenu: [
        isMac ? { role: 'close' } : { role: 'quit' }
      ]
    },
    // { role: 'editMenu' }
    {
      label: i18n.t('Edit'),
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        ...(isMac ? [
          { role: 'pasteAndMatchStyle' },
          { role: 'delete' },
          { role: 'selectAll' },
          { type: 'separator' },
          {
            label: i18n.t('Speech'),
            submenu: [
              { role: 'startspeaking' },
              { role: 'stopspeaking' }
            ]
          }
        ] : [
          { role: 'delete' },
          { type: 'separator' },
          { role: 'selectAll' }
        ])
      ]
    },
    // { role: 'viewMenu' }
    {
      label: i18n.t('View'),
      submenu: [
        { role: 'reload' },
        { role: 'forcereload' },
        { type: 'separator' },
        { role: 'resetzoom' },
        { role: 'zoomin' },
        { role: 'zoomout' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },
    // { role: 'windowMenu' }
    {
      label: i18n.t('Window'),
      submenu: [
        { role: 'minimize' },
        { role: 'zoom' },
        ...(isMac ? [
          { type: 'separator' },
          { role: 'front' },
          { type: 'separator' },
          { role: 'window' }
        ] : [
          { role: 'close' }
        ])
      ]
    },
    {
      label: i18n.t('About'),
      submenu: [
        {
          label: i18n.t('About InVEST'),
          click: () => openAboutWindow(parentWindow, isDevMode),
        },
        {
          label: i18n.t('Report a problem'),
          click: () => openReportWindow(parentWindow, isDevMode),
        },
        {
          label: i18n.t('Open Notebook'),
          click: () => openJupyterLab(parentWindow, isDevMode, jupyterExe),
        }
      ],
    },
  ];
  return template;
}

function createWindow(parentWindow, isDevMode) {
  const devModeArg = isDevMode ? '--devMode' : '';
  const win = new BrowserWindow({
    parent: parentWindow,
    width: 700,
    height: 800,
    frame: true,
    webPreferences: {
      minimumFontSize: 12,
      preload: path.join(__dirname, '../preload/preload.js'),
      defaultEncoding: 'UTF-8',
      additionalArguments: [devModeArg, `--port=${process.env.PORT}`],
    },
  });
  setupContextMenu(win);
  win.setMenu(null);
  if (isDevMode) {
    win.webContents.openDevTools();
  }
  return win;
}

async function openJupyterLab(parentWindow, isDevMode, jupyterExe) {
  let labDir = process.resourcesPath;
  if (isDevMode) { labDir = 'resources/notebooks'; }
  const subprocess = createJupyterProcess(jupyterExe, labDir);
  const child = createWindow(parentWindow, isDevMode);
  await getJupyterIsReady();
  child.loadURL(`http://localhost:${process.env.JUPYTER_PORT}/?token=${process.env.JUPYTER_TOKEN}`);
  child.on('close', async (event) => {
    await shutdownPythonProcess(subprocess);
  });
  // TODO: what if entire app is quit instead of this window closing first
}

function openAboutWindow(parentWindow, isDevMode) {
  const child = createWindow(parentWindow, isDevMode);
  child.loadURL(path.join(BASE_URL, 'about.html'));
}

function openReportWindow(parentWindow, isDevMode) {
  logger.debug('PROBLEM REPORT: process dump');
  // There are some circular references in process object, so can't
  // stringify the whole object. Here's the useful bits.
  logger.debug(JSON.stringify(process.versions, null, 2));
  logger.debug(JSON.stringify(process.arch, null, 2));
  logger.debug(JSON.stringify(process.platform, null, 2));
  logger.debug(JSON.stringify(process.env, null, 2));

  const child = createWindow(parentWindow, isDevMode);
  child.loadURL(path.join(BASE_URL, 'report_a_problem.html'));
}
