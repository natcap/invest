import path from 'path';

import { app, BrowserWindow } from 'electron'; // eslint-disable-line import/no-extraneous-dependencies

import setupContextMenu from './setupContextMenu';
import BASE_URL from './baseUrl';
import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

const isMac = process.platform === 'darwin';

export default function menuTemplate(parentWindow, isDevMode) {
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
      label: 'File',
      submenu: [
        isMac ? { role: 'close' } : { role: 'quit' }
      ]
    },
    // { role: 'editMenu' }
    {
      label: 'Edit',
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
            label: 'Speech',
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
      label: 'View',
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
      label: 'Window',
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
      label: 'About',
      submenu: [
        {
          label: 'About InVEST',
          click: () => openAboutWindow(parentWindow, isDevMode),
        },
        {
          label: 'Report a problem',
          click: () => openReportWindow(parentWindow, isDevMode),
        },
      ],
    },
  ];
  return template;
}

function openAboutWindow(parentWindow, isDevMode) {
  const child = new BrowserWindow({
    parent: parentWindow,
    width: 700,
    height: 800,
    frame: true,
    webPreferences: {
      minimumFontSize: 12,
      preload: path.join(__dirname, '../preload/preload.js'),
    },
  });
  setupContextMenu(child);
  child.setMenu(null);
  child.loadURL(path.join(BASE_URL, 'about.html'));
  if (isDevMode) {
    child.webContents.openDevTools();
  }
}

function openReportWindow(parentWindow, isDevMode) {
  logger.debug('PROBLEM REPORT: process dump');
  // There are some circular references in process object, so can't
  // stringify the whole object. Here's the useful bits.
  logger.debug(JSON.stringify(process.versions, null, 2));
  logger.debug(JSON.stringify(process.arch, null, 2));
  logger.debug(JSON.stringify(process.platform, null, 2));
  logger.debug(JSON.stringify(process.env, null, 2));
  const child = new BrowserWindow({
    parent: parentWindow,
    width: 700,
    height: 800,
    frame: true,
    webPreferences: {
      minimumFontSize: 12,
      preload: path.join(__dirname, '..', 'preload/preload.js'),
      defaultEncoding: 'UTF-8',
    },
  });
  setupContextMenu(child);
  child.setMenu(null);
  child.loadURL(path.join(BASE_URL, 'report_a_problem.html'));
  if (isDevMode) {
    child.webContents.openDevTools();
  }
}
