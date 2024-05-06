/**
 * @jest-environment node
 */

import fs from 'fs';
import path from 'path';
import os from 'os';
import glob from 'glob';
import { spawn, spawnSync } from 'child_process';

import rimraf from 'rimraf';
import puppeteer from 'puppeteer-core';

import pkg from '../../package.json';
import { APP_HAS_RUN_TOKEN } from '../../src/main/setupCheckFirstRun';

jest.setTimeout(240000);
const PORT = 9009;
const WAIT_TO_CLICK = 300; // ms
let ELECTRON_PROCESS;
let BROWSER;

// For ease of automated testing, run the app from the 'unpacked' directory
// to avoid need to install first on windows or extract on mac.
let BINARY_PATH;
// append to this prefix and the image will be uploaded to github artifacts
// E.g. page.screenshot({ path: `${SCREENSHOT_PREFIX}screenshot.png` })
let SCREENSHOT_PREFIX;
// We'll clear this token before launching the app so we can have a
// predictable startup page.
let APP_HAS_RUN_TOKEN_PATH;

// On GHA macos, invest validation can time-out reading from os.tmpdir
// So on GHA, use the homedir instead.
const rootDir = process.env.CI ? os.homedir() : os.tmpdir();
const TMP_DIR = fs.mkdtempSync(path.join(rootDir, 'data-'));
const TMP_AOI_PATH = path.join(TMP_DIR, 'aoi.geojson');
const TEST_PLUGIN_GIT_URL = 'https://github.com/emlys/demo-invest-plugin.git';
const testRaster = path.join(__dirname, 'dem.tif');
const TYPE_DELAY = 10;

if (process.platform === 'darwin') {
  // https://github.com/electron-userland/electron-builder/issues/2724#issuecomment-375850150
  [BINARY_PATH] = glob.sync('./dist/mac*/*.app/Contents/MacOS/InVEST*');
  SCREENSHOT_PREFIX = path.join(
    os.homedir(), 'Library/Logs', pkg.name, 'invest-workbench-'
  );
  APP_HAS_RUN_TOKEN_PATH = path.join(
    os.homedir(), 'Library/Application Support', pkg.name, APP_HAS_RUN_TOKEN
  );
} else if (process.platform === 'win32') {
  [BINARY_PATH] = glob.sync('./dist/win-unpacked/InVEST*.exe');
  SCREENSHOT_PREFIX = path.join(
    os.homedir(), 'AppData/Roaming', pkg.name, 'logs/invest-workbench-'
  );
  APP_HAS_RUN_TOKEN_PATH = path.join(
    os.homedir(), 'AppData/Roaming', pkg.name, APP_HAS_RUN_TOKEN
  );
}

if (!fs.existsSync(BINARY_PATH)) {
  throw new Error(`Binary file not found: ${BINARY_PATH}`);
}
fs.accessSync(BINARY_PATH, fs.constants.X_OK);

function makeAOI() {
  /* eslint-disable */
  const geojson = {
    "type": "FeatureCollection",
    "name": "aoi",
    "features": [
      {
        "type": "Feature",
        "properties": { "id": 1 },
        "geometry": {
          "type": "Polygon",
          "coordinates": [ [
            [ -123, 45 ],
            [ -123, 45.2 ],
            [ -122.8, 45.2 ],
            [ -122.8, 45 ],
            [ -123, 45 ]
          ] ]
        }
      }
    ]
  }
  /* eslint-enable */
  fs.writeFileSync(TMP_AOI_PATH, JSON.stringify(geojson));
}

beforeAll(() => {
  makeAOI();
});
afterAll(() => {
  rimraf(TMP_DIR, (error) => { if (error) { throw error; } });
});

// errors are not thrown from an async beforeAll
// https://github.com/facebook/jest/issues/8688
beforeEach(() => {
  try { fs.unlinkSync(APP_HAS_RUN_TOKEN_PATH); } catch {}
  // start the invest app and forward stderr to console
  ELECTRON_PROCESS = spawn(
    `"${BINARY_PATH}"`,
    // these are chromium args
    [`--remote-debugging-port=${PORT}`],
    {
      shell: true,
      env: { ...process.env, PUPPETEER: true }
    }
  );
  ELECTRON_PROCESS.stderr.on('data', (data) => {
    console.log(`${data}`);
  });
  const stdOutCallback = async (data) => {
    // Here's how we know the electron window is ready to connect to
    if (`${data}`.match('main window loaded')) {
      try {
        BROWSER = await puppeteer.connect({
          browserURL: `http://127.0.0.1:${PORT}`,
          defaultViewport: null,
          protocolTimeout: 300000,
        });
      } catch (e) {
        console.log(e);
      }
      ELECTRON_PROCESS.stdout.removeListener('data', stdOutCallback);
    }
  };
  ELECTRON_PROCESS.stdout.on('data', stdOutCallback);

  // clear old screenshots
  glob.glob(`${SCREENSHOT_PREFIX}*.png`, (err, files) => {
    files.forEach((file) => fs.unlinkSync(file));
  });
});

afterEach(async () => {
  try {
    const pages = await BROWSER.pages();
    await Promise.all(pages.map(page => page.close()));
    await BROWSER.close();
  } catch (error) {
    console.log(BINARY_PATH);
    console.error(error);
    // Normally BROWSER.close() will kill this process
    ELECTRON_PROCESS.kill();
  }
});

test('Run a real invest model', async () => {
  // On GHA MacOS, we seem to have to wait a long time for the browser
  // to be ready. Maybe related to https://github.com/natcap/invest-workbench/issues/158
  let i = 0;
  while (!BROWSER || !BROWSER.isConnected()) {
    i++;
    await new Promise(r => setTimeout(r, 1000));
  }
  console.log(`waited ${i} seconds for pptr to connect`);
  // find the mainWindow's index.html, not the splashScreen's splash.html
  const target = await BROWSER.waitForTarget(
    (target) => target.url().endsWith('index.html')
  );
  const page = await target.page();
  page.on('error', (err) => {
    console.log(err);
  });
  await page.screenshot({ path: `${SCREENSHOT_PREFIX}1-page-load.png` });

  const downloadModal = await page.waitForSelector('.modal-dialog');
  const downloadModalCancel = await downloadModal.waitForSelector(
    'aria/[name="Cancel"][role="button"]');
  await page.waitForTimeout(WAIT_TO_CLICK); // waiting for click handler to be ready
  await downloadModalCancel.click();
  // We need to get the modelButton from w/in this list-group because there
  // are buttons with the same name in the Recent Jobs container.
  const investModels = await page.waitForSelector('.invest-list-group');
  await page.screenshot({ path: `${SCREENSHOT_PREFIX}2-models-list.png` });

  // Setting up Recreation model because it has very few data requirements
  const modelButton = await investModels.waitForSelector(
    'aria/[name="Visitation: Recreation and Tourism"][role="button"]');
  await modelButton.click();
  await page.screenshot({ path: `${SCREENSHOT_PREFIX}3-model-tab.png` });

  const argsForm = await page.waitForSelector('.args-form');

  const workspace = await argsForm.waitForSelector(
    'aria/[name="Workspace"][role="textbox"]');
  await workspace.type(TMP_DIR, { delay: TYPE_DELAY });
  const aoi = await argsForm.waitForSelector(
    'aria/[name="Area Of Interest"][role="textbox"]');
  await aoi.type(TMP_AOI_PATH, { delay: TYPE_DELAY });
  const startYear = await argsForm.waitForSelector(
    'aria/[name="Start Year"][role="textbox"]');
  await startYear.type('2008', { delay: TYPE_DELAY });
  const endYear = await argsForm.waitForSelector(
    'aria/[name="End Year"][role="textbox"]');
  await endYear.type('2012', { delay: TYPE_DELAY });
  await page.screenshot({ path: `${SCREENSHOT_PREFIX}4-complete-setup-form.png` });

  const sidebar = await page.waitForSelector('.invest-sidebar-col');

  // Button is disabled until validation completes
  const runButton = await sidebar.waitForSelector(
    '.btn-primary:not([disabled])');
  await runButton.click();

  await page.waitForSelector('#invest-tab-tab-log.active');
  await page.screenshot({ path: `${SCREENSHOT_PREFIX}5-active-log-tab.png` });

  // Cancel button does not appear until after invest has confirmed
  // it is running. So extra timeout on the query:
  const cancelButton = await sidebar.waitForSelector(
    'aria/[name="Cancel Run"][role="button"]', { timeout: 15000 });
  await cancelButton.click();
  await sidebar.waitForSelector('text/Run Canceled');
  await page.waitForSelector('aria/[name="Open Workspace"][role="button"]');
  await page.screenshot({ path: `${SCREENSHOT_PREFIX}6-run-canceled.png` });
}, 240000); // >2x the sum of all the max timeouts within this test

test('Check local userguide links', async () => {
  // On GHA MacOS, we seem to have to wait a long time for the browser
  // to be ready. Maybe related to https://github.com/natcap/invest-workbench/issues/158
  let i = 0;
  while (!BROWSER || !BROWSER.isConnected()) {
    i++;
    await new Promise(r => setTimeout(r, 1000));
  }
  console.log(`waited ${i} seconds for pptr to connect`);
  // find the mainWindow's index.html, not the splashScreen's splash.html
  const target = await BROWSER.waitForTarget(
    (target) => target.url().endsWith('index.html')
  );
  const page = await target.page();
  page.on('error', (err) => {
    console.log(err);
  });
  const downloadModal = await page.waitForSelector('.modal-dialog');
  const downloadModalCancel = await downloadModal.waitForSelector(
    'aria/[name="Cancel"][role="button"]');
  await page.waitForTimeout(WAIT_TO_CLICK); // waiting for click handler to be ready
  await downloadModalCancel.click();

  const investList = await page.waitForSelector('.invest-list-group');
  const modelButtons = await investList.$$('aria/[role="button"]');

  await page.waitForTimeout(WAIT_TO_CLICK); // first btn click does not register w/o this pause
  for (const btn of modelButtons) {
    await btn.click();
    const link = await page.waitForSelector('text/User\'s Guide');
    await page.waitForTimeout(WAIT_TO_CLICK); // link.click() not working w/o this pause
    const hrefHandle = await link.getProperty('href');
    const hrefValue = await hrefHandle.jsonValue();
    await link.click();
    const ugTarget = await BROWSER.waitForTarget(
      (target) => target.url() === hrefValue
    );
    const ugPage = await ugTarget.page();
    try {
      await ugPage.waitForSelector('text/Table of Contents');
    } catch {
      throw new Error(`${hrefValue} not found`);
    }

    await ugPage.close();
    const tab = await page.waitForSelector('.nav-item');
    const closeTabBtn = await tab.waitForSelector('aria/[role="button"]');
    await closeTabBtn.click();
    await page.waitForTimeout(100); // allow for Home Tab to be visible again
  }
});

test('Install and run a plugin', async () => {
  // On GHA MacOS, we seem to have to wait a long time for the browser
  // to be ready. Maybe related to https://github.com/natcap/invest-workbench/issues/158
  let i = 0;
  while (!BROWSER || !BROWSER.isConnected()) {
    i++;
    await new Promise(r => setTimeout(r, 1000));
  }
  console.log(`waited ${i} seconds for pptr to connect`);
  // find the mainWindow's index.html, not the splashScreen's splash.html
  const target = await BROWSER.waitForTarget(
    (target) => target.url().endsWith('index.html')
  );
  const page = await target.page();
  page.on('error', (err) => {
    console.log(err);
  });
  await page.screenshot({ path: `${SCREENSHOT_PREFIX}1-page-load.png` });
  const downloadModal = await page.waitForSelector('.modal-dialog');
  const downloadModalCancel = await downloadModal.waitForSelector(
    'aria/[name="Cancel"][role="button"]'
  );
  await page.waitForTimeout(WAIT_TO_CLICK); // waiting for click handler to be ready
  await downloadModalCancel.click();

  const addPluginButton = await page.waitForSelector('div ::-p-text(Add a plugin)');
  await addPluginButton.click();
  console.log('clicked add plugin');
  const urlInputField = await page.waitForSelector('input[name=url]');
  console.log('found url field');
  await urlInputField.type(TEST_PLUGIN_GIT_URL, { delay: TYPE_DELAY });
  console.log('typed into input field');
  const submitButton = await page.waitForSelector('button[name=submit]');
  console.log('found submit button');
  console.log(submitButton);
  await submitButton.click();
  console.log('clicked submit');
  const pluginButton = await page.waitForSelector("button[name='Foo Model']");
  await pluginButton.evaluate((b) => b.click());

  await page.waitForSelector('div ::-p-text(Starting up model...)');
  console.log('starting up model');
  const argsForm = await page.waitForSelector('.args-form');
  console.log('found args form');
  const workspace = await argsForm.waitForSelector(
    'aria/[name="Workspace"][role="textbox"]'
  );
  console.log('found workspace');
  await workspace.type(TMP_DIR, { delay: TYPE_DELAY });
  const rasterInput = await argsForm.waitForSelector(
    'aria/[name="Input Raster"][role="textbox"]'
  );
  await rasterInput.type(testRaster, { delay: TYPE_DELAY });
  const numberInput = await argsForm.waitForSelector(
    'aria/[name="Multiplication Factor"][role="textbox"]'
  );
  await numberInput.type('2', { delay: TYPE_DELAY });

  const sidebar = await page.waitForSelector('.invest-sidebar-col');
  const runButton = await sidebar.waitForSelector('.btn-primary:not([disabled])');
  await runButton.click();
  await page.waitForSelector('#invest-tab-tab-log.active');
  await page.waitForSelector('div ::-p-text(Model Complete)');
}, 500000);

const testWin = process.platform === 'win32' ? test : test.skip;
/* Test for duplicate application launch.
We have the binary path, so now let's launch a new subprocess with the same binary
The test is that the subprocess exits within a certain reasonable timeout.
Single instance lock caused the app to crash on macOS, and also
is less important because mac generally won't open multiple instances */
testWin('App re-launch will exit and focus on first instance', async () => {
  let i = 0;
  while (!BROWSER || !BROWSER.isConnected()) {
    i++;
    await new Promise(r => setTimeout(r, 1000));
  }
  console.log(`waited ${i} seconds for pptr to connect`);

  // Open another instance of the Workbench application.
  // This should return quickly.  The test timeout is there in case the new i
  // process hangs for some reason.
  const otherElectronProcess = spawnSync(
    `"${BINARY_PATH}"`, [`--remote-debugging-port=${PORT}`],
    { shell: true }
  );

  // When another instance is already open, we expect an exit code of 1.
  expect(otherElectronProcess.status).toBe(1);
});
