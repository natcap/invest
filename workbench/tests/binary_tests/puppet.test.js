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
import { getDocument, queries, waitFor } from 'pptr-testing-library';

import pkg from '../../package.json';
import { APP_HAS_RUN_TOKEN } from '../../src/main/setupCheckFirstRun';

jest.setTimeout(240000); // This test takes ~20 seconds, but sometimes longer
const PORT = 9009;
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

if (process.platform === 'darwin') {
  // https://github.com/electron-userland/electron-builder/issues/2724#issuecomment-375850150
  [BINARY_PATH] = glob.sync('./dist/mac/*.app/Contents/MacOS/InVEST*');
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

// errors are not thrown from an async beforeAll
// https://github.com/facebook/jest/issues/8688
beforeAll(() => {
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
          browserURL: `http://localhost:${PORT}`,
          defaultViewport: null,
        });
      } catch (e) {
        console.log(e);
      }
      ELECTRON_PROCESS.stdout.removeListener('data', stdOutCallback);
    }
  };
  ELECTRON_PROCESS.stdout.on('data', stdOutCallback);

  // set up test data
  makeAOI();

  // clear old screenshots
  glob.glob(`${SCREENSHOT_PREFIX}*.png`, (err, files) => {
    files.forEach((file) => fs.unlinkSync(file));
  });
});

afterAll(async () => {
  try {
    await BROWSER.close();
  } catch (error) {
    console.log(BINARY_PATH);
    console.error(error);
  }

  rimraf(TMP_DIR, (error) => { if (error) { throw error; } });
  ELECTRON_PROCESS.removeAllListeners();
  ELECTRON_PROCESS.kill();
});

test('Run a real invest model', async () => {
  const { findByText, findByLabelText, findByRole } = queries;
  // On GHA MacOS, we seem to have to wait a long time for the browser
  // to be ready. Maybe related to https://github.com/natcap/invest-workbench/issues/158
  await waitFor(() => {
    expect(BROWSER && BROWSER.isConnected()).toBeTruthy();
  }, { timeout: 60000 });
  // find the mainWindow's index.html, not the splashScreen's splash.html
  const target = await BROWSER.waitForTarget(
    (target) => target.url().endsWith('index.html')
  );
  const page = await target.page();
  page.on('error', (err) => {
    console.log(err);
  });
  const doc = await getDocument(page);
  await page.screenshot({ path: `${SCREENSHOT_PREFIX}1-page-load.png` });

  const extraTime = 5000; // long timeouts finding the first elements
  const downloadModalCancel = await findByRole(
    doc, 'button', { name: 'Cancel' }, { timeout: extraTime }
  );
  await downloadModalCancel.click();
  // We need to get the modelButton from w/in this list-group because there
  // are buttons with the same name in the Recent Jobs container.
  const investModels = await page.$('.invest-list-group');
  await page.screenshot({ path: `${SCREENSHOT_PREFIX}2-models-list.png` });

  // Setting up Recreation model because it has very few data requirements
  const modelButton = await findByRole(
    investModels, 'button', { name: /Visitation/ }
  );
  await modelButton.click();
  await page.screenshot({ path: `${SCREENSHOT_PREFIX}3-model-tab.png` });

  const argsForm = await page.waitForSelector('.args-form');
  const typeDelay = 10;
  const workspace = await findByRole(
    argsForm, 'textbox', { name: /Workspace/i }
  );
  await workspace.type(TMP_DIR, { delay: typeDelay });
  const aoi = await findByRole(
    argsForm, 'textbox', { name: /area of interest/i }
  );
  await aoi.type(TMP_AOI_PATH, { delay: typeDelay });
  const startYear = await findByRole(
    argsForm, 'textbox', { name: /start year/i }
  );
  await startYear.type('2008', { delay: typeDelay });
  const endYear = await findByRole(
    argsForm, 'textbox', {name: /end year/i }
  );
  await endYear.type('2012', { delay: typeDelay });
  await page.screenshot({ path: `${SCREENSHOT_PREFIX}4-complete-setup-form.png` });

  // Button is disabled until validation completes
  const sidebar = await page.$('.invest-sidebar-col');
  const runButton = await findByRole(sidebar, 'button', { name: 'Run' });
  await waitFor(async () => {
    const isEnabled = await page.evaluate(
      (btn) => !btn.disabled,
      runButton
    );
    expect(isEnabled).toBe(true);
  }, { timeout: 10000 }); // waiting for validation
  await runButton.click();

  const logTab = await findByText(doc, 'Log');
  await waitFor(async () => {
    const prop = await logTab.getProperty('className');
    const vals = await prop.jsonValue();
    expect(vals.includes('active')).toBeTruthy();
  });
  await page.screenshot({ path: `${SCREENSHOT_PREFIX}5-active-log-tab.png` });

  // Cancel button does not appear until after invest has confirmed
  // it is running. So extra timeout on the query:
  const cancelButton = await findByRole(sidebar,
    'button', { name: 'Cancel Run' }, { timeout: 5000 });
  await cancelButton.click();
  expect(await findByText(sidebar, 'Run Canceled'));
  expect(await findByText(sidebar, 'Open Workspace'));
  await page.screenshot({ path: `${SCREENSHOT_PREFIX}6-run-canceled.png` });
}, 240000); // >2x the sum of all the max timeouts within this test

// Test for duplicate application launch.
// We have the binary path, so now let's launch a new subprocess with the same binary
// The test is that the subprocess exits within a certain reasonable timeout.
// Also verify that window 1 has focus.
test('App re-launch will exit and focus on first instance', async () => {
  if (process.platform === 'win32') {
    await waitFor(() => {
      expect(BROWSER.isConnected()).toBeTruthy();
    });

    // Open another instance of the Workbench application.
    // This should return quickly.  The test timeout is there in case the new i
    // process hangs for some reason.
    const otherElectronProcess = spawnSync(
      `"${BINARY_PATH}"`, [`--remote-debugging-port=${PORT}`],
      { shell: true }
    );

    // When another instance is already open, we expect an exit code of 1.
    expect(otherElectronProcess.status).toBe(1);
  } else {
    // Single instance lock caused the app to crash on macOS, and also
    // is less important because mac generally won't open multiple instances
    console.log("Skipping this test because we're not on Windows");
  }
});
