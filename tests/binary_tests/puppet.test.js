import fs from 'fs';
import path from 'path';
import os from 'os';
import glob from 'glob';
import fetch from 'node-fetch';
import { spawn, spawnSync } from 'child_process';
import puppeteer from 'puppeteer-core';
import { getDocument, queries, waitFor } from 'pptr-testing-library';

jest.setTimeout(120000); // This test takes ~15 seconds, but longer in CI
const PORT = 9009;
const TMP_DIR = fs.mkdtempSync('tests/data/_');
const TMP_AOI_PATH = path.join(TMP_DIR, 'aoi.geojson');
let ELECTRON_PROCESS;
let BROWSER;
// append to this filename and the image will be uploaded to github artifacts
// E.g. page.screenshot({ path: `${SCREENSHOT_PREFIX}screenshot.png` })
const SCREENSHOT_PREFIX = path.join(
  os.homedir(), 'AppData/Roaming/invest-workbench/invest-workbench-'
);

// For ease of automated testing, run the app from the 'unpacked' directory
// to avoid need to install first on windows or extract on mac.
let BINARY_PATH;
if (process.platform === 'darwin') {
  // https://github.com/electron-userland/electron-builder/issues/2724#issuecomment-375850150
  [BINARY_PATH] = glob.sync('./dist/mac/*.app/Contents/MacOS/InVEST*');
} else if (process.platform === 'win32') {
  [BINARY_PATH] = glob.sync('./dist/win-unpacked/InVEST*.exe');
} else {
  BINARY_PATH = './dist/linux-unpacked/invest-workbench';
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
beforeAll(async () => {
  // start the invest app and forward stderr to console
  ELECTRON_PROCESS = spawn(
    `"${BINARY_PATH}"`,
    // remote-debugging-port is a chromium arg
    [`--remote-debugging-port=${PORT}`],
    { shell: true }
  );
  ELECTRON_PROCESS.stderr.on('data', (data) => {
    console.log(`${data}`);
  });

  // get data about the remote debugging endpoint
  // so we don't make the next fetch too early
  await new Promise((resolve) => setTimeout(resolve, 20000));
  const res = await fetch(`http://localhost:${PORT}/json/version`);
  const data = JSON.parse(await res.text());
  BROWSER = await puppeteer.connect({
    browserWSEndpoint: data.webSocketDebuggerUrl, // this works
    // browserURL: `http://localhost:${PORT}`,    // this also works
    defaultViewport: null
  });
  // set up test data
  makeAOI();
});

afterAll(async () => {
  try {
    await BROWSER.close();
  } catch (error) {
    console.log(BINARY_PATH);
    console.error(error);
  }

  // being extra careful with recursive rm
  if (TMP_DIR.startsWith('tests/data')) {
    fs.rmdirSync(TMP_DIR, { recursive: true });
  }
  const wasKilled = ELECTRON_PROCESS.kill();
  console.log(`electron process was killed: ${wasKilled}`);
});

test('Run a real invest model', async () => {
  const { findByText, findByLabelText, findByRole } = queries;
  await waitFor(() => {
    expect(BROWSER.isConnected()).toBeTruthy();
  });
  // find the mainWindow's index.html, not the splashScreen's splash.html
  const target = await BROWSER.waitForTarget(
    (target) => target.url().endsWith('index.html')
  );
  const page = await target.page();
  const doc = await getDocument(page);

  // Setting up Recreation model because it has very few data requirements
  const investTable = await findByRole(doc, 'table');
  const button = await findByRole(investTable, 'button', { name: /Visitation/ });
  button.click();

  const runButton = await findByRole(doc, 'button', { name: 'Run' });
  const typeDelay = 10;
  const workspace = await findByLabelText(doc, /Workspace/);
  await workspace.type(TMP_DIR, { delay: typeDelay });
  const aoi = await findByLabelText(doc, /area of interest/);
  await aoi.type(TMP_AOI_PATH, { delay: typeDelay });
  const startYear = await findByLabelText(doc, /start year/);
  await startYear.type('2008', { delay: typeDelay });
  const endYear = await findByLabelText(doc, /end year/);
  await endYear.type('2012', { delay: typeDelay });
  
  // Button is disabled until validation completes
  await waitFor(async () => {
    const isEnabled = await page.evaluate(
      (btn) => !btn.disabled,
      runButton
    );
    expect(isEnabled).toBe(true);
  });
  page.screenshot({ path: `${SCREENSHOT_PREFIX}before-run-click.png` });
  await runButton.click();
  const logTab = await findByText(doc, 'Log');
  // Log tab is not active until after the invest logfile is opened
  await waitFor(async () => {
    const prop = await logTab.getProperty('className');
    const vals = await prop.jsonValue();
    expect(vals.includes('active')).toBeTruthy();
  });

  const cancelButton = await findByText(doc, 'Cancel Run');
  await cancelButton.click();
  await waitFor(async () => {
    expect(await findByText(doc, 'Run Canceled'));
    expect(await findByText(doc, 'Open Workspace'));
  });
}, 50000); // 10x default timeout: sometimes expires in GHA

// Test for duplicate application launch.
// We have the binary path, so now let's launch a new subprocess with the same binary
// The test is that the subprocess exits within a certain reasonable timeout.
// Also verify that window 1 has focus.
test('App re-launch will exit and focus on first instance', async () => {
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
});
