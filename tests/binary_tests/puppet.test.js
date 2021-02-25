import fs from 'fs';
import path from 'path';
import glob from 'glob';
import fetch from 'node-fetch';
import { spawn } from 'child_process';
import puppeteer from 'puppeteer-core';
import { getDocument, queries, waitFor } from 'pptr-testing-library';

jest.setTimeout(120000); // This test takes ~15 seconds, but longer in CI
const PORT = 9009;

// For ease of automated testing, run the app from the 'unpacked' directory
// to avoid need to install first on windows or extract on mac.
let binaryPath;
if (process.platform === 'darwin') {
  // https://github.com/electron-userland/electron-builder/issues/2724#issuecomment-375850150
  [binaryPath] = glob.sync('./dist/mac/*.app/Contents/MacOS/InVEST*');
} else if (process.platform === 'win32') {
  [binaryPath] = glob.sync('./dist/win-unpacked/InVEST*.exe');
} else {
  binaryPath = './dist/linux-unpacked/invest-workbench';
}

console.log(binaryPath);
fs.accessSync(binaryPath, fs.constants.X_OK);
const TMP_DIR = fs.mkdtempSync('tests/data/_');
const TMP_AOI_PATH = path.join(TMP_DIR, 'aoi.geojson');
let electronProcess;
let browser;

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
  electronProcess = spawn(
    // remote-debugging-port is a chromium arg
    `"${binaryPath}"`, [`--remote-debugging-port=${PORT}`],
    { shell: true }
  );
  electronProcess.stderr.on('data', (data) => {
    console.log(`${data}`);
  });

  // get data about the remote debugging endpoint
  // so we don't make the next fetch too early
  await new Promise(resolve => setTimeout(resolve, 10000));
  const res = await fetch(`http://localhost:${PORT}/json/version`);
  const data = JSON.parse(await res.text());

  // connect to the debugging endpoint
  browser = await puppeteer.connect({
    browserWSEndpoint: data.webSocketDebuggerUrl, // this works
    // browserURL: `http://localhost:${PORT}`,    // this also works
    defaultViewport: { width: 1000, height: 800 },
  });
  // set up test data
  makeAOI();
});

afterAll(async () => {
  try {
    await browser.close();
  } catch (error) {
    console.log(binaryPath);
    console.error(error);
  }

  // being extra careful with recursive rm
  if (TMP_DIR.startsWith('tests/data')) {
    fs.rmdirSync(TMP_DIR, { recursive: true });
  }
  // I thought this business would be necessary to kill the spawned shell
  // process running electron - since that's how we kill a similar spawned
  // subprocess in the app, but actually it is not.
  // if (electronProcess.pid) {
  //   console.log(electronProcess.pid)
  //   if (process.platform !== 'win32') {
  //     process.kill(-electronProcess.pid, 'SIGTERM');
  //   } else {
  //     exec(`taskkill /pid ${electronProcess.pid} /t /f`)
  //   }
  // }
  const wasKilled = electronProcess.kill();
  console.log(`electron process was killed: ${wasKilled}`);
});

test('Run a real invest model', async () => {
  const { findByText, findByLabelText, findByRole } = queries;
  await waitFor(() => {
    expect(browser.isConnected()).toBeTruthy();
  });
  const pages = (await browser.pages());
  // find the mainWindow's index.html, not the splashScreen's splash.html
  let page;
  pages.forEach((p) => {
    if (p.url().endsWith('index.html')) {
      page = p;
    }
  });
  const doc = await getDocument(page);

  // Setting up Recreation model because it has very few data requirements
  const investTable = await findByRole(doc, 'table');
  const button = await findByRole(investTable, 'button', { name: /Visitation/ });
  button.click();

  const workspace = await findByLabelText(doc, /Workspace/);
  await workspace.type(TMP_DIR, { delay: 10 });
  const aoi = await findByLabelText(doc, /area of interest/);
  await aoi.type(TMP_AOI_PATH, { delay: 10 });
  const startYear = await findByLabelText(doc, /start year/);
  await startYear.type('2008', { delay: 10 });
  const endYear = await findByLabelText(doc, /end year/);
  await endYear.type('2012', { delay: 10 });

  const runButton = await findByText(doc, 'Run');
  // Button is disabled until validation completes
  await waitFor(async () => {
    const isEnabled = await page.evaluate(
      (btn) => !btn.disabled,
      runButton
    );
    expect(isEnabled).toBeTruthy();
  });

  runButton.click();
  const logTab = await findByText(doc, 'Log');
  // Log tab is not active until after the invest logfile is opened
  await waitFor(async () => {
    const prop = await logTab.getProperty('className');
    const vals = await prop.jsonValue();
    expect(vals.includes('active')).toBeTruthy();
  }, 18000); // 4x default timeout: sometimes this expires unmet in GHA

  const cancelButton = await findByText(doc, 'Cancel Run');
  cancelButton.click();
  await waitFor(async () => {
    expect(await findByText(doc, 'Run Canceled'));
    expect(await findByText(doc, 'Open Workspace'));
  });
});
