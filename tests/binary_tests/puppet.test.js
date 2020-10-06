import fs from 'fs';
import path from 'path';
import glob from 'glob';
import fetch from 'node-fetch';
import { remote } from 'electron';
import { spawn, spawnSync } from 'child_process';
import puppeteer from 'puppeteer-core';
import { getDocument, queries, waitFor } from 'pptr-testing-library';

import { cleanupDir } from '../../src/utils';
import { getFlaskIsReady } from '../../src/server_requests';

jest.setTimeout(250000) // This test takes ~15 seconds, but longer in CI
const PORT = 9009;

// let binaryPath = glob.sync('./dist/invest-workbench_*@(zip|exe|AppImage)')[0]
// For ease of automated testing, run the app from the 'unpacked' directory
// to avoid need to install first on windows or extract on mac.
let binaryPath;
if (process.platform === 'darwin') {
  // https://github.com/electron-userland/electron-builder/issues/2724#issuecomment-375850150
  console.log(glob.sync('./dist/mac/*'))
  binaryPath = glob.sync('./dist/mac/InVEST*')[0]
} else if (process.platform === 'win32') {
  binaryPath = glob.sync('./dist/win-unpacked/InVEST*.exe')[0]
} else {
  binaryPath = glob.sync('./dist/linux-unpacked/InVEST*.AppImage')[0]
}
// if (binaryPath.endsWith('.zip')) {
//   // The MacOS exe needs to be extracted first
//   spawnSync('unzip', [binaryPath, '-d', './dist/'])
//   binaryPath = glob.sync('./dist/*.app')[0]
// }
console.log(binaryPath)
fs.accessSync(binaryPath, fs.constants.X_OK)
const TMP_DIR = fs.mkdtempSync('tests/data/_')
const TMP_AOI_PATH = path.join(TMP_DIR, 'aoi.geojson')
let electronProcess;
let browser;

function makeAOI() {
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
  fs.writeFileSync(TMP_AOI_PATH, JSON.stringify(geojson))
}

// errors are not thrown from an async beforeAll
// https://github.com/facebook/jest/issues/8688
beforeAll(async () => {
  electronProcess = spawn(
    `"${binaryPath}"`, [`--remote-debugging-port=${PORT}`],
    { shell: true },
  );
  electronProcess.stderr.on('data', (data) => {
    console.log(`${data}`)
  });
  // so we don't make the next fetch too early
  await new Promise(resolve => setTimeout(resolve, 5000)) 
  const res = await fetch(`http://localhost:${PORT}/json/version`);
  const data = JSON.parse(await res.text());
  browser = await puppeteer.connect({
    browserWSEndpoint: data.webSocketDebuggerUrl,  // this works
    // browserURL: `http://localhost:${PORT}`,     // this also works
    defaultViewport: { width: 1000, height: 800 },
  });
  makeAOI()
})

afterAll(async () => {
  try {
    await browser.close();
  } catch (error) {
    console.log(binaryPath);
    console.error(error);
  }
  cleanupDir(TMP_DIR);
  electronProcess.kill();
})

test('Run a real invest model', async () => {
  const { findByText, findByLabelText } = queries;
  console.log(browser);
  await waitFor(() => {
    expect(browser.isConnected()).toBeTruthy();
  })
  let page = (await browser.pages())[0];
  const doc = await getDocument(page);

  // Setting up Recreation model because it has very few data requirements
  const button = await findByText(doc, /Visitation/);
  button.click()
  const workspace = await findByLabelText(doc, /Workspace/);
  await workspace.type(TMP_DIR, { delay: 10 })
  const aoi = await findByLabelText(doc, /Area of Interest/);
  await aoi.type(TMP_AOI_PATH, { delay: 10 })
  const startYear = await findByLabelText(doc, /Start Year/);
  await startYear.type('2008', { delay: 10 })
  const endYear = await findByLabelText(doc, /End Year/);
  await endYear.type('2012', { delay: 10 })
  
  const executeButton = await findByText(doc, 'Execute');
  // Button is disabled until validation completes
  await waitFor(async () => {
    const isEnabled = await page.evaluate((button) => {
      return !button.disabled
    }, executeButton)
    expect(isEnabled).toBeTruthy()
  })  
  
  executeButton.click();
  const logTab = await findByText(doc, 'Log');
  // Log tab is not active until after the invest logfile is opened
  await waitFor(async () => {
    const prop = await logTab.getProperty('className');
    const vals = await prop.jsonValue();
    expect(vals.includes('active')).toBeTruthy();
  })

  const cancelButton = await findByText(doc, 'Cancel Run');
  cancelButton.click();
  await waitFor(async () => {
    expect(await findByText(doc, 'Open Workspace'));
    // expect(await findByText(doc, 'Run Canceled'));
  })
})
