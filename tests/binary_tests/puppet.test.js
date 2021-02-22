import fs from 'fs';
import path from 'path';
import glob from 'glob';
import fetch from 'node-fetch';
import { spawn } from 'child_process';
import puppeteer from 'puppeteer-core';
import { getDocument, queries, waitFor } from 'pptr-testing-library';

jest.setTimeout(60000); // This test takes ~15 seconds, but longer in CI
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
  console.log('beforeAll');
  // electronProcess = spawn(
  //   `"${binaryPath}"`, [`--remote-debugging-port=${PORT}`],
  //   { shell: true }
  // );
  // electronProcess.stderr.on('data', (data) => {
  //   console.log(`${data}`);
  // });
  // // so we don't make the next fetch too early
  // await new Promise((resolve) => setTimeout(resolve, 5000));
  // const res = await fetch(`http://localhost:${PORT}/json/version`);
  // const data = JSON.parse(await res.text());
  // browser = await puppeteer.connect({
  //   browserWSEndpoint: data.webSocketDebuggerUrl, // this works
  //   // browserURL: `http://localhost:${PORT}`,    // this also works
  //   defaultViewport: { width: 1000, height: 800 },
  // });
  // makeAOI();
});

afterAll(async () => {
  console.log('afterAll');
//   try {
//     await browser.close();
//   } catch (error) {
//     console.log(binaryPath);
//     console.error(error);
//   }
//   console.log('should be done with tests');
//   // being extra careful with recursive rm
//   // if (TMP_DIR.startsWith('tests/data')) {
//   //   fs.rmdirSync(TMP_DIR, { recursive: true });
//   // }
//   // I thought this business would be necessary to kill the spawned shell
//   // process running electron - since that's how we kill a similar spawned
//   // subprocess in the app, but actually it is not.
//   // if (electronProcess.pid) {
//   //   console.log(electronProcess.pid)
//   //   if (process.platform !== 'win32') {
//   //     process.kill(-electronProcess.pid, 'SIGTERM');
//   //   } else {
//   //     exec(`taskkill /pid ${electronProcess.pid} /t /f`)
//   //   }
//   // }
//   // const wasKilled = electronProcess.kill();
//   // console.log(`electron process was killed: ${wasKilled}`);
});

test('Run a real invest model', async () => {
  // const { findByText, findByLabelText, findByRole } = queries;
  console.log('test');
  electronProcess = spawn(
    `"${binaryPath}"`, [`--remote-debugging-port=${PORT}`],
    { shell: true }
  );
  console.log('after test');
  // await waitFor(() => {
  //   expect(browser.isConnected()).toBeTruthy();
  // });
});
