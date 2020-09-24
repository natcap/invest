import glob from 'glob';
import { remote } from 'electron';
import { spawn } from 'child_process';
import puppeteer from 'puppeteer-core';
import { getDocument, queries, waitFor } from 'pptr-testing-library';

jest.setTimeout(60000)

const PORT = 9009;
// const binaryPath = glob.sync('/home/dmf/Downloads/invest-desktop_*')[0]
const binaryPath = '/home/dmf/projects/invest-workbench/dist/linux-unpacked/invest-electron'
const SAMPLE_DATA_JSON = 'home/dmf/projects/invest/data/invest-sample-data/coastal_vuln_grandbahama.invs.json'
let electronProcess;
let browser;

beforeAll(async () => {
  electronProcess = spawn(
    binaryPath, [`--remote-debugging-port=${PORT}`],
    { shell: true },
  );

  await new Promise(resolve => { setTimeout(resolve, 5000) });
  browser = await puppeteer.connect({
    browserURL: `http://localhost:${PORT}`,
    defaultViewport: { width: 1000, height: 800 },
  });
  // browser = await puppeteer.launch({
  //   executeablePath: binaryPath,
  // }) 
})

// afterAll(async () => {
//   await browser.close() 
//   electronProcess.kill()
// })

test('something', async () => {
  const { getByText, findByText } = queries;
  await new Promise(resolve => { setTimeout(resolve, 1000) });
  // console.log(browser.isConnected());
  const page = (await browser.pages())[0];
  const doc = await getDocument(page);
  // const button = await getByText(doc, 'Load Parameters');
  // // console.log(button);
  // const mockDialogData = {
  //   filePaths: [SAMPLE_DATA_JSON]
  // }
  // remote.dialog.showOpenDialog.mockResolvedValue(mockDialogData)
  // button.click();
  const recentJobCard = await findByText(
    doc, '/home/dmf/projects/invest-workbench/runs/rec-sample544'
  );
  recentJobCard.click();
  const executeButton = await findByText(doc, 'Execute');
  // Button is disabled until validation completes
  await waitFor(async () => {
    const isEnabled = await page.evaluate((button) => {
      return !button.disabled
    }, executeButton)
    expect(isEnabled).toBeTruthy()
  })  
  
  executeButton.click();
  // await new Promise(resolve => { setTimeout(resolve, 5000) });
  const logTab = await findByText(doc, 'Log');
  await waitFor(async () => {
    const prop = await logTab.getProperty('className');
    const vals = await prop.jsonValue();
    expect(vals.includes('active')).toBeTruthy();
    // expect(false).toBeTruthy()
  })
})
