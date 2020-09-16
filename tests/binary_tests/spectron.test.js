import { Application } from 'spectron';
import glob from 'glob';

jest.setTimeout(60000)

// const appPath = glob.sync('dist/invest-desktop_*');
const appPath = glob.sync('/home/dmf/Downloads/invest-desktop_*');
console.log(appPath)
let app;

beforeAll(() => {
  app = new Application({
    path: appPath[0],
    // args: ["src/specmain.js"],
    // args: ["src/main.js"],
    // webdriverOptions: {execArgv: "-r @babel/register"},
    // chromeDriverArgs: ['--disable-dev-shm-usage', '--headless'],
    chromeDriverLogPath: 'chromeDriver.log'
  });
});



beforeEach(async () => {
  return await app.start()
})

afterAll(async () => {
  if (app.isRunning()) {
    return await app.stop()
  }
})

test('browser window opens', () => {
  expect(app.isRunning()).toBeTruthy();
	expect(app.browserWindow.isVisible()).toBeTruthy()
})

test.only('execute and cancel an invest run', async () => {
  console.log(app.isRunning());
  // await app.client.$('button=Load Parameters').click(); // works
  // const loadButton = await app.client.$('button=Load Parameters');
  // await loadButton.click(); // no work
  await app.client.click('button=Load Parameters');
})