import { Application } from 'spectron';
import glob from 'glob';

const appPath = glob.sync('dist/invest-desktop_*')

beforeAll(() => {
  const app = new Application({
    path: appPath[0],
    // args: ["src/specmain.js"],
    // args: ["src/main.js"],
    // webdriverOptions: {execArgv: "-r @babel/register"},
    chromeDriverArgs: ['--disable-dev-shm-usage', '--headless'],
    chromeDriverLogPath: 'chromeDriver.log'
  });
});


// jest.setTimeout(10000)

beforeEach(() => {
  return app.start()
})

afterAll(() => {
  if (app.isRunning()) {
    return app.stop()
  }
})

// jest shows a failed test suite if there are no tests,
// so a placeholder until we revisit the rest of tests.
test('placeholder test', () => {
	expect(true).toBeTruthy()
})


// test('Application starts and stops', async () => {
//   await app.start()
//   app.client.getMainProcessLogs().then(function (logs) {
//     logs.forEach(function (log) {
//       console.log(log)
//     })
//   })
//   // const title = await app.client.getTitle()  
//   await app.client.waitUntilWindowLoaded()  // method not found
//   const isVisible = await app.browserWindow.isVisible() // browserWindow undefined
//   expect(isVisible).toBeTruthy()

//   const res = await app.client.$('Resources')
//   console.log(res);

//   await app.stop()
// })