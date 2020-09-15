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



// beforeEach(async () => {
//   return await app.start()
// })

// afterAll(async () => {
//   if (app.isRunning()) {
//     return await app.stop()
//   }
// })

test('somthing', async () => {
  await app.start();
  expect(app.isRunning()).toBeTruthy();
  await app.stop();
	// expect(app.browserWindow.isVisible()).toBeTruthy()
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