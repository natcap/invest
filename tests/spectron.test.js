import { Application } from 'spectron';

const app = new Application({
  path: 'node_modules/electron/dist/electron',
  args: ["src/specmain.js"],
  // args: ["src/main.js"],
  webdriverOptions: {execArgv: "-r @babel/register"},
  chromeDriverArgs: ['--disable-dev-shm-usage', '--headless'],
  chromeDriverLogPath: 'chromeDriver.log'
})

jest.setTimeout(10000)

// beforeEach(() => {
//   return app.start()
// })

afterAll(() => {
  if (app.isRunning()) {
    return app.stop()
  }
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