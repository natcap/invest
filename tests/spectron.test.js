import { Application } from 'spectron';

const app = new Application({
  path: 'node_modules/electron/dist/electron',
  args: ['-r @babel/register src/main.js'],
  // chromeDriverArgs: ['remote-debugging-port=9222'],
  chromeDriverLogPath: 'chromeDriver.log'
})

jest.setTimeout(10000)

// beforeEach(() => {
//   return app.start()
// })

// afterEach(() => {
//   if (app.isRunning()) {
//     return app.stop()
//   }
// })


test('Application starts', async () => {
  console.log('1')
  await app.start()
  // app.client.getMainProcessLogs().then(function (logs) {
  //   logs.forEach(function (log) {
  //     console.log(log)
  //   })
  // })
  console.log('2')
  const title = app.client.getTitle()  
  console.log(title)
  await app.stop()
})