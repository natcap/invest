import { Application } from 'spectron';

const app = new Application({
  path: 'node_modules/electron/dist/electron',
  args: ['src/main.js'],
  // chromeDriverArgs: ["--disable-dev-shm-usage", "--no-sandbox", "--headless"],
  chromeDriverLogPath: 'chromeDriver.log'
})

app.client.getMainProcessLogs().then(function (logs) {
  logs.forEach(function (log) {
    console.log(log)
  })
})

jest.setTimeout(10000)

beforeEach(() => {
  return app.start()
})

afterEach(() => {
  // if (app.isRunning()) {
  return app.stop()
  // }
})

test('Application starts', async () => {
  await new Promise(resolve => setTimeout(resolve, 8000)); // sleep while app starts
  console.log(app.client.getTitle())
  expect(true)
}, 10000)