import { Application } from 'spectron';

// ['disable-dev-shm-usage', 'no-sandbox', 'headless']
const app = new Application({
  path: 'node_modules/electron/dist/electron',
  args: ['main.js'],
  chromeDriverArgs: ["--disable-dev-shm-usage", "--no-sandbox", "--headless"]
})

jest.setTimeout(10000)

beforeEach(() => {
  return app.start()
})

afterEach(() => {
  if (app.isRunning()) {
    return app.stop()
  }
})

test('Application starts', () => {
  console.log(app.client.getTitle())
}, 10000)