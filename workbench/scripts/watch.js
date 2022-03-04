const { createServer } = require('vite');
const path = require('path');

const sharedConfig = {
  build: {
    watch: {},
  },
};

async function watch() {
  const server = await createServer({
    ...sharedConfig,
    configFile: path.join(__dirname, '../src/renderer/vite.config.js')
  });

  await server.listen();
  server.printUrls();
}

watch();