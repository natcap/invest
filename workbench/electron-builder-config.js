const { execFileSync } = require('child_process');

const OS = process.platform;
const ARCH = process.arch;
const EXT = OS === 'win32' ? 'exe' : 'dmg';

// Uniquely identify the changeset we're building & packaging.
const investVersion = execFileSync(
  '../dist/invest/invest', ['--version']
).toString().trim();

// the appID may not display anywhere, but seems to control if the
// install overwrites pre-existing or creates a new install directory.
// It deliberately varies by invest version.
const APP_ID = `NaturalCapitalProject.Invest.Workbench.${investVersion}`;

// productName controls the install dirname & app name
// We might want to remove the workbench version from this name
const PRODUCT_NAME = `InVEST ${investVersion} Workbench`;
const ARTIFACT_NAME = `invest_${investVersion}_workbench_${OS}_${ARCH}.${EXT}`;

const config = {
  extraMetadata: {
    main: 'build/main/main.js',
  },
  extraResources: [
    {
      from: '../dist/invest',
      to: 'invest',
    },
    {
      from: 'resources/storage_token.txt',
      to: 'storage_token.txt',
    },
  ],
  extraFiles: [{
    from: '../LICENSE.txt',
    to: 'LICENSE.txt',
  }],
  appId: APP_ID,
  productName: PRODUCT_NAME,
  artifactName: ARTIFACT_NAME,
  mac: {
    category: 'public.app-category.business',
    icon: 'resources/invest-in-shadow-white.png',
    target: 'dmg',
  },
  win: {
    target: 'nsis',
    icon: 'resources/invest-in-shadow-white.png',
  },
  nsis: {
    oneClick: false,
  },
  files: [
    'build/**/*',
    'node_modules/**/*',
  ],
  publish: null // undocumented. does what you would expect ['never'] to do
};

module.exports = config;
