const { execFileSync } = require('child_process');
const pkg = require('./package');

const OS = process.platform;
const ARCH = process.arch;
const EXT = OS === 'win32' ? 'exe' : 'dmg';

// the appID may not display anywhere, but seems to control if the
// install overwrites pre-existing or creates a new install directory.
// It deliberately only varies by invest version (not workbench version)
const APP_ID = `NaturalCapitalProject.Invest.Workbench.${pkg.invest.version}`;

// Uniquely identify the changeset we're building & packaging.
const workbenchVersion = execFileSync('git', ['describe', '--tags'])
  .toString().trim();

// productName controls the install dirname & app name
// We might want to remove the workbench version from this name
const PRODUCT_NAME = `InVEST ${pkg.invest.version} Workbench ${workbenchVersion}`;
const ARTIFACT_NAME = `invest_${pkg.invest.version}_workbench_${workbenchVersion}_${OS}_${ARCH}.${EXT}`;

const config = {
  extraMetadata: {
    main: 'build/main.js'
  },
  extraResources: [
    {
      from: 'build/invest',
      to: 'invest',
    },
  ],
  appId: APP_ID,
  productName: PRODUCT_NAME,
  artifactName: ARTIFACT_NAME,
  mac: {
    category: 'public.app-category.business',
    icon: 'resources/invest-in-shadow-white.png',
    target: 'dmg',
  },
  linux: {
    target: [
      'AppImage',
    ],
    icon: 'resources/invest-in-shadow-white.png',
    category: 'Science',
  },
  win: {
    target: 'nsis',
    icon: 'resources/invest-in-shadow-white.png'
  },
  nsis: {
    oneClick: false,
  },
  files: [
    'build/**/*',
    'node_modules/**/*',
  ],
};

module.exports = config;
