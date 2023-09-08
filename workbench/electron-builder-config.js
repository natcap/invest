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

// this version appears as a footer in the NSIS installer & uninstaller.
// It includes package.json version by default, but that is meaningless for us.
// We can override that, but also must maintain semver compliance, so
// always trim to this format.
const installerVersion = investVersion.match(/[0-9]+\.[0-9]+\.[0-9]+/)[0];

const config = {
  extraMetadata: {
    main: 'build/main/main.js',
    version: installerVersion,
  },
  extraResources: [
    {
      from: '../dist/invest',
      to: 'invest',
    },
    {
      from: '../dist/userguide',
      to: 'documentation',
    },
    {
      from: 'resources/storage_token.txt',
      to: 'storage_token.txt',
    },
    {
      from: '../LICENSE.txt',
      to: 'LICENSE.InVEST.txt',
    },
    {
      from: '../NOTICE.txt',
      to: 'NOTICE.InVEST.txt',
    },
  ],
  appId: APP_ID,
  productName: PRODUCT_NAME,
  artifactName: ARTIFACT_NAME,
  mac: {
    category: 'public.app-category.business',
    icon: 'resources/InVEST-2-574x574.ico',
    target: 'dmg',
  },
  win: {
    target: 'nsis',
    icon: 'resources/InVEST-2-256x256.ico',
  },
  nsis: {
    allowToChangeInstallationDirectory: true,
    createDesktopShortcut: false,
    installerHeader: 'resources/InVEST-header-wcvi-rocks.bmp',
    oneClick: false,
    uninstallDisplayName: PRODUCT_NAME,
    license: 'build/license_en.txt',
  },
  files: [
    'build/**/*',
    'node_modules/**/*',
  ],
  publish: null // undocumented. does what you would expect ['never'] to do
};

module.exports = config;
