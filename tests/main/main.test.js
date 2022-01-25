/**
 * @jest-environment node
 */

/* Tests for main process code running in a node env.
 * Tests requiring a DOM do not belong here, they are
 * in tests/renderer/*.test.js.
*/

import fs from 'fs';
import os from 'os';
import path from 'path';
import { execFileSync } from 'child_process';

import { app, ipcMain } from 'electron';
import yazl from 'yazl';
import rimraf from 'rimraf';
import GettextJS from 'gettext.js';
import fetch from 'node-fetch';

import {
  createWindow,
  destroyWindow,
  removeIpcMainListeners
} from '../../src/main/main';
import {
  checkFirstRun,
  APP_HAS_RUN_TOKEN
} from '../../src/main/setupCheckFirstRun';
import {
  createPythonFlaskProcess,
  getFlaskIsReady
} from '../../src/main/createPythonFlaskProcess';
import findInvestBinaries from '../../src/main/findInvestBinaries';
import extractZipInplace from '../../src/main/extractZipInplace';
import { ipcMainChannels } from '../../src/main/ipcMainChannels';
import investUsageLogger from '../../src/main/investUsageLogger';

// mock out the global gettext function - avoid setting up translation
global.window._ = (x) => x;

jest.mock('node-fetch');
jest.mock('child_process');
jest.mock('../../src/main/createPythonFlaskProcess');

beforeEach(() => {
  execFileSync.mockReturnValue('foo');
  createPythonFlaskProcess.mockImplementation(() => {});
  getFlaskIsReady.mockResolvedValue(true);
  // These vars are only defined in an electron environment and our
  // app expects them to be defined.
  process.defaultApp = 'test'; // imitates dev mode
  process.resourcesPath = 'path/to/electron/package';
});

describe('checkFirstRun', () => {
  const tokenPath = path.join(app.getPath(), APP_HAS_RUN_TOKEN);
  beforeEach(() => {
    try {
      fs.unlinkSync(tokenPath);
    } catch {}
  });

  afterEach(() => {
    try {
      fs.unlinkSync(tokenPath);
    } catch {}
  });

  test('should return true & create token if token does not exist', () => {
    expect(fs.existsSync(tokenPath)).toBe(false);
    expect(checkFirstRun()).toBe(true);
    expect(fs.existsSync(tokenPath)).toBe(true);
  });

  test('should return false if token already exists', () => {
    fs.writeFileSync(tokenPath, '');
    expect(checkFirstRun()).toBe(false);
  });
});

describe('findInvestBinaries', () => {
  const ext = (process.platform === 'win32') ? '.exe' : '';
  const filename = `invest${ext}`;
  test('should point to build folder in dev mode', () => {
    const isDevMode = true;
    const exePath = findInvestBinaries(isDevMode);
    expect(exePath).toBe(path.join('build', 'invest', filename));
  });
  test('should point to resourcesPath in production', async () => {
    const isDevMode = false;
    const exePath = findInvestBinaries(isDevMode);
    expect(exePath).toBe(path.join(process.resourcesPath, 'invest', filename));
  });
  test('should throw if the invest exe is bad', async () => {
    execFileSync.mockImplementation(() => {
      throw new Error('error from invest --version');
    });
    const isDevMode = false;
    expect(() => findInvestBinaries(isDevMode)).toThrow();
  });
});

describe('extractZipInplace', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'data-'));
  const zipPath = path.join(root, 'output.zip');
  let level1Dir;
  let level2Dir;
  let file1Path;
  let file2Path;
  let doneZipping = false;

  beforeEach((done) => {
    level1Dir = fs.mkdtempSync(path.join(root, 'level1'));
    level2Dir = fs.mkdtempSync(path.join(level1Dir, 'level2'));
    file1Path = path.join(level1Dir, 'file1');
    file2Path = path.join(level2Dir, 'file2');
    fs.closeSync(fs.openSync(file1Path, 'w'));
    fs.closeSync(fs.openSync(file2Path, 'w'));

    const zipfile = new yazl.ZipFile();
    // adding the deeper file first, so extract function needs to
    // deal with extracting to non-existent directories.
    zipfile.addFile(file2Path, path.relative(root, file2Path));
    zipfile.addFile(file1Path, path.relative(root, file1Path));
    zipfile.outputStream.pipe(
      fs.createWriteStream(zipPath)
    ).on('close', () => {
      // being extra careful with recursive rm
      if (level1Dir.startsWith(path.join(root, 'level1'))) {
        rimraf(level1Dir, (error) => {
          if (error) {
            throw error;
          }
          doneZipping = true;
          done();
        });
      }
    });
    zipfile.end();
  });

  afterEach(() => {
    fs.rmSync(root, { recursive: true, force: true });
  });

  test('should extract recursively', async () => {
    expect(doneZipping).toBe(true);
    // The expected state after the setup, before extraction
    expect(fs.existsSync(zipPath)).toBe(true);
    expect(fs.existsSync(file1Path)).toBe(false);
    expect(fs.existsSync(file2Path)).toBe(false);

    expect(await extractZipInplace(zipPath)).toBe(true);

    // And the expected state after extraction
    expect(fs.existsSync(file1Path)).toBe(true);
    expect(fs.existsSync(file2Path)).toBe(true);
  });
});

describe('createWindow', () => {
  test('should register various ipcMain listeners', async () => {
    await createWindow();
    const expectedHandleChannels = [
      ipcMainChannels.SHOW_OPEN_DIALOG,
      ipcMainChannels.SHOW_SAVE_DIALOG,
      ipcMainChannels.IS_FIRST_RUN,
      ipcMainChannels.SET_LANGUAGE,
      ipcMainChannels.GET_N_CPUS,
    ];
    const expectedOnChannels = [
      ipcMainChannels.DOWNLOAD_URL,
      ipcMainChannels.INVEST_RUN,
      ipcMainChannels.INVEST_KILL,
      ipcMainChannels.INVEST_READ_LOG,
      ipcMainChannels.GETTEXT,
      ipcMainChannels.SHOW_ITEM_IN_FOLDER,
    ];
    // Even with mocking, the 'on' method is a real event handler,
    // so we can get it's registered events from the EventEmitter.
    const registeredOnChannels = ipcMain.eventNames();
    // for 'handle', we query the mock's calls.
    const registeredHandleChannels = ipcMain.handle.mock.calls.map(
      (item) => item[0]
    );
    expect(registeredHandleChannels.sort())
      .toEqual(expectedHandleChannels.sort());
    expect(registeredOnChannels.sort())
      .toEqual(expectedOnChannels.sort());
    removeIpcMainListeners();
    expect(ipcMain.eventNames()).toEqual([]);
    destroyWindow();
  });
});

describe('Translation', () => {
  const i18n = new GettextJS();
  const testLanguage = 'es';
  const messageCatalog = {
    '': {
      language: testLanguage,
      'plural-forms': 'nplurals=2; plural=(n!=1);',
    },
    Open: 'σρєи',
    Language: 'ℓαиgυαgє',
  };

  beforeAll(async () => {
    getInvestModelNames.mockResolvedValue({});

    i18n.loadJSON(messageCatalog, 'messages');
    console.log(i18n.gettext('Language'));

    // mock out the relevant IPC channels
    ipcRenderer.invoke.mockImplementation((channel, arg) => {
      if (channel === ipcMainChannels.SET_LANGUAGE) {
        i18n.setLocale(arg);
      }
      return Promise.resolve();
    });

    ipcRenderer.sendSync.mockImplementation((channel, arg) => {
      if (channel === ipcMainChannels.GETTEXT) {
        return i18n.gettext(arg);
      }
      return undefined;
    });

    // this is the same setup that's done in src/renderer/index.js (out of test scope)
    ipcRenderer.invoke(ipcMainChannels.SET_LANGUAGE, 'en');
    global.window._ = ipcRenderer.sendSync.bind(null, ipcMainChannels.GETTEXT);
  });
  afterAll(async () => {
    jest.resetAllMocks();
  });

  test('Text rerenders in new language when language setting changes', async () => {
    const {
      findByText,
      getByText,
      findByLabelText,
    } = render(<App />);

    fireEvent.click(await findByLabelText('settings'));
    let languageInput = await findByLabelText('Language', { exact: false });
    expect(languageInput).toHaveValue('en');

    fireEvent.change(languageInput, { target: { value: testLanguage } });

    // text within the settings modal component should be translated
    languageInput = await findByLabelText(messageCatalog.Language, { exact: false });
    expect(languageInput).toHaveValue(testLanguage);

    // text should also be translated in other components
    // such as the Open button (visible in background)
    await findByText(messageCatalog.Open);

    // text without a translation in the message catalog should display in the default English
    expect(getByText('Logging threshold')).toBeDefined();

    // resetting language should re-render components in English
    fireEvent.click(getByText('Reset to Defaults'));
    expect(await findByText('Language')).toBeDefined();
    expect(await findByText('Open')).toBeDefined();
  });
});

describe('investUsageLogger', () => {
  test('sends requests with correct payload', () => {
    const modelPyname = 'natcap.invest.carbon';
    const args = {
      workspace_dir: 'foo',
      aoi: 'bar',
    };
    const investStdErr = '';
    const usageLogger = investUsageLogger();

    usageLogger.start(modelPyname, args);
    expect(fetch.mock.calls).toHaveLength(1);
    const startPayload = JSON.parse(fetch.mock.calls[0][1].body);

    usageLogger.exit(investStdErr);
    expect(fetch.mock.calls).toHaveLength(2);
    const exitPayload = JSON.parse(fetch.mock.calls[1][1].body);

    expect(startPayload.session_id).toBe(exitPayload.session_id);
    expect(startPayload.model_pyname).toBe(modelPyname);
    expect(JSON.parse(startPayload.model_args)).toMatchObject(args);
    expect(startPayload.invest_interface).toContain('Workbench');
    expect(exitPayload.status).toBe(investStdErr);
  });
});
