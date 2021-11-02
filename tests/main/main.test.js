/* Tests for main process code and integration tests for front-end
features that depend on a BrowserWindow instance, since that is
instantiated by main.
*/

import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';

import { app, ipcMain, ipcRenderer } from 'electron';
import React from 'react';
import {
  fireEvent, render, waitFor,
} from '@testing-library/react';
import '@testing-library/jest-dom';
import yazl from 'yazl';
import rimraf from 'rimraf';
import gettext_js from 'gettext.js';

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
  getFlaskIsReady,
} from '../../src/main/createPythonFlaskProcess';
import findInvestBinaries from '../../src/main/findInvestBinaries';
import extractZipInplace from '../../src/main/extractZipInplace';
import { ipcMainChannels } from '../../src/main/ipcMainChannels';
import { getInvestModelNames } from '../../src/renderer/server_requests';
import App from '../../src/renderer/app';
import {
  clearSettingsStore,
  getSettingsValue,
} from '../../src/renderer/components/SettingsModal/SettingsStorage';

// mock out the global gettext function - avoid setting up translation
global.window._ = x => x;

jest.mock('child_process');
execFileSync.mockReturnValue('foo');
jest.mock('../../src/main/createPythonFlaskProcess');
createPythonFlaskProcess.mockImplementation(() => {});
jest.mock('../../src/renderer/server_requests');
getFlaskIsReady.mockResolvedValue(true);

// These vars are only defined in an electron environment and our
// app expects them to be defined.
process.defaultApp = 'test'; // imitates dev mode
process.resourcesPath = 'path/to/electron/package';

describe('checkFirstRun', () => {
  const tokenPath = path.join(app.getPath(), APP_HAS_RUN_TOKEN);
  beforeEach(() => {
    try {
      fs.unlinkSync(tokenPath);
    } catch {}
  });

  afterAll(() => {
    try {
      fs.unlinkSync(tokenPath);
    } catch {}
  });

  it('should return true & create token if token does not exist', () => {
    expect(fs.existsSync(tokenPath)).toBe(false);
    expect(checkFirstRun()).toBe(true);
    expect(fs.existsSync(tokenPath)).toBe(true);
  });

  it('should return false if token already exists', () => {
    fs.writeFileSync(tokenPath, '');
    expect(checkFirstRun()).toBe(false);
  });
});

describe('findInvestBinaries', () => {
  afterAll(() => {
    execFileSync.mockReset();
  });
  const ext = (process.platform === 'win32') ? '.exe' : '';
  const filename = `invest${ext}`;
  it('should point to build folder in dev mode', () => {
    const isDevMode = true;
    const exePath = findInvestBinaries(isDevMode);
    expect(exePath).toBe(path.join('build', 'invest', filename));
  });
  it('should point to resourcesPath in production', async () => {
    const isDevMode = false;
    const exePath = findInvestBinaries(isDevMode);
    expect(exePath)
      .toBe(path.join(process.resourcesPath, 'invest', filename));
  });
  it('should throw if the invest exe is bad', async () => {
    execFileSync.mockImplementation(() => {
      throw new Error('error from invest --version');
    });
    const isDevMode = false;
    expect(() => findInvestBinaries(isDevMode)).toThrow();
  });
});

describe('extractZipInplace', () => {
  const root = path.join('tests', 'data');
  const zipPath = path.join(root, 'output.zip');
  let level1Dir;
  let level2Dir;
  let file1Path;
  let file2Path;
  let doneZipping = false;

  beforeEach(() => {
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
      if (level1Dir.startsWith(path.join('tests', 'data', 'level1'))) {
        rimraf(level1Dir, (error) => { if (error) { throw error; } });
      }
      doneZipping = true;
    });
    zipfile.end();
  });

  afterEach(() => {
    fs.unlinkSync(zipPath);
    // being extra careful with recursive rm
    if (level1Dir.startsWith(path.join('tests', 'data', 'level1'))) {
      rimraf(level1Dir, (error) => { if (error) { throw error; } });
    }
  });

  it('should extract recursively', async () => {
    await waitFor(() => {
      expect(doneZipping).toBe(true);
      // The expected state after the setup, before extraction
      expect(fs.existsSync(zipPath)).toBe(true);
      expect(fs.existsSync(file1Path)).toBe(false);
      expect(fs.existsSync(file2Path)).toBe(false);
    });

    expect(await extractZipInplace(zipPath)).toBe(true);

    // And the expected state after extraction
    await waitFor(() => {
      expect(fs.existsSync(file1Path)).toBe(true);
      expect(fs.existsSync(file2Path)).toBe(true);
    });
  });
});

describe('createWindow', () => {
  beforeEach(async () => {
    jest.clearAllMocks();
  });
  afterEach(() => {
    jest.clearAllMocks();
  });
  it('should register various ipcMain listeners', async () => {
    const expectedHandleChannels = [
      ipcMainChannels.SHOW_OPEN_DIALOG,
      ipcMainChannels.SHOW_SAVE_DIALOG,
      ipcMainChannels.IS_FIRST_RUN,
      ipcMainChannels.SET_LANGUAGE,
    ];
    const expectedOnChannels = [
      ipcMainChannels.DOWNLOAD_URL,
      ipcMainChannels.INVEST_RUN,
      ipcMainChannels.INVEST_KILL,
      ipcMainChannels.INVEST_READ_LOG,
      ipcMainChannels.SHOW_CONTEXT_MENU,
      ipcMainChannels.GETTEXT,
    ];
    createWindow();
    await waitFor(() => {
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
    });
    removeIpcMainListeners();
    await waitFor(() => {
      expect(ipcMain.eventNames()).toEqual([]);
    });
  });
});

describe('Integration tests for Download Sample Data Modal', () => {
  beforeAll(async () => {
    await createWindow();
  });
  beforeEach(async () => {
    getInvestModelNames.mockResolvedValue({});
  });
  afterAll(() => {
    destroyWindow();
  });
  afterEach(async () => {
    await clearSettingsStore();
    jest.resetAllMocks();
  });

  test('Modal does not display when app has been run before', async () => {
    const { queryByText } = render(<App />);
    const modalTitle = await queryByText('Download InVEST sample data');
    expect(modalTitle).toBeNull();
  });

  test('Modal displays immediately on user`s first run', async () => {
    const {
      findByText,
      getByText,
    } = render(<App isFirstRun />);

    const modalTitle = await findByText('Download InVEST sample data');
    expect(modalTitle).toBeInTheDocument();
    fireEvent.click(getByText('Cancel'));
    await waitFor(() => {
      expect(modalTitle).not.toBeInTheDocument();
    });
  });

  test('Download starts, updates progress, & stores location', async () => {
    const dialogData = {
      filePaths: ['foo/directory'],
    };
    ipcRenderer.invoke.mockResolvedValue(dialogData);

    const {
      findByRole,
      findAllByRole,
    } = render(<App isFirstRun />);

    const allCheckBoxes = await findAllByRole('checkbox');
    const downloadButton = await findByRole('button', { name: 'Download' });
    fireEvent.click(downloadButton);
    const nURLs = allCheckBoxes.length - 1; // all except Select All
    await waitFor(async () => {
      expect(await getSettingsValue('sampleDataDir'))
        .toBe(dialogData.filePaths[0]);
    });
    const progressBar = await findByRole('progressbar');
    expect(progressBar).toHaveTextContent(`Downloading 1 of ${nURLs}`);
    // We don't have mocks that take us all the way through to a complete
    // download, when the progress bar would become a 'Download Complete' alert
  });

  test('Cancel does not store a sampleDataDir value', async () => {
    const spy = jest.spyOn(ipcRenderer, 'send');

    const { findByRole } = render(<App isFirstRun />);

    const existingValue = await getSettingsValue('sampleDataDir');
    const cancelButton = await findByRole('button', { name: 'Cancel' });
    fireEvent.click(cancelButton);

    await waitFor(() => {
      expect(spy).toHaveBeenCalledTimes(0);
    });
    await waitFor(async () => {
      const value = await getSettingsValue('sampleDataDir');
      expect(value).toBe(existingValue);
    });
  });
});

describe('Translation', () => {
  const i18n = new gettext_js();
  const testLanguage = 'es';
  const messageCatalog = {
    '': {
      'language': testLanguage,
      'plural-forms': 'nplurals=2; plural=(n!=1);'
    },
    'Open': 'σρєи',
    'Language': 'ℓαиgυαgє'
  };

  beforeAll(async () => {
    getInvestModelNames.mockResolvedValue([]);

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
        return i18n.gettext(arg)
      }
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
      queryByText,
      findByTitle,
      findByLabelText,
    } = render(<App />);

    fireEvent.click(await findByTitle('settings'));
    let languageInput = await findByLabelText('Language', { exact: false });
    expect(languageInput).toHaveValue('en');

    fireEvent.change(languageInput, { target: { value: testLanguage } });

    // text within the settings modal component should be translated
    languageInput = await findByLabelText(messageCatalog['Language'], { exact: false });
    expect(languageInput).toHaveValue(testLanguage);

    // text should also be translated in other components
    // such as the Open button (visible in background)
    const openButton = await findByText(messageCatalog['Open']);

    // text without a translation in the message catalog should display in the default English
    expect(getByText('Logging threshold')).toBeDefined();

    // resetting language should re-render components in English
    fireEvent.click(getByText('Reset'));
    expect(await findByText('Language')).toBeDefined();
    expect(await findByText('Open')).toBeDefined();
  });
});
