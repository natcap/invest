import fs from 'fs';
import path from 'path';
import events from 'events';
import { spawn, exec } from 'child_process';
import Stream from 'stream';

import GettextJS from 'gettext.js';
import React from 'react';
import { ipcRenderer } from 'electron';
import {
  render, waitFor, within
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

import App from '../../src/renderer/app';
import {
  getInvestModelNames,
  getSpec,
  fetchValidation,
  fetchDatastackFromFile
} from '../../src/renderer/server_requests';
import InvestJob from '../../src/renderer/InvestJob';
import {
  getSettingsValue, saveSettingsStore
} from '../../src/renderer/components/SettingsModal/SettingsStorage';
import { ipcMainChannels } from '../../src/main/ipcMainChannels';
import {
  setupInvestRunHandlers,
  setupInvestLogReaderHandler,
} from '../../src/main/setupInvestHandlers';
import writeInvestParameters from '../../src/main/writeInvestParameters';
import { removeIpcMainListeners } from '../../src/main/main';

// It's quite a pain to dynamically mock a const from a module,
// here we do it by importing as another object, then
// we can overwrite the object we want to mock later
// https://stackoverflow.com/questions/42977961/how-to-mock-an-exported-const-in-jest
import * as uiConfig from '../../src/renderer/ui_config';

jest.mock('node-fetch');
jest.mock('child_process');
jest.mock('../../src/renderer/server_requests');
jest.mock('../../src/main/writeInvestParameters');

const MOCK_MODEL_TITLE = 'Carbon';
const MOCK_MODEL_RUN_NAME = 'carbon';
const MOCK_INVEST_LIST = {
  [MOCK_MODEL_TITLE]: {
    model_name: MOCK_MODEL_RUN_NAME,
  },
};
const MOCK_VALIDATION_VALUE = [[['workspace_dir'], 'invalid because']];

const SAMPLE_SPEC = {
  model_name: MOCK_MODEL_TITLE,
  pyname: 'natcap.invest.carbon',
  userguide: 'carbonstorage.html',
  args: {
    workspace_dir: {
      name: 'Workspace',
      about: 'help text',
      type: 'directory',
    },
    carbon_pools_path: {
      name: 'Carbon Pools',
      about: 'help text',
      type: 'csv',
    },
  },
};

function mockUISpec(spec, modelName) {
  return {
    [modelName]: { order: [Object.keys(spec.args)] },
  };
}

// Because we mock UI_SPEC without using jest's API
// we also need to reset it without jest's API.
const { UI_SPEC } = uiConfig;
afterEach(() => {
  uiConfig.UI_SPEC = UI_SPEC;
});

describe('Various ways to open and close InVEST models', () => {
  beforeEach(async () => {
    getInvestModelNames.mockResolvedValue(MOCK_INVEST_LIST);
    getSpec.mockResolvedValue(SAMPLE_SPEC);
    fetchValidation.mockResolvedValue(MOCK_VALIDATION_VALUE);
    uiConfig.UI_SPEC = mockUISpec(SAMPLE_SPEC, MOCK_MODEL_RUN_NAME);
  });

  afterEach(async () => {
    await InvestJob.clearStore(); // because a test calls InvestJob.saveJob()
  });

  test('Clicking an invest model button renders SetupTab', async () => {
    const { findByText, findByRole } = render(
      <App />
    );

    const carbon = await findByRole(
      'button', { name: MOCK_MODEL_TITLE }
    );
    userEvent.click(carbon);
    const executeButton = await findByRole('button', { name: /Run/ });
    expect(executeButton).toBeDisabled();
    const setupTab = await findByText('Setup');
    expect(setupTab.classList.contains('active')).toBeTruthy();
    expect(getSpec).toHaveBeenCalledTimes(1);
  });

  test('Clicking a recent job renders SetupTab', async () => {
    const workspacePath = 'my_workspace';
    const argsValues = {
      workspace_dir: workspacePath,
    };
    const mockJob = new InvestJob({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Sequestration',
      argsValues: argsValues,
      status: 'success',
    });
    await InvestJob.saveJob(mockJob);

    const { findByText, findByLabelText, findByRole } = render(
      <App />
    );

    const recentJobCard = await findByText(
      argsValues.workspace_dir
    );
    userEvent.click(recentJobCard);
    const executeButton = await findByRole('button', { name: /Run/ });
    expect(executeButton).toBeDisabled();
    const setupTab = await findByText('Setup');
    expect(setupTab.classList.contains('active')).toBeTruthy();

    // Expect some arg values that were loaded from the saved job:
    const input = await findByLabelText(SAMPLE_SPEC.args.workspace_dir.name);
    expect(input).toHaveValue(
      argsValues.workspace_dir
    );
  });

  test('Open File: Dialog callback renders SetupTab', async () => {
    const mockDialogData = {
      filePaths: ['foo.json'],
    };
    const mockDatastack = {
      args: {
        carbon_pools_path: 'Carbon/carbon_pools_willamette.csv',
      },
      module_name: 'natcap.invest.carbon',
      model_run_name: 'carbon',
      model_human_name: 'Carbon',
    };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);
    fetchDatastackFromFile.mockResolvedValue(mockDatastack);

    const { findByText, findByLabelText, findByRole } = render(
      <App />
    );

    const openButton = await findByRole('button', { name: 'Open' });
    expect(openButton).not.toBeDisabled();
    userEvent.click(openButton);
    const executeButton = await findByRole('button', { name: /Run/ });
    expect(executeButton).toBeDisabled();
    const setupTab = await findByText('Setup');
    const input = await findByLabelText(
      SAMPLE_SPEC.args.carbon_pools_path.name
    );
    expect(setupTab.classList.contains('active')).toBeTruthy();
    expect(input).toHaveValue(mockDatastack.args.carbon_pools_path);
  });

  test('Open File: Dialog callback is canceled', async () => {
    // Resembles callback data if the dialog was canceled
    const mockDialogData = {
      filePaths: [],
    };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);

    const { findByRole } = render(
      <App />
    );

    const openButton = await findByRole('button', { name: 'Open' });
    userEvent.click(openButton);
    const homeTab = await findByRole('tabpanel', { name: 'home tab' });
    // expect we're on the same tab we started on instead of switching to Setup
    expect(homeTab.classList.contains('active')).toBeTruthy();
    // These are the calls that would have triggered if a file was selected
    expect(fetchDatastackFromFile).toHaveBeenCalledTimes(0);
    expect(getSpec).toHaveBeenCalledTimes(0);
  });

  test('Open three tabs and close them', async () => {
    const {
      findByRole,
      findAllByRole,
      queryAllByRole,
    } = render(<App />);

    const carbon = await findByRole(
      'button', { name: MOCK_MODEL_TITLE }
    );
    const homeTab = await findByRole('tabpanel', { name: 'home tab' });

    // Open a model tab and expect that it's active
    userEvent.click(carbon);
    let modelTabs = await findAllByRole('tab', { name: MOCK_MODEL_TITLE });
    expect(modelTabs).toHaveLength(1); // one carbon tab open
    const tab1 = modelTabs[0];
    const tab1EventKey = tab1.getAttribute('data-rb-event-key');
    expect(tab1.classList.contains('active')).toBeTruthy();
    expect(homeTab.classList.contains('active')).toBeFalsy();

    // Open a second model tab and expect that it's active
    userEvent.click(homeTab);
    userEvent.click(carbon);
    modelTabs = await findAllByRole('tab', { name: MOCK_MODEL_TITLE });
    expect(modelTabs).toHaveLength(2); // 2 carbon tabs open
    const tab2 = modelTabs[1];
    const tab2EventKey = tab2.getAttribute('data-rb-event-key');
    expect(tab2.classList.contains('active')).toBeTruthy();
    expect(tab1.classList.contains('active')).toBeFalsy();
    expect(homeTab.classList.contains('active')).toBeFalsy();
    // make sure that we switched away from the first tab
    expect(tab2EventKey).not.toEqual(tab1EventKey);

    // Open a third model tab and expect that it's active
    userEvent.click(homeTab);
    userEvent.click(carbon);
    modelTabs = await findAllByRole('tab', { name: MOCK_MODEL_TITLE });
    expect(modelTabs).toHaveLength(3); // 3 carbon tabs open
    const tab3 = modelTabs[2];
    const tab3EventKey = tab3.getAttribute('data-rb-event-key');
    expect(tab3.classList.contains('active')).toBeTruthy();
    expect(tab2.classList.contains('active')).toBeFalsy();
    expect(tab1.classList.contains('active')).toBeFalsy();
    expect(homeTab.classList.contains('active')).toBeFalsy();
    // make sure that we switched away from the first model tabs
    expect(tab3EventKey).not.toEqual(tab2EventKey);
    expect(tab3EventKey).not.toEqual(tab1EventKey);

    // Click the close button on the middle tab
    const tab2CloseButton = await within(tab2.closest('.nav-item'))
      .getByRole('button', { name: new RegExp(`close ${MOCK_MODEL_TITLE}`) });
    userEvent.click(tab2CloseButton);
    // Now there should only be 2 model tabs open
    modelTabs = await findAllByRole('tab', { name: MOCK_MODEL_TITLE });
    expect(modelTabs).toHaveLength(2);
    // Should have switched to tab3, the next tab to the right
    expect(tab3.classList.contains('active')).toBeTruthy();
    expect(tab1.classList.contains('active')).toBeFalsy();

    // Click the close button on the right tab
    const tab3CloseButton = await within(tab3.closest('.nav-item'))
      .getByRole('button', { name: new RegExp(`close ${MOCK_MODEL_TITLE}`) });
    userEvent.click(tab3CloseButton);
    // Now there should only be 1 model tab open
    modelTabs = await findAllByRole('tab', { name: MOCK_MODEL_TITLE });
    expect(modelTabs).toHaveLength(1);
    // No model tabs to the right, so it should switch to the next tab to the left.
    expect(tab1.classList.contains('active')).toBeTruthy();
    expect(homeTab.classList.contains('active')).toBeFalsy();

    // Click the close button on the last tab
    const tab1CloseButton = await within(tab1.closest('.nav-item'))
      .getByRole('button', { name: new RegExp(`close ${MOCK_MODEL_TITLE}`) });
    userEvent.click(tab1CloseButton);
    // Now there should be no model tabs open.
    modelTabs = await queryAllByRole('tab', { name: MOCK_MODEL_TITLE });
    expect(modelTabs).toHaveLength(0);
    // No more model tabs, so it should switch back to the home tab.
    expect(homeTab.classList.contains('active')).toBeTruthy();
  });
});

describe('Display recently executed InVEST jobs on Home tab', () => {
  beforeEach(() => {
    getInvestModelNames.mockResolvedValue({});
  });

  afterEach(async () => {
    await InvestJob.clearStore();
  });

  test('Recent Jobs: each has a button', async () => {
    const job1 = new InvestJob({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Sequestration',
      argsValues: {
        workspace_dir: 'work1',
      },
      status: 'success',
    });
    await InvestJob.saveJob(job1);
    const job2 = new InvestJob({
      modelRunName: 'sdr',
      modelHumanName: 'Sediment Ratio Delivery',
      argsValues: {
        workspace_dir: 'work2',
        results_suffix: 'suffix',
      },
      status: 'error',
      finalTraceback: 'ValueError ...',
    });
    const recentJobs = await InvestJob.saveJob(job2);
    const initialJobs = [job1, job2];

    const { getByText } = render(<App />);

    await waitFor(() => {
      initialJobs.forEach((job, idx) => {
        const recent = recentJobs[idx];
        const card = getByText(job.modelHumanName)
          .closest('button');
        expect(within(card).getByText(job.argsValues.workspace_dir))
          .toBeInTheDocument();
        if (job.status === 'success') {
          expect(getByText('\u{2705}'))
            .toBeInTheDocument();
        }
        if (job.status === 'error' && job.finalTraceback) {
          expect(getByText(job.finalTraceback))
            .toBeInTheDocument();
        }
        if (job.argsValues.results_suffix) {
          expect(getByText(job.argsValues.results_suffix))
            .toBeInTheDocument();
        }
        // The timestamp is not part of the initial object, but should
        // in the saved object
        expect(within(card).getByText(recent.humanTime))
          .toBeInTheDocument();
      });
    });
  });

  test('Recent Jobs: placeholder if there are no recent jobs', async () => {
    const { findByText } = render(
      <App />
    );

    const node = await findByText(/Set up a model from a sample datastack file/);
    expect(node).toBeInTheDocument();
  });

  test('Recent Jobs: cleared by button', async () => {
    const job1 = new InvestJob({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Sequestration',
      argsValues: {
        workspace_dir: 'work1',
      },
      status: 'success',
    });
    const recentJobs = await InvestJob.saveJob(job1);

    const { getByText, findByText, getByRole } = render(<App />);

    await waitFor(() => {
      recentJobs.forEach((job) => {
        expect(getByText(job.argsValues.workspace_dir))
          .toBeTruthy();
      });
    });
    userEvent.click(getByRole('button', { name: 'settings' }));
    userEvent.click(getByText('Clear Recent Jobs'));
    const node = await findByText(/Set up a model from a sample datastack file/);
    expect(node).toBeInTheDocument();
  });
});

describe('InVEST global settings: dialog interactions', () => {
  const nWorkersLabelText = 'Taskgraph n_workers parameter';
  const loggingLabelText = 'Logging threshold';
  const tgLoggingLabelText = 'Taskgraph logging threshold';
  const languageLabelText = 'Language';

  beforeEach(async () => {
    getInvestModelNames.mockResolvedValue({});
    ipcRenderer.invoke.mockImplementation((channel) => {
      if (channel === ipcMainChannels.IS_DEV_MODE) {
        return Promise.resolve(true); // mock dev mode so that language dropdown is rendered
      }
      return Promise.resolve();
    });
  });

  test('Invest settings save on change', async () => {
    const nWorkersLabel = 'Threaded task management (0)';
    const nWorkersValue = '0';
    const loggingLevel = 'DEBUG';
    const tgLoggingLevel = 'DEBUG';
    const languageValue = 'es';

    const {
      getByText, getByRole, getByLabelText, findByRole,
    } = render(
      <App />
    );

    userEvent.click(await findByRole('button', { name: 'settings' }));
    const nWorkersInput = getByLabelText(nWorkersLabelText, { exact: false });
    const loggingInput = getByLabelText(loggingLabelText);
    const tgLoggingInput = getByLabelText(tgLoggingLabelText);
    const languageInput = getByLabelText(languageLabelText, { exact: false });

    userEvent.selectOptions(nWorkersInput, [getByText(nWorkersLabel)]);
    await waitFor(() => { expect(nWorkersInput).toHaveValue(nWorkersValue); });
    userEvent.selectOptions(loggingInput, [loggingLevel]);
    await waitFor(() => { expect(loggingInput).toHaveValue(loggingLevel); });
    userEvent.selectOptions(tgLoggingInput, [tgLoggingLevel]);
    await waitFor(() => { expect(tgLoggingInput).toHaveValue(tgLoggingLevel); });
    userEvent.selectOptions(languageInput, [languageValue]);
    await waitFor(() => { expect(languageInput).toHaveValue(languageValue); });
    userEvent.click(getByRole('button', { name: 'close settings' }));

    // Check values were saved in app and in store
    userEvent.click(await findByRole('button', { name: 'settings' }));
    await waitFor(() => {
      expect(nWorkersInput).toHaveValue(nWorkersValue);
      expect(loggingInput).toHaveValue(loggingLevel);
      expect(tgLoggingInput).toHaveValue(tgLoggingLevel);
      expect(languageInput).toHaveValue(languageValue);
    });
    expect(await getSettingsValue('nWorkers')).toBe(nWorkersValue);
    expect(await getSettingsValue('loggingLevel')).toBe(loggingLevel);
    expect(await getSettingsValue('taskgraphLoggingLevel')).toBe(tgLoggingLevel);
    expect(await getSettingsValue('language')).toBe(languageValue);
  });

  test('Load invest settings from storage and test Reset', async () => {
    const defaultSettings = {
      nWorkers: '-1',
      loggingLevel: 'INFO',
      taskgraphLoggingLevel: 'ERROR',
      language: 'en',
    };
    const expectedSettings = {
      nWorkers: '0',
      loggingLevel: 'ERROR',
      taskgraphLoggingLevel: 'INFO',
      language: 'es',
    };

    await saveSettingsStore(expectedSettings);

    const {
      getByText, getByLabelText, findByRole,
    } = render(
      <App />
    );

    userEvent.click(await findByRole('button', { name: 'settings' }));
    const nWorkersInput = getByLabelText(nWorkersLabelText, { exact: false });
    const loggingInput = getByLabelText(loggingLabelText);
    const tgLoggingInput = getByLabelText(tgLoggingLabelText);
    const languageInput = getByLabelText(languageLabelText, { exact: false });

    // Test that the invest settings were loaded in from store.
    await waitFor(() => {
      expect(nWorkersInput).toHaveValue(expectedSettings.nWorkers);
      expect(loggingInput).toHaveValue(expectedSettings.loggingLevel);
      expect(tgLoggingInput).toHaveValue(expectedSettings.tgLoggingLevel);
      expect(languageInput).toHaveValue(expectedSettings.language);
    });

    // Test Reset sets values to default
    userEvent.click(getByText('Reset to Defaults'));
    await waitFor(() => {
      expect(nWorkersInput).toHaveValue(defaultSettings.nWorkers);
      expect(loggingInput).toHaveValue(defaultSettings.loggingLevel);
      expect(tgLoggingInput).toHaveValue(defaultSettings.tgLoggingLevel);
      expect(languageInput).toHaveValue(defaultSettings.language);
    });
  });

  test('Access sampledata download Modal from settings', async () => {
    const {
      findByText, findByRole, queryByText,
    } = render(
      <App />
    );

    const settingsBtn = await findByRole('button', { name: 'settings' });
    userEvent.click(settingsBtn);
    userEvent.click(
      await findByRole('button', { name: 'Download Sample Data' })
    );

    expect(await findByText('Download InVEST sample data'))
      .toBeInTheDocument();
    expect(queryByText('Settings')).toBeNull();
  });
});

describe('InVEST subprocess testing', () => {
  const spec = {
    args: {
      workspace_dir: {
        name: 'Workspace',
        type: 'directory',
      },
      results_suffix: {
        name: 'Suffix',
        type: 'freestyle_string',
      },
    },
    model_name: 'EcoModel',
    pyname: 'natcap.invest.dot',
    userguide: 'foo.html',
  };
  const modelName = 'carbon';
  // nothing is written to the fake workspace in these tests,
  // and we mock validation, so this dir need not exist.
  const fakeWorkspace = 'foo_dir';
  const logfilePath = path.join(fakeWorkspace, 'invest-log.txt');
  // invest always emits this message, and the workbench always listens for it:
  const stdOutLogfileSignal = `Writing log messages to [${logfilePath}]`;
  const stdOutText = 'hello from invest';
  const investExe = 'foo';

  // We need to mock child_process.spawn to return a mock invest process,
  // And we also need an interface to that mocked process in each test
  // so we can push to stdout, stderr, etc, just like invest would.
  function getMockedInvestProcess() {
    const mockInvestProc = new events.EventEmitter();
    mockInvestProc.pid = -9999999; // not a plausible pid
    mockInvestProc.stdout = new Stream.Readable({
      read: () => {},
    });
    mockInvestProc.stderr = new Stream.Readable({
      read: () => {},
    });
    if (process.platform !== 'win32') {
      jest.spyOn(process, 'kill')
        .mockImplementation(() => {
          mockInvestProc.emit('exit', null);
        });
    } else {
      exec.mockImplementation(() => {
        mockInvestProc.emit('exit', null);
      });
    }
    spawn.mockImplementation(() => mockInvestProc);
    return mockInvestProc;
  }

  beforeAll(() => {
    setupInvestRunHandlers(investExe);
    setupInvestLogReaderHandler();
  });

  afterAll(() => {
    removeIpcMainListeners();
  });

  beforeEach(() => {
    getSpec.mockResolvedValue(spec);
    fetchValidation.mockResolvedValue([]);
    getInvestModelNames.mockResolvedValue(
      { Carbon: { model_name: modelName } }
    );
    uiConfig.UI_SPEC = mockUISpec(spec, modelName);
    // mock the request to write the datastack file. Actually write it
    // because the app will clean it up when invest exits.
    writeInvestParameters.mockImplementation((payload) => {
      fs.writeFileSync(payload.filepath, 'foo');
      return Promise.resolve({ text: () => 'foo' });
    });
  });

  afterEach(async () => {
    await InvestJob.clearStore();
  });

  test('exit without error - expect log display', async () => {
    const mockInvestProc = getMockedInvestProcess();
    const {
      findByText,
      findByLabelText,
      findByRole,
      getByRole,
      queryByText,
    } = render(<App />);

    const carbon = await findByRole(
      'button', { name: MOCK_MODEL_TITLE }
    );
    userEvent.click(carbon);
    const workspaceInput = await findByLabelText(
      `${spec.args.workspace_dir.name}`
    );
    userEvent.type(workspaceInput, fakeWorkspace);
    const execute = await findByRole('button', { name: /Run/ });
    userEvent.click(execute);
    await waitFor(() => expect(execute).toBeDisabled());

    // logfile signal on stdout listener is how app knows the process started
    mockInvestProc.stdout.push(stdOutText);
    mockInvestProc.stdout.push(stdOutLogfileSignal);
    const logTab = await findByText('Log');
    await waitFor(() => {
      expect(logTab.classList.contains('active')).toBeTruthy();
    });
    expect(await findByText(stdOutText, { exact: false }))
      .toBeInTheDocument();
    expect(queryByText('Model Complete')).toBeNull();
    expect(queryByText('Open Workspace')).toBeNull();

    mockInvestProc.emit('exit', 0); // 0 - exit w/o error
    expect(await findByText('Model Complete')).toBeInTheDocument();
    expect(await findByText('Open Workspace')).toBeEnabled();
    expect(execute).toBeEnabled();

    // A recent job card should be rendered
    await getByRole('button', { name: 'InVEST' }).click();
    const homeTab = await getByRole('tabpanel', { name: 'home tab' });
    const cardText = await within(homeTab)
      .findByText(fakeWorkspace);
    expect(cardText).toBeInTheDocument();
  });

  test('exit with error - expect log & alert display', async () => {
    const mockInvestProc = getMockedInvestProcess();
    const {
      findByText,
      findByLabelText,
      findByRole,
      getByRole,
    } = render(<App />);

    const carbon = await findByRole(
      'button', { name: MOCK_MODEL_TITLE }
    );
    userEvent.click(carbon);
    const workspaceInput = await findByLabelText(
      `${spec.args.workspace_dir.name}`
    );
    userEvent.type(workspaceInput, fakeWorkspace);

    const execute = await findByRole('button', { name: /Run/ });
    userEvent.click(execute);

    // To test that we can parse the finalTraceback even after extra data
    const someStdErr = 'something went wrong';
    const finalTraceback = 'ValueError';
    const pyInstallerErr = "[12345] Failed to execute script 'cli' due to unhandled exception!";
    const allStdErr = `${someStdErr}\n${finalTraceback}\n${pyInstallerErr}\n`;

    mockInvestProc.stdout.push(stdOutText);
    mockInvestProc.stdout.push(stdOutLogfileSignal);
    mockInvestProc.stderr.push(allStdErr);
    const logTab = await findByText('Log');
    await waitFor(() => {
      expect(logTab.classList.contains('active')).toBeTruthy();
    });

    expect(await findByText(stdOutText, { exact: false }))
      .toBeInTheDocument();

    mockInvestProc.emit('exit', 1); // 1 - exit w/ error

    // Only finalTraceback text should be rendered in a red alert
    const alert = await findByRole('alert');
    await waitFor(() => {
      expect(alert).toHaveTextContent(new RegExp(`^${finalTraceback}`));
      expect(alert).not.toHaveTextContent(someStdErr);
      expect(alert).toHaveClass('alert-danger');
    });
    expect(await findByRole('button', { name: 'Open Workspace' }))
      .toBeEnabled();

    // A recent job card should be rendered
    await getByRole('button', { name: 'InVEST' }).click();
    const homeTab = await getByRole('tabpanel', { name: 'home tab' });
    const cardText = await within(homeTab)
      .findByText(fakeWorkspace);
    expect(cardText).toBeInTheDocument();
  });

  test('user terminates process - expect log & alert display', async () => {
    const mockInvestProc = getMockedInvestProcess();
    const {
      findByText,
      findByLabelText,
      findByRole,
      getByRole,
      queryByText,
    } = render(<App />);

    const carbon = await findByRole(
      'button', { name: MOCK_MODEL_TITLE }
    );
    userEvent.click(carbon);
    const workspaceInput = await findByLabelText(
      `${spec.args.workspace_dir.name}`
    );
    userEvent.type(workspaceInput, fakeWorkspace);

    const execute = await findByRole('button', { name: /Run/ });
    userEvent.click(execute);

    // stdout listener is how the app knows the process started
    // Canel button only appears after this signal.
    mockInvestProc.stdout.push(stdOutText);
    expect(queryByText('Cancel Run')).toBeNull();
    mockInvestProc.stdout.push(stdOutLogfileSignal);
    const logTab = await findByText('Log');
    await waitFor(() => {
      expect(logTab.classList.contains('active')).toBeTruthy();
    });
    expect(await findByText(stdOutText, { exact: false }))
      .toBeInTheDocument();

    const cancelButton = await findByText('Cancel Run');
    userEvent.click(cancelButton);
    expect(await findByText('Open Workspace'))
      .toBeEnabled();
    expect(await findByRole('alert'))
      .toHaveTextContent('Run Canceled');

    // A recent job card should be rendered
    await getByRole('button', { name: 'InVEST' }).click();
    const homeTab = await getByRole('tabpanel', { name: 'home tab' });
    const cardText = await within(homeTab)
      .findByText(fakeWorkspace);
    expect(cardText).toBeInTheDocument();
  });

  test('Run & re-run a job - expect new log display', async () => {
    const mockInvestProc = getMockedInvestProcess();
    const {
      findByText,
      findByLabelText,
      findByRole,
    } = render(<App />);

    const carbon = await findByRole(
      'button', { name: MOCK_MODEL_TITLE }
    );
    userEvent.click(carbon);
    const workspaceInput = await findByLabelText(
      `${spec.args.workspace_dir.name}`
    );
    userEvent.type(workspaceInput, fakeWorkspace);

    const execute = await findByRole('button', { name: /Run/ });
    userEvent.click(execute);

    // stdout listener is how the app knows the process started
    mockInvestProc.stdout.push(stdOutText);
    mockInvestProc.stdout.push(stdOutLogfileSignal);
    let logTab = await findByText('Log');
    await waitFor(() => {
      expect(logTab.classList.contains('active')).toBeTruthy();
    });
    expect(await findByText(stdOutText, { exact: false }))
      .toBeInTheDocument();

    const cancelButton = await findByText('Cancel Run');
    userEvent.click(cancelButton);
    expect(await findByText('Open Workspace'))
      .toBeEnabled();

    // Now the second invest process:
    const anotherInvestProc = getMockedInvestProcess();
    // Now click away from Log, re-run, and expect the switch
    // back to the new log
    const setupTab = await findByText('Setup');
    userEvent.click(setupTab);
    userEvent.click(execute);
    const newStdOutText = 'this is new stdout text';
    anotherInvestProc.stdout.push(newStdOutText);
    anotherInvestProc.stdout.push(stdOutLogfileSignal);
    logTab = await findByText('Log');
    await waitFor(() => {
      expect(logTab.classList.contains('active')).toBeTruthy();
    });
    expect(await findByText(newStdOutText, { exact: false }))
      .toBeInTheDocument();
    anotherInvestProc.emit('exit', 0);
    // Give it time to run the listener before unmounting.
    await new Promise((resolve) => setTimeout(resolve, 300));
  });

  test('Load Recent run & re-run it - expect new log display', async () => {
    const mockInvestProc = getMockedInvestProcess();
    const argsValues = {
      workspace_dir: fakeWorkspace,
    };
    const mockJob = new InvestJob({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Sequestration',
      argsValues: argsValues,
      status: 'success',
      logfile: logfilePath,
    });
    await InvestJob.saveJob(mockJob);

    const {
      findByText,
      findByRole,
      queryByText,
    } = render(<App />);

    const recentJobCard = await findByText(
      argsValues.workspace_dir
    );
    userEvent.click(recentJobCard);
    userEvent.click(await findByText('Log'));
    // We don't need to have a real logfile in order to test that LogTab
    // is trying to read from a file instead of from stdout
    expect(await findByText(/Logfile is missing/)).toBeInTheDocument();

    // Now re-run from the same InvestTab component and expect
    // LogTab is displaying the new invest process stdout
    const setupTab = await findByText('Setup');
    userEvent.click(setupTab);
    const execute = await findByRole('button', { name: /Run/ });
    userEvent.click(execute);
    mockInvestProc.stdout.push(stdOutText);
    mockInvestProc.stdout.push(stdOutLogfileSignal);
    const logTab = await findByText('Log');
    await waitFor(() => {
      expect(logTab.classList.contains('active')).toBeTruthy();
    });
    expect(await findByText(stdOutText, { exact: false }))
      .toBeInTheDocument();
    expect(queryByText('Logfile is missing')).toBeNull();
    mockInvestProc.emit('exit', 0);
    // Give it time to run the listener before unmounting.
    await new Promise((resolve) => setTimeout(resolve, 300));
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

    // mock out the relevant IPC channels
    ipcRenderer.invoke.mockImplementation((channel, arg) => {
      if (channel === ipcMainChannels.SET_LANGUAGE) {
        i18n.setLocale(arg);
      }
      if (channel === ipcMainChannels.IS_DEV_MODE) {
        return Promise.resolve(true); // mock dev mode so that language dropdown is rendered
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

  test('Text rerenders in new language when language setting changes', async () => {
    const {
      findByText,
      getByText,
      findByLabelText,
    } = render(<App />);

    userEvent.click(await findByLabelText('settings'));
    let languageInput = await findByLabelText('Language', { exact: false });
    expect(languageInput).toHaveValue('en');

    userEvent.selectOptions(languageInput, testLanguage);

    // text within the settings modal component should be translated
    languageInput = await findByLabelText(messageCatalog.Language, { exact: false });
    expect(languageInput).toHaveValue(testLanguage);

    // text should also be translated in other components
    // such as the Open button (visible in background)
    await findByText(messageCatalog.Open);

    // text without a translation in the message catalog should display in the default English
    expect(getByText('Logging threshold')).toBeDefined();

    // resetting language should re-render components in English
    userEvent.click(getByText('Reset to Defaults'));
    expect(await findByText('Language')).toBeDefined();
    expect(await findByText('Open')).toBeDefined();
  });
});
