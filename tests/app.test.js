import path from 'path';
import fs from 'fs';
import events from 'events';
import os from 'os';
import { spawn } from 'child_process';
import Stream from 'stream';
import React from 'react';
import { remote } from 'electron';
import {
  fireEvent, render, waitFor, within
} from '@testing-library/react';
import '@testing-library/jest-dom';

import { fileRegistry } from '../src/constants';
import InvestTab from '../src/components/InvestTab';
import App from '../src/app';
import {
  getInvestModelNames, getSpec, fetchValidation, fetchDatastackFromFile
} from '../src/server_requests';
import InvestJob from '../src/InvestJob';
import SAMPLE_SPEC from './data/carbon_args_spec.json';

jest.mock('child_process');
jest.mock('../src/server_requests');

const MOCK_MODEL_LIST_KEY = 'Carbon';
const MOCK_MODEL_RUN_NAME = 'carbon';
const MOCK_INVEST_LIST = {
  [MOCK_MODEL_LIST_KEY]: {
    internal_name: MOCK_MODEL_RUN_NAME
  }
};
const MOCK_VALIDATION_VALUE = [[['workspace_dir'], 'invalid because']];

afterAll(async () => {
  await InvestJob.clearStore();
  jest.resetAllMocks();
});

describe('Various ways to open and close InVEST models', () => {
  beforeAll(() => {
    getInvestModelNames.mockResolvedValue(MOCK_INVEST_LIST);
    getSpec.mockResolvedValue(SAMPLE_SPEC);
    fetchValidation.mockResolvedValue(MOCK_VALIDATION_VALUE);
  });
  afterEach(async () => {
    jest.clearAllMocks(); // clears usage data, does not reset/restore
    await InvestJob.clearStore(); // should call because a test calls job.save()
  });

  test('Clicking an invest model button renders SetupTab', async () => {
    const { findByText, findByRole } = render(
      <App investExe="foo" />
    );

    const carbon = await findByRole('button', { name: MOCK_MODEL_LIST_KEY });
    fireEvent.click(carbon);
    const executeButton = await findByRole('button', { name: /Run/ });
    expect(executeButton).toBeDisabled();
    const setupTab = await findByText('Setup');
    expect(setupTab.classList.contains('active')).toBeTruthy();
    expect(getSpec).toHaveBeenCalledTimes(1);
  });

  test('Clicking a recent job renders SetupTab', async () => {
    const workspacePath = 'my_workspace';
    const argsValues = {
      workspace_dir: workspacePath
    };
    const mockJob = new InvestJob({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Sequestration',
      argsValues: argsValues,
      status: 'success',
      humanTime: '3/5/2020, 10:43:14 AM',
    });
    await mockJob.save();

    const { findByText, findByLabelText, findByRole } = render(
      <App investExe="foo" />
    );

    const recentJobCard = await findByText(
      argsValues.workspace_dir
    );
    fireEvent.click(recentJobCard);
    const executeButton = await findByRole('button', { name: /Run/ });
    expect(executeButton).toBeDisabled();
    const setupTab = await findByText('Setup');
    expect(setupTab.classList.contains('active')).toBeTruthy();

    // Expect some arg values that were loaded from the saved job:
    const input = await findByLabelText(/Workspace/);
    expect(input).toHaveValue(
      argsValues.workspace_dir
    );
  });

  test('Open File: Dialog callback renders SetupTab', async () => {
    const mockDialogData = {
      filePaths: ['foo.json']
    };
    const mockDatastack = {
      args: {
        carbon_pools_path: 'Carbon/carbon_pools_willamette.csv',
      },
      module_name: 'natcap.invest.carbon',
      model_run_name: 'carbon',
      model_human_name: 'Carbon',
    };
    remote.dialog.showOpenDialog.mockResolvedValue(mockDialogData);
    fetchDatastackFromFile.mockResolvedValue(mockDatastack);

    const { findByText, findByLabelText, findByRole } = render(
      <App investExe="foo" />
    );

    const openButton = await findByText('Open');
    fireEvent.click(openButton);
    const executeButton = await findByRole('button', { name: /Run/ });
    expect(executeButton).toBeDisabled();
    const setupTab = await findByText('Setup');
    const input = await findByLabelText(/Carbon Pools/);
    expect(setupTab.classList.contains('active')).toBeTruthy();
    expect(input).toHaveValue(mockDatastack.args.carbon_pools_path);
  });

  test('Open File: Dialog callback is canceled', async () => {
    // Resembles callback data if the dialog was canceled
    const mockDialogData = {
      filePaths: []
    };
    remote.dialog.showOpenDialog.mockResolvedValue(mockDialogData);

    const { findByText, findByRole } = render(
      <App investExe="foo" />
    );

    const openButton = await findByText('Open');
    fireEvent.click(openButton);
    const homeTab = await findByRole('tabpanel', {name: /Home/});
    // expect we're on the same tab we started on instead of switching to Setup
    expect(homeTab.classList.contains('active')).toBeTruthy();
    // These are the calls that would have triggered if a file was selected
    expect(fetchDatastackFromFile).toHaveBeenCalledTimes(0);
    expect(getSpec).toHaveBeenCalledTimes(0);
  });

  test('Opening and closing multiple InVEST models', async () => {
    const {
      findByText,
      findByTitle,
      findByRole,
      findAllByText,
    } = render(<App investExe="foo" />);

    // Open first model
    const modelA = await findByRole('button', { name: MOCK_MODEL_LIST_KEY });
    fireEvent.click(modelA);
    const tabPanelA = await findByTitle(MOCK_MODEL_LIST_KEY);
    const setupTabA = await within(tabPanelA).findByText('Setup');
    expect(setupTabA.classList.contains('active')).toBeTruthy();
    expect(within(tabPanelA).queryByRole('button', { name: /Run/ }))
      .toBeInTheDocument();
    within(tabPanelA).queryAllByText(/Save to/).forEach((saveButton) => {
      expect(saveButton).toBeInTheDocument();
    });
    expect(getSpec).toHaveBeenCalledTimes(1);

    // Open another model (via Open button for convenience)
    const mockDialogData = {
      filePaths: ['foo.json']
    };
    const mockDatastack = {
      module_name: 'natcap.invest.party',
      model_run_name: 'party',
      model_human_name: 'Party Time',
      args: {
        carbon_pools_path: 'Carbon/carbon_pools_willamette.csv',
      }
    };
    remote.dialog.showOpenDialog.mockResolvedValue(mockDialogData);
    fetchDatastackFromFile.mockResolvedValue(mockDatastack);
    const openButton = await findByText('Open');
    fireEvent.click(openButton);
    const tabPanelB = await findByTitle(mockDatastack.model_human_name);
    const setupTabB = await within(tabPanelB).findByText('Setup');
    expect(setupTabB.classList.contains('active')).toBeTruthy();
    expect(within(tabPanelB).queryByRole('button', { name: /Run/ }))
      .toBeInTheDocument();
    within(tabPanelB).queryAllByText(/Save to/).forEach((saveButton) => {
      expect(saveButton).toBeInTheDocument();
    });
    expect(getSpec).toHaveBeenCalledTimes(2);

    // Close one open model
    const closeButtonArray = await findAllByText('x', { exact: true });
    fireEvent.click(closeButtonArray[1]);
    expect(setupTabB).not.toBeInTheDocument();
    expect(setupTabA.classList.contains('active')).toBeTruthy();

    // Close the other open model
    fireEvent.click(closeButtonArray[0]);
    expect(setupTabA).not.toBeInTheDocument();
    const homeTab = await findByRole('tabpanel', {name: /Home/});
    expect(homeTab.classList.contains('active')).toBeTruthy();
  });
});

describe('Display recently executed InVEST jobs', () => {
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
        workspace_dir: 'work1'
      },
      status: 'success',
      humanTime: '3/5/2020, 10:43:14 AM',
    });
    let recentJobs = await job1.save();
    const job2 = new InvestJob({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Sequestration',
      argsValues: {
        workspace_dir: 'work2'
      },
      status: 'success',
      humanTime: '3/5/2020, 10:43:14 AM',
    });
    recentJobs = await job2.save();

    const { getByText } = render(<App investExe="foo" />);

    await waitFor(() => {
      recentJobs.forEach((job) => {
        expect(getByText(job.argsValues.workspace_dir))
          .toBeTruthy();
      });
    });
  });

  test('Recent Jobs: placeholder if there are no recent jobs', async () => {
    const { findByText } = render(
      <App investExe="foo" />
    );

    const node = await findByText(/No recent InVEST runs/);
    expect(node).toBeInTheDocument();
  });

  test('Recent Jobs: cleared by button', async () => {
    const job1 = new InvestJob({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Sequestration',
      argsValues: {
        workspace_dir: 'work1'
      },
      status: 'success',
      humanTime: '3/5/2020, 10:43:14 AM',
    });
    const recentJobs = await job1.save();

    const { getByText, findByText, getByTitle } = render(<App investExe="foo" />);

    await waitFor(() => {
      recentJobs.forEach((job) => {
        expect(getByText(job.argsValues.workspace_dir))
          .toBeTruthy();
      });
    });
    fireEvent.click(getByTitle('settings'));
    fireEvent.click(getByText('Clear'));
    const node = await findByText(/No recent InVEST runs/);
    expect(node).toBeInTheDocument();
  });
});

describe('InVEST global settings: dialog interactions', () => {
  beforeEach(() => {
    getInvestModelNames.mockResolvedValue({});
  });
  afterEach(() => {
    jest.resetAllMocks();
  });
  test('Set the python logging level to pass to the invest CLI', async () => {
    const DEFAULT = 'INFO';
    const { getByText, getByLabelText, getByTitle } = render(
      <App investExe="foo" />
    );

    // Check the default settings
    fireEvent.click(getByTitle('settings'));
    await waitFor(() => {
      // waiting because the selected value depends on passed props
      expect(getByText(DEFAULT).selected).toBeTruthy();
    });

    // Change the select input and cancel -- expect default selected
    fireEvent.change(getByLabelText('Logging threshold'),
      { target: { value: 'DEBUG' } });
    fireEvent.click(getByText('Cancel'));
    // fireEvent.click(getByText('Settings'));  // why is this unecessary?
    expect(getByText(DEFAULT).selected).toBeTruthy();

    // Change the select input and save -- expect new value selected
    fireEvent.change(getByLabelText('Logging threshold'),
      { target: { value: 'DEBUG' } });
    fireEvent.click(getByText('Save Changes'));
    // fireEvent.click(getByText('Settings'));  // why is this unecessary?
    expect(getByText('DEBUG').selected).toBeTruthy();
  });

  test('Set the invest n_workers parameter', async () => {
    const defaultValue = '-1';
    const newValue = '2';
    const badValue = 'a';
    const labelText = 'Taskgraph n_workers parameter';

    const { getByText, getByLabelText, getByTitle } = render(
      <App investExe="foo" />
    );

    fireEvent.click(getByTitle('settings'));
    const input = getByLabelText(labelText, { exact: false });

    // Check the default settings
    await waitFor(() => {
      // waiting because the text value depends on passed props
      expect(input).toHaveValue(defaultValue);
    });

    // Change the value and cancel -- expect default value
    fireEvent.change(input, { target: { value: newValue } });
    fireEvent.click(getByText('Cancel'));
    expect(input).toHaveValue(defaultValue);

    // Change the value and save -- expect new value selected
    fireEvent.change(input, { target: { value: newValue } });
    expect(input).toHaveValue(newValue);
    // The real test: still newValue after saving and re-opening
    fireEvent.click(getByText('Save Changes'));
    fireEvent.click(getByTitle('settings'));
    await waitFor(() => { // the value to test is inherited through props
      expect(input).toHaveValue(newValue);
    });

    // Change to bad value -- expect invalid signal
    fireEvent.change(input, { target: { value: badValue } });
    expect(input.classList.contains('is-invalid')).toBeTruthy();
    expect(getByText('Save Changes')).toBeDisabled();
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
      }
    },
    model_name: 'EcoModel',
    module: 'natcap.invest.dot',
  };


  const dummyTextToLog = JSON.stringify(spec.args);
  let fakeWorkspace;
  let logfilePath;
  let mockInvestProc;

  beforeEach(() => {
    fakeWorkspace = fs.mkdtempSync(path.join('tests/data', 'data-'));
    // Need to reset these streams since mockInvestProc is shared by tests
    // and the streams apparently receive the EOF signal in each test.
    mockInvestProc = new events.EventEmitter();
    mockInvestProc.pid = -9999999; // a value that is not a plausible pid
    mockInvestProc.stdout = new Stream.Readable({
      read: () => {},
    });
    mockInvestProc.stderr = new Stream.Readable({
      read: () => {},
    });
    getSpec.mockResolvedValue(spec);
    fetchValidation.mockResolvedValue([]);
    getInvestModelNames.mockResolvedValue(
      { Carbon: { internal_name: 'carbon' } }
    );

    spawn.mockImplementation((exe, cmdArgs, options) => {
      // To simulate an invest model run, write a logfile to the workspace
      // with an expected filename pattern.
      const timestamp = new Date().toLocaleTimeString(
        'en-US', { hour12: false }
      ).replace(/:/g, '_');
      const logfileName = `InVEST-natcap.invest.model-log-9999-99-99--${timestamp}.txt`;
      logfilePath = path.join(fakeWorkspace, logfileName);
      // line-ending is critical; the log is read with `tail.on('line'...)`
      fs.writeFileSync(logfilePath, dummyTextToLog + os.EOL);
      return mockInvestProc;
    });

    // mock out the whole UI config module
    // brackets around spec.model_name turns it into a valid literal key
    const mockUISpec = {[spec.model_name]: {order: [Object.keys(spec.args)]}};
    jest.mock('../src/ui_config', () => mockUISpec);
  });

  afterEach(async () => {
    mockInvestProc = null;
    // being extra careful with recursive rm
    if (fakeWorkspace.startsWith('tests/data')) {
      fs.rmdirSync(fakeWorkspace, { recursive: true });
    }
    await InvestJob.clearStore();
    jest.resetAllMocks();
    jest.resetModules();
  });

  test('exit without error - expect log display', async () => {
    const {
      findByText,
      findByLabelText,
      findByRole,
      queryByText,
      unmount
    } = render(<App investExe="foo" />);

    const carbon = await findByRole('button', { name: MOCK_MODEL_LIST_KEY });
    fireEvent.click(carbon);
    const workspaceInput = await findByLabelText(
      RegExp(`${spec.args.workspace_dir.name}`)
    );
    fireEvent.change(workspaceInput, { target: { value: fakeWorkspace } });
    const execute = await findByRole('button', { name: /Run/ });
    fireEvent.click(execute);
    await waitFor(() => {
      expect(execute).toBeDisabled();
    });

    // stdout listener is how the app knows the process started
    mockInvestProc.stdout.push('hello from stdout');
    const logTab = await findByText('Log');
    expect(logTab.classList.contains('active')).toBeTruthy();
    // some text from the logfile should be rendered:
    expect(await findByText(dummyTextToLog, { exact: false }))
      .toBeInTheDocument();
    expect(queryByText('Model Complete')).toBeNull();
    expect(queryByText('Open Workspace')).toBeNull();
    // Job should already be saved to recent jobs database w/ status:
    const recentJobCards = await findByLabelText('Recent InVEST Runs:');
    expect(await within(recentJobCards).findByText('running'))
      .toBeInTheDocument();

    mockInvestProc.emit('exit', 0); // 0 - exit w/o error
    expect(await findByText('Model Complete')).toBeInTheDocument();
    expect(await findByText('Open Workspace')).toBeEnabled();
    expect(execute).toBeEnabled();

    // A recent job card should be rendered w/ updated status
    const cardText = await within(recentJobCards)
      .findByText(`${path.resolve(fakeWorkspace)}`);
    expect(cardText).toBeInTheDocument();
    expect(within(recentJobCards).queryByText('running'))
      .toBeNull();
    // Normally we don't explicitly unmount the rendered components,
    // but in this case we're 'watching' a file that the afterEach()
    // wants to remove. Unmounting triggers an 'unwatch' of the logfile
    // before afterEach cleanup, avoiding an error.
    unmount();
  });

  test('exit with error - expect log display', async () => {
    const {
      findByText,
      findByLabelText,
      findByRole,
      unmount,
    } = render(<App investExe="foo" />);

    const carbon = await findByRole('button', { name: MOCK_MODEL_LIST_KEY });
    fireEvent.click(carbon);
    const workspaceInput = await findByLabelText(
      RegExp(`${spec.args.workspace_dir.name}`)
    );
    fireEvent.change(workspaceInput, { target: { value: fakeWorkspace } });

    const execute = await findByRole('button', { name: /Run/ });
    fireEvent.click(execute);

    const errorMessage = 'fail';
    // Emit some stdout, some stderr, then pause and exit with error
    mockInvestProc.stdout.push('hello from stdout');
    mockInvestProc.stderr.push(errorMessage);
    const logTab = await findByText('Log');
    expect(logTab.classList.contains('active'))
      .toBeTruthy();

    // some text from the logfile should be rendered:
    expect(await findByText(dummyTextToLog, { exact: false }))
      .toBeInTheDocument();

    await new Promise((resolve) => setTimeout(resolve, 2000));
    mockInvestProc.emit('exit', 1); // 1 - exit w/ error

    // stderr text should be rendered in a red alert
    expect(await findByText(errorMessage))
      .toHaveClass('alert-danger');
    expect(await findByText('Open Workspace'))
      .toBeEnabled();

    // A recent job card should be rendered
    const cardText = await within(
      await findByLabelText('Recent InVEST Runs:')
    ).findByText(`${path.resolve(fakeWorkspace)}`);
    expect(cardText).toBeInTheDocument();
    unmount();
  });

  test('user terminates process - expect log display', async () => {
    const spy = jest.spyOn(InvestTab.prototype, 'terminateInvestProcess')
      .mockImplementation(() => {
        mockInvestProc.emit('exit', null);
      });

    const {
      findByText,
      findByLabelText,
      findByRole,
      unmount,
    } = render(<App investExe="foo" />);

    const carbon = await findByRole('button', { name: MOCK_MODEL_LIST_KEY });
    fireEvent.click(carbon);
    const workspaceInput = await findByLabelText(
      RegExp(`${spec.args.workspace_dir.name}`)
    );
    fireEvent.change(workspaceInput, { target: { value: fakeWorkspace } });

    const execute = await findByRole('button', { name: /Run/ });
    fireEvent.click(execute);

    // stdout listener is how the app knows the process started
    mockInvestProc.stdout.push('hello from stdout');
    const logTab = await findByText('Log');
    expect(logTab.classList.contains('active')).toBeTruthy();

    // some text from the logfile should be rendered:
    expect(await findByText(dummyTextToLog, { exact: false }))
      .toBeInTheDocument();

    const cancelButton = await findByText('Cancel Run');
    fireEvent.click(cancelButton);
    expect(await findByText('Open Workspace'))
      .toBeEnabled();

    // A recent job card should be rendered
    const cardText = await within(
      await findByLabelText('Recent InVEST Runs:')
    ).findByText(`${path.resolve(fakeWorkspace)}`);
    expect(cardText).toBeInTheDocument();
    unmount();
    spy.mockRestore();
  });

  test('re-run a job - expect new log display', async () => {
    const spy = jest.spyOn(InvestTab.prototype, 'terminateInvestProcess')
      .mockImplementation(() => {
        mockInvestProc.emit('exit', null);
      });

    const {
      findByText,
      findByLabelText,
      findByRole,
      unmount,
    } = render(<App investExe="foo" />);

    const carbon = await findByRole('button', { name: MOCK_MODEL_LIST_KEY });
    fireEvent.click(carbon);
    const workspaceInput = await findByLabelText(
      RegExp(`${spec.args.workspace_dir.name}`)
    );
    fireEvent.change(workspaceInput, { target: { value: fakeWorkspace } });

    const execute = await findByRole('button', { name: /Run/ });
    fireEvent.click(execute);

    // stdout listener is how the app knows the process started
    mockInvestProc.stdout.push('hello from stdout');
    let logTab = await findByText('Log');
    expect(logTab.classList.contains('active')).toBeTruthy();

    // some text from the logfile should be rendered:
    expect(await findByText(dummyTextToLog, { exact: false }))
      .toBeInTheDocument();

    const cancelButton = await findByText('Cancel Run');
    fireEvent.click(cancelButton);
    expect(await findByText('Open Workspace'))
      .toBeEnabled();

    // Now click away from Log, re-run, and expect the switch
    // back to the new log
    const setupTab = await findByText('Setup');
    fireEvent.click(setupTab);
    fireEvent.click(execute);
    // firing execute re-assigns mockInvestProc via the spawn mock,
    // but we need to wait for that before pushing to it's stdout.
    // Since the production code cannot 'await spawn()',
    // we do this manual timeout instead.
    await new Promise((resolve) => setTimeout(resolve, 500));
    mockInvestProc.stdout.push('hello from stdout');
    logTab = await findByText('Log');
    await waitFor(() => {
      expect(logTab.classList.contains('active')).toBeTruthy();
    });
    mockInvestProc.emit('exit', 0);
    // Give it time to run the listener before unmounting.
    await new Promise((resolve) => setTimeout(resolve, 300));
    unmount();
    spy.mockRestore();
  });
});
describe('Tab closing and switching', () => {
  beforeAll(() => {
    getInvestList.mockResolvedValue(MOCK_INVEST_LIST);
    getSpec.mockResolvedValue(SAMPLE_SPEC);
    fetchValidation.mockResolvedValue(MOCK_VALIDATION_VALUE);
  });
  afterEach(async () => {
    jest.clearAllMocks(); // clears usage data, does not reset/restore
    await InvestJob.clearStore(); // should call because a test calls job.save()
  });

  test('', async () => {
    const { 
      getByRole,
      findByText, 
      findByRole, 
      findAllByRole, 
      queryAllByRole } = render(
      <App investExe="foo" />
    );

    const carbon = await findByRole('button', { name: MOCK_MODEL_LIST_KEY });
    const homeTab = await findByText('Home');

    // Open a model tab and expect that it's active
    fireEvent.click(carbon);
    let modelTabs = await findAllByRole('tab', { name: /Carbon/ });
    expect(modelTabs.length).toEqual(1);  // one carbon tab open
    const tab1 = modelTabs[0];
    const tab1EventKey = tab1.getAttribute('data-rb-event-key');
    expect(tab1.classList.contains('active')).toBeTruthy();
    expect(homeTab.classList.contains('active')).toBeFalsy();

    // Open a second model tab and expect that it's active
    fireEvent.click(homeTab);
    fireEvent.click(carbon);
    modelTabs = await findAllByRole('tab', { name: /Carbon/ });
    expect(modelTabs.length).toEqual(2);  // 2 carbon tabs open
    const tab2 = modelTabs[1];
    const tab2EventKey = tab2.getAttribute('data-rb-event-key');
    expect(tab2.classList.contains('active')).toBeTruthy();
    expect(tab1.classList.contains('active')).toBeFalsy();
    expect(homeTab.classList.contains('active')).toBeFalsy();
    // make sure that we switched away from the first tab
    expect(tab2EventKey).not.toEqual(tab1EventKey);

    // Open a third model tab and expect that it's active
    fireEvent.click(homeTab);
    fireEvent.click(carbon);
    modelTabs = await findAllByRole('tab', { name: /Carbon/ });
    expect(modelTabs.length).toEqual(3);  // 3 carbon modelTabs open
    const tab3 = modelTabs[2];
    const tab3EventKey = tab3.getAttribute('data-rb-event-key');
    expect(tab3.classList.contains('active')).toBeTruthy();
    expect(tab2.classList.contains('active')).toBeFalsy();
    expect(tab1.classList.contains('active')).toBeFalsy();
    expect(homeTab.classList.contains('active')).toBeFalsy();
    // make sure that we switched away from the first modelTabs
    expect(tab3EventKey).not.toEqual(tab2EventKey);
    expect(tab3EventKey).not.toEqual(tab1EventKey);

    console.log(tab1EventKey, tab2EventKey, tab3EventKey);

    // Click the close button on the middle tab
    const tab2CloseButton = await within(tab2).getByRole('button', { name: /x/ });
    fireEvent.click(tab2CloseButton);
    // Now there should only be 2 model modelTabs open
    modelTabs = await findAllByRole('tab', { name: /Carbon/ });
    expect(modelTabs.length).toEqual(2);
    // Should have switched to tab3, the next tab to the right
    expect(tab3.classList.contains('active')).toBeTruthy();
    expect(tab1.classList.contains('active')).toBeFalsy();

    // Click the close button on the right tab
    const tab3CloseButton = await within(tab3).getByRole('button', { name: /x/ });
    fireEvent.click(tab3CloseButton);
    // Now there should only be 1 model tab open
    modelTabs = await findAllByRole('tab', { name: /Carbon/ });
    expect(modelTabs.length).toEqual(1);
    // No modelTabs to the right, so it should switch to the next tab to the left.
    expect(tab1.classList.contains('active')).toBeTruthy();
    expect(homeTab.classList.contains('active')).toBeFalsy();

    // Click the close button on the last tab
    const tab1CloseButton = await within(tab1).getByRole('button', { name: /x/ });
    fireEvent.click(tab1CloseButton);
    // Now there should be no model modelTabs open.
    modelTabs = await queryAllByRole('tab', { name: /Carbon/ });
    expect(modelTabs.length).toEqual(0);
    // No more modelTabs, so it should switch back to the home tab.
    expect(homeTab.classList.contains('active')).toBeTruthy();
  });
});
