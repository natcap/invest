import path from 'path';
import fs from 'fs';
import events from 'events';
import os from 'os';
import { spawn } from 'child_process';
jest.mock('child_process');
import Stream from 'stream';

import React from 'react';
import { remote } from 'electron';
import { fireEvent, render, waitFor, within } from '@testing-library/react'
import '@testing-library/jest-dom'

import InvestJob from '../src/InvestJob';
import App from '../src/app';
import { 
  getInvestList, getSpec, fetchValidation, fetchDatastackFromFile
} from '../src/server_requests';
jest.mock('../src/server_requests');
import Job from '../src/Job';
import { fileRegistry } from '../src/constants';
import { cleanupDir } from '../src/utils';
import SAMPLE_SPEC from './data/carbon_args_spec.json';

const MOCK_MODEL_LIST_KEY = 'Carbon';
const MOCK_MODEL_RUN_NAME = 'carbon';
const MOCK_INVEST_LIST = {
  [MOCK_MODEL_LIST_KEY]: { 
    internal_name: MOCK_MODEL_RUN_NAME
  }
}
const MOCK_VALIDATION_VALUE = [[['workspace_dir'], 'invalid because']];

afterAll(() => {
  // cache dir accumulates files when Run is clicked
  fs.readdirSync(fileRegistry.CACHE_DIR).forEach(filename => {
    fs.unlinkSync(path.join(fileRegistry.CACHE_DIR, filename))
  })
  jest.resetAllMocks()
})

describe('Various ways to open and close InVEST models', () => {
  beforeAll(() => {
    getInvestList.mockResolvedValue(MOCK_INVEST_LIST);
    getSpec.mockResolvedValue(SAMPLE_SPEC);
    fetchValidation.mockResolvedValue(MOCK_VALIDATION_VALUE);
  })
  afterEach(async () => {
    jest.clearAllMocks() // clears usage data, does not reset/restore
    await Job.clearStore(); // should call because a test calls job.save()
  })

  test('Clicking an invest model button renders SetupTab', async () => {
    const { findByText, findByRole } = render(
      <App
        investExe='foo'
      />
    );

    const carbon = await findByRole('button', {name: MOCK_MODEL_LIST_KEY});
    fireEvent.click(carbon);
    const executeButton = await findByRole('button', {name: /Run/});
    expect(executeButton).toBeDisabled();
    const setupTab = await findByText('Setup');
    expect(setupTab.classList.contains('active')).toBeTruthy();  
    expect(getSpec).toHaveBeenCalledTimes(1);
  })

  test('Clicking a recent job renders SetupTab', async () => {
    const workspacePath = 'my_workspace';
    const mockJob = new Job({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Sequestration',
      argsValues: {
        workspace_dir: workspacePath
      },
      workspace: { directory: workspacePath, suffix: null },
      status: 'success',
      humanTime: '3/5/2020, 10:43:14 AM',
    })
    mockJob.save();

    const { findByText, findByLabelText, findByRole } = render(
      <App
        investExe='foo'
      />
    );

    const recentJobCard = await findByText(
      mockJob.metadata.workspace.directory
    );
    fireEvent.click(recentJobCard);
    const executeButton = await findByRole('button', {name: /Run/});
    expect(executeButton).toBeDisabled();
    const setupTab = await findByText('Setup');
    expect(setupTab.classList.contains('active')).toBeTruthy();

    // Expect some arg values that were loaded from the saved job:
    const input = await findByLabelText(/Workspace/);
    expect(input).toHaveValue(
      mockJob.metadata.workspace.directory
    );
  })

  test('LoadParameters: Dialog callback renders SetupTab', async () => {
    const mockDialogData = {
      filePaths: ['foo.json']
    }
    const mockDatastack = {
      args: {
        carbon_pools_path: "Carbon/carbon_pools_willamette.csv", 
      },
      module_name: 'natcap.invest.carbon',
      model_run_name: 'carbon',
      model_human_name: 'Carbon',
    }
    remote.dialog.showOpenDialog.mockResolvedValue(mockDialogData)
    fetchDatastackFromFile.mockResolvedValue(mockDatastack)
    
    const { findByText, findByLabelText, findByRole } = render(
      <App
        investExe='foo'
      />
    );

    const loadButton = await findByText('Load Parameters');
    fireEvent.click(loadButton);
    const executeButton = await findByRole('button', {name: /Run/});
    expect(executeButton).toBeDisabled();
    const setupTab = await findByText('Setup');
    const input = await findByLabelText(/Carbon Pools/);
    expect(setupTab.classList.contains('active')).toBeTruthy();
    expect(input).toHaveValue(mockDatastack.args.carbon_pools_path)
  })

  test('LoadParameters: Dialog callback does nothing when canceled', async () => {
    // this resembles the callback data if the dialog is canceled instead of 
    // a file selected.
    const mockDialogData = {
      filePaths: []
    }
    remote.dialog.showOpenDialog.mockResolvedValue(mockDialogData)
    
    const { findByText, findByLabelText } = render(
      <App
        investExe='foo'
      />
    );

    const loadButton = await findByText('Load Parameters');
    const homeTab = await findByText('Home');
    fireEvent.click(loadButton);
    // expect we're on the same tab we started on instead of switching to Setup
    expect(homeTab.classList.contains('active')).toBeTruthy();
    // These are the calls that would have triggered if a file was selected
    expect(fetchDatastackFromFile).toHaveBeenCalledTimes(0)
    expect(getSpec).toHaveBeenCalledTimes(0)
  })

  test('Opening and closing multiple InVEST models', async () => {
    const {
      findByText,
      findByTitle,
      findByRole,
      findAllByText,
      queryByText,
    } = render(
      <App
        investExe='foo'
      />
    );

    // Open first model
    const modelA = await findByRole('button', {name: MOCK_MODEL_LIST_KEY});
    fireEvent.click(modelA);
    const tabPanelA = await findByTitle(MOCK_MODEL_LIST_KEY);
    const setupTabA = await within(tabPanelA).findByText('Setup');
    expect(setupTabA.classList.contains('active')).toBeTruthy();  
    expect(within(tabPanelA).queryByRole('button', {name: /Run/}))
      .toBeInTheDocument();
    within(tabPanelA).queryAllByText(/Save to/).forEach(saveButton => {
      expect(saveButton).toBeInTheDocument();
    })
    expect(getSpec).toHaveBeenCalledTimes(1);

    // Open another model (via Load button for convenience)
    const mockDialogData = {
      filePaths: ['foo.json']
    }
    const mockDatastack = {
      module_name: 'natcap.invest.party',
      model_run_name: 'party',
      model_human_name: 'Party Time',
      args: {
        carbon_pools_path: "Carbon/carbon_pools_willamette.csv", 
      }
    }
    remote.dialog.showOpenDialog.mockResolvedValue(mockDialogData);
    fetchDatastackFromFile.mockResolvedValue(mockDatastack);
    const loadButton = await findByText('Load Parameters');
    fireEvent.click(loadButton);
    const tabPanelB = await findByTitle(mockDatastack.model_human_name);
    const setupTabB = await within(tabPanelB).findByText('Setup');
    expect(setupTabB.classList.contains('active')).toBeTruthy();
    expect(within(tabPanelB).queryByRole('button', {name: /Run/}))
      .toBeInTheDocument();
    within(tabPanelB).queryAllByText(/Save to/).forEach(saveButton => {
      expect(saveButton).toBeInTheDocument();
    })
    expect(getSpec).toHaveBeenCalledTimes(2);

    // Close one open model
    const closeButtonArray = await findAllByText('x', { exact: true });
    fireEvent.click(closeButtonArray[1]);
    expect(setupTabB).not.toBeInTheDocument();
    expect(setupTabA.classList.contains('active')).toBeTruthy();
    
    // Close the other open model
    fireEvent.click(closeButtonArray[0]);
    expect(setupTabA).not.toBeInTheDocument();
    const homeTab = await findByText('Home');
    expect(homeTab.classList.contains('active')).toBeTruthy();
  })
})


describe('Display recently executed InVEST jobs', () => {
  beforeEach(() => {
    getInvestList.mockResolvedValue({});
  })
  afterEach(async () => {
    await Job.clearStore();
  })
  
  test('Recent Jobs: each has a button', async () => {
    // TODO: This json is tracked in the repo for purposes of this test.
    const job1 = new Job({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Sequestration',
      argsValues: {
        workspace_dir: 'work1'
      },
      workspace: { directory: 'work1', suffix: null },
      status: 'success',
      humanTime: '3/5/2020, 10:43:14 AM',
    });
    let recentJobs = await job1.save();
    const job2 = new Job({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Sequestration',
      argsValues: {
        workspace_dir: 'work2'
      },
      workspace: { directory: 'work2', suffix: null },
      status: 'success',
      humanTime: '3/5/2020, 10:43:14 AM',
    });
    recentJobs = await job2.save();
    
    const { getByText } = render(
      <App
        investExe='foo'/>);
    
    await waitFor(() => {
      recentJobs.forEach((job) => {
        expect(getByText(job.workspace.directory))
          .toBeTruthy();
      })
    })
  })

  test('Recent Jobs: Some text if the database does not exist', async () => {
    const { findByText } = render(
      <App
        investExe='foo'/>);

    const node = await findByText(/No recent InVEST runs/)
    expect(node).toBeInTheDocument()
  })
})

describe('InVEST global settings: dialog interactions', () => {
  beforeEach(() => {
    getInvestList.mockResolvedValue({});
  })
  afterEach(() => {
    jest.resetAllMocks();
  })
  test('Set the python logging level to pass to the invest CLI', async () => {
    const DEFAULT = 'INFO';
    const { getByText, getByLabelText } = render(
      <App
        investExe='foo'/>);

    // Check the default settings
    fireEvent.click(getByText('Settings'));
    await waitFor(() => { 
      // waiting because the selected value depends on passed props
      expect(getByText(DEFAULT).selected).toBeTruthy();
    })

    // Change the select input and cancel -- expect default selected
    fireEvent.change(getByLabelText('Logging threshold'),
      { target: { value: 'DEBUG' } })
    fireEvent.click(getByText('Cancel'));
    // fireEvent.click(getByText('Settings'));  // why is this unecessary?
    expect(getByText(DEFAULT).selected).toBeTruthy();

    // Change the select input and save -- expect new value selected
    fireEvent.change(getByLabelText('Logging threshold'),
      { target: { value: 'DEBUG' } })
    fireEvent.click(getByText('Save Changes'));
    // fireEvent.click(getByText('Settings'));  // why is this unecessary?
    expect(getByText('DEBUG').selected).toBeTruthy();
  })

  test('Set the invest n_workers parameter', async () => {
    const defaultValue = '-1';
    const newValue = '2'
    const badValue = 'a'
    const labelText = 'Taskgraph n_workers parameter'

    const { getByText, getByLabelText } = render(
      <App
        investExe='foo'/>);

    fireEvent.click(getByText('Settings'));
    const input = getByLabelText(labelText, { exact: false })
    
    // Check the default settings
    await waitFor(() => { 
      // waiting because the text value depends on passed props
      expect(input).toHaveValue(defaultValue);
    })

    // Change the value and cancel -- expect default value
    fireEvent.change(input, { target: { value: newValue } })
    fireEvent.click(getByText('Cancel'));
    expect(input).toHaveValue(defaultValue);

    // Change the value and save -- expect new value selected
    fireEvent.change(input, { target: { value: newValue } })
    expect(input).toHaveValue(newValue); // of course, we just set it
    // The real test: still newValue after saving and re-opening
    fireEvent.click(getByText('Save Changes'));
    fireEvent.click(getByText('Settings'));
    await waitFor(() => {  // the value to test is inherited through props
      expect(input).toHaveValue(newValue);
    })

    // Change to bad value -- expect invalid signal
    fireEvent.change(input, { target: { value: badValue } })
    expect(input.classList.contains('is-invalid')).toBeTruthy();
    expect(getByText('Save Changes')).toBeDisabled();
  })
})

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
    model_name: 'Eco Model',
    module: 'natcap.invest.dot',
  }

  const dummyTextToLog = JSON.stringify(spec.args)
  const logfileName = 'InVEST-natcap.invest.model-log-9999-99-99--99_99_99.txt'
  let fakeWorkspace;
  let mockInvestProc;

  beforeEach(() => {
    fakeWorkspace = fs.mkdtempSync(path.join(
      'tests/data', 'data-'))
    const logfilePath = path.join(fakeWorkspace, logfileName)
    // Need to reset these streams since mockInvestProc is shared by tests
    // and the streams apparently receive the EOF signal in each test.
    mockInvestProc = new events.EventEmitter();
    mockInvestProc.stdout = new Stream.Readable({
      read() {},
    })
    mockInvestProc.stderr = new Stream.Readable({
      read() {}
    })
    getSpec.mockResolvedValue(spec);
    fetchValidation.mockResolvedValue([]);
    getInvestList.mockResolvedValue(
      {Carbon: {internal_name: 'carbon'}});
    
    spawn.mockImplementation((exe, cmdArgs, options) => {
      // To simulate an invest model run, write a logfile to the workspace
      // The line-ending is critical; the log is read with `tail.on('line'...)`
      fs.writeFileSync(logfilePath, dummyTextToLog + os.EOL)
      return mockInvestProc
    })
  })
  
  afterEach(() => {
    cleanupDir(fakeWorkspace);
    jest.resetAllMocks();
  })
  
  test('exit without error - expect log display', async () => {
    const {
      findByText, findByLabelText, findByRole, queryByText, unmount
    } = render(
      <App
        investExe='foo'/>);

    const carbon = await findByRole('button', {name: MOCK_MODEL_LIST_KEY});
    fireEvent.click(carbon);
    const workspaceInput = await findByLabelText(
      RegExp(`${spec.args.workspace_dir.name}`)
    );
    fireEvent.change(workspaceInput, { target: { value: fakeWorkspace } })
    const execute = await findByRole('button', {name: /Run/});
    fireEvent.click(execute);
    await waitFor(() => {
      expect(execute).toBeDisabled();
    })
    
    // stdout listener is how the app knows the process started
    mockInvestProc.stdout.push('hello from stdout');
    const logTab = await findByText('Log');
    expect(logTab.classList.contains('active')).toBeTruthy();
    // some text from the logfile should be rendered:
    expect(await findByText(dummyTextToLog, { exact: false }))
      .toBeInTheDocument();
    expect(queryByText('Model Complete')).toBeNull()
    expect(queryByText('Open Workspace')).toBeNull()
    
    mockInvestProc.emit('exit', 0)  // 0 - exit w/o error
    expect(await findByText('Model Complete')).toBeInTheDocument();
    expect(await findByText('Open Workspace')).toBeEnabled();
    expect(execute).toBeEnabled();

    // A recent job card should be rendered
    const cardText = await within(
      await findByLabelText('Recent InVEST Runs:')
    ).findByText(`${path.resolve(fakeWorkspace)}`)
    expect(cardText).toBeInTheDocument()
    // Normally we don't explicitly unmount the rendered components,
    // but in this case we're 'watching' a file that the afterEach()
    // wants to remove. Unmounting triggers an 'unwatch' of the logfile
    // before afterEach cleanup, avoiding an error.
    unmount();
  })

  test('exit with error - expect log display', async () => {
    const {
      findByText, findByLabelText, findByRole, unmount, ...utils
    } = render(
      <App
        jobDatabase={fileRegistry.JOBS_DATABASE}
        investExe='foo'/>);

    const carbon = await findByRole('button', {name: MOCK_MODEL_LIST_KEY});
    fireEvent.click(carbon);
    const workspaceInput = await findByLabelText(
      RegExp(`${spec.args.workspace_dir.name}`))
    fireEvent.change(workspaceInput, { target: { value: fakeWorkspace } })
    
    const execute = await findByRole('button', {name: /Run/});
    fireEvent.click(execute);
    
    const errorMessage = 'fail'
    // Emit some stdout, some stderr, then pause and exit with error
    mockInvestProc.stdout.push('hello from stdout');
    mockInvestProc.stderr.push(errorMessage);
    const logTab = await findByText('Log');
    expect(logTab.classList.contains('active')).toBeTruthy();
    // some text from the logfile should be rendered:
    expect(await findByText(dummyTextToLog, { exact: false }))
      .toBeInTheDocument();
    await new Promise(resolve => setTimeout(resolve, 2000))
    mockInvestProc.emit('exit', 1)  // 1 - exit w/ error
    // stderr text should be rendered in a red alert
    expect(await findByText(errorMessage)).toHaveClass('alert-danger');
    expect(await findByText('Open Workspace')).toBeEnabled();
    // A recent job card should be rendered
    const cardText = await within(
      await findByLabelText('Recent InVEST Runs:')
    ).findByText(`${path.resolve(fakeWorkspace)}`);
    expect(cardText).toBeInTheDocument();
    unmount();
  })

  test('user terminates process - expect log display', async () => {
    const spy = jest.spyOn(InvestJob.prototype, 'terminateInvestProcess')
      .mockImplementation(() => {
        mockInvestProc.emit('exit', null)
      })

    const {
      findByText, findByLabelText, findByRole, unmount, ...utils
    } = render(
      <App
        jobDatabase={fileRegistry.JOBS_DATABASE}
        investExe='foo'/>);

    const carbon = await findByRole('button', {name: MOCK_MODEL_LIST_KEY});
    fireEvent.click(carbon);
    const workspaceInput = await findByLabelText(
      RegExp(`${spec.args.workspace_dir.name}`))
    fireEvent.change(workspaceInput, { target: { value: fakeWorkspace } })
    
    const execute = await findByRole('button', {name: /Run/});
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
    expect(await findByText('Open Workspace')).toBeEnabled();
    // A recent job card should be rendered
    const cardText = await within(
      await findByLabelText('Recent InVEST Runs:')
    ).findByText(`${path.resolve(fakeWorkspace)}`);
    expect(cardText).toBeInTheDocument();
    unmount();
    spy.mockRestore();
  })
})
