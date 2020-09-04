import path from 'path';
import fs from 'fs';
import events from 'events';
import os from 'os';
import { spawn } from 'child_process';
jest.mock('child_process');
import Stream from 'stream';

import React from 'react';
import { fireEvent, render, waitFor, within } from '@testing-library/react'
import '@testing-library/jest-dom'

import App from '../src/app';
import { getInvestList, getSpec, fetchValidation } from '../src/server_requests';
jest.mock('../src/server_requests');
import { fileRegistry } from '../src/constants';
import { cleanupDir } from '../src/utils'

import SAMPLE_SPEC from './data/carbon_args_spec.json';
const MOCK_INVEST_LIST = {
  Carbon: { 
    internal_name: 'carbon'
  }
}
const MOCK_VALIDATION_VALUE = [[['workspace_dir'], 'invalid because']];
const MOCK_JOB_DATABASE_PATH = path.join(fileRegistry.CACHE_DIR, 'mock_job_database.json')
const MOCK_JOB_DATA_PATH = path.join(fileRegistry.CACHE_DIR, 'mock_job_data.json');
const MOCK_JOB_ID = 'job1hash';
const MOCK_RECENT_JOBS_DB = { 
  [MOCK_JOB_ID]: {
    model: "carbon",
    workspace: { "directory": "carbon_workspace", "suffix": null },
    jobDataPath: MOCK_JOB_DATA_PATH,
    status: "success",
    humanTime: "3/5/2020, 10:43:14 AM",
    systemTime: 1583259376573.759,
  },
};

beforeAll(() => {
  // Setting up the files that would exist if there are saved jobs.
  const job = {
    jobID: MOCK_JOB_ID,
    modelRunName: MOCK_RECENT_JOBS_DB[MOCK_JOB_ID].model,
    argsValues: {
      workspace_dir: MOCK_RECENT_JOBS_DB[MOCK_JOB_ID].workspace.directory
    },
    workspace: MOCK_RECENT_JOBS_DB[MOCK_JOB_ID].workspace,
    logfile: 'foo-log.txt',
    status: MOCK_RECENT_JOBS_DB[MOCK_JOB_ID].status,
  };
  fs.writeFileSync(MOCK_JOB_DATA_PATH, JSON.stringify(job), 'utf8');
  fs.writeFileSync(MOCK_JOB_DATABASE_PATH, JSON.stringify(MOCK_RECENT_JOBS_DB), 'utf8');
})

afterAll(() => {
  // cache dir accumulates files when Execute is clicked
  fs.readdirSync(fileRegistry.CACHE_DIR).forEach(filename => {
    fs.unlinkSync(path.join(fileRegistry.CACHE_DIR, filename))
  })
  jest.resetAllMocks()
})

test('Clicking an invest model button renders SetupTab', async () => {
  getInvestList.mockResolvedValue(MOCK_INVEST_LIST);
  getSpec.mockResolvedValue(SAMPLE_SPEC);
  fetchValidation.mockResolvedValue(MOCK_VALIDATION_VALUE);

  const { findByText } = render(
    <App
      jobDatabase={'foodb.json'}
      investExe='foo'
    />
  );

  const carbon = await findByText('Carbon');
  fireEvent.click(carbon);
  const executeButton = await findByText('Execute');
  const setupTab = await findByText('Setup');
  expect(executeButton).toBeTruthy();
  expect(executeButton).toBeDisabled();  // depends on the mocked fetchValidation
  expect(setupTab.classList.contains('active')).toBeTruthy();  
  expect(getSpec).toHaveBeenCalledTimes(1);
})

test('Clicking a recent job renders SetupTab', async () => {
  getInvestList.mockResolvedValue(MOCK_INVEST_LIST);
  fetchValidation.mockResolvedValue(MOCK_VALIDATION_VALUE);
  getSpec.mockResolvedValue(SAMPLE_SPEC);

  const { findByText, findByLabelText } = render(
    <App
      jobDatabase={MOCK_JOB_DATABASE_PATH}
      investExe='foo'
    />
  );

  const recentJobCard = await findByText(
    MOCK_RECENT_JOBS_DB[MOCK_JOB_ID].workspace.directory
  );
  fireEvent.click(recentJobCard);
  const executeButton = await findByText('Execute');
  const setupTab = await findByText('Setup');
  expect(executeButton).toBeTruthy();
  expect(executeButton).toBeDisabled(); // depends on the mocked fetchValidation
  expect(setupTab.classList.contains('active')).toBeTruthy();

  // Expect some arg values that were loaded from the saved job:
  const input = await findByLabelText(/Workspace/);
  expect(input).toHaveValue(
    MOCK_RECENT_JOBS_DB[MOCK_JOB_ID].workspace.directory
  );
})

test('Recent Jobs: each has a button', async () => {
  getInvestList.mockResolvedValue({});
  // This is a special json containing data used for testing
  const testJobsDatabase = path.join(__dirname, './data/jobdb.json');
  const { getByText } = render(
    <App
      jobDatabase={testJobsDatabase}
      investExe='foo'/>);
  const db = JSON.parse(fs.readFileSync(testJobsDatabase));

  await waitFor(() => {
    Object.keys(db).forEach(job => {
      expect(getByText(db[job].workspace.directory))
        .toBeTruthy();
    })
  })
})

test('Recent Jobs: database is missing', async () => {
  getInvestList.mockResolvedValue({});
  const testJobsDatabase = 'foo.json';
  const { findByText } = render(
    <App
      jobDatabase={testJobsDatabase}
      investExe='foo'/>);

  const node = await findByText(/No recent InVEST runs/)
  expect(node).toBeInTheDocument()
})

test('Settings dialog interactions: logging level', async () => {
  getInvestList.mockResolvedValue({});
  const DEFAULT = 'INFO';

  const { getByText, getByLabelText } = render(
    <App
      jobDatabase={fileRegistry.JOBS_DATABASE}
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

test('Settings dialog interactions: n workers', async () => {
  getInvestList.mockResolvedValue({});
  const defaultValue = '-1';
  const newValue = '2'
  const badValue = 'a'
  const labelText = 'Taskgraph n_workers parameter'

  const { getByText, getByLabelText } = render(
    <App
      jobDatabase={fileRegistry.JOBS_DATABASE}
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
  
  test('Invest subprocess - exit without error', async () => {
    const { getByText, getByLabelText, unmount, ...utils } = render(
      <App
        jobDatabase={fileRegistry.JOBS_DATABASE}
        investExe='foo'/>);

    const carbon = await utils.findByText('Carbon');
    fireEvent.click(carbon);
    const workspaceInput = await utils.findByLabelText(
      RegExp(`${spec.args.workspace_dir.name}`))
    fireEvent.change(workspaceInput, { target: { value: fakeWorkspace } })
    const execute = await utils.findByText('Execute');
    fireEvent.click(execute);
    
    // Emit some stdout because our program listens for stdout
    // in order to know the invest subprocess has spawned,
    // signalling that an invest logfile now exists in the workspace.
    mockInvestProc.stdout.push('hello from stdout')
    await waitFor(() => {
      expect(getByText('Log').classList.contains('active')).toBeTruthy();
      // some text from the logfile should be rendered:
      expect(getByText(dummyTextToLog, { exact: false }))
        .toBeInTheDocument();
    });
    mockInvestProc.emit('close', 0)  // 0 - exit w/o error
    await waitFor(() => {
      expect(getByText('Model Completed')).toBeInTheDocument();
    })
    // A recent job card should be rendered
    const { findByText } = within(getByLabelText('Recent InVEST Runs:'))
    const cardText = await findByText(`${path.resolve(fakeWorkspace)}`)
    expect(cardText).toBeInTheDocument()
    // Normally we don't explicitly unmount the rendered components,
    // but in this case we're 'watching' a file that the afterEach()
    // wants to remove. Unmounting triggers an 'unwatch' of the logfile
    // before afterEach cleanup, avoiding an error.
    unmount();
  })

  test('Invest subprocess - exit with error', async () => {
    const { getByText, getByLabelText, unmount, ...utils } = render(
      <App
        jobDatabase={fileRegistry.JOBS_DATABASE}
        investExe='foo'/>);

    const carbon = await utils.findByText('Carbon');
    fireEvent.click(carbon);
    const workspaceInput = await utils.findByLabelText(
      RegExp(`${spec.args.workspace_dir.name}`))
    fireEvent.change(workspaceInput, { target: { value: fakeWorkspace } })
    
    const execute = await utils.findByText('Execute');
    fireEvent.click(execute);
    
    const errorMessage = 'fail'
    // Emit some stdout, some stderr, then pause and exit with error
    mockInvestProc.stdout.push('hello from stdout')
    mockInvestProc.stderr.push(errorMessage)
    await new Promise(resolve => setTimeout(resolve, 2000))
    mockInvestProc.emit('close', 1)  // 1 - exit w/ error
    
    await waitFor(() => {
      expect(getByText('Log').classList.contains('active')).toBeTruthy();
      // some text from the logfile should be rendered:
      expect(getByText(dummyTextToLog, { exact: false }))
        .toBeInTheDocument();
      // stderr text should be rendered 
      expect(getByText(errorMessage)).toHaveClass('alert-danger');
    });
    // A recent job card should be rendered
    const { findByText } = within(getByLabelText('Recent InVEST Runs:'))
    const cardText = await findByText(`${path.resolve(fakeWorkspace)}`)
    expect(cardText).toBeInTheDocument()
    unmount()
  })
})
