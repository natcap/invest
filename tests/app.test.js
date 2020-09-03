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


afterAll(() => {
  // cache dir accumulates files when Execute is clicked
  fs.readdirSync(fileRegistry.CACHE_DIR).forEach(filename => {
    fs.unlinkSync(path.join(fileRegistry.CACHE_DIR, filename))
  })
  jest.resetAllMocks()
})

test('Recent Sessions: each has a button', async () => {
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

test('Recent Sessions: database is missing', async () => {
  getInvestList.mockResolvedValue({});
  const testJobsDatabase = 'foo.json';
  const { findByText } = render(
    <App
      jobDatabase={testJobsDatabase}
      investExe='foo'/>);

  const node = await findByText(/No recent sessions/)
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
    // A recent session card should be rendered
    const { findByText } = within(getByLabelText('Recent Sessions:'))
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
    // A recent session card should be rendered
    const { findByText } = within(getByLabelText('Recent Sessions:'))
    const cardText = await findByText(`${path.resolve(fakeWorkspace)}`)
    expect(cardText).toBeInTheDocument()
    unmount()
  })

})
