import path from 'path';
import fs from 'fs';
import React from 'react';
import { remote } from 'electron';
import { fireEvent, render, waitFor } from '@testing-library/react'
import '@testing-library/jest-dom'

import App from '../src/app';
import { getInvestList, getFlaskIsReady } from '../src/server_requests';
jest.mock('../src/server_requests');
import { fileRegistry } from '../src/constants';
import { cleanupDir } from '../src/utils'

getFlaskIsReady.mockResolvedValue('Flask ready');
getInvestList.mockResolvedValue({});

afterAll(() => {
    cleanupDir(fileRegistry.TEMP_DIR)
    cleanupDir(fileRegistry.CACHE_DIR)
})

test('Recent Sessions: each has a button', async () => {
  // This is a special json containing data used for testing
  const testJobsDatabase = path.join(__dirname, './data/jobdb.json');
  const { getByText, getByLabelText, debug } = render(
    <App jobDatabase={testJobsDatabase}/>);
  const db = JSON.parse(fs.readFileSync(testJobsDatabase));

  await waitFor(() => {
    Object.keys(db).forEach(job => {
      expect(getByText(db[job].workspace.directory))
        .toBeTruthy();
    })
  })
})

test('Settings dialog interactions: logging level', async () => {
  
  const DEFAULT = 'INFO';

  const { getByText, getByLabelText, debug } = render(
    <App appdata={fileRegistry.JOBS_DATABASE}/>);

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
  const defaultValue = '-1';
  const newValue = '2'
  const badValue = 'a'
  const labelText = 'Taskgraph n_workers parameter'

  const { getByText, getByLabelText, debug } = render(
    <App appdata={fileRegistry.JOBS_DATABASE}/>);

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