import events from 'events';
// import sinon from 'sinon';
import React from 'react';
import { fireEvent, render,
         wait, waitForElement } from '@testing-library/react'
import '@testing-library/jest-dom'
// import child_process from 'child_process';

import { InvestJob } from '../src/InvestJob';
import SAMPLE_SPEC from './data/carbon_args_spec.json';
import { getSpec, saveToPython, writeParametersToFile,
         fetchValidation } from '../src/server_requests';
jest.mock('../src/server_requests');
// import { findMostRecentLogfile } from '../src/utils';
// jest.mock('../src/utils');

const APP_DATA = './data/jobdb.json';
const MOCK_VALIDATION_VALUE = [[['workspace_dir'], 'invalid because']]
const MOCK_RECENT_SESSIONS_VALUE = 
  [ [ "job1",
      {
        "model": "carbon",
        "workspace": { "directory": "carbon_setup", "suffix": null },
        "statefile": "carbon_setup.json",
        "status": null,
        "humanTime": "3/5/2020, 10:43:14 AM",
        "systemTime": 1583259376573.759,
        "description": null } ] ]

test('Clicking an invest button renders SetupTab', async () => {
  getSpec.mockResolvedValue(SAMPLE_SPEC);
  fetchValidation.mockResolvedValue(MOCK_VALIDATION_VALUE);
  const spy = jest.spyOn(InvestJob.prototype, 'investGetSpec');

  const { getByText, debug } = render(
    <InvestJob 
      investList={{Carbon: {internal_name: 'carbon'}}}
      investSettings={null}
      recentSessions={[]}
      updateRecentSessions={() => {}}
      saveSettings={() => {}}
    />);
  const carbon = getByText('Carbon');
  fireEvent.click(carbon);  // Choosing a model from the list
  await wait(() => {
    const execute = getByText('Execute');
    // Expect a disabled Execute button and a visible SetupTab
    expect(execute).toBeTruthy();
    expect(execute).toBeDisabled();  // depends on the mocked fetchValidation
    expect(getByText('Setup').classList.contains('active')).toBeTruthy();
  });
  
  expect(spy).toHaveBeenCalledTimes(1);  // the click handler
  expect(getSpec).toHaveBeenCalledTimes(1);  // the wrapper around fetch
})

test('Clicking a recent session renders SetupTab', async () => {
  fetchValidation.mockResolvedValue(MOCK_VALIDATION_VALUE);
  const spy = jest.spyOn(InvestJob.prototype, 'loadState');

  const { getByText, debug } = render(
    <InvestJob 
      investList={{}}
      investSettings={null}
      recentSessions={MOCK_RECENT_SESSIONS_VALUE}
      updateRecentSessions={() => {}}
      saveSettings={() => {}}
    />);

  const recent = getByText('carbon_setup');
  fireEvent.click(recent);  // a recent session button
  await wait(() => {
    const execute = getByText('Execute');
    // Expect a disabled Execute button and a visible SetupTab
    expect(execute).toBeTruthy();
    expect(execute).toBeDisabled(); // depends on the mocked fetchValidation
    expect(getByText('Setup').classList.contains('active')).toBeTruthy();
    // TODO: toBeVisible doesn't work w/ the attributes on these nodes
    // expect(getByText('Setup')).toBeVisible();
    // expect(getByText('Resources')).not.toBeVisible();
  });
  
  expect(spy).toHaveBeenCalledTimes(1);  // called by the click handler
})

// test('Browsing for recent session renders SetupTab', async () => {
//   // TODO: This functionality might be dropped.
// })

test('Save Parameters/Python enable after model select ', async () => {

  const { getByText, debug } = render(
    <InvestJob 
      investList={{}}
      investSettings={null}
      recentSessions={MOCK_RECENT_SESSIONS_VALUE}
      updateRecentSessions={() => {}}
      saveSettings={() => {}}
    />);

  // Check the dropdown before any model setup
  fireEvent.click(getByText('Save'));
  await wait(() => {
    expect(getByText('Save parameters to JSON')).toBeDisabled();
    expect(getByText('Save to Python script')).toBeDisabled();
  })

  // Now load a model setup using a recent session
  fireEvent.click(getByText('carbon_setup'));
  await wait(() => {
    expect(getByText('Save parameters to JSON')).toBeEnabled();
    expect(getByText('Save to Python script')).toBeEnabled();
  });
})

// test('Execute launches a python subprocess', async () => {
//   /*
//   This functionality might be better tested in an end-end test,
//   maybe using spectron and the actual flask server. I have not
//   gotten this to work yet - mocking the spawn call and getting
//   the spawned object's .on.stdout callback to fire being the main
//   sticking point.
//   */
//   // investExecute() expects to find these args:
//   const spec = { args: {
//     workspace_dir: { 
//       name: 'Workspace', 
//       type: 'directory'},
//     results_suffix: {
//       name: 'suffix',
//       type: 'freestyle_string'} } }

//   const stdOutText = 'hello'
//   getSpec.mockResolvedValue(spec);
//   fetchValidation.mockResolvedValue([]);
//   // TODO: create this stuff dynamically here:
//   const fakeWorkspace = '../data';  // it contains this logfile:
//   findMostRecentLogfile.mockResolvedValue('invest_logfile.txt')

//   let spawnEvent = new events.EventEmitter();
//   spawnEvent.stdout = new events.EventEmitter();
//   spawnEvent.stderr = new events.EventEmitter();
//   sinon.stub(child_process, 'spawn').returns(spawnEvent);
//   // jest.mock('child_process');
//   // let mockSpawn = spawn.mockImplementation(() => {
//   //   // spawn('python', ['mock_invest.py', stdOutText]) // no errors
//   //   let spawnEvent = new events.EventEmitter();
//   //   spawnEvent.stdout = new events.EventEmitter();
//   //   spawnEvent.stderr = new events.EventEmitter();
//   //   return spawnEvent;
//   // })

//   const { getByText, getByLabelText, getByTestId, debug } = render(
//     <InvestJob 
//       investList={{Carbon: {internal_name: 'carbon'}}}
//       investSettings={{
//           nWorkers: '-1',
//           loggingLevel: 'INFO',
//         }}
//       recentSessions={[]}
//       updateRecentSessions={() => {}}
//       saveSettings={() => {}}
//     />);

//   const carbon = getByText('Carbon');
//   fireEvent.click(carbon);  // Choosing a model from the list
//   const input = await waitForElement(() => {
//     return getByLabelText(spec.args.workspace_dir.name)
//   })
//   fireEvent.change(input, { target: { value: fakeWorkspace } })
//   const execute = await waitForElement(() => getByText('Execute'))
//   fireEvent.click(execute);
//   console.log(spawnEvent);
//   spawnEvent.stdout.emit('data', stdOutText);

//   await wait(() => {
//     // debug()
//     // expect(getByText('Model Completed'))
//     expect(getByText('This is a fake invest logfile'))
//   });
// })