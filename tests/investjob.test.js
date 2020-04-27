import fs from 'fs';
import path from 'path';
import events from 'events';
import React from 'react';
import { remote } from 'electron';
import { fireEvent, render,
         wait, waitForElement } from '@testing-library/react';
import '@testing-library/jest-dom';

import { InvestJob } from '../src/InvestJob';
import SAMPLE_SPEC from './data/carbon_args_spec.json';
import { getSpec, saveToPython, writeParametersToFile,
         fetchValidation, fetchDatastackFromFile } from '../src/server_requests';
jest.mock('../src/server_requests');


const APP_DATA = './data/jobdb.json';
const MOCK_VALIDATION_VALUE = [[['workspace_dir'], 'invalid because']]
const MOCK_RECENT_SESSIONS_VALUE = 
  [ [ "job1",
      {
        "model": "carbon",
        "workspace": { "directory": "carbon_setup", "suffix": null },
        "statefile": "tests/data/carbon_setup.json",
        "status": null,
        "humanTime": "3/5/2020, 10:43:14 AM",
        "systemTime": 1583259376573.759,
        "description": null } ] ]

beforeEach(() => {
  jest.resetAllMocks(); 
  // Careful with reset because "resetting a spy results
  // in a function with no return value". I had been using spies to observe
  // function calls, but not to mock return values. Spies used for that 
  // purpose should be 'restored' not 'reset'. Do that inside the test as-needed.
})

test('Clicking an invest button renders SetupTab', async () => {
  getSpec.mockResolvedValue(SAMPLE_SPEC);
  fetchValidation.mockResolvedValue(MOCK_VALIDATION_VALUE);

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
  
  expect(getSpec).toHaveBeenCalledTimes(1);  // the wrapper around fetch
})

test('Clicking a recent session renders SetupTab', async () => {
  fetchValidation.mockResolvedValue(MOCK_VALIDATION_VALUE);

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
})

test('LoadParameters: Dialog callback renders SetupTab', async () => {
  const mockDialogData = {
    filePaths: ['foo.json']
  }
  const mockDatastack = {
    module_name: 'natcap.invest.carbon',
    args: {
      carbon_pools_path: "Carbon/carbon_pools_willamette.csv", 
    }
  }
  remote.dialog.showOpenDialog.mockResolvedValue(mockDialogData)
  fetchDatastackFromFile.mockResolvedValue(mockDatastack)
  getSpec.mockResolvedValue(SAMPLE_SPEC);
  fetchValidation.mockResolvedValue(MOCK_VALIDATION_VALUE);

  const { getByText, getByLabelText, debug } = render(
    <InvestJob 
      investList={{Carbon: {internal_name: 'carbon'}}}
      investSettings={null}
      recentSessions={[]}
      updateRecentSessions={() => {}}
      saveSettings={() => {}}
    />);
  const loadButton = getByText('Load Parameters');
  fireEvent.click(loadButton);
  await wait(() => {
    // Expect a disabled Execute button and a visible SetupTab
    const execute = getByText('Execute');
    expect(execute).toBeDisabled();  // depends on the mocked fetchValidation
    expect(getByText('Setup').classList.contains('active')).toBeTruthy();
    expect(getByLabelText('Carbon Pools')).toHaveValue(
        mockDatastack.args.carbon_pools_path)
  });
  // And now that we know setup has loaded, expect the values from the datastack
  // TODO: expect some global validation errors
})

test('LoadParameters: Dialog callback does nothing when canceled', async () => {
  // this resembles the callback data if the dialog is canceled instead of 
  // a file selected.
  const mockDialogData = {
    filePaths: [] // if(array.length) is how we check if filePaths exist.
  }
  remote.dialog.showOpenDialog.mockResolvedValue(mockDialogData)

  const { getByText, getByLabelText, debug } = render(
    <InvestJob 
      investList={{Carbon: {internal_name: 'carbon'}}}
      investSettings={null}
      recentSessions={[]}
      updateRecentSessions={() => {}}
      saveSettings={() => {}}
    />);
  const loadButton = getByText('Load Parameters');
  fireEvent.click(loadButton);
  await wait(() => {
    // expect we're on the same tab we started on instead of switching to Setup
    expect(getByText('Home').classList.contains('active')).toBeTruthy();
  });
  // These are the calls that would have triggered if a file was selected
  expect(fetchDatastackFromFile).toHaveBeenCalledTimes(0)
  expect(getSpec).toHaveBeenCalledTimes(0)
  expect(fetchValidation).toHaveBeenCalledTimes(0)
})


test('SaveParameters/Python enable after model select ', async () => {
  getSpec.mockResolvedValue(SAMPLE_SPEC);
  fetchValidation.mockResolvedValue([]);
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

test('SaveParametersButton: requests endpoint with correct payload ', async () => {
  // mock the server call, instead just returning
  // the payload. At least we can assert the payload is what 
  // the flask endpoint needs to build the json file.
  writeParametersToFile.mockImplementation((payload) => {
    return payload
  })

  const mockDialogData = {
    filePath: 'foo.json'
  }
  remote.dialog.showSaveDialog.mockResolvedValue(mockDialogData)
  fetchValidation.mockResolvedValue([]);

  const { getByText, findByText, debug } = render(
    <InvestJob 
      investList={{}}
      investSettings={{nWorkers: '-1'}}
      recentSessions={MOCK_RECENT_SESSIONS_VALUE}
      updateRecentSessions={() => {}}
      saveSettings={() => {}}
    />);

  fireEvent.click(getByText('carbon_setup'));
  fireEvent.click(getByText('Save')); // click the dropdown to mount the next button
  const saveButton = await findByText('Save parameters to JSON')
  fireEvent.click(saveButton);

  await wait(() => {
    expect(Object.keys(writeParametersToFile.mock.results[0].value).includes(
      ['parameterSetPath', 'moduleName', 'args', 'relativePaths']))
    expect(writeParametersToFile).toHaveBeenCalledTimes(1)
  })
})

test('SavePythonButton: requests endpoint with correct payload', async () => {
  // mock the server call, instead just returning
  // the payload. At least we can assert the payload is what 
  // the flask endpoint needs to build the python script.
  saveToPython.mockImplementation((payload) => {
    return payload
  })
  const mockDialogData = { filePath: 'foo.py' }
  remote.dialog.showSaveDialog.mockResolvedValue(mockDialogData)
  fetchValidation.mockResolvedValue([]);

  const { getByText, findByText, debug } = render(
    <InvestJob 
      investList={{}}
      investSettings={{nWorkers: '-1'}}
      recentSessions={MOCK_RECENT_SESSIONS_VALUE}
      updateRecentSessions={() => {}}
      saveSettings={() => {}}
    />);

  fireEvent.click(getByText('carbon_setup'));
  fireEvent.click(getByText('Save')); // click the dropdown to mount the next button
  const saveButton = await findByText('Save to Python script')
  fireEvent.click(saveButton);

  await wait(() => {
    expect(Object.keys(saveToPython.mock.results[0].value).includes(
      ['filepath', 'modelname', 'pyname', 'args']))
    expect(saveToPython).toHaveBeenCalledTimes(1)
  })
})

test('SaveParametersButton: Dialog callback does nothing when canceled', async () => {
  // this resembles the callback data if the dialog is canceled instead of 
  // a save file selected.
  const mockDialogData = {
    filePath: ''
  }
  remote.dialog.showSaveDialog.mockResolvedValue(mockDialogData)
  getSpec.mockResolvedValue(SAMPLE_SPEC)
  fetchValidation.mockResolvedValue(MOCK_VALIDATION_VALUE)
  // Spy on this method so we can assert it was never called.
  // Don't forget to restore! Otherwise the beforeEach will 'resetAllMocks'
  // will silently turn this spy into a function that returns nothing.
  const spy = jest.spyOn(InvestJob.prototype, 'argsToJsonFile')

  const { getByText, findByText, debug } = render(
    <InvestJob 
      investList={{Carbon: {internal_name: 'carbon'}}}
      investSettings={null}
      recentSessions={[]}
      updateRecentSessions={() => {}}
      saveSettings={() => {}}
    />);
  
  fireEvent.click(getByText('Carbon'));
  fireEvent.click(getByText('Save')); // click the dropdown to mount the next button
  const saveButton = await findByText('Save parameters to JSON')
  fireEvent.click(saveButton);

  // These are the calls that would have triggered if a file was selected
  expect(spy).toHaveBeenCalledTimes(0)
  spy.mockRestore() // restores to unmocked implementation
})

test('SavePythonButton: Dialog callback does nothing when canceled', async () => {
  // this resembles the callback data if the dialog is canceled instead of 
  // a save file selected.
  const mockDialogData = {
    filePath: ''
  }
  remote.dialog.showSaveDialog.mockResolvedValue(mockDialogData)
  getSpec.mockResolvedValue(SAMPLE_SPEC)
  fetchValidation.mockResolvedValue(MOCK_VALIDATION_VALUE)
  // Spy on this method so we can assert it was never called.
  // Don't forget to restore! Otherwise the beforeEach will 'resetAllMocks'
  // will silently turn this spy into a function that returns nothing.
  const spy = jest.spyOn(InvestJob.prototype, 'savePythonScript')

  const { getByText, findByText, debug } = render(
    <InvestJob 
      investList={{Carbon: {internal_name: 'carbon'}}}
      investSettings={null}
      recentSessions={[]}
      updateRecentSessions={() => {}}
      saveSettings={() => {}}
    />);
  
  fireEvent.click(getByText('Carbon'));
  fireEvent.click(getByText('Save')); // click the dropdown to mount the next button
  const saveButton = await findByText('Save to Python script')
  fireEvent.click(saveButton);

  // These are the calls that would have triggered if a file was selected
  expect(spy).toHaveBeenCalledTimes(0)
  spy.mockRestore() // restores to unmocked implementation
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