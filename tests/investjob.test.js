import React from 'react';
import { remote } from 'electron';
import { fireEvent, render, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

import InvestJob from '../src/InvestJob';
import SetupTab from '../src/components/SetupTab';
import { getSpec, saveToPython, writeParametersToFile,
         fetchValidation, fetchDatastackFromFile } from '../src/server_requests';
jest.mock('../src/server_requests');
import { fileRegistry } from '../src/constants';

import SAMPLE_SPEC from './data/carbon_args_spec.json';
const MOCK_VALIDATION_VALUE = [[['workspace_dir'], 'invalid because']]
const MOCK_RECENT_JOBS_VALUE = 
  [ [ "job1",
      {
        "model": "carbon",
        "workspace": { "directory": "carbon_setup", "suffix": null },
        "statefile": "tests/data/carbon_setup.json",
        "status": null,
        "humanTime": "3/5/2020, 10:43:14 AM",
        "systemTime": 1583259376573.759,
        "description": null } ] ]


function renderInvestJob() {
  /* Render an InvestJob component with the minimal props 
  * needed for these tests.
  */
  const { getByText, getByLabelText, ...utils } = render(
    <InvestJob
      investExe=''
      investList={{Carbon: {internal_name: 'carbon'}}}
      investSettings={{nWorkers: '-1', loggingLevel: 'INFO'}}
      recentJobs={MOCK_RECENT_JOBS_VALUE}
      jobDatabase={fileRegistry.JOBS_DATABASE}
      updateRecentJobs={() => {}}
      saveSettings={() => {}}
    />);
  return { getByText, getByLabelText, utils }
}

beforeEach(() => {
  jest.resetAllMocks();
  // Careful with reset because "resetting a spy results
  // in a function with no return value". I had been using spies to observe
  // function calls, but not to mock return values. Spies used for that 
  // purpose should be 'restored' not 'reset'. Do that inside the test as-needed.
})

// test('Clicking an invest button renders SetupTab', async () => {
//   getSpec.mockResolvedValue(SAMPLE_SPEC);
//   fetchValidation.mockResolvedValue(MOCK_VALIDATION_VALUE);

//   const { getByText, getByLabelText, utils } = renderInvestJob()

//   const carbon = getByText('Carbon');
//   fireEvent.click(carbon);  // Choosing a model from the list
//   const execute = await utils.findByText('Execute');
//   await waitFor(() => {
//     // Expect a disabled Execute button and a visible SetupTab
//     expect(execute).toBeTruthy();
//     expect(execute).toBeDisabled();  // depends on the mocked fetchValidation
//     expect(getByText('Setup').classList.contains('active')).toBeTruthy();
//   });
  
//   expect(getSpec).toHaveBeenCalledTimes(1);  // the wrapper around fetch
// })

// test('Clicking a recent job renders SetupTab', async () => {
//   fetchValidation.mockResolvedValue(MOCK_VALIDATION_VALUE);
//   const mockDatastack = {
//     module_name: 'natcap.invest.carbon',
//     args: {
//       workspace_dir: "carbon-sample", 
//     }
//   }
//   getSpec.mockResolvedValue(SAMPLE_SPEC)
//   fetchDatastackFromFile.mockResolvedValue(mockDatastack)

//   const { getByText, getByLabelText, utils } = renderInvestJob()

//   const recent = getByText('carbon_setup');
//   fireEvent.click(recent);  // a recent job button
//   const execute = await utils.findByText('Execute');
//   await waitFor(() => {
//     // Expect a disabled Execute button and a visible SetupTab
//     expect(execute).toBeTruthy();
//     expect(execute).toBeDisabled(); // depends on the mocked fetchValidation
//     expect(getByText('Setup').classList.contains('active')).toBeTruthy();
//     // Expect some values that were loaded from the saved state:
//     expect(getByLabelText(/Workspace/)).toHaveValue('carbon-sample')
//   });
// })

test('Loading a recent job when the invest logfile is missing', async () => {
  /* We should get an alert saying nothing can be loaded. */
  const spy = jest.spyOn(window, 'alert').mockImplementation(() => {});
  fetchDatastackFromFile.mockResolvedValue(undefined)
  getSpec.mockResolvedValue(SAMPLE_SPEC);

  const { getByText, getByLabelText, utils } = renderInvestJob()

  const recent = getByText('carbon_setup');
  fireEvent.click(recent);  // a recent job button
  await waitFor(() => {
    expect(spy).toHaveBeenCalledTimes(1)
  });
  spy.mockRestore()
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

  const { getByText, getByLabelText, utils } = renderInvestJob()

  const loadButton = getByText('Load Parameters');
  fireEvent.click(loadButton);
  const execute = await utils.findByText('Execute');
  await waitFor(() => {
    // Expect a disabled Execute button and a visible SetupTab
    expect(execute).toBeDisabled();  // depends on the mocked fetchValidation
    expect(getByText('Setup').classList.contains('active')).toBeTruthy();
    expect(getByLabelText(/Carbon Pools/))
      .toHaveValue(mockDatastack.args.carbon_pools_path)
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

  const { getByText, getByLabelText, utils } = renderInvestJob()

  const loadButton = getByText('Load Parameters');
  fireEvent.click(loadButton);
  await waitFor(() => {
    // expect we're on the same tab we started on instead of switching to Setup
    expect(getByText('Home').classList.contains('active')).toBeTruthy();
  });
  // These are the calls that would have triggered if a file was selected
  expect(fetchDatastackFromFile).toHaveBeenCalledTimes(0)
  expect(getSpec).toHaveBeenCalledTimes(0)
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

  getSpec.mockResolvedValue(SAMPLE_SPEC)
  fetchValidation.mockResolvedValue([]);

  const { getByText, getByLabelText, utils } = renderInvestJob()

  fireEvent.click(getByText('Carbon'));
  const saveDropdown = await utils.findByText('Save Parameters')
  fireEvent.click(saveDropdown); // click the dropdown to mount the next button
  const saveButton = await utils.findByText('Save parameters to JSON')
  fireEvent.click(saveButton);

  await waitFor(() => {
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

  getSpec.mockResolvedValue(SAMPLE_SPEC)
  fetchValidation.mockResolvedValue([]);

  const { getByText, getByLabelText, utils } = renderInvestJob()

  fireEvent.click(getByText('Carbon'));
  const saveDropdown = await utils.findByText('Save Parameters')
  fireEvent.click(saveDropdown)
  const saveButton = await utils.findByText('Save to Python script')
  fireEvent.click(saveButton);

  await waitFor(() => {
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

  const { getByText, getByLabelText, utils } = renderInvestJob()
  
  fireEvent.click(getByText('Carbon'));
  const saveDropdown = await utils.findByText('Save Parameters')
  fireEvent.click(saveDropdown); // click the dropdown to mount the next button
  const saveButton = await utils.findByText('Save parameters to JSON')
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
  const spy = jest.spyOn(SetupTab.prototype, 'savePythonScript')

  const { getByText, getByLabelText, utils } = renderInvestJob()
  
  fireEvent.click(getByText('Carbon'));
  const saveDropdown = await utils.findByText('Save Parameters')
  fireEvent.click(saveDropdown); // click the dropdown to mount the next button
  const saveButton = await utils.findByText('Save to Python script')
  fireEvent.click(saveButton);

  // These are the calls that would have triggered if a file was selected
  expect(spy).toHaveBeenCalledTimes(0)
  spy.mockRestore() // restores to unmocked implementation
})

test('Test various ways for repeated re-renders of SetupTab', async () => {
  /** Test the response of the Setup Tab from the various 
  * ways in which a user can trigger clearing and re-initializing the setup.
  */
  getSpec.mockResolvedValue(SAMPLE_SPEC);
  fetchValidation.mockResolvedValue(MOCK_VALIDATION_VALUE);
  const mockDatastack = {
    module_name: 'natcap.invest.carbon',
    args: {
      workspace_dir: "carbon-sample", 
    }
  }
  fetchDatastackFromFile.mockResolvedValue(mockDatastack)

  const { getByText, getByLabelText, utils } = renderInvestJob()

  // 1. Loading from a recent job
  fireEvent.click(getByText('carbon_setup'));  // a recent job button
  const execute = await utils.findByText('Execute');
  await waitFor(() => {
    // Expect a disabled Execute button and a visible SetupTab
    expect(execute).toBeTruthy();
    expect(execute).toBeDisabled(); // depends on the mocked fetchValidation
    expect(getByText('Setup').classList.contains('active')).toBeTruthy();
    // Expect some values that were loaded from the saved state:
    expect(getByLabelText(/Workspace/))
      .toHaveValue(mockDatastack.args.workspace_dir)
  });

  // 2. Choosing a model from the list
  fireEvent.click(getByText('Carbon'));  
  await waitFor(() => {
    // Expect the values that were previously loaded were cleared
    expect(getByLabelText(/Workspace/)).toHaveValue('')
  });

  // 3. Using the Load Parameters Button
  const mockDialogData = {
    filePaths: ['foo.json']
  }
  remote.dialog.showOpenDialog.mockResolvedValue(mockDialogData)
  
  fireEvent.click(getByText('Load Parameters'));
  await waitFor(() => {
    expect(getByLabelText(/Workspace/))
      .toHaveValue(mockDatastack.args.workspace_dir)
  });
})

