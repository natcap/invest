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


/** Render an InvestJob component with the minimal props needed for tests. */
function renderInvestJob() {
  const { findByText, findByLabelText, ...utils } = render(
    <InvestJob
      investExe=''
      modelRunName='carbon'
      argsInitValues={undefined}
      logfile={undefined}
      jobStatus={undefined}
      investSettings={{ nWorkers: '-1', loggingLevel: 'INFO' }}
      saveJob={() => {}}
    />);
  return { findByText, findByLabelText, utils }
}

describe('Save model setup tests', () => {

  afterAll(() => {
    jest.resetAllMocks();
    // Careful with reset because "resetting a spy results
    // in a function with no return value". I had been using spies to observe
    // function calls, but not to mock return values. Spies used for that 
    // purpose should be 'restored' not 'reset'. Do that inside the test as-needed.
  })

  const spec = {
    module: 'natcap.invest.foo',
    args: {
      arg: { 
        name: 'Workspace',
        type: 'directory',
        about: 'this is a workspace'
      }
    }
  };
  getSpec.mockResolvedValue(spec)
  fetchValidation.mockResolvedValue([]);

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

    const { findByText } = renderInvestJob();

    const saveDropdown = await findByText('Save Parameters')
    fireEvent.click(saveDropdown);
    const saveButton = await findByText('Save parameters to JSON')
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

    const { findByText } = renderInvestJob()

    const saveDropdown = await findByText('Save Parameters')
    fireEvent.click(saveDropdown)
    const saveButton = await findByText('Save to Python script')
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
    // Spy on this method so we can assert it was never called.
    // Don't forget to restore! Otherwise a 'resetAllMocks'
    // can silently turn this spy into a function that returns nothing.
    const spy = jest.spyOn(InvestJob.prototype, 'argsToJsonFile')

    const { findByText } = renderInvestJob()
    
    const saveDropdown = await findByText('Save Parameters')
    fireEvent.click(saveDropdown);
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
    // Spy on this method so we can assert it was never called.
    // Don't forget to restore! Otherwise the beforeEach will 'resetAllMocks'
    // will silently turn this spy into a function that returns nothing.
    const spy = jest.spyOn(SetupTab.prototype, 'savePythonScript')

    const { findByText } = renderInvestJob()
    
    const saveDropdown = await findByText('Save Parameters')
    fireEvent.click(saveDropdown); // click the dropdown to mount the next button
    const saveButton = await findByText('Save to Python script')
    fireEvent.click(saveButton);

    // These are the calls that would have triggered if a file was selected
    expect(spy).toHaveBeenCalledTimes(0)
    spy.mockRestore() // restores to unmocked implementation
  })
})

// TODO: move this to apptests and make it handle rendering multipe InvestJob's
// // including closeTab, etc.
// test('Test various ways for repeated re-renders of SetupTab', async () => {
//   /** Test the response of the Setup Tab from the various 
//   * ways in which a user can trigger clearing and re-initializing the setup.
//   */
//   getSpec.mockResolvedValue(SAMPLE_SPEC);
//   fetchValidation.mockResolvedValue(MOCK_VALIDATION_VALUE);
//   const mockDatastack = {
//     module_name: 'natcap.invest.carbon',
//     args: {
//       workspace_dir: "carbon-sample", 
//     }
//   }
//   fetchDatastackFromFile.mockResolvedValue(mockDatastack)

//   const { getByText, getByLabelText, utils } = renderInvestJob()

//   // 1. Loading from a recent job
//   fireEvent.click(getByText('carbon_setup'));  // a recent job button
//   const execute = await utils.findByText('Execute');
//   await waitFor(() => {
//     // Expect a disabled Execute button and a visible SetupTab
//     expect(execute).toBeTruthy();
//     expect(execute).toBeDisabled(); // depends on the mocked fetchValidation
//     expect(getByText('Setup').classList.contains('active')).toBeTruthy();
//     // Expect some values that were loaded from the saved state:
//     expect(getByLabelText(/Workspace/))
//       .toHaveValue(mockDatastack.args.workspace_dir)
//   });

//   // 2. Choosing a model from the list
//   fireEvent.click(getByText('Carbon'));  
//   await waitFor(() => {
//     // Expect the values that were previously loaded were cleared
//     expect(getByLabelText(/Workspace/)).toHaveValue('')
//   });

//   // 3. Using the Load Parameters Button
//   const mockDialogData = {
//     filePaths: ['foo.json']
//   }
//   remote.dialog.showOpenDialog.mockResolvedValue(mockDialogData)
  
//   fireEvent.click(getByText('Load Parameters'));
//   await waitFor(() => {
//     expect(getByLabelText(/Workspace/))
//       .toHaveValue(mockDatastack.args.workspace_dir)
//   });
// })

