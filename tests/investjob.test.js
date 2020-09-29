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
      const results = writeParametersToFile.mock.results[0].value
      expect(Object.keys(results)).toEqual(expect.arrayContaining(
        ['parameterSetPath', 'moduleName', 'relativePaths', 'args']
      ));
      const args = JSON.parse(results.args);
      const argKeys = Object.keys(args);
      const expectedKeys = Object.keys(spec.args);
      expectedKeys.push('n_workers'); // never in the spec, always in the args dict
      expect(argKeys).toEqual(expect.arrayContaining(expectedKeys));
      argKeys.forEach((key) => {
        expect(typeof args[key]).toBe('string');
      });
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
