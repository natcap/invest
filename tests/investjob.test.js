import path from 'path';
import fs from 'fs';
import React from 'react';
import { remote } from 'electron';
import { fireEvent, render, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

import InvestJob from '../src/InvestJob';
import SetupTab from '../src/components/SetupTab';
jest.mock('../src/server_requests');
import { getSpec, saveToPython, writeParametersToFile,
         fetchValidation, fetchDatastackFromFile } from '../src/server_requests';
import { fileRegistry } from '../src/constants';
import Job from '../src/Job';

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
  const job = new Job({
    modelRunName: 'carbon',
    modelHumanName: 'Carbon Model',
  })
  job.setProperty('navID', 'carbon456asdf');
  const {
    findByText,
    findAllByText,
    findByLabelText,
    findByRole,
    queryAllByText,
    ...utils
  } = render(
    <InvestJob
      job={job}
      investExe=''
      investSettings={{ nWorkers: '-1', loggingLevel: 'INFO' }}
      saveJob={() => {}}
    />);
  return {
    findByText,
    findAllByText,
    findByLabelText,
    findByRole,
    queryAllByText,
    utils
  }
}

describe('Save InVEST Model Setup Buttons', () => {

  const spec = {
    module: 'natcap.invest.foo',
    model_name: 'Foo Model',
    args: {
      workspace: { 
        name: 'Workspace',
        type: 'directory',
        about: 'this is a workspace'
      },
      port: {
        name: 'Port',
        type: 'number',
      }
    }
  };

  const uiSpecFilePath = path.join(
    fileRegistry.INVEST_UI_DATA, `${spec.module}.json`
  );
  const uiSpec = {
    workspace: { order: 0.1 },
    port: { order: 'hidden' }
  }
  fs.writeFileSync(uiSpecFilePath, JSON.stringify(uiSpec));

  // args expected to be in the saved JSON / Python dictionary
  const expectedArgKeys = [];
  Object.keys(spec.args).forEach((key) => {
    if (uiSpec[key].order !== 'hidden') { expectedArgKeys.push(key) }
  })
  expectedArgKeys.push('n_workers'); // never in the spec, always in the args dict

  getSpec.mockResolvedValue(spec)
  fetchValidation.mockResolvedValue([]);

  afterAll(() => {
    fs.unlinkSync(uiSpecFilePath);
    jest.resetAllMocks();
    // Careful with reset because "resetting a spy results
    // in a function with no return value". I had been using spies to observe
    // function calls, but not to mock return values. Spies used for that 
    // purpose should be 'restored' not 'reset'. Do that inside the test as-needed.
  });

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

    const saveButton = await findByText('Save to JSON')
    fireEvent.click(saveButton);
    
    await waitFor(() => {
      const results = writeParametersToFile.mock.results[0].value
      expect(Object.keys(results)).toEqual(expect.arrayContaining(
        ['parameterSetPath', 'moduleName', 'relativePaths', 'args']
      ));
      Object.keys(results).forEach(key => {
        expect(results[key]).not.toBe(undefined);
      });
      const args = JSON.parse(results.args);
      const argKeys = Object.keys(args);
      expect(argKeys).toEqual(expect.arrayContaining(expectedArgKeys));
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

    const saveButton = await findByText('Save to Python script')
    fireEvent.click(saveButton);

    await waitFor(() => {
      const results = saveToPython.mock.results[0].value
      expect(Object.keys(results)).toEqual(expect.arrayContaining(
        ['filepath', 'modelname', 'pyname', 'args']
      ));
      Object.keys(results).forEach(key => {
        expect(results[key]).not.toBe(undefined);
      });
      const args = JSON.parse(results.args);
      const argKeys = Object.keys(args);
      expect(argKeys).toEqual(expect.arrayContaining(expectedArgKeys));
      argKeys.forEach((key) => {
        expect(typeof args[key]).toBe('string');
      });
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
    
    const saveButton = await findByText('Save to JSON')
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
    
    const saveButton = await findByText('Save to Python script')
    fireEvent.click(saveButton);

    // These are the calls that would have triggered if a file was selected
    expect(spy).toHaveBeenCalledTimes(0)
    spy.mockRestore() // restores to unmocked implementation
  })
})

describe('InVEST Run Button', () => {

  const spec = {
    module: 'natcap.invest.foo',
    model_name: 'Foo Model',
    args: {
      a: { 
        name: 'afoo', 
        type: 'freestyle_string'
      },
      b: {
        name: 'bfoo', 
        type: 'number'
      },
      c: {
        name: 'cfoo',
        type: 'csv'
      }
    }
  };

  beforeAll(() => {
    getSpec.mockResolvedValue(spec)
  })

  afterAll(() => {
    jest.resetAllMocks();
  });


  test('Changing inputs trigger validation & enable/disable Run', async () => {
    /*
    This tests that changes to input values trigger validation. 
    The fetchValidation return value is always mocked, but then this
    also tests that validation results correctly enable/disable the 
    Run button and display feedback messages on invalid inputs.
    */
    let invalidFeedback = 'is a required key'
    fetchValidation.mockResolvedValue([[['a', 'b'], invalidFeedback]])

    const {
      findByText,
      findAllByText,
      findByLabelText,
      findByRole,
      queryAllByText
    } = renderInvestJob();

    const runButton = await findByRole('button', {name: /Run/});
    expect(runButton).toBeDisabled();
    // The inputs are invalid so the invalid feedback message is present.
    // But, the inputs have not yet been touched, so the message is hidden
    // by CSS 'display: none'. Unfortunately, the bootstrap stylesheet is
    // not loaded in this testing DOM, so cannot assert the message is not visible.
    const invalidInputs = await findAllByText(invalidFeedback, { exact: false });
    invalidInputs.forEach(element => {
      expect(element).toBeInTheDocument()
      // Would be nice if these worked, but they do not:
      // expect(element).not.toBeVisible()
      // expect(element).toHaveStyle('display: none')
    });

    const a = await findByLabelText(RegExp(`${spec.args.a.name}`))
    const b = await findByLabelText(RegExp(`${spec.args.b.name}`))
    const c = await findByLabelText(RegExp(`${spec.args.c.name}`))

    // These new values will be valid - Run should enable
    fetchValidation.mockResolvedValue([])
    fireEvent.change(a, { target: { value: 'foo' } })
    fireEvent.change(b, { target: { value: 1 } })
    await waitFor(() => {
      expect(runButton).toBeEnabled();
    })
    // Now that inputs are valid, feedback message should be cleared:
    // Note: Can't put this inside wait - it will timeout waiting to be not null.
    // But it does rely on waiting for the change event to propogate. 
    // Putting it after the above `await` works.
    queryAllByText(invalidFeedback, { exact: false }).forEach(element => {
      expect(element).toBeNull()
    })

    // This new value will be invalid - Run should disable again
    invalidFeedback = 'must be a number';
    fetchValidation.mockResolvedValue([[['b'], invalidFeedback]])
    fireEvent.change(b, { target: { value: 'one' } })  // triggers validation
    await waitFor(() => {
      expect(runButton).toBeDisabled();
    })
    expect(await findByText(invalidFeedback, { exact: false })).toBeInTheDocument()
    // fetchValidation.mockReset();
  })

})

