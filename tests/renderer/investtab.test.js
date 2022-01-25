import React from 'react';
import { ipcRenderer, shell } from 'electron';
import { render, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';

import InvestTab from '../../src/renderer/components/InvestTab';
import SetupTab from '../../src/renderer/components/SetupTab';
import {
  getSpec,
  saveToPython,
  writeParametersToFile,
  fetchValidation,
  fetchDatastackFromFile
} from '../../src/renderer/server_requests';
import InvestJob from '../../src/renderer/InvestJob';
import setupDialogs from '../../src/main/setupDialogs';
import { removeIpcMainListeners } from '../../src/main/main';

// mock out the global gettext function - avoid setting up translation
global.window._ = x => x;

jest.mock('../../src/renderer/server_requests');

const UI_CONFIG_PATH = '../../src/renderer/ui_config';

const DEFAULT_JOB = new InvestJob({
  modelRunName: 'carbon',
  modelHumanName: 'Carbon Model',
});

function mockUISpec(spec) {
  return {
    [DEFAULT_JOB.modelRunName]: { order: [Object.keys(spec.args)] }
  };
}

function renderInvestTab(job = DEFAULT_JOB) {
  const { ...utils } = render(
    <InvestTab
      job={job}
      jobID="carbon456asdf"
      investSettings={{ nWorkers: '-1', loggingLevel: 'INFO' }}
      saveJob={() => {}}
      updateJobProperties={() => {}}
    />
  );
  return utils;
}

describe('Sidebar Alert renders with data from a recent run', () => {
  const spec = {
    pyname: 'natcap.invest.foo',
    model_name: 'Foo Model',
    args: {
      workspace: {
        name: 'Workspace',
        type: 'directory',
        about: 'this is a workspace',
      },
    },
  };

  beforeEach(() => {
    getSpec.mockResolvedValue(spec);
    fetchValidation.mockResolvedValue([]);
    const mockSpec = spec; // jest.mock not allowed to ref out-of-scope var
    jest.mock(UI_CONFIG_PATH, () => mockUISpec(mockSpec));
    setupDialogs();
  });

  afterEach(() => {
    removeIpcMainListeners();
  });

  test('final Traceback displays', async () => {
    const job = new InvestJob({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Model',
      status: 'error',
      argsValues: {},
      logfile: 'foo.txt',
      finalTraceback: 'ValueError:',
    });

    const { findByRole } = renderInvestTab(job);
    expect(await findByRole('alert'))
      .toHaveTextContent(job.finalTraceback);
  });

  test('Model Complete displays if status was success', async () => {
    const job = new InvestJob({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Model',
      status: 'success',
      argsValues: {},
      logfile: 'foo.txt',
      finalTraceback: '',
    });

    const { findByRole } = renderInvestTab(job);
    expect(await findByRole('alert'))
      .toHaveTextContent('Model Complete');
  });

  test('Model Complete displays even with non-fatal stderr', async () => {
    const job = new InvestJob({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Model',
      status: 'success',
      argsValues: {},
      logfile: 'foo.txt',
      finalTraceback: 'Error that did not actually raise an exception',
    });

    const { findByRole, queryByText } = renderInvestTab(job);
    expect(await findByRole('alert'))
      .toHaveTextContent('Model Complete');
    expect(queryByText(job.finalTraceback))
      .toBeNull();
  });

  test('Open Workspace button is available on success', async () => {
    const job = new InvestJob({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Model',
      status: 'success',
      argsValues: {},
      logfile: 'foo.txt',
    });

    const { findByRole } = renderInvestTab(job);
    const openWorkspace = await findByRole('button', { name: 'Open Workspace' })
    openWorkspace.click();
    expect(shell.showItemInFolder).toHaveBeenCalledTimes(1);
  });

  test('Open Workspace button is available on error', async () => {
    const job = new InvestJob({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Model',
      status: 'error',
      argsValues: {},
      logfile: 'foo.txt',
    });

    const { findByRole } = renderInvestTab(job);
    const openWorkspace = await findByRole('button', { name: 'Open Workspace' })
    openWorkspace.click();
    expect(shell.showItemInFolder).toHaveBeenCalledTimes(1);
  });
});

describe('Save InVEST Model Setup Buttons', () => {
  const spec = {
    pyname: 'natcap.invest.foo',
    model_name: 'Foo Model',
    args: {
      workspace: {
        name: 'Workspace',
        type: 'directory',
        about: 'this is a workspace',
      },
      port: {
        name: 'Port',
        type: 'number',
      },
    },
  };

  // args expected to be in the saved JSON / Python dictionary
  const expectedArgKeys = ['workspace', 'n_workers'];

  beforeEach(async () => {
    getSpec.mockResolvedValue(spec);
    fetchValidation.mockResolvedValue([]);
    const mockSpec = spec;
    jest.mock(UI_CONFIG_PATH, () => mockUISpec(mockSpec));
  });

  test('SaveParametersButton: requests endpoint with correct payload', async () => {
    // mock the server call, instead just returning
    // the payload. At least we can assert the payload is what
    // the flask endpoint needs to build the json file.
    writeParametersToFile.mockImplementation(
      (payload) => payload
    );
    const mockDialogData = { filePath: 'foo.json' };
    ipcRenderer.invoke.mockResolvedValueOnce(mockDialogData);

    const { findByText } = renderInvestTab();
    const saveButton = await findByText('Save to JSON');
    userEvent.click(saveButton);

    await waitFor(() => {
      const results = writeParametersToFile.mock.results[0].value;
      expect(Object.keys(results)).toEqual(expect.arrayContaining(
        ['parameterSetPath', 'moduleName', 'relativePaths', 'args']
      ));
      Object.keys(results).forEach((key) => {
        expect(results[key]).not.toBeUndefined();
      });
      const args = JSON.parse(results.args);
      const argKeys = Object.keys(args);
      expect(argKeys).toEqual(expect.arrayContaining(expectedArgKeys));
      argKeys.forEach((key) => {
        expect(typeof args[key]).toBe('string');
      });
      expect(writeParametersToFile).toHaveBeenCalledTimes(1);
    });
  });

  test('SavePythonButton: requests endpoint with correct payload', async () => {
    // mock the server call, instead just returning
    // the payload. At least we can assert the payload is what
    // the flask endpoint needs to build the python script.
    saveToPython.mockImplementation(
      (payload) => payload
    );
    const mockDialogData = { filePath: 'foo.py' };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);

    const { findByText } = renderInvestTab();

    const saveButton = await findByText('Save to Python script');
    userEvent.click(saveButton);

    await waitFor(() => {
      const results = saveToPython.mock.results[0].value;
      expect(Object.keys(results)).toEqual(expect.arrayContaining(
        ['filepath', 'modelname', 'args']
      ));
      expect(typeof results.filepath).toBe('string');
      expect(typeof results.modelname).toBe('string');
      // guard against a common mistake of passing a model title
      expect(results.modelname.split(' ')).toHaveLength(1);

      expect(results.args).not.toBeUndefined();
      const args = JSON.parse(results.args);
      const argKeys = Object.keys(args);
      expect(argKeys).toEqual(expect.arrayContaining(expectedArgKeys));
      argKeys.forEach((key) => {
        expect(typeof args[key]).toBe('string');
      });
      expect(saveToPython).toHaveBeenCalledTimes(1);
    });
  });

  test('Load Parameters Button: loads parameters', async () => {
    const mockDatastack = {
      module_name: spec.pyname,
      args: {
        workspace: 'myworkspace',
        port: '9999',
      },
    };
    fetchDatastackFromFile.mockResolvedValue(mockDatastack);
    const mockDialogData = {
      filePaths: ['foo.json']
    };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);
    const { findByText, findByLabelText, queryByText } = renderInvestTab();

    const loadButton = await findByText('Load parameters from file');
    // test the tooltip before we click
    userEvent.hover(loadButton);
    const hoverText = 'Browse to a datastack (.json) or InVEST logfile (.txt)';
    expect(await findByText(hoverText)).toBeInTheDocument();
    userEvent.unhover(loadButton);
    await waitFor(() => {
      expect(queryByText(hoverText)).toBeNull();
    });
    userEvent.click(loadButton);

    const input1 = await findByLabelText(spec.args.workspace.name);
    expect(input1).toHaveValue(mockDatastack.args.workspace);
    const input2 = await findByLabelText(spec.args.port.name);
    expect(input2).toHaveValue(mockDatastack.args.port);
  });

  test('SaveParametersButton: Dialog callback does nothing when canceled', async () => {
    // this resembles the callback data if the dialog is canceled instead of
    // a save file selected.
    const mockDialogData = {
      filePath: ''
    };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);
    const spy = jest.spyOn(SetupTab.prototype, 'saveJsonFile');

    const { findByText } = renderInvestTab();

    const saveButton = await findByText('Save to JSON');
    userEvent.click(saveButton);

    // These are the calls that would have triggered if a file was selected
    expect(spy).toHaveBeenCalledTimes(0);
  });

  test('SavePythonButton: Dialog callback does nothing when canceled', async () => {
    // this resembles the callback data if the dialog is canceled instead of
    // a save file selected.
    const mockDialogData = {
      filePath: ''
    };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);
    const spy = jest.spyOn(SetupTab.prototype, 'savePythonScript');

    const { findByText } = renderInvestTab();

    const saveButton = await findByText('Save to Python script');
    userEvent.click(saveButton);

    // These are the calls that would have triggered if a file was selected
    expect(spy).toHaveBeenCalledTimes(0);
  });

  test('Load Parameters Button: does nothing when canceled', async () => {
    // this resembles the callback data if the dialog is canceled instead of
    // a save file selected.
    const mockDialogData = {
      filePaths: ['']
    };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);
    const spy = jest.spyOn(SetupTab.prototype, 'loadParametersFromFile');

    const { findByText } = renderInvestTab();

    const loadButton = await findByText('Load parameters from file');
    userEvent.click(loadButton);

    // These are the calls that would have triggered if a file was selected
    expect(spy).toHaveBeenCalledTimes(0);
  });
});

describe('InVEST Run Button', () => {
  const spec = {
    pyname: 'natcap.invest.bar',
    model_name: 'Bar Model',
    args: {
      a: {
        name: 'abar',
        type: 'freestyle_string',
      },
      b: {
        name: 'bbar',
        type: 'number',
      },
      c: {
        name: 'cbar',
        type: 'csv',
      },
    },
  };

  beforeEach(() => {
    getSpec.mockResolvedValue(spec);
    const mockSpec = spec;
    jest.mock(UI_CONFIG_PATH, () => mockUISpec(mockSpec));
  });

  test('Changing inputs trigger validation & enable/disable Run', async () => {
    let invalidFeedback = 'is a required key';
    fetchValidation.mockResolvedValue([[['a', 'b'], invalidFeedback]]);

    const {
      findByLabelText,
      findByRole,
    } = renderInvestTab();

    const runButton = await findByRole('button', { name: /Run/ });
    expect(runButton).toBeDisabled();

    const a = await findByLabelText(`${spec.args.a.name}`);
    const b = await findByLabelText(`${spec.args.b.name}`);

    expect(a).toHaveClass('is-invalid');
    expect(b).toHaveClass('is-invalid');

    // These new values will be valid - Run should enable
    fetchValidation.mockResolvedValue([]);
    userEvent.type(a, 'foo');
    userEvent.type(b, '1');
    await waitFor(() => {
      expect(runButton).toBeEnabled();
    });

    // This new value will be invalid - Run should disable again
    invalidFeedback = 'must be a number';
    fetchValidation.mockResolvedValue([[['b'], invalidFeedback]]);
    userEvent.type(b, 'one');
    await waitFor(() => {
      expect(runButton).toBeDisabled();
    });
  });
});
