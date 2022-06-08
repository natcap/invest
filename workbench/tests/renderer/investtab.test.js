import crypto from 'crypto';

import React from 'react';
import { ipcRenderer, shell } from 'electron';
import { render, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';

import InvestTab from '../../src/renderer/components/InvestTab';
import SetupTab from '../../src/renderer/components/SetupTab';
import {
  archiveDatastack,
  getSpec,
  saveToPython,
  writeParametersToFile,
  fetchValidation,
  fetchDatastackFromFile
} from '../../src/renderer/server_requests';
import InvestJob from '../../src/renderer/InvestJob';
import setupDialogs from '../../src/main/setupDialogs';
import setupOpenExternalUrl from '../../src/main/setupOpenExternalUrl';
import { removeIpcMainListeners } from '../../src/main/main';

// It's quite a pain to dynamically mock a const from a module,
// here we do it by importing as another object, then
// we can overwrite the object we want to mock later
// https://stackoverflow.com/questions/42977961/how-to-mock-an-exported-const-in-jest
import * as uiConfig from '../../src/renderer/ui_config';

jest.mock('../../src/renderer/server_requests');

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
  const tabID = crypto.randomBytes(4).toString('hex');
  const { ...utils } = render(
    <InvestTab
      job={job}
      tabID={tabID}
      investSettings={{ nWorkers: '-1', loggingLevel: 'INFO', taskgraphLoggingLevel: 'ERROR' }}
      saveJob={() => {}}
      updateJobProperties={() => {}}
    />
  );
  return utils;
}

// Because we mock UI_SPEC without using jest's API
// we alse need to a reset it without jest's API.
const { UI_SPEC } = uiConfig;
afterEach(() => {
  uiConfig.UI_SPEC = UI_SPEC;
});

describe('Run status Alert renders with data from a recent run', () => {
  const spec = {
    pyname: 'natcap.invest.foo',
    model_name: 'Foo Model',
    userguide: 'foo.html',
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
    uiConfig.UI_SPEC = mockUISpec(spec);
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

describe('Sidebar Buttons', () => {
  const spec = {
    pyname: 'natcap.invest.foo',
    model_name: 'Foo Model',
    userguide: 'foo.html',
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

  beforeEach(async () => {
    getSpec.mockResolvedValue(spec);
    fetchValidation.mockResolvedValue([]);
    uiConfig.UI_SPEC = mockUISpec(spec);
    setupOpenExternalUrl();
  });

  afterEach(() => {
    removeIpcMainListeners();
  });

  test('Save to JSON: requests endpoint with correct payload', async () => {
    const response = 'saved';
    writeParametersToFile.mockResolvedValue(response);
    const mockDialogData = { filePath: 'foo.json' };
    ipcRenderer.invoke.mockResolvedValueOnce(mockDialogData);

    const { findByText, findByRole } = renderInvestTab();
    const saveButton = await findByText('Save to JSON');
    userEvent.click(saveButton);

    expect(await findByRole('alert')).toHaveTextContent(response);
    const payload = writeParametersToFile.mock.calls[0][0];
    expect(Object.keys(payload)).toEqual(expect.arrayContaining(
      ['filepath', 'moduleName', 'relativePaths', 'args']
    ));
    Object.keys(payload).forEach((key) => {
      expect(payload[key]).not.toBeUndefined();
    });
    const args = JSON.parse(payload.args);
    const argKeys = Object.keys(args);
    expect(argKeys).toEqual(
      expect.arrayContaining(Object.keys(spec.args).concat('n_workers'))
    );
    argKeys.forEach((key) => {
      expect(typeof args[key]).toBe('string');
    });
    expect(writeParametersToFile).toHaveBeenCalledTimes(1);
  });

  test('Save to Python script: requests endpoint with correct payload', async () => {
    const response = 'saved';
    saveToPython.mockResolvedValue(response);
    const mockDialogData = { filePath: 'foo.py' };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);

    const { findByText, findByRole } = renderInvestTab();

    const saveButton = await findByText('Save to Python script');
    userEvent.click(saveButton);

    expect(await findByRole('alert')).toHaveTextContent(response);
    const payload = saveToPython.mock.calls[0][0];
    expect(Object.keys(payload)).toEqual(expect.arrayContaining(
      ['filepath', 'modelname', 'args']
    ));
    expect(typeof payload.filepath).toBe('string');
    expect(typeof payload.modelname).toBe('string');
    // guard against a common mistake of passing a model title
    expect(payload.modelname.split(' ')).toHaveLength(1);

    expect(payload.args).not.toBeUndefined();
    const args = JSON.parse(payload.args);
    const argKeys = Object.keys(args);
    expect(argKeys).toEqual(
      expect.arrayContaining(Object.keys(spec.args).concat('n_workers'))
    );
    argKeys.forEach((key) => {
      expect(typeof args[key]).toBe('string');
    });
    expect(saveToPython).toHaveBeenCalledTimes(1);
  });

  test('Save datastack: requests endpoint with correct payload', async () => {
    const response = 'saved';
    archiveDatastack.mockImplementation(() => new Promise(
      (resolve) => {
        setTimeout(() => resolve(response), 100);
      }
    ));
    const mockDialogData = { filePath: 'data.tgz' };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);

    const { findByText, findByRole, getByRole } = renderInvestTab();

    const saveButton = await findByText('Save datastack');
    userEvent.click(saveButton);

    expect(await findByRole('alert')).toHaveTextContent('archiving...');
    await waitFor(() => {
      expect(getByRole('alert')).toHaveTextContent(response);
    });
    const payload = archiveDatastack.mock.calls[0][0];
    expect(Object.keys(payload)).toEqual(expect.arrayContaining(
      ['filepath', 'moduleName', 'args']
    ));
    expect(typeof payload.filepath).toBe('string');
    expect(typeof payload.moduleName).toBe('string');
    // guard against a common mistake of passing a model title
    expect(payload.moduleName.split(' ')).toHaveLength(1);

    expect(payload.args).not.toBeUndefined();
    const args = JSON.parse(payload.args);
    const argKeys = Object.keys(args);
    expect(argKeys).toEqual(
      expect.arrayContaining(Object.keys(spec.args))
    );
    argKeys.forEach((key) => {
      expect(typeof args[key]).toBe('string');
    });
    expect(archiveDatastack).toHaveBeenCalledTimes(1);
  });

  test('Multiple Save Clicks: each triggers a unique alert', async () => {
    const response = 'saved';
    archiveDatastack.mockImplementation(() => new Promise(
      (resolve) => {
        setTimeout(() => resolve(response), 100);
      }
    ));
    saveToPython.mockResolvedValue(response);
    const mockDialogData = { filePath: 'foo' };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);

    const { findByText, getAllByRole, queryByRole } = renderInvestTab();

    const saveDatastackButton = await findByText('Save datastack');
    const savePythonButton = await findByText('Save to Python script');
    userEvent.click(saveDatastackButton);
    userEvent.click(savePythonButton);
    await waitFor(() => {
      expect(getAllByRole('alert')).toHaveLength(2);
    });
    await waitFor(() => {
      expect(queryByRole('alert')).toBeNull();
    }, { timeout: 3000 }); // alerts disappear after 2 seconds
  });

  test('Load parameters from file: loads parameters', async () => {
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

    // Render with a completed model run so we can navigate to Log Tab
    // and assert that Loading new params toggles back to Setup Tab
    const job = new InvestJob({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Model',
      status: 'success',
      argsValues: {},
      logfile: 'foo.txt',
      finalTraceback: '',
    });
    const { findByText, findByLabelText, findByRole } = renderInvestTab(job);

    userEvent.click(await findByRole('tab', { name: 'Log' }));
    const loadButton = await findByText('Load parameters from file');
    userEvent.click(loadButton);

    const setupTab = await findByRole('tab', { name: 'Setup' });
    expect(setupTab.classList.contains('active')).toBeTruthy();

    const input1 = await findByLabelText(spec.args.workspace.name);
    expect(input1).toHaveValue(mockDatastack.args.workspace);
    const input2 = await findByLabelText(spec.args.port.name);
    expect(input2).toHaveValue(mockDatastack.args.port);
  });

  test.each([
    ['Load parameters from file', 'loadParametersFromFile'],
    ['Save to Python script', 'savePythonScript'],
    ['Save to JSON', 'saveJsonFile'],
    ['Save datastack', 'saveDatastack']
  ])('%s: does nothing when canceled', async (label, method) => {
    // callback data if the OS dialog was canceled
    const mockDialogData = {
      filePaths: ['']
    };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);
    const spy = jest.spyOn(SetupTab.prototype, method);

    const { findByText } = renderInvestTab();

    const loadButton = await findByText(label);
    userEvent.click(loadButton);

    // Calls that would have triggered if a file was selected
    expect(spy).toHaveBeenCalledTimes(0);
  });

  test.each([
    ['Load parameters from file'],
    ['Save to Python script'],
    ['Save to JSON'],
    ['Save datastack'],
  ])('%s: has hover text', async (label) => {
    const {
      findByText,
      findByRole,
      queryByRole,
    } = renderInvestTab();
    const loadButton = await findByText(label);
    userEvent.hover(loadButton);
    expect(await findByRole('tooltip')).toBeInTheDocument();
    userEvent.unhover(loadButton);
    await waitFor(() => {
      expect(queryByRole('tooltip')).toBeNull();
    });
  });

  test('User Guide link opens externally', async () => {
    const { findByRole } = renderInvestTab();
    const link = await findByRole('link', { name: /user's guide/i });
    userEvent.click(link);
    await waitFor(() => {
      expect(shell.openExternal).toHaveBeenCalledTimes(1);
    });
  });

  test('Forum link opens externally', async () => {
    const { findByRole } = renderInvestTab();
    const link = await findByRole('link', { name: /frequently asked questions/i });
    userEvent.click(link);
    await waitFor(() => {
      expect(shell.openExternal).toHaveBeenCalledTimes(1);
    });
  });
});

describe('InVEST Run Button', () => {
  const spec = {
    pyname: 'natcap.invest.bar',
    model_name: 'Bar Model',
    userguide: 'bar.html',
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
    uiConfig.UI_SPEC = mockUISpec(spec);
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
