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
import setupOpenLocalHtml from '../../src/main/setupOpenLocalHtml';
import { removeIpcMainListeners } from '../../src/main/main';
import { ipcMainChannels } from '../../src/main/ipcMainChannels';

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

describe('Run status Alert renders with status from a recent run', () => {
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

  test.each([
    ['success', 'Model Complete'],
    ['error', 'Error: see log for details'],
    ['canceled', 'Run Canceled'],
  ])('status message displays on %s', async (status, message) => {
    const job = new InvestJob({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Model',
      status: status,
      argsValues: {},
      logfile: 'foo.txt',
    });

    const { findByRole } = renderInvestTab(job);
    expect(await findByRole('alert'))
      .toHaveTextContent(message);
  });

  test.each([
    'success', 'error', 'canceled',
  ])('Open Workspace button is available on %s', async (status) => {
    const job = new InvestJob({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Model',
      status: status,
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
    setupOpenLocalHtml();
  });

  afterEach(() => {
    removeIpcMainListeners();
  });

  test('Save to JSON: requests endpoint with correct payload', async () => {
    const response = 'saved';
    writeParametersToFile.mockResolvedValue(response);
    const mockDialogData = { canceled: false, filePath: 'foo.json' };
    ipcRenderer.invoke.mockResolvedValueOnce(mockDialogData);

    const { findByText, findByLabelText, findByRole } = renderInvestTab();
    const saveAsButton = await findByText('Save as...');
    await userEvent.click(saveAsButton);
    const jsonOption = await findByLabelText((content, element) => content.startsWith('Parameters only'));
    await userEvent.click(jsonOption);
    const saveButton = await findByRole('button', { name: 'Save' });
    await userEvent.click(saveButton);

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
      expect.arrayContaining(Object.keys(spec.args))
    );
    argKeys.forEach((key) => {
      expect(typeof args[key]).toBe('string');
    });
    expect(writeParametersToFile).toHaveBeenCalledTimes(1);
  });

  test('Save to Python script: requests endpoint with correct payload', async () => {
    const response = 'saved';
    saveToPython.mockResolvedValue(response);
    const mockDialogData = { canceled: false, filePath: 'foo.py' };
    ipcRenderer.invoke.mockResolvedValueOnce(mockDialogData);

    const { findByText, findByLabelText, findByRole } = renderInvestTab();
    const saveAsButton = await findByText('Save as...');
    await userEvent.click(saveAsButton);
    const pythonOption = await findByLabelText((content, element) => content.startsWith('Python script'));
    await userEvent.click(pythonOption);
    const saveButton = await findByRole('button', { name: 'Save' });
    await userEvent.click(saveButton);

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
      expect.arrayContaining(Object.keys(spec.args))
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
        setTimeout(() => resolve(response), 500);
      }
    ));
    const mockDialogData = { canceled: false, filePath: 'data.tgz' };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);

    const { findByText, findByLabelText, findByRole, getByRole } = renderInvestTab();
    const saveAsButton = await findByText('Save as...');
    await userEvent.click(saveAsButton);
    const datastackOption = await findByLabelText((content, element) => content.startsWith('Parameters and data'));
    await userEvent.click(datastackOption);
    const saveButton = await findByRole('button', { name: 'Save' });
    await userEvent.click(saveButton);

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
      canceled: false,
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
    });
    const { findByText, findByLabelText, findByRole } = renderInvestTab(job);

    await userEvent.click(await findByRole('tab', { name: 'Log' }));
    const loadButton = await findByText('Load parameters from file');
    await userEvent.click(loadButton);

    const setupTab = await findByRole('tab', { name: 'Setup' });
    expect(setupTab.classList.contains('active')).toBeTruthy();

    const input1 = await findByLabelText(spec.args.workspace.name);
    expect(input1).toHaveValue(mockDatastack.args.workspace);
    const input2 = await findByLabelText(spec.args.port.name);
    expect(input2).toHaveValue(mockDatastack.args.port);
  });

  test('Load parameters from file does nothing when canceled', async () => {
    // callback data if the OS dialog was canceled
    const mockDialogData = {
      canceled: true,
      filePaths: []
    };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);
    const spy = jest.spyOn(SetupTab.WrappedComponent.prototype, 'loadParametersFromFile');

    const { findByText } = renderInvestTab();

    const loadButton = await findByText('Load parameters from file');
    await userEvent.click(loadButton);

    // Calls that would have triggered if a file was selected
    expect(spy).toHaveBeenCalledTimes(0);
  });

  test.each([
    ['Parameters only', 'saveJsonFile'],
    ['Parameters and data', 'saveDatastack'],
    ['Python script', 'savePythonScript']
  ])('%s: does nothing when canceled', async (label, method) => {
    // callback data if the OS dialog was canceled
    const mockDialogData = {
      canceled: true,
      filePaths: []
    };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);
    const spy = jest.spyOn(SetupTab.WrappedComponent.prototype, method);

    const { findByText, findByLabelText, findByRole } = renderInvestTab();
    const saveAsButton = await findByText('Save as...');
    await userEvent.click(saveAsButton);
    const option = await findByLabelText((content, element) => content.startsWith(label));
    await userEvent.click(option);
    const saveButton = await findByRole('button', { name: 'Save' });
    await userEvent.click(saveButton);

    // Calls that would have triggered if a file was selected
    expect(spy).toHaveBeenCalledTimes(0);
  });

  test('Load parameters button has hover text', async () => {
    const {
      findByText,
      findByRole,
      queryByRole,
    } = renderInvestTab();
    const loadButton = await findByText('Load parameters from file');
    await userEvent.hover(loadButton);
    expect(await findByRole('tooltip')).toBeInTheDocument();
    await userEvent.unhover(loadButton);
    await waitFor(() => {
      expect(queryByRole('tooltip')).toBeNull();
    });
  });

  test('User Guide link sends IPC to main', async () => {
    // It seemed impossible to spy on an instance of BrowserWindow
    // and its call to .loadUrl(), given our setup in __mocks__/electron.js,
    // so this will have to suffice:
    const spy = jest.spyOn(ipcRenderer, 'send')
      .mockImplementation(() => Promise.resolve());

    const { findByRole } = renderInvestTab();
    const link = await findByRole('link', { name: /user's guide/i });
    await userEvent.click(link);
    await waitFor(() => {
      const calledChannels = spy.mock.calls.map(call => call[0]);
      expect(calledChannels).toContain(ipcMainChannels.OPEN_LOCAL_HTML);
    });
  });

  test('Forum link opens externally', async () => {
    const { findByRole } = renderInvestTab();
    const link = await findByRole('link', { name: /frequently asked questions/i });
    await userEvent.click(link);
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
    await userEvent.type(a, 'foo');
    await userEvent.type(b, '1');
    await waitFor(() => {
      expect(runButton).toBeEnabled();
    });

    // This new value will be invalid - Run should disable again
    invalidFeedback = 'must be a number';
    fetchValidation.mockResolvedValue([[['b'], invalidFeedback]]);
    await userEvent.type(b, 'one');
    await waitFor(() => {
      expect(runButton).toBeDisabled();
    });
  });
});
