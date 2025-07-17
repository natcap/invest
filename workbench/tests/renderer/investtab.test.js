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
  fetchDatastackFromFile,
  fetchArgsEnabled,
  getDynamicDropdowns
} from '../../src/renderer/server_requests';
import InvestJob from '../../src/renderer/InvestJob';
import setupDialogs from '../../src/main/setupDialogs';
import setupOpenExternalUrl from '../../src/main/setupOpenExternalUrl';
import setupOpenLocalHtml from '../../src/main/setupOpenLocalHtml';
import { removeIpcMainListeners } from '../../src/main/main';
import { ipcMainChannels } from '../../src/main/ipcMainChannels';

jest.mock('../../src/renderer/server_requests');

const DEFAULT_JOB = new InvestJob({
  modelID: 'carbon',
  modelTitle: 'Carbon Model',
});

function renderInvestTab(job = DEFAULT_JOB) {
  const tabID = crypto.randomBytes(4).toString('hex');
  const { ...utils } = render(
    <InvestTab
      job={job}
      tabID={tabID}
      updateJobProperties={() => {}}
      investList={{
        carbon: { modelTitle: 'Carbon Model', type: 'core' },
        foo: { modelTitle: 'Foo Model', type: 'plugin' },
      }}
    />
  );
  return utils;
}

describe('Run status Alert renders with status from a recent run', () => {
  const spec = {
    pyname: 'natcap.invest.foo',
    model_title: 'Foo Model',
    userguide: 'foo.html',
    input_field_order: [['workspace']],
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
    fetchArgsEnabled.mockResolvedValue({ workspace: true });
    getDynamicDropdowns.mockResolvedValue({});
    setupDialogs();
  });

  afterEach(() => {
    removeIpcMainListeners();
    jest.resetAllMocks();
  });

  test.each([
    ['success', 'Model Complete'],
    ['error', 'Error: see log for details'],
    ['canceled', 'Run Canceled'],
  ])('status message displays on %s', async (status, message) => {
    // mock a defined value for ipcMainChannels.INVEST_SERVE so the tab loads
    ipcRenderer.invoke.mockResolvedValueOnce('foo');

    const job = new InvestJob({
      modelID: 'carbon',
      modelTitle: 'Carbon Model',
      status: status,
      argsValues: {},
      logfile: 'foo.txt',
      type: 'core',
    });

    const { findByRole } = renderInvestTab(job);
    expect(await findByRole('alert'))
      .toHaveTextContent(message);
  });

  test.each([
    'success', 'error', 'canceled',
  ])('Open Workspace button is available on %s', async (status) => {
    // mock a defined value for ipcMainChannels.INVEST_SERVE so the tab loads
    ipcRenderer.invoke.mockResolvedValueOnce('foo');
    const job = new InvestJob({
      modelID: 'carbon',
      modelTitle: 'Carbon Model',
      status: status,
      argsValues: {},
      logfile: 'foo.txt',
    });

    const { findByRole } = renderInvestTab(job);
    const openWorkspaceBtn = await findByRole('button', { name: 'Open Workspace' });
    expect(openWorkspaceBtn).toBeTruthy();
  });
});

describe('Open Workspace button', () => {
  const spec = {
    pyname: 'natcap.invest.foo',
    model_name: 'Foo Model',
    userguide: 'foo.html',
    args: {},
    input_field_order: [],
  };

  const baseJob = {
    ...DEFAULT_JOB,
    status: 'success',
  };

  beforeEach(() => {
    getSpec.mockResolvedValue(spec);
    fetchValidation.mockResolvedValue([]);
    getDynamicDropdowns.mockResolvedValue({});
    setupDialogs();
  });

  afterEach(() => {
    removeIpcMainListeners();
  });

  test('should open workspace', async () => {
    const job = {
      ...baseJob,
      argsValues: {
        workspace_dir: '/workspace',
      },
    };

    jest.spyOn(ipcRenderer, 'invoke');

    const { findByRole } = renderInvestTab(job);
    const openWorkspaceBtn = await findByRole('button', { name: 'Open Workspace' })
    openWorkspaceBtn.click();

    expect(ipcRenderer.invoke).toHaveBeenCalledTimes(1);
    expect(ipcRenderer.invoke).toHaveBeenCalledWith(ipcMainChannels.OPEN_PATH, job.argsValues.workspace_dir);
  });

  test('should present an error message to the user if workspace cannot be opened (e.g., if it does not exist)', async () => {
    const job = {
      ...baseJob,
      status: 'error',
      argsValues: {
        workspace_dir: '/nonexistent-workspace',
      },
    };

    jest.spyOn(ipcRenderer, 'invoke');
    ipcRenderer.invoke.mockResolvedValue('Error opening workspace');

    const { findByRole } = renderInvestTab(job);
    const openWorkspaceBtn = await findByRole('button', { name: 'Open Workspace' });
    openWorkspaceBtn.click();

    expect(ipcRenderer.invoke).toHaveBeenCalledTimes(1);
    expect(ipcRenderer.invoke).toHaveBeenCalledWith(ipcMainChannels.OPEN_PATH, job.argsValues.workspace_dir);

    const errorModal = await findByRole('dialog', { name: 'Error opening workspace'});
    expect(errorModal).toBeTruthy();
  });
});

describe('Sidebar Buttons', () => {
  const spec = {
    model_id: 'foo',
    pyname: 'natcap.invest.foo',
    model_title: 'Foo Model',
    userguide: 'foo.html',
    input_field_order: [['workspace', 'port']],
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
    fetchArgsEnabled.mockResolvedValue({ workspace: true, port: true });
    getDynamicDropdowns.mockResolvedValue({});
    setupOpenExternalUrl();
    setupOpenLocalHtml();
    ipcRenderer.invoke.mockImplementation((channel) => {
      if (channel === ipcMainChannels.SHOW_SAVE_DIALOG) {
        return { canceled: false, filePath: 'foo.json' };
      }
      return {};
    });
  });

  afterEach(() => {
    removeIpcMainListeners();
  });

  test('Save to JSON: requests endpoint with correct payload', async () => {
    const response = {
      message: 'saved',
      error: false,
    };
    writeParametersToFile.mockResolvedValueOnce(response);
    const mockDialogData = { canceled: false, filePath: 'foo.json' };
    ipcRenderer.invoke.mockResolvedValueOnce(mockDialogData);

    const { findByText, findByLabelText, findByRole } = renderInvestTab();
    const saveAsButton = await findByText('Save as...');
    await userEvent.click(saveAsButton);
    const jsonOption = await findByLabelText((content) => content.startsWith('Parameters only'));
    await userEvent.click(jsonOption);
    const saveButton = await findByRole('button', { name: 'Save' });
    await userEvent.click(saveButton);

    const payload = writeParametersToFile.mock.calls[0][0];
    expect(Object.keys(payload)).toEqual(expect.arrayContaining(
      ['filepath', 'model_id', 'relativePaths', 'args']
    ));
    Object.keys(payload).forEach((key) => {
      expect(payload[key]).toBeDefined();
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
    saveToPython.mockResolvedValueOnce(response);
    const mockDialogData = { canceled: false, filePath: 'foo.py' };
    ipcRenderer.invoke.mockResolvedValueOnce(mockDialogData);

    const { findByText, findByLabelText, findByRole } = renderInvestTab();
    const saveAsButton = await findByText('Save as...');
    await userEvent.click(saveAsButton);
    const pythonOption = await findByLabelText((content) => content.startsWith('Python script'));
    await userEvent.click(pythonOption);
    const saveButton = await findByRole('button', { name: 'Save' });
    await userEvent.click(saveButton);

    const payload = saveToPython.mock.calls[0][0];
    expect(Object.keys(payload)).toEqual(expect.arrayContaining(
      ['filepath', 'model_id', 'args']
    ));
    expect(typeof payload.filepath).toBe('string');
    expect(typeof payload.model_id).toBe('string');
    // guard against a common mistake of passing a model title
    expect(payload.model_id.split(' ')).toHaveLength(1);

    expect(payload.args).toBeDefined();
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
    const response = {
      message: 'saved',
      error: false,
    };
    archiveDatastack.mockResolvedValueOnce(response);
    const mockDialogData = { canceled: false, filePath: 'data.tgz' };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);

    const { findByText, findByLabelText, findByRole } = renderInvestTab();
    const saveAsButton = await findByText('Save as...');
    await userEvent.click(saveAsButton);
    const datastackOption = await findByLabelText((content) => content.startsWith('Parameters and data'));
    await userEvent.click(datastackOption);
    const saveButton = await findByRole('button', { name: 'Save' });
    await userEvent.click(saveButton);

    const payload = archiveDatastack.mock.calls[0][0];
    expect(Object.keys(payload)).toEqual(expect.arrayContaining(
      ['filepath', 'model_id', 'args']
    ));
    expect(typeof payload.filepath).toBe('string');
    expect(typeof payload.model_id).toBe('string');
    // guard against a common mistake of passing a model title
    expect(payload.model_id.split(' ')).toHaveLength(1);

    expect(payload.args).toBeDefined();
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

  test.each([
    [
      'Parameters only',
      writeParametersToFile,
      {message: 'Parameter set saved', error: false}
    ],
    [
      'Parameters and data',
      archiveDatastack,
      {message: 'Datastack archive created', error: false}
    ],
    [
      'Python script',
      saveToPython,
      'Python script saved'
    ]
  ])('%s: renders success message', async (label, method, response) => {
    ipcRenderer.invoke.mockResolvedValueOnce({canceled: false, filePath: 'example.txt'});
    if (method == archiveDatastack) {
      method.mockImplementationOnce(() => new Promise(
        (resolve) => {
          setTimeout(() => resolve(response), 500);
        }
      ));
    } else {
      method.mockResolvedValueOnce(response);
    }

    const { findByText, findByLabelText, findByRole } = renderInvestTab();
    const saveAsButton = await findByText('Save as...');
    await userEvent.click(saveAsButton);
    const option = await findByLabelText((content) => content.startsWith(label));
    await userEvent.click(option);
    const saveButton = await findByRole('button', { name: 'Save' });
    await userEvent.click(saveButton);

    const saveAlert = await findByRole('alert');
    if (method == archiveDatastack) {
      expect(saveAlert).toHaveTextContent('archiving...');
    }
    await waitFor(() => {
      expect(saveAlert).toHaveTextContent(response.message ?? response);
    });
    expect(saveAlert).toHaveClass('alert-success');
  });

  test.each([
    [
      'Parameters only',
      writeParametersToFile,
      {message: 'Error saving parameter set', error: true}
    ],
    [
      'Parameters and data',
      archiveDatastack,
      {message: 'Error creating datastack archive', error: true}
    ],
  ])('%s: renders error message', async (label, method, response) => {
    ipcRenderer.invoke.mockResolvedValueOnce({canceled: false, filePath: 'example.txt'});
    method.mockResolvedValueOnce(response);

    const { findByText, findByLabelText, findByRole } = renderInvestTab();
    const saveAsButton = await findByText('Save as...');
    await userEvent.click(saveAsButton);
    const option = await findByLabelText((content) => content.startsWith(label));
    await userEvent.click(option);
    const saveButton = await findByRole('button', { name: 'Save' });
    await userEvent.click(saveButton);

    const saveAlert = await findByRole('alert');
    expect(saveAlert).toHaveTextContent(response.message);
    expect(saveAlert).toHaveClass('alert-danger');
  });

  test('Save errors are cleared when save modal opens', async () => {
    ipcRenderer.invoke.mockResolvedValueOnce({canceled: false, filePath: 'example.txt'});
    writeParametersToFile.mockResolvedValueOnce({message: 'Error saving parameter set', error: true});

    // Trigger error alert
    const { findByText, findByLabelText, findByRole, queryByRole } = renderInvestTab();
    const saveAsButton = await findByText('Save as...');
    await userEvent.click(saveAsButton);
    const jsonOption = await findByLabelText((content) => content.startsWith('Parameters only'));
    await userEvent.click(jsonOption);
    const saveButton = await findByRole('button', { name: 'Save' });
    await userEvent.click(saveButton);
    expect(await findByRole('alert')).toHaveClass('alert-danger');

    // Re-open save modal
    await userEvent.click(saveAsButton);
    expect(queryByRole('alert')).toBe(null);
  });

  test('Load parameters from file: loads parameters', async () => {
    const mockDatastack = {
      model_id: 'foo',
      args: {
        workspace: 'myworkspace',
        port: '9999',
      },
    };
    fetchDatastackFromFile.mockResolvedValue(mockDatastack);
    const mockDialogData = {
      canceled: false,
      filePaths: ['foo.json'],
    };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);

    // Render with a completed model run so we can navigate to Log Tab
    // and assert that Loading new params toggles back to Setup Tab
    const job = new InvestJob({
      modelID: 'foo',
      modelTitle: 'Foo Model',
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

    const input1 = await findByLabelText((content) => content.startsWith(spec.args.workspace.name));
    expect(input1).toHaveValue(mockDatastack.args.workspace);
    const input2 = await findByLabelText((content) => content.startsWith(spec.args.port.name));
    expect(input2).toHaveValue(mockDatastack.args.port);
  });

  test('Load parameters from datastack: tgz asks for extract location', async () => {
    const mockDatastack = {
      model_id: 'carbon',
      args: {
        workspace: 'myworkspace',
        port: '9999',
      },
    };
    fetchDatastackFromFile.mockResolvedValue(mockDatastack);
    const mockDialogData = {
      canceled: false,
      filePaths: ['foo.tgz'],
    };
    ipcRenderer.invoke.mockImplementation((channel, options) => {
      if (channel === ipcMainChannels.SHOW_OPEN_DIALOG) {
        return Promise.resolve(mockDialogData);
      }
      if (channel === ipcMainChannels.CHECK_FILE_PERMISSIONS) {
        return Promise.resolve(true);
      }
      return Promise.resolve(undefined);
    });

    const job = new InvestJob({
      modelID: 'carbon',
      modelTitle: 'Carbon Model',
      argsValues: {},
    });
    const { findByText, findByLabelText } = renderInvestTab(job);

    const loadButton = await findByText('Load parameters from file');
    await userEvent.click(loadButton);

    const input1 = await findByLabelText((content) => content.startsWith(spec.args.workspace.name));
    expect(input1).toHaveValue(mockDatastack.args.workspace);
    const input2 = await findByLabelText((content) => content.startsWith(spec.args.port.name));
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

  test('Plugin Documentation link points to userguide URL from plugin model spec and invokes OPEN_EXTERNAL_URL', async () => {
    const spy = jest.spyOn(ipcRenderer, 'send')
      .mockImplementation(() => Promise.resolve());

    const { findByRole, queryByRole } = renderInvestTab(new InvestJob({
      modelID: 'foo',
      modelTitle: 'Foo Model',
    }));
    const ugLink = await queryByRole('link', { name: /user's guide/i });
    expect(ugLink).toBeNull();
    const docsLink = await findByRole('link', { name: /plugin documentation/i });
    expect(docsLink.getAttribute('href')).toEqual(spec.userguide);
    await userEvent.click(docsLink);
    await waitFor(() => {
      const calledChannels = spy.mock.calls.map(call => call[0]);
      expect(calledChannels).toContain(ipcMainChannels.OPEN_EXTERNAL_URL);
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
    model_title: 'Bar Model',
    userguide: 'bar.html',
    input_field_order: [['a', 'b', 'c']],
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
    fetchArgsEnabled.mockResolvedValue({ a: true, b: true, c: true });
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

    const a = await findByLabelText((content) => content.startsWith(spec.args.a.name));
    const b = await findByLabelText((content) => content.startsWith(spec.args.b.name));

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
