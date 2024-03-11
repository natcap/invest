import fs from 'fs';
import path from 'path';
import events from 'events';
import { spawn, exec } from 'child_process';
import Stream from 'stream';

import React from 'react';
import {
  render, waitFor, within, act
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import fetch from 'node-fetch';

import {
  setupInvestRunHandlers,
  setupInvestLogReaderHandler,
} from '../../src/main/setupInvestHandlers';
import writeInvestParameters from '../../src/main/writeInvestParameters';
import { removeIpcMainListeners } from '../../src/main/main';

import App from '../../src/renderer/app';
import {
  getInvestModelNames,
  getSpec,
  fetchValidation,
  fetchArgsEnabled,
} from '../../src/renderer/server_requests';
import InvestJob from '../../src/renderer/InvestJob';

const MOCK_MODEL_TITLE = 'Carbon';

jest.mock('node-fetch');
jest.mock('child_process');
jest.mock('../../src/renderer/server_requests');
jest.mock('../../src/main/writeInvestParameters');

describe('InVEST subprocess testing', () => {
  const spec = {
    args: {
      workspace_dir: {
        name: 'Workspace',
        type: 'directory',
      },
      results_suffix: {
        name: 'Suffix',
        type: 'freestyle_string',
      },
    },
    ui_spec: {
      order: [['workspace_dir', 'results_suffix']],
    },
    model_name: 'EcoModel',
    pyname: 'natcap.invest.dot',
    userguide: 'foo.html',
  };
  const modelName = 'carbon';
  // nothing is written to the fake workspace in these tests,
  // and we mock validation, so this dir need not exist.
  const fakeWorkspace = 'foo_dir';
  const logfilePath = path.join(fakeWorkspace, 'invest-log.txt');
  // invest always emits this message, and the workbench always listens for it:
  const stdOutLogfileSignal = `Writing log messages to [${logfilePath}]`;
  const stdOutText = 'hello from invest';
  const investExe = 'foo';

  // We need to mock child_process.spawn to return a mock invest process,
  // And we also need an interface to that mocked process in each test
  // so we can push to stdout, stderr, etc, just like invest would.
  function getMockedInvestProcess() {
    const mockInvestProc = new events.EventEmitter();
    mockInvestProc.pid = -9999999; // not a plausible pid
    mockInvestProc.stdout = new Stream.Readable({
      read: () => {},
    });
    mockInvestProc.stderr = new Stream.Readable({
      read: () => {},
    });
    if (process.platform !== 'win32') {
      jest.spyOn(process, 'kill')
        .mockImplementation(() => {
          mockInvestProc.emit('exit', null);
        });
    } else {
      exec.mockImplementation(() => {
        mockInvestProc.emit('exit', null);
      });
    }
    spawn.mockImplementation(() => mockInvestProc);
    return mockInvestProc;
  }

  beforeAll(() => {
    setupInvestRunHandlers(investExe);
    setupInvestLogReaderHandler();
    // mock request/response for invest usage-logging
    const response = {
      ok: true,
      text: async () => 'foo',
    };
    fetch.mockResolvedValue(response);
  });

  afterAll(() => {
    removeIpcMainListeners();
  });

  beforeEach(() => {
    getSpec.mockResolvedValue(spec);
    fetchValidation.mockResolvedValue([]);
    fetchArgsEnabled.mockResolvedValue({
      workspace_dir: true, results_suffix: true
    });
    getInvestModelNames.mockResolvedValue(
      { Carbon: { model_name: modelName } }
    );
    // mock the request to write the datastack file. Actually write it
    // because the app will clean it up when invest exits.
    writeInvestParameters.mockImplementation((payload) => {
      fs.writeFileSync(payload.filepath, 'foo');
      return Promise.resolve({ text: () => 'foo' });
    });
  });

  afterEach(async () => {
    await InvestJob.clearStore();
  });

  test('exit without error - expect log display', async () => {
    const mockInvestProc = getMockedInvestProcess();
    const {
      findByText,
      findByLabelText,
      findByRole,
      getByRole,
      queryByText,
    } = render(<App />);

    const carbon = await findByRole(
      'button', { name: MOCK_MODEL_TITLE }
    );
    await userEvent.click(carbon);
    const workspaceInput = await findByLabelText(
      `${spec.args.workspace_dir.name}`
    );
    await userEvent.type(workspaceInput, fakeWorkspace);
    const execute = await findByRole('button', { name: /Run/ });
    await userEvent.click(execute);
    await waitFor(() => expect(execute).toBeDisabled());

    act(() => {
      // logfile signal on stdout listener is how app knows the process started
      mockInvestProc.stdout.push(stdOutText);
      mockInvestProc.stdout.push(stdOutLogfileSignal);
    });
    const logTab = await findByText('Log');
    await waitFor(() => {
      expect(logTab.classList.contains('active')).toBeTruthy();
    });
    expect(await findByText(stdOutText, { exact: false }))
      .toBeInTheDocument();
    expect(queryByText('Model Complete')).toBeNull();
    expect(queryByText('Open Workspace')).toBeNull();

    act(() => {
      mockInvestProc.emit('exit', 0); // 0 - exit w/o error
    });
    expect(await findByRole('alert')).toHaveTextContent('Model Complete');
    expect(await findByText('Open Workspace')).toBeEnabled();
    expect(await findByText(/\u2705/)).toBeInTheDocument();
    await waitFor(() => expect(execute).toBeEnabled());

    // A recent job card should be rendered
    await userEvent.click(getByRole('button', { name: 'InVEST' }));
    const homeTab = getByRole('tabpanel', { name: 'home tab' });
    const cardText = await within(homeTab).findByText(fakeWorkspace);
    expect(cardText).toBeInTheDocument();
  });

  test('exit with error - expect log & alert display', async () => {
    const mockInvestProc = getMockedInvestProcess();
    const {
      findByText,
      findByLabelText,
      findByRole,
      getByRole,
    } = render(<App />);

    const carbon = await findByRole(
      'button', { name: MOCK_MODEL_TITLE }
    );
    await userEvent.click(carbon);
    const workspaceInput = await findByLabelText(
      `${spec.args.workspace_dir.name}`
    );
    await userEvent.type(workspaceInput, fakeWorkspace);

    const execute = await findByRole('button', { name: /Run/ });
    await userEvent.click(execute);

    act(() => {
      mockInvestProc.stdout.push(stdOutText);
      mockInvestProc.stdout.push(stdOutLogfileSignal);
      mockInvestProc.stderr.push('some error');
    });
    const logTab = await findByText('Log');
    await waitFor(() => {
      expect(logTab.classList.contains('active')).toBeTruthy();
    });

    expect(await findByText(stdOutText, { exact: false }))
      .toBeInTheDocument();

    act(() => {
      mockInvestProc.emit('exit', 1); // 1 - exit w/ error
    });

    const alert = await findByRole('alert');
    await waitFor(() => {
      expect(alert).toHaveTextContent('Error: see log for details');
      expect(alert).toHaveClass('alert-danger');
    });
    expect(await findByRole('button', { name: 'Open Workspace' }))
      .toBeEnabled();

    // A recent job card should be rendered
    await userEvent.click(getByRole('button', { name: 'InVEST' }));
    const homeTab = await getByRole('tabpanel', { name: 'home tab' });
    const cardText = await within(homeTab)
      .findByText(fakeWorkspace);
    expect(cardText).toBeInTheDocument();
  });

  test('user terminates process - expect log & alert display', async () => {
    const mockInvestProc = getMockedInvestProcess();
    const {
      findByText,
      findByLabelText,
      findByRole,
      getByRole,
      queryByText,
    } = render(<App />);

    const carbon = await findByRole(
      'button', { name: MOCK_MODEL_TITLE }
    );
    await userEvent.click(carbon);
    const workspaceInput = await findByLabelText(
      `${spec.args.workspace_dir.name}`
    );
    await userEvent.type(workspaceInput, fakeWorkspace);

    const execute = await findByRole('button', { name: /Run/ });
    await userEvent.click(execute);

    act(() => {
      // stdout listener is how the app knows the process started
      // Canel button only appears after this signal.
      mockInvestProc.stdout.push(stdOutText);
    });
    expect(queryByText('Cancel Run')).toBeNull();
    act(() => {
      mockInvestProc.stdout.push(stdOutLogfileSignal);
    });
    const logTab = await findByText('Log');
    await waitFor(() => {
      expect(logTab.classList.contains('active')).toBeTruthy();
    });
    expect(await findByText(stdOutText, { exact: false }))
      .toBeInTheDocument();

    const cancelButton = await findByText('Cancel Run');
    await userEvent.click(cancelButton);
    expect(await findByText('Open Workspace'))
      .toBeEnabled();
    expect(await findByRole('alert'))
      .toHaveTextContent('Run Canceled');

    // A recent job card should be rendered
    await userEvent.click(getByRole('button', { name: 'InVEST' }));
    const homeTab = await getByRole('tabpanel', { name: 'home tab' });
    const cardText = await within(homeTab)
      .findByText(fakeWorkspace);
    expect(cardText).toBeInTheDocument();
  });

  test('Run & re-run a job - expect new log display', async () => {
    const mockInvestProc = getMockedInvestProcess();
    const {
      findByText,
      findByLabelText,
      findByRole,
    } = render(<App />);

    const carbon = await findByRole(
      'button', { name: MOCK_MODEL_TITLE }
    );
    await userEvent.click(carbon);
    const workspaceInput = await findByLabelText(
      `${spec.args.workspace_dir.name}`
    );
    await userEvent.type(workspaceInput, fakeWorkspace);

    const execute = await findByRole('button', { name: /Run/ });
    await userEvent.click(execute);

    act(() => {
      // stdout listener is how the app knows the process started
      mockInvestProc.stdout.push(stdOutText);
      mockInvestProc.stdout.push(stdOutLogfileSignal);
    });
    let logTab = await findByText('Log');
    await waitFor(() => {
      expect(logTab.classList.contains('active')).toBeTruthy();
    });
    expect(await findByText(stdOutText, { exact: false }))
      .toBeInTheDocument();

    const cancelButton = await findByText('Cancel Run');
    await userEvent.click(cancelButton);
    expect(await findByText('Open Workspace'))
      .toBeEnabled();

    // Now the second invest process:
    const anotherInvestProc = getMockedInvestProcess();
    // Now click away from Log, re-run, and expect the switch
    // back to the new log
    const setupTab = await findByText('Setup');
    await userEvent.click(setupTab);
    await userEvent.click(execute);
    const newStdOutText = 'this is new stdout text';
    act(() => {
      anotherInvestProc.stdout.push(newStdOutText);
      anotherInvestProc.stdout.push(stdOutLogfileSignal);
    });
    logTab = await findByText('Log');
    await waitFor(() => {
      expect(logTab.classList.contains('active')).toBeTruthy();
    });
    expect(await findByText(newStdOutText, { exact: false }))
      .toBeInTheDocument();
    act(() => {
      anotherInvestProc.emit('exit', 0);
    });
    expect(await findByText('Open Workspace'))
      .toBeEnabled();
  });

  test('Load Recent run & re-run it - expect new log display', async () => {
    const mockInvestProc = getMockedInvestProcess();
    const argsValues = {
      workspace_dir: fakeWorkspace,
    };
    const mockJob = new InvestJob({
      modelRunName: 'carbon',
      modelHumanName: 'Carbon Sequestration',
      argsValues: argsValues,
      status: 'success',
      logfile: logfilePath,
    });
    await InvestJob.saveJob(mockJob);

    const {
      findByText,
      findByRole,
      queryByText,
    } = render(<App />);

    const recentJobCard = await findByText(
      argsValues.workspace_dir
    );
    await userEvent.click(recentJobCard);
    await userEvent.click(await findByText('Log'));
    // We don't need to have a real logfile in order to test that LogTab
    // is trying to read from a file instead of from stdout
    expect(await findByText(/Logfile is missing/)).toBeInTheDocument();

    // Now re-run from the same InvestTab component and expect
    // LogTab is displaying the new invest process stdout
    const setupTab = await findByText('Setup');
    await userEvent.click(setupTab);
    const execute = await findByRole('button', { name: /Run/ });
    await userEvent.click(execute);
    act(() => {
      mockInvestProc.stdout.push(stdOutText);
      mockInvestProc.stdout.push(stdOutLogfileSignal);
    });
    const logTab = await findByText('Log');
    await waitFor(() => {
      expect(logTab.classList.contains('active')).toBeTruthy();
    });
    expect(await findByText(stdOutText, { exact: false }))
      .toBeInTheDocument();
    expect(queryByText('Logfile is missing')).toBeNull();
    act(() => {
      mockInvestProc.emit('exit', 0);
    });
    expect(await findByText('Open Workspace'))
      .toBeEnabled();
  });
});
