import { ipcRenderer } from 'electron';
import React from 'react';
import {
  act, within, render, waitFor
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { ipcMainChannels } from '../../src/main/ipcMainChannels';
import App from '../../src/renderer/app';
import {
  getSpec,
  getInvestModelNames,
  fetchArgsEnabled,
  fetchValidation
} from '../../src/renderer/server_requests';

jest.mock('../../src/renderer/server_requests');

describe('Add plugin modal', () => {
  beforeEach(() => {
    getSpec.mockResolvedValue({
      model_name: 'foo',
      pyname: 'natcap.invest.foo',
      userguide: '',
      args: {
        workspace_dir: {
          name: 'Workspace',
          about: 'help text',
          type: 'directory',
        },
        input_path: {
          name: 'Input raster',
          about: 'help text',
          type: 'raster',
        },
      },
      ui_spec: {
        order: [['workspace_dir', 'input_path']],
      },
    });

    fetchArgsEnabled.mockResolvedValue({
      workspace_dir: true,
      input_path: true,
    });
    fetchValidation.mockResolvedValue([]);
    getInvestModelNames.mockResolvedValue({});
  });

  test('Interface to add a plugin', async () => {
    const spy = ipcRenderer.invoke.mockImplementation((channel, setting) => {
      if (channel === ipcMainChannels.GET_SETTING) {
        if (setting === 'plugins') {
          return Promise.resolve({
            foo: {
              model_name: 'Foo',
              type: 'plugin',
            },
          });
        }
      }
      return Promise.resolve();
    });
    const {
      findByText, findByLabelText, findByRole, queryByRole,
    } = render(<App />);

    const managePluginsButton = await findByText('Manage plugins');
    userEvent.click(managePluginsButton);

    const urlField = await findByLabelText('Add a plugin');
    await userEvent.type(urlField, 'fake url', { delay: 0 });
    const submitButton = await findByText('Add');
    userEvent.click(submitButton);

    await findByText('Loading...');
    const calledChannels = spy.mock.calls.map((call) => call[0]);
    await waitFor(() => {
      expect(calledChannels).toContain(ipcMainChannels.ADD_PLUGIN);
    });
    // expect the plugin dialog to have disappeared
    await waitFor(() => expect(queryByRole('dialog')).toBeNull());
    const pluginButton = await findByRole('button', { name: /Foo/ });
    // assert that the 'plugin' badge is displayed
    await waitFor(() => expect(within(pluginButton).getByText('Plugin')).toBeInTheDocument());
  });

  test('Open and run a plugin', async () => {
    ipcRenderer.invoke.mockImplementation((channel, setting) => {
      if (channel === ipcMainChannels.GET_SETTING) {
        if (setting === 'plugins') {
          return Promise.resolve({
            foo: {
              model_name: 'Foo',
              type: 'plugin',
            },
          });
        }
      } else if (channel === ipcMainChannels.LAUNCH_PLUGIN_SERVER) {
        return 1;
      }
      return Promise.resolve();
    });
    const spy = jest.spyOn(ipcRenderer, 'send');
    const { findByRole } = render(<App />);
    const pluginButton = await findByRole('button', { name: /Foo/ });
    await act(async () => {
      userEvent.click(pluginButton);
    });
    const executeButton = await findByRole('button', { name: /Run/ });
    expect(executeButton).toBeEnabled();
    // Nothing is really different about plugin tabs on the renderer side, so
    // this test is pretty basic.
    await userEvent.click(executeButton);

    await waitFor(() => {
      expect(spy).toHaveBeenCalledWith(
        ipcMainChannels.INVEST_RUN,
        'foo',
        'natcap.invest.foo',
        { input_path: '', workspace_dir: '' },
        expect.anything()
      );
    });
  });

  test('Remove a plugin', async () => {
    let plugins = {
      foo: {
        model_name: 'Foo',
        type: 'plugin',
      },
    };
    const spy = ipcRenderer.invoke.mockImplementation((channel, setting) => {
      if (channel === ipcMainChannels.GET_SETTING) {
        if (setting === 'plugins') {
          return Promise.resolve(plugins);
        }
      } else if (channel === ipcMainChannels.REMOVE_PLUGIN) {
        plugins = {};
      }
      return Promise.resolve();
    });
    const {
      findByText, getByRole, findByLabelText, queryByRole,
    } = render(<App />);

    const managePluginsButton = await findByText('Manage plugins');
    userEvent.click(managePluginsButton);

    const pluginDropdown = await findByLabelText('Remove a plugin');
    await userEvent.selectOptions(pluginDropdown, [getByRole('option', { name: 'Foo' })]);

    const submitButton = await findByText('Remove');
    userEvent.click(submitButton);
    await waitFor(() => {
      expect(spy.mock.calls.map((call) => call[0])).toContain(ipcMainChannels.REMOVE_PLUGIN);
    });
    // expect the plugin to have disappeared from the model list and the dropdown
    await waitFor(() => expect(queryByRole('button', { name: /Foo/ })).toBeNull());
    await waitFor(() => expect(queryByRole('option', { name: /Foo/ })).toBeNull());
  });
});
