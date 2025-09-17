import React from 'react';
import { ipcRenderer } from 'electron';
import '@testing-library/jest-dom';
import {
  within, render, waitFor,
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { ipcMainChannels } from '../../src/main/ipcMainChannels';
import {
  getSpec,
  getInvestModelIDs,
  fetchArgsEnabled,
  fetchValidation
} from '../../src/renderer/server_requests';
import App from '../../src/renderer/app';

jest.mock('../../src/renderer/server_requests');

describe('Add plugin modal', () => {
  beforeEach(() => {
    getSpec.mockResolvedValue({
      model_id: 'foo',
      model_title: 'Foo',
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
      input_field_order: [['workspace_dir', 'input_path']],
    });

    fetchArgsEnabled.mockResolvedValue({
      workspace_dir: true,
      input_path: true,
    });
    fetchValidation.mockResolvedValue([]);
    getInvestModelIDs.mockResolvedValue({});
  });

  describe('"Add a plugin" form validation', () => {
    let spy;

    beforeEach(async () => {
      spy = ipcRenderer.invoke.mockImplementation((channel, setting) => {
        if (channel === ipcMainChannels.GET_SETTING) {
          if (setting === 'plugins') {
            return Promise.resolve({
              foo: {
                modelTitle: 'Foo',
                type: 'plugin',
              },
            });
          }
        } else if (channel === ipcMainChannels.HAS_MSVC) {
          return Promise.resolve(true);
        }
        return Promise.resolve();
      });
    });

    test('Should render an error on submit if git URL is empty', async () => {
      const {
        findByText, findByLabelText, findByRole,
      } = render(<App />);

      await userEvent.click(await findByRole('button', { name: 'menu' }));
      const managePluginsButton = await findByText(/Manage plugins/i);
      await userEvent.click(managePluginsButton);

      const userAcknowledgmentCheckbox = await findByLabelText(/I acknowledge and accept/i);
      await userEvent.click(userAcknowledgmentCheckbox);

      const submitButton = await findByText('Add');
      await userEvent.click(submitButton);

      const missingUrlError = await findByText('Error: URL is required.');
      expect(missingUrlError).toBeInTheDocument();

      expect(spy).not.toHaveBeenCalledWith(ipcMainChannels.ADD_PLUGIN);
    });

    test('Should render an error on submit if local path is empty', async () => {
      const {
        findByText, findByLabelText, findByRole,
      } = render(<App />);

      await userEvent.click(await findByRole('button', { name: 'menu' }));
      const managePluginsButton = await findByText(/Manage plugins/i);
      await userEvent.click(managePluginsButton);

      const sourceType = await findByLabelText('Install from');
      await userEvent.selectOptions(sourceType, 'local path');

      const userAcknowledgmentCheckbox = await findByLabelText(/I acknowledge and accept/i);
      await userEvent.click(userAcknowledgmentCheckbox);

      const submitButton = await findByText('Add');
      await userEvent.click(submitButton);

      const missingPathError = await findByText('Error: Path is required.');
      expect(missingPathError).toBeInTheDocument();

      expect(spy).not.toHaveBeenCalledWith(ipcMainChannels.ADD_PLUGIN);
    });

    test('Should render an error on submit if user acknowledgment is unchecked', async () => {
      const {
        findByText, findByLabelText, findByRole,
      } = render(<App />);

      await userEvent.click(await findByRole('button', { name: 'menu' }));
      const managePluginsButton = await findByText(/Manage plugins/i);
      await userEvent.click(managePluginsButton);

      const urlField = await findByLabelText('Git URL');
      await userEvent.type(urlField, 'fake url', { delay: 0 });

      const submitButton = await findByText('Add');
      await userEvent.click(submitButton);

      const userAcknowledgmentError = await findByText(/Error: Before installing a plugin/i);
      expect(userAcknowledgmentError).toBeInTheDocument();

      expect(spy).not.toHaveBeenCalledWith(ipcMainChannels.ADD_PLUGIN);
    });
  });

  test('Add a plugin: success', async () => {
    // mocking the plugins data in the settings store is how
    // we mock a successfull plugin installation
    const spy = ipcRenderer.invoke.mockImplementation((channel, setting) => {
      if (channel === ipcMainChannels.GET_SETTING) {
        if (setting === 'plugins') {
          return Promise.resolve({
            foo: {
              modelTitle: 'Foo',
              type: 'plugin',
            },
          });
        }
      } else if (channel === ipcMainChannels.HAS_MSVC) {
        return Promise.resolve(true);
      }
      return Promise.resolve();
    });
    const {
      findByText, findByLabelText, findByRole,
    } = render(<App />);

    await userEvent.click(await findByRole('button', { name: 'menu' }));
    const managePluginsButton = await findByText(/Manage plugins/i);
    await userEvent.click(managePluginsButton);

    const urlField = await findByLabelText('Git URL');
    await userEvent.type(urlField, 'fake url', { delay: 0 });
    const userAcknowledgmentCheckbox = await findByLabelText(/I acknowledge and accept/i);
    await userEvent.click(userAcknowledgmentCheckbox);

    const submitButton = await findByText('Add');
    // The following click event is not awaited because we want to expect the
    // 'loading' status, which  is only present before the click handler
    // fully resolves.
    userEvent.click(submitButton);
    await findByText('Adding plugin');

    await waitFor(() => {
      const calledChannels = spy.mock.calls.map((call) => call[0]);
      expect(calledChannels).toContain(ipcMainChannels.ADD_PLUGIN);
    });
    // close the modal
    const overlay = await findByRole('dialog');
    await userEvent.click(overlay);
    const pluginButton = await findByRole('button', { name: /Foo/ });
    // assert that the 'plugin' badge is displayed
    await waitFor(() => expect(within(pluginButton).getByText('Plugin')).toBeInTheDocument());
  });

  test('Add a plugin: failure with error displayed', async () => {
    const errorString = 'Failed to clone repository.';
    const spy = ipcRenderer.invoke.mockImplementation((channel) => {
      if (channel === ipcMainChannels.HAS_MSVC) {
        return Promise.resolve(true);
      }
      if (channel === ipcMainChannels.ADD_PLUGIN) {
        return Promise.reject(
          new Error(errorString)
        );
      }
      return Promise.resolve();
    });
    const {
      findByText, findByLabelText, findByRole,
    } = render(<App />);

    await userEvent.click(await findByRole('button', { name: 'menu' }));
    const managePluginsButton = await findByText(/Manage plugins/i);
    await userEvent.click(managePluginsButton);

    const urlField = await findByLabelText('Git URL');
    await userEvent.type(urlField, 'fake url', { delay: 0 });
    const userAcknowledgmentCheckbox = await findByLabelText(/I acknowledge and accept/i);
    await userEvent.click(userAcknowledgmentCheckbox);

    const submitButton = await findByText('Add');
    await userEvent.click(submitButton);

    await waitFor(() => {
      const calledChannels = spy.mock.calls.map((call) => call[0]);
      expect(calledChannels).toContain(ipcMainChannels.ADD_PLUGIN);
    });
    await findByText(new RegExp(errorString));
  });

  test('Open and run a plugin', async () => {
    ipcRenderer.invoke.mockImplementation((channel, setting) => {
      if (channel === ipcMainChannels.GET_SETTING) {
        if (setting === 'plugins') {
          return Promise.resolve({
            foo: {
              modelTitle: 'Foo',
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
    await userEvent.click(pluginButton);
    const executeButton = await findByRole('button', { name: /Run/ });
    expect(executeButton).toBeEnabled();
    // Nothing is really different about plugin tabs on the renderer side, so
    // this test is pretty basic.
    await userEvent.click(executeButton);

    await waitFor(() => {
      expect(spy).toHaveBeenCalledWith(
        ipcMainChannels.INVEST_RUN,
        'foo',
        { input_path: '', workspace_dir: '' },
        expect.anything()
      );
    });
  });

  test('Remove a plugin', async () => {
    let plugins = {
      foo: {
        modelTitle: 'Foo',
        type: 'plugin',
        version: '1.0',
      },
    };
    const spy = ipcRenderer.invoke.mockImplementation((channel, setting) => {
      if (channel === ipcMainChannels.GET_SETTING) {
        if (setting === 'plugins') {
          return Promise.resolve(plugins);
        }
      } else if (channel === ipcMainChannels.REMOVE_PLUGIN) {
        plugins = {};
      } else if (channel === ipcMainChannels.HAS_MSVC) {
        return Promise.resolve(true);
      }
      return Promise.resolve();
    });
    const {
      findByText, findByRole, getByRole, findByLabelText, queryByRole,
    } = render(<App />);

    // open the plugin first, to make sure it doesn't cause a crash when removing
    const pluginButton = await findByRole('button', { name: /Foo/ });
    await userEvent.click(pluginButton);

    await userEvent.click(await findByRole('button', { name: 'menu' }));
    const managePluginsButton = await findByText(/Manage plugins/i);
    await userEvent.click(managePluginsButton);

    const pluginDropdown = await findByLabelText('Plugin name');
    await userEvent.selectOptions(pluginDropdown, [getByRole('option', { name: 'Foo (1.0)' })]);

    const submitButton = await findByText('Remove');
    await userEvent.click(submitButton);
    await waitFor(() => {
      expect(spy.mock.calls.map((call) => call[0])).toContain(ipcMainChannels.REMOVE_PLUGIN);
    });
    // expect the plugin to have disappeared from the model list and the dropdown
    await waitFor(() => expect(queryByRole('button', { name: /Foo/ })).toBeNull());
    await waitFor(() => expect(queryByRole('option', { name: /Foo/ })).toBeNull());
  });
});
