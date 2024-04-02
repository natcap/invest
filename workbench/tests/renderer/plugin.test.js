import { ipcRenderer } from 'electron';
import React from 'react';
import {
  act, within, render, waitFor
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { ipcMainChannels } from '../../src/main/ipcMainChannels';
import App from '../../src/renderer/app';
import { getSpec, fetchArgsEnabled, fetchValidation } from '../../src/renderer/server_requests';

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
  });

  test('Interface to add a plugin', async () => {
    const spy = jest.spyOn(ipcRenderer, 'invoke').mockImplementation((channel, setting) => {
      if (channel === ipcMainChannels.GET_SETTING) {
        if (setting === 'models') {
          return Promise.resolve({
            carbon: {
              model_name: 'Carbon',
              type: 'core',
            },
          });
        }
      }
      return Promise.resolve();
    });

    const { findByText, findByLabelText } = render(<App />);

    const addPluginButton = await findByText('Add a plugin');
    userEvent.click(addPluginButton);

    const urlField = await findByLabelText('Git URL');
    await act(async () => {
      userEvent.type(urlField, 'https://github.com/emlys/demo-invest-plugin.git', { delay: 0 });
    });
    const submitButton = await findByText('Add');
    userEvent.click(submitButton);

    await findByText('Loading...');
    await waitFor(() => {
      const calledChannels = spy.mock.calls.map((call) => call[0]);
      expect(calledChannels).toContain(ipcMainChannels.ADD_PLUGIN);
    });
  });

  test('Display a plugin in the list of models', async () => {
    ipcRenderer.invoke.mockImplementation((channel, setting) => {
      if (channel === ipcMainChannels.GET_SETTING) {
        if (setting === 'models') {
          return {
            carbon: {
              model_name: 'Carbon',
              type: 'core',
            },
            foo: {
              model_name: 'Foo',
              type: 'plugin',
            },
          };
        }
      }
      return undefined;
    });
    const { findByRole } = render(<App />);
    const pluginButton = await findByRole('button', { name: /Foo/ });
    // assert that the 'plugin' badge is displayed
    expect(within(pluginButton).getByText('Plugin')).toBeInTheDocument();
  });

  test('Open and run a plugin', async () => {
    ipcRenderer.invoke.mockImplementation((channel, setting) => {
      if (channel === ipcMainChannels.GET_SETTING) {
        if (setting === 'models') {
          return Promise.resolve({
            carbon: {
              model_name: 'Carbon',
              type: 'core',
            },
            foo: {
              model_name: 'Foo',
              type: 'plugin',
            },
          });
        }
      } else if (channel === ipcMainChannels.INVEST_SERVE) {
        return 0;
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
});
