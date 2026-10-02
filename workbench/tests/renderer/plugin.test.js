import React from 'react';
import { ipcRenderer } from 'electron';
import '@testing-library/jest-dom';
import {
  act, within, render, waitFor, fireEvent, createEvent,
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
import * as modalUtils from '../../src/renderer/components/PluginModal/PluginModalUtils';

jest.mock('../../src/renderer/server_requests');

const PLUGIN_SETTING_ITEM = {
  foo: {
    modelID: 'foo',
    modelTitle: 'Foo',
    type: 'plugin',
    source: 'git+example_url.git@1.0',
    sourceType: 'git_url',
    env: 'foo/bar/baz/',
    projectName: 'invest_foo_plugin',
    version: '1.0',
  }
}
const MULTIPLE_PLUGINS_SETTING_ITEM = {
  ...PLUGIN_SETTING_ITEM,
  bar: {
    modelID: 'bar',
    modelTitle: 'Bar',
    type: 'plugin',
    source: 'git+bar_example_url.git@1.0',
    sourceType: 'git_url',
    env: 'bar/baz/',
    projectName: 'invest_bar_plugin',
    version: '1.0',
  },
};

const PLUGIN_REGISTRY_DATA = [
  {
    'pyproject_toml_project_name': 'invest_bar_plugin',
    'invest_package_name': 'invest_bar_plugin',
    'plugin_name': 'Bar',
    'version': '1.0.1',
    'description': 'Bar baz (no foo).',
    'authors': [],
    'maintainers': ['Natural Capital Alliance Software Team'],
    'registry_url': 'invest-plugin-registry/plugins/invest_bar_plugin.html',
    'repository_url': 'bar_example_url.git',
    'documentation_url': 'bar_example_url/README.md',
    'issues_url': 'bar_example_url/issues',
    'license': 'Apache-2.0',
    'plugin_type': 'model_variant',
    'keywords': ['bar', 'foo'],
    'date_last_updated': '2026-06-16T23:27:41Z'
  }, {
    'pyproject_toml_project_name': 'invest_baz_plugin',
    'invest_package_name': 'invest_baz_plugin',
    'plugin_name': 'Baz',
    'version': '2.0.2',
    'description': 'Just baz.',
    'authors': [],
    'maintainers': ['Natural Capital Alliance Software Team'],
    'registry_url': 'invest-plugin-registry/plugins/invest_baz_plugin.html',
    'repository_url': 'baz_example_url.git',
    'documentation_url': 'baz_example_url/README.md',
    'issues_url': 'baz_example_url/issues',
    'license': 'Apache-2.0',
    'plugin_type': 'new_model',
    'keywords': ['baz'],
    'date_last_updated': '2026-06-16T23:27:41Z'
  }, {
    'pyproject_toml_project_name': 'invest_foo_plugin',
    'invest_package_name': 'invest_foo_plugin',
    'plugin_name': 'Foo',
    'version': '1.0',
    'description': 'Foo bar baz.',
    'authors': [],
    'maintainers': ['Natural Capital Alliance Software Team'],
    'registry_url': 'invest-plugin-registry/plugins/invest_foo_plugin.html',
    'repository_url': 'example_url.git',
    'documentation_url': 'example_url/README.md',
    'issues_url': 'example_url/issues',
    'license': 'Apache-2.0',
    'plugin_type': 'workflow',
    'keywords': ['foo'],
    'date_last_updated': '2026-06-16T23:27:41Z'
  }
];

describe('Plugin Manager modal', () => {
  beforeEach(() => {
    getSpec.mockResolvedValue({
      model_id: 'foo',
      model_title: 'Foo',
      userguide: '',
      args: {
        workspace_dir: {
          name: 'Workspace',
          about: 'help text',
          type: 'workspace',
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

    modalUtils.fetchRegistryData = jest.fn().mockResolvedValue([]);
  });

  describe('PluginRegistryTab', () => {
    let spy;

    beforeEach(async () => {
      modalUtils.fetchRegistryData = jest.fn()
        .mockResolvedValue(PLUGIN_REGISTRY_DATA);

      spy = ipcRenderer.invoke.mockImplementation((channel, setting) => {
        if (channel === ipcMainChannels.GET_SETTING) {
          if (setting === 'plugins') {
            return Promise.resolve(MULTIPLE_PLUGINS_SETTING_ITEM);
          }
        } else if (channel === ipcMainChannels.HAS_MSVC) {
          return Promise.resolve(true);
        }
        return Promise.resolve();
      });
    });

    test('Other version installed plugin shows message', async () => {
      const {
        findByText, findByRole, findAllByRole,
      } = render(<App />);

      await userEvent.click(await findByRole('button', { name: /add a plugin/i }));
      const modal = await findByRole('dialog');
      const tabpanels = await within(modal).findAllByRole('tabpanel');
      const pluginRegistryTab = tabpanels[0];

      const pluginDetailsPanes = await within(pluginRegistryTab)
        .findAllByRole('tabpanel');
      const barDetailsPane = pluginDetailsPanes[0];

      const otherVersionInstalledMsg = await within(barDetailsPane)
        .findByText('A different version of this plugin is already installed',
           { exact: false });
      expect(otherVersionInstalledMsg).toBeInTheDocument();
      const installBtn = await within(barDetailsPane)
        .queryByRole('button', { name: 'Install' });
      expect(installBtn).toBeInTheDocument();
    });

    test('Uninstalled plugin shows install button', async () => {
      const {
        findByRole, findAllByRole,
      } = render(<App />);

      await userEvent.click(await findByRole('button', { name: /add a plugin/i }));
      const modal = await findByRole('dialog');
      const tabpanels = await within(modal).findAllByRole('tabpanel');
      const pluginRegistryTab = tabpanels[0];

      const pluginDetailsPanes = await within(pluginRegistryTab)
        .findAllByRole('tabpanel');
      const bazDetailsPane = pluginDetailsPanes[1];

      const installBtn = await within(bazDetailsPane)
        .queryByRole('button', { name: 'Install' });
      expect(installBtn).toBeInTheDocument();
    });

    test('Installed plugin shows already installed message', async () => {
      const {
        findByText, findByRole, findAllByRole,
      } = render(<App />);

      await userEvent.click(await findByRole('button', { name: /add a plugin/i }));
      const modal = await findByRole('dialog');
      const tabpanels = await within(modal).findAllByRole('tabpanel');
      const pluginRegistryTab = tabpanels[0];

      const pluginDetailsPanes = await within(pluginRegistryTab)
        .findAllByRole('tabpanel')
      const fooDetailsPane = pluginDetailsPanes[2];

      const alreadyInstalledMsg = await within(fooDetailsPane)
        .findByText('This plugin is installed!');
      expect(alreadyInstalledMsg).toBeInTheDocument();
      await waitFor(() => expect(within(fooDetailsPane)
        .queryByRole('button', { name: 'Install' })).toBeNull());
    });
  });

  test('InstalledPluginsTab: Remove a plugin', async () => {
    let plugins = PLUGIN_SETTING_ITEM;
    const spy = ipcRenderer.invoke.mockImplementation((channel, setting) => {
      if (channel === ipcMainChannels.GET_SETTING) {
        if (setting === 'plugins') {
          return Promise.resolve(plugins);
        }
      } else if (channel === ipcMainChannels.REMOVE_PLUGIN) {
        // after REMOVE_PLUGIN there will be a subsequent call to GET_SETTING,
        // so this effectively replaces the mocked settings data
        plugins = {};
      } else if (channel === ipcMainChannels.HAS_MSVC) {
        return Promise.resolve(true);
      } else if (channel === ipcMainChannels.LAUNCH_PLUGIN_SERVER) {
        return 1111; // a fake PID
      }
      return Promise.resolve();
    });
    const {
      findByText, findByRole, findAllByRole, queryByRole,
    } = render(<App />);

    // open the plugin first, to make sure it doesn't cause a crash when removing
    const pluginButton = await findByRole('button', { name: /Foo/ });
    await userEvent.click(pluginButton);

    await userEvent.click(await findByRole('button', { name: /add a plugin/i }));
    const installedPluginsNav = await findByRole('tab', { name: /installed plugins/i });
    await userEvent.click(installedPluginsNav);

    const modal = await findByRole('dialog');
    const tabpanels = await within(modal).findAllByRole('tabpanel');
    const manualInstallTab = tabpanels[1];

    const submitButton = await within(manualInstallTab).findByText('Uninstall');
    await userEvent.click(submitButton);
    await waitFor(() => {
      expect(spy.mock.calls.map((call) => call[0])).toContain(ipcMainChannels.REMOVE_PLUGIN);
    });
    // expect the plugin to have disappeared from the model list and the dropdown
    await waitFor(() => expect(queryByRole('button', { name: /Foo/ })).toBeNull());
    await waitFor(() => expect(within(manualInstallTab)
      .queryByRole('heading', { name: /Foo/ })).toBeNull());
  });

  describe('ManualInstallTab: form validation', () => {
    let spy;

    beforeEach(async () => {
      spy = ipcRenderer.invoke.mockImplementation((channel, setting) => {
        if (channel === ipcMainChannels.GET_SETTING) {
          if (setting === 'plugins') {
            return Promise.resolve(PLUGIN_SETTING_ITEM);
          }
        } else if (channel === ipcMainChannels.HAS_MSVC) {
          return Promise.resolve(true);
        }
        return Promise.resolve();
      });
    });

    test('Should render an error on submit if git URL is empty', async () => {
      const {
        findByText, findByLabelText, findByRole, findAllByRole,
      } = render(<App />);

      await userEvent.click(await findByRole('button', { name: /add a plugin/i }));
      const manualInstallNav = await findByRole('tab', { name: /manual install/i });
      await userEvent.click(manualInstallNav);

      const modal = await findByRole('dialog');
      const tabpanels = await within(modal).findAllByRole('tabpanel');
      const manualInstallTab = tabpanels[2];

      const userAcknowledgmentCheckbox = await within(manualInstallTab)
        .findByLabelText(/I acknowledge and accept/i);
      await userEvent.click(userAcknowledgmentCheckbox);

      const submitButton = await within(manualInstallTab).findByText('Install');
      await userEvent.click(submitButton);

      const missingUrlError = await within(manualInstallTab)
        .findByText('Error: URL is required.');
      expect(missingUrlError).toBeInTheDocument();

      expect(spy).not.toHaveBeenCalledWith(ipcMainChannels.ADD_PLUGIN);
    });

    test('Should render an error on submit if local path is empty', async () => {
      const {
        findByText, findByLabelText, findByRole, findAllByRole,
      } = render(<App />);

      await userEvent.click(await findByRole('button', { name: /add a plugin/i }));
      const manualInstallNav = await findByRole('tab', { name: /manual install/i });
      await userEvent.click(manualInstallNav);

      const modal = await findByRole('dialog');
      const tabpanels = await within(modal).findAllByRole('tabpanel');
      const manualInstallTab = tabpanels[2];

      await userEvent.click(await within(manualInstallTab)
        .findByRole('radio', { name: 'local path' }));

      const userAcknowledgmentCheckbox = await within(manualInstallTab)
        .findByLabelText(/I acknowledge and accept/i);
      await userEvent.click(userAcknowledgmentCheckbox);

      const submitButton = await within(manualInstallTab).findByText('Install');
      await userEvent.click(submitButton);

      const missingPathError = await within(manualInstallTab)
        .findByText('Error: Path is required.');
      expect(missingPathError).toBeInTheDocument();

      expect(spy).not.toHaveBeenCalledWith(ipcMainChannels.ADD_PLUGIN);
    });

    test('Should render an error on submit if user acknowledgment is unchecked', async () => {
      const {
        findByText, findByLabelText, findByRole, findAllByRole,
      } = render(<App />);

      await userEvent.click(await findByRole('button', { name: /add a plugin/i }));
      const manualInstallNav = await findByRole('tab', { name: /manual install/i });
      await userEvent.click(manualInstallNav);

      const modal = await findByRole('dialog');
      const tabpanels = await within(modal).findAllByRole('tabpanel');
      const manualInstallTab = tabpanels[2];

      const urlField = await within(manualInstallTab).findByLabelText('Git URL');
      await userEvent.type(urlField, 'fake url', { delay: 0 });

      const submitButton = await within(manualInstallTab).findByText('Install');
      await userEvent.click(submitButton);

      const userAcknowledgmentError = await within(manualInstallTab)
        .findByText(/Error: Before installing a plugin/i);
      expect(userAcknowledgmentError).toBeInTheDocument();

      expect(spy).not.toHaveBeenCalledWith(ipcMainChannels.ADD_PLUGIN);
    });
  });

  describe('ManualInstallTab: install behavior', () => {
    test('Add a plugin: success', async () => {
      // mocking the plugins data in the settings store is how
      // we mock a successfull plugin installation
      const spy = ipcRenderer.invoke.mockImplementation((channel, setting) => {
        if (channel === ipcMainChannels.GET_SETTING) {
          if (setting === 'plugins') {
            return Promise.resolve(PLUGIN_SETTING_ITEM);
          }
        } else if (channel === ipcMainChannels.HAS_MSVC) {
          return Promise.resolve(true);
        }
        return Promise.resolve();
      });
      const {
        findByText, findByLabelText, findByRole, findAllByRole,
      } = render(<App />);

      await userEvent.click(await findByRole('button', { name: /add a plugin/i }));
      const manualInstallNav = await findByRole('tab', { name: /manual install/i });
      await userEvent.click(manualInstallNav);

      const modal = await findByRole('dialog');
      const tabpanels = await within(modal).findAllByRole('tabpanel');
      const manualInstallTab = tabpanels[2];

      const urlField = await within(manualInstallTab).findByLabelText('Git URL');
      await userEvent.type(urlField, 'fake url', { delay: 0 });
      const userAcknowledgmentCheckbox = await within(manualInstallTab)
        .findByLabelText(/I acknowledge and accept/i);
      await userEvent.click(userAcknowledgmentCheckbox);

      const submitButton = await within(manualInstallTab).findByText('Install');
      // The following click event is not awaited because we want to expect the
      // 'loading' status, which  is only present before the click handler
      // fully resolves.
      act(() => {
        userEvent.click(submitButton);
      });

      await within(manualInstallTab).findByText('Installing...');
      await waitFor(() => {
        const calledChannels = spy.mock.calls.map((call) => call[0]);
        expect(calledChannels).toContain(ipcMainChannels.ADD_PLUGIN);
      });
      // close the modal
      const overlay = await findByRole('dialog');
      await userEvent.click(overlay);
      const pluginButton = await findByRole('button', { name: /Foo/ });
      // assert that the 'plugin' badge is displayed
      await waitFor(() => expect(within(pluginButton).getByText('Plugin'))
        .toBeInTheDocument());
    });

    test('Add a plugin: failure with error displayed', async () => {
      const errorString = 'Failed to clone repository.';
      const spy = ipcRenderer.invoke.mockImplementation((channel, setting) => {
        if (channel === ipcMainChannels.HAS_MSVC) {
          return Promise.resolve(true);
        }
        if (channel === ipcMainChannels.ADD_PLUGIN) {
          return Promise.reject(
            new Error(errorString)
          );
        }
        if (channel === ipcMainChannels.GET_SETTING) {
          if (setting === 'plugins') {
            return Promise.resolve({});
          }
        }
        return Promise.resolve();
      });
      const {
        findByText, findByLabelText, findByRole, findAllByRole,
      } = render(<App />);

      await userEvent.click(await findByRole('button', { name: /add a plugin/i }));
      const manualInstallNav = await findByRole('tab', { name: /manual install/i });
      await userEvent.click(manualInstallNav);

      const modal = await findByRole('dialog');
      const tabpanels = await within(modal).findAllByRole('tabpanel');
      const manualInstallTab = tabpanels[2];

      const urlField = await within(manualInstallTab).findByLabelText('Git URL');
      await userEvent.type(urlField, 'fake url', { delay: 0 });
      const userAcknowledgmentCheckbox = await within(manualInstallTab)
        .findByLabelText(/I acknowledge and accept/i);
      await userEvent.click(userAcknowledgmentCheckbox);

      const submitButton = await within(manualInstallTab).findByText('Install');
      await userEvent.click(submitButton);
      // act(() => {
      //   userEvent.click(submitButton);
      // });

      await waitFor(() => {
        const calledChannels = spy.mock.calls.map((call) => call[0]);
        expect(calledChannels).toContain(ipcMainChannels.ADD_PLUGIN);
      });
      await within(manualInstallTab).findByText(new RegExp(errorString));
    });

    test('Drag-and-drop populates the local plugin path', async () => {
      ipcRenderer.invoke.mockImplementation((channel, setting) => {
        if (channel === ipcMainChannels.GET_SETTING) {
          if (setting === 'plugins') {
            return Promise.resolve({});
          }
        } else if (channel === ipcMainChannels.HAS_MSVC) {
          return Promise.resolve(true);
        }

        return Promise.resolve();
      });

      const {
        findByText, findByRole, findByLabelText, findAllByRole,
      } = render(<App />);

      await userEvent.click(await findByRole('button', { name: /add a plugin/i }));
      const manualInstallNav = await findByRole('tab', { name: /manual install/i });
      await userEvent.click(manualInstallNav);

      const modal = await findByRole('dialog');
      const tabpanels = await within(modal).findAllByRole('tabpanel');
      const manualInstallTab = tabpanels[2];

      await userEvent.click(await within(manualInstallTab)
        .findByRole('radio', { name: 'local path' }));

      const input = await within(manualInstallTab).findByLabelText('Local absolute path');
      const file = new File([], 'plugin-dir');

      fireEvent.dragEnter(input, {
        dataTransfer: { files: [file] },
      });

      expect(input).toHaveClass('input-dragging');

      fireEvent.drop(input, {
        dataTransfer: { files: [file] },
      });

      await waitFor(() => {
        expect(input).not.toHaveClass('input-dragging');
        expect(input).toHaveValue('plugin-dir');
        expect(input).toHaveFocus();
      });
    });

    test('Drag-leave removes input-dragging from a filepath input', async () => {
      ipcRenderer.invoke.mockImplementation((channel, setting) => {
        if (channel === ipcMainChannels.GET_SETTING) {
          if (setting === 'plugins') {
            return Promise.resolve(PLUGIN_SETTING_ITEM);
          }
        } else if (channel === ipcMainChannels.HAS_MSVC) {
          return Promise.resolve(true);
        }

        return Promise.resolve();
      });

      const {
        findByText, findByRole, findByLabelText,
      } = render(<App />);

      await userEvent.click(await findByRole('button', { name: /add a plugin/i }));
      const manualInstallNav = await findByRole('tab', { name: /manual install/i });
      await userEvent.click(manualInstallNav);

      const modal = await findByRole('dialog');
      const tabpanels = await within(modal).findAllByRole('tabpanel');
      const manualInstallTab = tabpanels[2];

      await userEvent.click(await within(manualInstallTab)
        .findByRole('radio', { name: 'local path' }));

      const input = await within(manualInstallTab).findByLabelText('Local absolute path');
      const file = new File([], 'plugin-directory');

      fireEvent.dragEnter(input, {
        dataTransfer: {
          files: [file],
        },
      });

      expect(input).toHaveClass('input-dragging');

      fireEvent.dragLeave(input, {
        dataTransfer: {
          files: [file],
        },
      });

      expect(input).not.toHaveClass('input-dragging');
    });

    test('Drag-and-drop on the git URL input does nothing', async () => {
      ipcRenderer.invoke.mockImplementation((channel, setting) => {
        if (channel === ipcMainChannels.GET_SETTING) {
          if (setting === 'plugins') {
            return Promise.resolve({});
          }
        } else if (channel === ipcMainChannels.HAS_MSVC) {
          return Promise.resolve(true);
        }

        return Promise.resolve();
      });

      const {
        findByText, findByRole, findByLabelText,
      } = render(<App />);

      await userEvent.click(await findByRole('button', { name: /add a plugin/i }));
      const manualInstallNav = await findByRole('tab', { name: /manual install/i });
      await userEvent.click(manualInstallNav);

      const modal = await findByRole('dialog');
      const tabpanels = await within(modal).findAllByRole('tabpanel');
      const manualInstallTab = tabpanels[2];

      const input = await within(manualInstallTab).findByLabelText('Git URL');

      const file = new File([], 'plugin-directory');
      const dropEvent = createEvent.drop(input, {
        dataTransfer: {
          files: [file],
        },
      });

      const preventDefaultSpy = jest.spyOn(dropEvent, 'preventDefault');

      fireEvent(input, dropEvent);

      expect(preventDefaultSpy).toHaveBeenCalled();
      expect(input).toHaveValue('');
    });
  });

  describe('AdvancedSettingsTab', () => {
    test('Change the conda executable', async () => {
      ipcRenderer.invoke.mockImplementation((channel, setting) => {
        if (channel === ipcMainChannels.GET_SETTING) {
          if (setting === 'plugins') {
            return Promise.resolve({});
          } else if (setting === 'micromamba') {
            return Promise.resolve('micromamba')
          }
        } else if (channel === ipcMainChannels.SHOW_OPEN_DIALOG) {
          return Promise.resolve({ filePaths: ['foo'] })
        } else if (channel === ipcMainChannels.HAS_MSVC) {
          return Promise.resolve(true);
        }
        return Promise.resolve();
      });
      const {
        findByText, findByRole, findByLabelText, findAllByRole,
      } = render(<App />);

      const spy = jest.spyOn(ipcRenderer, 'send');
      await userEvent.click(await findByRole('button', { name: /add a plugin/i }));
      const advancedSettingsNav = await findByRole('tab', { name: /advanced settings/i });
      await userEvent.click(advancedSettingsNav);

      const input = await findByLabelText('Conda or mamba executable');
      await userEvent.clear(input);
      await userEvent.type(input, 'conda');
      await waitFor(() => expect(input).toHaveValue('conda'));

      await userEvent.click(await findByRole('button', { name: /browse for conda executable/ }));
      await waitFor(() => { expect(input).toHaveValue('foo'); });

      const file = new File([], 'my-conda');
      fireEvent.dragEnter(input, {dataTransfer: {files: [file]}});
      expect(input).toHaveClass('input-dragging');
      fireEvent.drop(input, {dataTransfer: {files: [file]}});

      await waitFor(() => {
        expect(input).not.toHaveClass('input-dragging');
        expect(input).toHaveValue('my-conda');
        expect(input).toHaveFocus();
      });

      const div = await findByLabelText(
        'Configure conda executable')
      const saveButton = await within(div).findByRole('button', { name: /Save/ });
      await userEvent.click(saveButton);
      await waitFor(() => {
        expect(spy).toHaveBeenCalledWith(
          ipcMainChannels.SET_SETTING,
          'userDefinedMicromamba',
          'my-conda'
        );
      });

      const resetButton = await within(div).findByRole('button', { name: /Reset/ });
      await userEvent.click(resetButton);
      await waitFor(() => { expect(input).toHaveValue('micromamba'); });
    });

    test('Change a plugin env', async () => {
      ipcRenderer.invoke.mockImplementation((channel, setting) => {
        if (channel === ipcMainChannels.GET_SETTING) {
          if (setting === 'plugins') {
            return Promise.resolve(PLUGIN_SETTING_ITEM);
          } else if (setting === 'plugins.foo.env') {
            return Promise.resolve(PLUGIN_SETTING_ITEM.foo.env);
          }
        } else if (channel === ipcMainChannels.SHOW_OPEN_DIALOG) {
          return Promise.resolve({ filePaths: ['/path/to/my_env'] });
        } else if (channel === ipcMainChannels.HAS_MSVC) {
          return Promise.resolve(true);
        }
        return Promise.resolve();
      });
      const {
        findByText, findByRole, findByLabelText,
      } = render(<App />);

      const spy = jest.spyOn(ipcRenderer, 'send');
      await userEvent.click(await findByRole('button', { name: /add a plugin/i }));
      const advancedSettingsNav = await findByRole('tab', { name: /advanced settings/i });
      await userEvent.click(advancedSettingsNav);

      const div = await findByLabelText('Configure plugin environments')
      const input = await within(div).findByLabelText('foo');
      expect(input).toHaveValue(PLUGIN_SETTING_ITEM.foo.env);
      await userEvent.clear(input);
      await userEvent.type(input, 'my_env');
      await waitFor(() => expect(input).toHaveValue('my_env'));

      await userEvent.click(await findByRole('button', { name: /browse for env/ }));
      await waitFor(() => { expect(input).toHaveValue('/path/to/my_env'); });

      const saveButton = await within(div).findByRole('button', { name: /Save/ });
      await userEvent.click(saveButton);
      await waitFor(() => {
        expect(spy).toHaveBeenCalledWith(
          ipcMainChannels.SET_SETTING,
          'plugins.foo.userDefinedEnv',
          '/path/to/my_env'
        );
      });

      const resetButton = await within(div).findByRole('button', { name: /Reset/ });
      await userEvent.click(resetButton);
      await waitFor(() => { expect(input).toHaveValue(PLUGIN_SETTING_ITEM.foo.env); });
    });

    test('Drag-and-drop populates a plugin environment input', async () => {
      ipcRenderer.invoke.mockImplementation((channel, setting) => {
        if (channel === ipcMainChannels.GET_SETTING) {
          if (setting === 'plugins') {
            return Promise.resolve(PLUGIN_SETTING_ITEM);
          }
          if (setting === 'micromamba') {
            return Promise.resolve('micromamba');
          }
        } else if (channel === ipcMainChannels.HAS_MSVC) {
          return Promise.resolve(true);
        }

        return Promise.resolve();
      });

      const {
        findByText, findByRole, findByLabelText,
      } = render(<App />);

      await userEvent.click(await findByRole('button', { name: /add a plugin/i }));
      const advancedSettingsNav = await findByRole('tab', { name: /advanced settings/i });
      await userEvent.click(advancedSettingsNav);

      const div = await findByLabelText(
        'Configure plugin environments'
      );
      const input = await within(div).findByLabelText('foo');

      const file = new File([], 'plugin-environment');

      fireEvent.dragEnter(input, {
        dataTransfer: {
          files: [file],
        },
      });

      expect(input).toHaveClass('input-dragging');

      fireEvent.drop(input, {
        dataTransfer: {
          files: [file],
        },
      });

      await waitFor(() => {
        expect(input).not.toHaveClass('input-dragging');
        expect(input).toHaveValue('plugin-environment');
        expect(input).toHaveFocus();
      });
    });
  });

  test('Open and run a plugin', async () => {
    ipcRenderer.invoke.mockImplementation((channel, setting) => {
      if (channel === ipcMainChannels.GET_SETTING) {
        if (setting === 'plugins') {
          return Promise.resolve(PLUGIN_SETTING_ITEM);
        }
      } else if (channel === ipcMainChannels.LAUNCH_PLUGIN_SERVER) {
        return 1111; // a fake PID
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
});
