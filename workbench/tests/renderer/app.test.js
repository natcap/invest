import React from 'react';
import { ipcRenderer } from 'electron';
import {
  render, waitFor, within
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

import App from '../../src/renderer/app';
import {
  getInvestModelIDs,
  getSpec,
  fetchValidation,
  fetchDatastackFromFile,
  fetchArgsEnabled,
  getGeoMetaMakerProfile,
} from '../../src/renderer/server_requests';
import InvestJob from '../../src/renderer/InvestJob';
import { ipcMainChannels } from '../../src/main/ipcMainChannels';
import pkg from '../../package.json';

jest.mock('../../src/renderer/server_requests');

const MOCK_MODEL_TITLE = 'Carbon';
const MOCK_MODEL_ID = 'carbon';
const MOCK_INVEST_LIST = {
  [MOCK_MODEL_ID]: {
    model_title: MOCK_MODEL_TITLE,
  },
};
const MOCK_VALIDATION_VALUE = [[['workspace_dir'], 'invalid because']];

const SAMPLE_SPEC = {
  model_name: MOCK_MODEL_TITLE,
  pyname: 'natcap.invest.carbon',
  userguide: 'carbonstorage.html',
  args: {
    workspace_dir: {
      name: 'Workspace',
      about: 'help text',
      type: 'directory',
    },
    carbon_pools_path: {
      name: 'Carbon Pools',
      about: 'help text',
      type: 'csv',
    },
  },
  input_field_order: [['workspace_dir', 'carbon_pools_path']],
};

describe('Various ways to open and close InVEST models', () => {
  beforeEach(async () => {
    getInvestModelIDs.mockResolvedValue(MOCK_INVEST_LIST);
    getSpec.mockResolvedValue(SAMPLE_SPEC);
    fetchValidation.mockResolvedValue(MOCK_VALIDATION_VALUE);
    fetchArgsEnabled.mockResolvedValue({
      workspace_dir: true, carbon_pools_path: true
    });
  });

  afterEach(async () => {
    await InvestJob.clearStore(); // because a test calls InvestJob.saveJob()
  });

  test('Clicking an invest model button renders SetupTab', async () => {
    const { findByText, findByRole } = render(
      <App />
    );

    const carbon = await findByRole(
      'button', { name: MOCK_MODEL_TITLE }
    );
    await userEvent.click(carbon);
    const executeButton = await findByRole('button', { name: /Run/ });
    expect(executeButton).toBeDisabled();
    const setupTab = await findByText('Setup');
    expect(setupTab.classList.contains('active')).toBeTruthy();
    expect(getSpec).toHaveBeenCalledTimes(1);
    const navTab = await findByRole('tab', { name: MOCK_MODEL_TITLE });
    await userEvent.hover(navTab);
    await findByRole('tooltip', { name: MOCK_MODEL_TITLE });
  });

  test('Clicking a recent job renders SetupTab', async () => {
    const workspacePath = 'my_workspace';
    const argsValues = {
      workspace_dir: workspacePath,
    };
    const mockJob = new InvestJob({
      modelID: 'carbon',
      modelTitle: 'Carbon Sequestration',
      argsValues: argsValues,
      status: 'success',
      type: 'core',
    });
    await InvestJob.saveJob(mockJob);

    const { findByText, findByLabelText, findByRole } = render(
      <App />
    );

    const recentJobCard = await findByText(
      argsValues.workspace_dir
    );
    await userEvent.click(recentJobCard);
    const executeButton = await findByRole('button', { name: /Run/ });
    expect(executeButton).toBeDisabled();
    const setupTab = await findByText('Setup');
    expect(setupTab.classList.contains('active')).toBeTruthy();

    // Expect some arg values that were loaded from the saved job:
    const input = await findByLabelText((content) => content.startsWith(SAMPLE_SPEC.args.workspace_dir.name));
    expect(input).toHaveValue(
      argsValues.workspace_dir
    );
  });

  test('Open File: Dialog callback renders SetupTab', async () => {
    const mockDialogData = {
      canceled: false,
      filePaths: ['foo.json'],
    };
    const mockDatastack = {
      args: {
        carbon_pools_path: 'Carbon/carbon_pools_willamette.csv',
      },
      model_id: 'carbon',
      model_title: 'Carbon',
    };
    ipcRenderer.invoke.mockImplementation((channel) => {
      if (channel === ipcMainChannels.GET_SETTING) {
        return Promise.resolve();
      }
      return mockDialogData;
    });
    fetchDatastackFromFile.mockResolvedValue(mockDatastack);

    const { findByText, findByLabelText, findByRole } = render(
      <App />
    );

    const openButton = await findByRole(
      'button', { name: /browse to a datastack or invest logfile/i });
    expect(openButton).not.toBeDisabled();
    await userEvent.click(openButton);
    const executeButton = await findByRole('button', { name: /Run/ });
    expect(executeButton).toBeDisabled();
    const setupTab = await findByText('Setup');
    const input = await findByLabelText(
      (content) => content.startsWith(SAMPLE_SPEC.args.carbon_pools_path.name)
    );
    expect(setupTab.classList.contains('active')).toBeTruthy();
    expect(input).toHaveValue(mockDatastack.args.carbon_pools_path);
  });

  test('Open File: Dialog callback is canceled', async () => {
    // Resembles callback data if the dialog was canceled
    const mockDialogData = {
      canceled: true,
      filePaths: [],
    };
    ipcRenderer.invoke.mockImplementation((channel) => {
      if (channel === ipcMainChannels.GET_SETTING) {
        return Promise.resolve();
      }
      return mockDialogData;
    });

    const { findByRole } = render(
      <App />
    );

    const openButton = await findByRole(
      'button', { name: /browse to a datastack or invest logfile/i });
    await userEvent.click(openButton);
    const homeTab = await findByRole('tabpanel', { name: 'home tab' });
    // expect we're on the same tab we started on instead of switching to Setup
    expect(homeTab.classList.contains('active')).toBeTruthy();
    // These are the calls that would have triggered if a file was selected
    expect(fetchDatastackFromFile).toHaveBeenCalledTimes(0);
    expect(getSpec).toHaveBeenCalledTimes(0);
  });

  test('Open three tabs and close them', async () => {
    const {
      findByRole,
      findAllByRole,
      queryAllByRole,
    } = render(<App />);

    const carbon = await findByRole(
      'button', { name: MOCK_MODEL_TITLE }
    );
    const homeTab = await findByRole('tabpanel', { name: 'home tab' });

    // Open a model tab and expect that it's active
    await userEvent.click(carbon);
    let modelTabs = await findAllByRole('tab', { name: MOCK_MODEL_TITLE });
    expect(modelTabs).toHaveLength(1); // one carbon tab open
    const tab1 = modelTabs[0];
    const tab1EventKey = tab1.getAttribute('data-rb-event-key');
    expect(tab1.classList.contains('active')).toBeTruthy();
    expect(homeTab.classList.contains('active')).toBeFalsy();

    // Open a second model tab and expect that it's active
    await userEvent.click(homeTab);
    await userEvent.click(carbon);
    modelTabs = await findAllByRole('tab', { name: MOCK_MODEL_TITLE });
    expect(modelTabs).toHaveLength(2); // 2 carbon tabs open
    const tab2 = modelTabs[1];
    const tab2EventKey = tab2.getAttribute('data-rb-event-key');
    expect(tab2.classList.contains('active')).toBeTruthy();
    expect(tab1.classList.contains('active')).toBeFalsy();
    expect(homeTab.classList.contains('active')).toBeFalsy();
    // make sure that we switched away from the first tab
    expect(tab2EventKey).not.toEqual(tab1EventKey);

    // Open a third model tab and expect that it's active
    await userEvent.click(homeTab);
    await userEvent.click(carbon);
    modelTabs = await findAllByRole('tab', { name: MOCK_MODEL_TITLE });
    expect(modelTabs).toHaveLength(3); // 3 carbon tabs open
    const tab3 = modelTabs[2];
    const tab3EventKey = tab3.getAttribute('data-rb-event-key');
    expect(tab3.classList.contains('active')).toBeTruthy();
    expect(tab2.classList.contains('active')).toBeFalsy();
    expect(tab1.classList.contains('active')).toBeFalsy();
    expect(homeTab.classList.contains('active')).toBeFalsy();
    // make sure that we switched away from the first model tabs
    expect(tab3EventKey).not.toEqual(tab2EventKey);
    expect(tab3EventKey).not.toEqual(tab1EventKey);

    // Click the close button on the middle tab
    const tab2CloseButton = await within(tab2.closest('.nav-item'))
      .getByRole('button', { name: new RegExp(`close ${MOCK_MODEL_TITLE}`) });
    await userEvent.click(tab2CloseButton);
    // Now there should only be 2 model tabs open
    modelTabs = await findAllByRole('tab', { name: MOCK_MODEL_TITLE });
    expect(modelTabs).toHaveLength(2);
    // Should have switched to tab3, the next tab to the right
    expect(tab3.classList.contains('active')).toBeTruthy();
    expect(tab1.classList.contains('active')).toBeFalsy();

    // Click the close button on the right tab
    const tab3CloseButton = await within(tab3.closest('.nav-item'))
      .getByRole('button', { name: new RegExp(`close ${MOCK_MODEL_TITLE}`) });
    await userEvent.click(tab3CloseButton);
    // Now there should only be 1 model tab open
    modelTabs = await findAllByRole('tab', { name: MOCK_MODEL_TITLE });
    expect(modelTabs).toHaveLength(1);
    // No model tabs to the right, so it should switch to the next tab to the left.
    expect(tab1.classList.contains('active')).toBeTruthy();
    expect(homeTab.classList.contains('active')).toBeFalsy();

    // Click the close button on the last tab
    const tab1CloseButton = await within(tab1.closest('.nav-item'))
      .getByRole('button', { name: new RegExp(`close ${MOCK_MODEL_TITLE}`) });
    await userEvent.click(tab1CloseButton);
    // Now there should be no model tabs open.
    modelTabs = await queryAllByRole('tab', { name: MOCK_MODEL_TITLE });
    expect(modelTabs).toHaveLength(0);
    // No more model tabs, so it should switch back to the home tab.
    expect(homeTab.classList.contains('active')).toBeTruthy();
  });
});

describe('Display recently executed InVEST jobs on Home tab', () => {
  beforeEach(() => {
    getInvestModelIDs.mockResolvedValue(MOCK_INVEST_LIST);
  });

  afterEach(async () => {
    await InvestJob.clearStore();
  });

  test('Recent Jobs: each has a button', async () => {
    const job1 = new InvestJob({
      modelID: MOCK_MODEL_ID,
      modelTitle: 'Carbon Sequestration',
      argsValues: {
        workspace_dir: 'work1',
      },
      status: 'success',
    });
    await InvestJob.saveJob(job1);
    const job2 = new InvestJob({
      modelID: MOCK_MODEL_ID,
      modelTitle: 'Sediment Ratio Delivery',
      argsValues: {
        workspace_dir: 'work2',
        results_suffix: 'suffix',
      },
      status: 'error',
      type: 'core',
    });
    const recentJobs = await InvestJob.saveJob(job2);
    const initialJobs = [job1, job2];

    const { getByText } = render(<App />);

    await waitFor(() => {
      initialJobs.forEach((job, idx) => {
        const recent = recentJobs[idx];
        const card = getByText(job.argsValues.workspace_dir)
          .closest('button');
        expect(within(card).getByText(job.argsValues.workspace_dir))
          .toBeInTheDocument();
        if (job.status === 'success') {
          expect(getByText('Model Complete'))
            .toBeInTheDocument();
        }
        if (job.status === 'error') {
          expect(getByText(job.status))
            .toBeInTheDocument();
        }
        if (job.argsValues.results_suffix) {
          expect(getByText(job.argsValues.results_suffix))
            .toBeInTheDocument();
        }
        // The timestamp is not part of the initial object, but should
        // in the saved object
        expect(within(card).getByText(recent.humanTime))
          .toBeInTheDocument();
      });
    });
  });

  test('Recent Jobs: a job with incomplete data is skipped', async () => {
    const job1 = new InvestJob({
      modelID: MOCK_MODEL_ID,
      modelTitle: 'invest A',
      argsValues: {
        workspace_dir: 'dir',
      },
      status: 'success',
      type: 'core',
    });
    const job2 = new InvestJob({
      // argsValues is missing
      modelID: MOCK_MODEL_ID,
      modelTitle: 'invest B',
      status: 'success',
      type: 'core',
    });
    await InvestJob.saveJob(job1);
    await InvestJob.saveJob(job2);

    const { findByText, queryByText } = render(<App />);

    expect(await findByText(job1.modelTitle)).toBeInTheDocument();
    expect(queryByText(job2.modelTitle)).toBeNull();
  });

  test('Recent Jobs: a job from a deprecated model is not displayed', async () => {
    const job1 = new InvestJob({
      modelID: 'does not exist',
      modelTitle: 'invest A',
      argsValues: {
        workspace_dir: 'dir',
      },
      status: 'success',
      type: 'core',
    });
    await InvestJob.saveJob(job1);
    const { findByText, queryByText } = render(<App />);

    expect(queryByText(job1.modelTitle)).toBeNull();
    expect(await findByText('Welcome!'))
      .toBeInTheDocument();
  });

  test('Recent Jobs: placeholder if there are no recent jobs', async () => {
    const { findByText } = render(
      <App />
    );

    const node = await findByText('Welcome!');
    expect(node).toBeInTheDocument();
  });

  test('Recent Jobs: cleared by clear all button', async () => {
    const job1 = new InvestJob({
      modelID: MOCK_MODEL_ID,
      modelTitle: 'Carbon Sequestration',
      argsValues: {
        workspace_dir: 'work1',
      },
      status: 'success',
      // leave out the 'type' attribute to make sure it defaults to core
      // for backwards compatibility
    });
    const recentJobs = await InvestJob.saveJob(job1);

    const { getByText, findByText, getByRole } = render(<App />);

    await waitFor(() => {
      recentJobs.forEach((job) => {
        expect(getByText(job.argsValues.workspace_dir))
          .toBeTruthy();
      });
    });
    await userEvent.click(getByText(/clear all model runs/i));
    const node = await findByText('Welcome!');
    expect(node).toBeInTheDocument();
  });

  test('Recent Jobs: delete single job', async () => {
    const job1 = new InvestJob({
      modelID: MOCK_MODEL_ID,
      modelTitle: 'Carbon Sequestration',
      argsValues: {
        workspace_dir: 'work1',
      },
      status: 'success',
      // leave out the 'type' attribute to make sure it defaults to core
      // for backwards compatibility
    });
    const recentJobs = await InvestJob.saveJob(job1);

    const { getByText, findByText, getByRole } = render(<App />);

    await waitFor(() => {
      recentJobs.forEach((job) => {
        expect(getByText(job.argsValues.workspace_dir))
          .toBeTruthy();
      });
    });
    await userEvent.click(getByRole('button', { name: 'delete' }));
    const node = await findByText('Welcome!');
    expect(node).toBeInTheDocument();
  });
});

describe('Main menu interactions', () => {
  beforeEach(() => {
    getInvestModelIDs.mockResolvedValue(MOCK_INVEST_LIST);
  });

  test('Open sampledata download Modal from menu', async () => {
    const {
      findByText, findByRole,
    } = render(
      <App />
    );

    const dropdownBtn = await findByRole('button', { name: 'menu' });
    await userEvent.click(dropdownBtn);
    await userEvent.click(
      await findByRole('button', { name: /Download Sample Data/i })
    );

    expect(await findByText(/Download InVEST sample data/i))
      .toBeInTheDocument();
    await userEvent.click(
      await findByRole('button', { name: /close modal/i })
    );
  });

  test('Open Metadata Modal from menu', async () => {
    const {
      findByText, findByRole,
    } = render(
      <App />
    );

    const dropdownBtn = await findByRole('button', { name: 'menu' });
    await userEvent.click(dropdownBtn);
    await userEvent.click(
      await findByRole('button', { name: /Configure Metadata/i })
    );

    expect(await findByText(/contact information/i))
      .toBeInTheDocument();
    await userEvent.click(
      await findByRole('button', { name: /close modal/i })
    );
  });

  test('Open Plugins Modal from menu', async () => {
    ipcRenderer.invoke.mockImplementation((channel) => {
      if (channel === ipcMainChannels.HAS_MSVC) {
        return Promise.resolve(true);
      }
      return Promise.resolve();
    });

    const {
      findByText, findByRole,
    } = render(
      <App />
    );

    const dropdownBtn = await findByRole('button', { name: 'menu' });
    await userEvent.click(dropdownBtn);
    await userEvent.click(
      await findByRole('button', { name: /Manage Plugins/i })
    );

    expect(await findByText(/add a plugin/i))
      .toBeInTheDocument();
    await userEvent.click(
      await findByRole('button', { name: /close modal/i })
    );
  });

  test('Open Changelog Modal from menu', async () => {
    const currentVersion = pkg.version;
    const nonexistentVersion = '1.0.0';
    jest.spyOn(window, 'fetch')
      .mockResolvedValueOnce({
        ok: true,
        text: () => `
            <html>
              <head></head>
              <body>
                <section>
                  <h1>${currentVersion}</h1>
                </section>
                <section>
                  <h1>${nonexistentVersion}</h1>
                </section>
              </body>
            </html>
        `
      });

    const {
      findByText, findByRole,
    } = render(
      <App />
    );

    const dropdownBtn = await findByRole('button', { name: 'menu' });
    await userEvent.click(dropdownBtn);
    await userEvent.click(
      await findByRole('button', { name: /view changelog/i })
    );

    expect(await findByText(/new in this version/i))
      .toBeInTheDocument();
    await userEvent.click(
      await findByRole('button', { name: /close modal/i })
    );
  });

  test('Open Settings Modal from menu', async () => {
    const {
      findByText, findByRole,
    } = render(
      <App />
    );

    const dropdownBtn = await findByRole('button', { name: 'menu' });
    await userEvent.click(dropdownBtn);
    await userEvent.click(
      await findByRole('button', { name: /settings/i })
    );

    expect(await findByText(/invest settings/i))
      .toBeInTheDocument();
    await userEvent.click(
      await findByRole('button', { name: /close modal/i })
    );
  });
});
