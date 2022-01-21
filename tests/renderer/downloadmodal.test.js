import React from 'react';
import {
  fireEvent, render, waitFor
} from '@testing-library/react';
import '@testing-library/jest-dom';
import { ipcRenderer, BrowserWindow } from 'electron';

import {
  DataDownloadModal,
  DownloadProgressBar
} from '../../src/renderer/components/DataDownloadModal';
import sampledata_registry from '../../src/renderer/sampledata_registry.json';
import { getInvestModelNames } from '../../src/renderer/server_requests';
import App from '../../src/renderer/app';
import {
  clearSettingsStore,
  getSettingsValue,
} from '../../src/renderer/components/SettingsModal/SettingsStorage';
import setupDownloadHandlers from '../../src/main/setupDownloadHandlers';
import { removeIpcMainListeners } from '../../src/main/main';

jest.mock('../../src/renderer/server_requests');

const nModels = Object.keys(sampledata_registry).length;
const modelName = Object.keys(sampledata_registry)[0];

describe('Sample Data Download Form', () => {
  beforeEach(() => {
    getInvestModelNames.mockResolvedValue({});
  });

  test('Modal displays immediately on user`s first run', async () => {
    const {
      findByText,
      getByText,
    } = render(<App isFirstRun />);

    const modalTitle = await findByText('Download InVEST sample data');
    expect(modalTitle).toBeInTheDocument();
    fireEvent.click(getByText('Cancel'));
    await waitFor(() => {
      expect(modalTitle).not.toBeInTheDocument();
    });
  });

  test('Modal does not display when app has been run before', async () => {
    const { queryByText } = render(<App isFirstRun={false} />);
    const modalTitle = await queryByText('Download InVEST sample data');
    expect(modalTitle).toBeNull();
  });

  test('Checkbox initial state & interactions', () => {
    const {
      getByLabelText,
      getByRole,
      getAllByRole,
    } = render(
      <DataDownloadModal
        show={true}
        closeModal={() => {}}
        storeDownloadDir={() => {}}
      />
    );

    // All checked initially
    const allCheckBoxes = getAllByRole('checkbox');
    expect(allCheckBoxes).toHaveLength(nModels + 1); // +1 for Select All
    allCheckBoxes.forEach((box) => {
      expect(box).toBeChecked();
    });
    const downloadButton = getByRole('button', { name: 'Download' });
    expect(downloadButton).toBeEnabled();

    // Toggle all off using Select All
    const selectAllCheckbox = getByLabelText('Select All');
    fireEvent.click(selectAllCheckbox);
    allCheckBoxes.forEach((box) => {
      expect(box).not.toBeChecked();
    });
    expect(downloadButton).toBeDisabled();

    // Toggle all on using Select All
    fireEvent.click(selectAllCheckbox);
    allCheckBoxes.forEach((box) => {
      expect(box).toBeChecked();
    });

    // Toggle one off & on
    const modelCheckbox = getByLabelText(new RegExp(modelName));
    fireEvent.click(modelCheckbox);
    expect(modelCheckbox).not.toBeChecked();
    expect(selectAllCheckbox).not.toBeChecked();
    fireEvent.click(modelCheckbox);
    expect(modelCheckbox).toBeChecked();
  });

  test('Checkbox list matches the sampledata registry', () => {
    // The registry itself is validated during the build process
    // by the script called by `npm run fetch-invest`.
    const {
      getByLabelText,
      getAllByRole,
    } = render(
      <DataDownloadModal
        show={true}
        closeModal={() => {}}
        storeDownloadDir={() => {}}
      />
    );

    const allCheckBoxes = getAllByRole('checkbox');
    expect(allCheckBoxes).toHaveLength(nModels + 1); // +1 for Select All

    // Each checkbox is labeled by the model's name
    Object.keys(sampledata_registry).forEach((modelName) => {
      // some names have trailing parentheticals that trip up the query.
      const pattern = modelName.split('(')[0];
      const modelCheckbox = getByLabelText(new RegExp(pattern));
      expect(modelCheckbox).toBeChecked();
    });
  });
});

describe('DownloadProgressBar', () => {
  test('Displays progress before complete', () => {
    const nComplete = 5;
    const nTotal = 10;
    const { getByText } = render(
      <DownloadProgressBar
        downloadedNofN={[nComplete, nTotal]}
        expireAfter={5000}
      />
    );
    expect(getByText(/Downloading/)).toBeInTheDocument();
  });

  test('Displays message on complete, then disappears', async () => {
    const nComplete = 5;
    const nTotal = 5;
    const { getByText } = render(
      <DownloadProgressBar
        downloadedNofN={[nComplete, nTotal]}
        expireAfter={1000}
      />
    );
    const alert = getByText('Download Complete');
    expect(alert).toBeInTheDocument();
    await waitFor(() => {
      expect(alert).not.toBeInTheDocument();
    });
  });
});

describe('Integration tests with main process', () => {
  beforeEach(async () => {
    setupDownloadHandlers(new BrowserWindow());
    getInvestModelNames.mockResolvedValue({});
  });

  afterEach(async () => {
    removeIpcMainListeners();
    await clearSettingsStore();
  });

  test('Download: starts, updates progress, & stores location', async () => {
    const dialogData = {
      filePaths: ['foo/directory'],
    };
    ipcRenderer.invoke.mockResolvedValue(dialogData);

    const {
      findByRole,
      findAllByRole,
    } = render(<App isFirstRun />);

    const allCheckBoxes = await findAllByRole('checkbox');
    const downloadButton = await findByRole('button', { name: 'Download' });
    fireEvent.click(downloadButton);
    const nURLs = allCheckBoxes.length - 1; // all except Select All
    await waitFor(async () => {
      expect(await getSettingsValue('sampleDataDir'))
        .toBe(dialogData.filePaths[0]);
    });
    const progressBar = await findByRole('progressbar');
    expect(progressBar).toHaveTextContent(`Downloading 1 of ${nURLs}`);
    // We don't have mocks that take us all the way through to a complete
    // download, when the progress bar would become a 'Download Complete' alert
  });

  test('Cancel: does not store a sampleDataDir value', async () => {
    const { findByRole } = render(<App isFirstRun />);

    const existingValue = await getSettingsValue('sampleDataDir');
    const cancelButton = await findByRole('button', { name: 'Cancel' });
    fireEvent.click(cancelButton);

    await waitFor(async () => {
      const value = await getSettingsValue('sampleDataDir');
      expect(value).toBe(existingValue);
    });
  });
});
