import React from 'react';
import { ipcRenderer } from 'electron';
import {
  fireEvent, render, waitFor
} from '@testing-library/react';
import '@testing-library/jest-dom';

import { DataDownloadModal } from '../src/components/DataDownloadModal';
import sampledata_registry from '../src/sampledata_registry.json';

const nModels = Object.keys(sampledata_registry.Models).length;
const modelName = Object.keys(sampledata_registry.Models)[0];

describe('Sample Data Download Form', () => {
  test('Checkbox initial state & interactions', () => {
    const {
      getByLabelText,
      getByRole,
      getAllByRole,
    } = render(
      <DataDownloadModal
        show={true}
        storeDownloadDir={() => {}}
        defaultTargetPath="foo/dir"
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
        storeDownloadDir={() => {}}
        defaultTargetPath="foo/dir"
      />
    );

    const allCheckBoxes = getAllByRole('checkbox');
    expect(allCheckBoxes).toHaveLength(nModels + 1); // +1 for Select All

    // Each checkbox is labeled by the model's name
    Object.keys(sampledata_registry.Models).forEach((modelName) => {
      // some names have trailing parentheticals that trip up the query.
      const pattern = modelName.split('(')[0];
      const modelCheckbox = getByLabelText(new RegExp(pattern));
      expect(modelCheckbox).toBeChecked();
    });
  });
});
