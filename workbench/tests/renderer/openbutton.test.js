import React from 'react';
import { render, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import { ipcRenderer } from 'electron';

import OpenButton from '../../src/renderer/components/OpenButton';
import { fetchDatastackFromFile } from '../../src/renderer/server_requests';

jest.mock('../../src/renderer/server_requests');

test('Open File: displays a tooltip on hover', async () => {
  const { findByRole, findByText, queryByText } = render(
    <OpenButton
      openInvestModel={() => {}}
      batchUpdateArgs={() => {}}
    />
  );

  const openButton = await findByRole('button', { name: 'Open' });
  await userEvent.hover(openButton);
  const hoverText = 'Browse to a datastack (.json) or InVEST logfile (.txt)';
  expect(await findByText(hoverText)).toBeInTheDocument();
  await userEvent.unhover(openButton);
  await waitFor(() => {
    expect(queryByText(hoverText)).toBeNull();
  });
});

test('Open File: sends correct payload', async () => {
  const mockDatastack = {
    model_run_name: 'foo',
    model_human_name: 'Foo',
    args: {},
  };
  const filename = 'data.json';
  const mockDialogData = { canceled: false, filePaths: [filename] };
  ipcRenderer.invoke.mockResolvedValue(mockDialogData);
  fetchDatastackFromFile.mockResolvedValue(mockDatastack);
  const { findByRole } = render(
    <OpenButton
      openInvestModel={() => {}}
      batchUpdateArgs={() => {}}
    />
  );

  const openButton = await findByRole('button', { name: 'Open' });
  await userEvent.click(openButton);

  await waitFor(() => {
    expect(fetchDatastackFromFile).toHaveBeenCalled();
  });
  const payload = fetchDatastackFromFile.mock.calls[0][0];
  expect(Object.keys(payload)).toEqual(['filepath']);
  expect(payload['filepath']).toEqual(filename);
});
