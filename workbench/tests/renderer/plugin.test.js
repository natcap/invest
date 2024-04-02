import { ipcRenderer } from 'electron';
import React from 'react';
import userEvent from '@testing-library/user-event';
import { render, waitFor } from '@testing-library/react';
import { ipcMainChannels } from '../../src/main/ipcMainChannels';
import '@testing-library/jest-dom';
import App from '../../src/renderer/app';


describe('Add plugin modal', () => {
  beforeEach(() => {
    // getInvestModelNames.mockResolvedValue({});
  });

  test('Modal displays on button click', async () => {
    const { findByText, findByLabelText } = render(<App />);

    const addPluginButton = await findByText('Add plugin');
    const modalTitle = queryByText('Add a plugin');
    userEvent.click(addPluginButton);

    const urlField = await findByLabelText('Git URL');
    await userEvent.type(urlField, 'https://github.com/emlys/demo-invest-plugin.git', { delay: 0 });
    const submitButton = await findByText('Add');
    userEvent.click(submitButton);

    await findByText('Loading...');

    const spy = jest.spyOn(ipcRenderer, 'invoke')
      .mockImplementation(() => Promise.resolve());

    await waitFor(() => {
      const calledChannels = spy.mock.calls.map((call) => call[0]);
      expect(calledChannels).toContain(ipcMainChannels.ADD_PLUGIN);
    });
  });
});
