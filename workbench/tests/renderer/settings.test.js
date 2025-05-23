import React from 'react';
import { ipcRenderer } from 'electron';
import {
  render, waitFor
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

import {
  settingsStore,
  setupSettingsHandlers
} from '../../src/main/settingsStore';
import { ipcMainChannels } from '../../src/main/ipcMainChannels';
import { removeIpcMainListeners } from '../../src/main/main';

import { getSupportedLanguages } from '../../src/renderer/server_requests';
import SettingsModal from '../../src/renderer/components/SettingsModal';

jest.mock('../../src/renderer/server_requests');

describe('InVEST global settings: dialog interactions', () => {
  const nWorkersLabelText = 'Taskgraph n_workers parameter';
  const loggingLabelText = 'Logging threshold';
  const tgLoggingLabelText = 'Taskgraph logging threshold';
  const languageLabelText = 'Language';

  beforeAll(() => {
    setupSettingsHandlers();
  });

  afterAll(() => {
    removeIpcMainListeners();
  });

  beforeEach(async () => {
    getSupportedLanguages.mockResolvedValue({ en: 'english', es: 'spanish' });
  });

  test('Invest settings save on change', async () => {
    const nWorkersLabel = 'Threaded task management (0)';
    const nWorkersValue = 0;
    const loggingLevel = 'DEBUG';
    const tgLoggingLevel = 'DEBUG';
    const languageValue = 'es';
    const spyInvoke = jest.spyOn(ipcRenderer, 'invoke');

    const {
      getByText, getByLabelText, findByText,
    } = render(
      <SettingsModal
        show
        close={() => {}}
        nCPU={4}
      />
    );

    const nWorkersInput = getByLabelText(nWorkersLabelText, { exact: false });
    const loggingInput = getByLabelText(loggingLabelText);
    const tgLoggingInput = getByLabelText(tgLoggingLabelText);

    await userEvent.selectOptions(nWorkersInput, [getByText(nWorkersLabel)]);
    await waitFor(() => { expect(nWorkersInput).toHaveValue(nWorkersValue.toString()); });
    await userEvent.selectOptions(loggingInput, [loggingLevel]);
    await waitFor(() => { expect(loggingInput).toHaveValue(loggingLevel); });
    await userEvent.selectOptions(tgLoggingInput, [tgLoggingLevel]);
    await waitFor(() => { expect(tgLoggingInput).toHaveValue(tgLoggingLevel); });

    // Check values were saved
    expect(settingsStore.get('nWorkers')).toBe(nWorkersValue);
    expect(settingsStore.get('loggingLevel')).toBe(loggingLevel);
    expect(settingsStore.get('taskgraphLoggingLevel')).toBe(tgLoggingLevel);

    // language is handled differently; changing it triggers electron to restart
    const languageInput = getByLabelText(languageLabelText, { exact: false });
    await userEvent.selectOptions(languageInput, [languageValue]);
    await userEvent.click(await findByText('Change to spanish'));
    expect(spyInvoke)
      .toHaveBeenCalledWith(ipcMainChannels.CHANGE_LANGUAGE, languageValue);
  });
});
