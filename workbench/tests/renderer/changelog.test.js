import React from 'react';
import '@testing-library/jest-dom';
import { render } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import App from '../../src/renderer/app';
import pkg from '../../package.json';
import { getInvestModelIDs } from '../../src/renderer/server_requests';

jest.mock('../../src/renderer/server_requests');

const MOCK_MODEL_TITLE = 'Carbon';
const MOCK_MODEL_RUN_NAME = 'carbon';
const MOCK_INVEST_LIST = {
  [MOCK_MODEL_TITLE]: {
    model_name: MOCK_MODEL_RUN_NAME,
  },
};

describe('Changelog', () => {
  const currentVersion = pkg.version;
  const nonexistentVersion = 'nonexistent-version';
  beforeEach(() => {
    jest.spyOn(window, 'fetch')
      .mockResolvedValue({
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
    getInvestModelIDs.mockResolvedValue(MOCK_INVEST_LIST);
  });

  test('Changelog modal opens immediately on launch of a new version', async () => {
    const { findByRole } = render(<App isNewVersion />);
    const changelogModal = await findByRole('dialog', { name: 'New in this version' });
    expect(changelogModal).toBeInTheDocument();
  });

  test('On first run (of any version), Changelog modal opens after Download modal is closed', async () => {
    const { findByRole, getByText } = render(<App isFirstRun isNewVersion />);

    let changelogModalFound = true;
    try {
      await findByRole('dialog', { name: 'New in this version' });
    } catch {
      changelogModalFound = false;
    }
    expect(changelogModalFound).toBe(false);

    const downloadModal = await findByRole('dialog', { name: 'Download InVEST sample data' });
    expect(downloadModal).toBeInTheDocument();

    await userEvent.click(getByText('Cancel'));
    expect(downloadModal).not.toBeInTheDocument();
    const changelogModal = await findByRole('dialog', { name: 'New in this version' });
    expect(changelogModal).toBeInTheDocument();
  });

  test('Changelog modal does not open when current version has been run before', async () => {
    const { findByRole } = render(<App isNewVersion={false} />);
    let changelogModalFound = true;
    try {
      await findByRole('dialog', { name: 'New in this version' });
    } catch {
      changelogModalFound = false;
    }
    expect(changelogModalFound).toBe(false);
  });

  test('Changelog modal contains only content relevant to the current version', async () => {
    const { findByRole, queryByRole } = render(<App isNewVersion />);
    const currentVersionSectionHeading = await findByRole('heading', { name: currentVersion });
    expect(currentVersionSectionHeading).toBeInTheDocument();

    const nonexistentVersionSectionHeading = queryByRole('heading', { name: nonexistentVersion });
    expect(nonexistentVersionSectionHeading).not.toBeInTheDocument();
  });
});
