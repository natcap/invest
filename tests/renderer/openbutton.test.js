import React from 'react';
import { render, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';

import OpenButton from '../../src/renderer/components/OpenButton';

test('Open File: displays a tooltip on hover', async () => {
  const { findByRole, findByText, queryByText } = render(
    <OpenButton
      openInvestModel={() => {}}
      batchUpdateArgs={() => {}}
    />
  );

  const openButton = await findByRole('button', { name: 'Open' });
  userEvent.hover(openButton);
  const hoverText = 'Browse to a datastack (.json) or InVEST logfile (.txt)';
  expect(await findByText(hoverText)).toBeInTheDocument();
  userEvent.unhover(openButton);
  await waitFor(() => {
    expect(queryByText(hoverText)).toBeNull();
  });
});
