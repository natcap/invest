import React from 'react';
import { render, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';

import LoadButton from '../src/components/LoadButton';

test('Open File: displays a tooltip on hover', async () => {
  const { findByRole, findByText, queryByText } = render(
    <LoadButton
      openInvestModel={() => {}}
      batchUpdateArgs={() => {}}
    />
  );

  const openButton = await findByRole('button', { name: 'Open' });
  userEvent.hover(openButton);
  const hoverText = 'Browse to a datastack (.json) or invest logfile (.txt)';
  expect(await findByText(hoverText)).toBeInTheDocument();
  userEvent.unhover(openButton);
  await waitFor(() => {
    expect(queryByText(hoverText)).toBeNull();
  })
});
