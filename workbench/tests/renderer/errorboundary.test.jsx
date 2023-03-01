import React from 'react';
import { render } from '@testing-library/react';
import '@testing-library/jest-dom';

import ErrorBoundary from '../../src/renderer/components/ErrorBoundary';

function MyComponent() {
  throw new Error('From MyComponent');
}

test('Error Boundary: displays useful content', async () => {
  const { getByRole, getByText } = render(
    <ErrorBoundary>
      <MyComponent />
    </ErrorBoundary>
  );

  expect(getByText(/Something went wrong/)).toBeInTheDocument();
  expect(getByRole('button', { name: 'Find My Logs' })).toBeInTheDocument();
  expect(getByRole('link')).toHaveTextContent('community.naturalcapitalproject.org');
});
