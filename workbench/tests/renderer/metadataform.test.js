import React from 'react';
import { render, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';

import MetadataForm from '../../src/renderer/components/SettingsModal/MetadataForm';
import {
  getGeoMetaMakerProfile,
  setGeoMetaMakerProfile,
} from '../../src/renderer/server_requests';

jest.mock('../../src/renderer/server_requests');

test('Metadata form interact and submit', async () => {
  const startingName = 'Alice';
  const startingLicense = '';
  getGeoMetaMakerProfile.mockResolvedValue({
    contact: {
      individual_name: startingName,
      email: '',
      organization: '',
      position_name: '',
    },
    license: {
      title: startingLicense,
      path: '',
    },
  });
  setGeoMetaMakerProfile.mockResolvedValue({
    message: 'Metadata profile saved',
    error: false,
  });

  const user = userEvent.setup();
  const { findByRole, getByLabelText, getByRole } = render(
    <MetadataForm />
  );
  // The form should render with content from an existing profile
  const nameInput = getByLabelText('Full name');
  await waitFor(() => {
    expect(nameInput).toHaveValue(startingName);
  });
  const licenseInput = getByLabelText('Title');
  await waitFor(() => {
    expect(licenseInput).toHaveValue(startingLicense);
  });

  const name = 'Bob';
  const license = 'CC-BY-4.0';
  await user.clear(nameInput);
  await user.clear(licenseInput);
  await user.type(nameInput, name);
  await user.type(licenseInput, license);
  const submit = getByRole('button', { name: 'Save Metadata' });
  await user.click(submit);

  expect(await findByRole('alert'))
    .toHaveTextContent('Metadata profile saved');
  const payload = setGeoMetaMakerProfile.mock.calls[0][0];
  expect(Object.keys(payload)).toEqual(['contact', 'license']);
  expect(payload['contact']['individual_name']).toEqual(name);
  expect(payload['license']['title']).toEqual(license);
});
