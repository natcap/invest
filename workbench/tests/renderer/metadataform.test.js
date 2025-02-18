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
  const {
    findByRole,
    getByLabelText,
    getByRole,
    getByText,
  } = render(<MetadataForm />);

  // The form should render with content from an existing profile
  const nameInput = getByLabelText('Full name');
  await waitFor(() => {
    expect(nameInput).toHaveValue(startingName);
  });
  let licenseInput = getByLabelText('Title');
  await waitFor(() => {
    expect(licenseInput).toHaveValue(startingLicense);
  });

  const name = 'Bob';
  const license = 'CC-BY-4.0';
  await user.clear(nameInput);
  await user.clear(licenseInput);
  await user.type(nameInput, name);
  await user.type(licenseInput, license);

  // Exercise the "more info" button
  await user.click(getByRole('button', { name: /more info/i }));
  expect(getByText('Metadata for InVEST results')).toBeInTheDocument();
  await user.click(getByRole('button', { name: /hide info/i }));

  const submit = getByRole('button', { name: /save metadata/i });
  await user.click(submit);

  const alert = await findByRole('alert');
  expect(alert).toHaveTextContent('Metadata profile saved');

  const payload = setGeoMetaMakerProfile.mock.calls[0][0];
  expect(Object.keys(payload)).toEqual(['contact', 'license']);
  expect(payload['contact']['individual_name']).toEqual(name);
  expect(payload['license']['title']).toEqual(license);

  // The alert should go away if the form data changes
  licenseInput = getByLabelText('Title');
  await user.clear(licenseInput);
  await waitFor(() => {
    expect(alert).not.toBeInTheDocument();
  });
});

test('Metadata form error on submit', async () => {
  getGeoMetaMakerProfile.mockResolvedValue({
    contact: {
      individual_name: '',
      email: '',
      organization: '',
      position_name: '',
    },
    license: {
      title: '',
      path: '',
    },
  });
  const alertMessage = 'Something went wrong';
  setGeoMetaMakerProfile.mockResolvedValue({
    message: alertMessage,
    error: true,
  });

  const user = userEvent.setup();
  const {
    findByRole,
    getByLabelText,
    getByRole,
  } = render(<MetadataForm />);

  const submit = getByRole('button', { name: /save metadata/i });
  await user.click(submit);

  const alert = await findByRole('alert');
  expect(alert).toHaveTextContent(alertMessage);

  // The alert should persist until the form is re-submit
  const licenseInput = getByLabelText('Title');
  await user.type(licenseInput, 'foo');
  expect(alert).toHaveTextContent(alertMessage);

  const successMessage = 'success';
  setGeoMetaMakerProfile.mockResolvedValue({
    message: successMessage,
    error: false,
  });
  await user.click(submit);
  await waitFor(() => {
    expect(alert).toHaveTextContent(successMessage);
  });
});
