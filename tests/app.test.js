import React from 'react';
import { fireEvent, render, wait } from '@testing-library/react'

import App from '../src/app';
import { investList } from '../src/server_requests';
jest.mock('../src/server_requests');

test('Settings dialog interactions', async () => {
  investList.mockResolvedValue({});
  const DEFAULT = 'INFO';

  const { getByText, getByLabelText, debug } = render(<App />);

  // Check the default settings
  fireEvent.click(getByText('Settings'));
  await wait(() => { 
    // waiting is necessary, I think because the selected value
    // depends on passed props
    expect(getByText(DEFAULT).selected).toBeTruthy();
  })

  // Change the select input and cancel -- expect default selected
  fireEvent.change(getByLabelText('Logging threshold'),
    { target: { value: 'DEBUG' } })
  fireEvent.click(getByText('Cancel'));
  // fireEvent.click(getByText('Settings'));  // why is this unecessary?
  expect(getByText(DEFAULT).selected).toBeTruthy();

  // Change the select input and save -- expect new value selected
  fireEvent.change(getByLabelText('Logging threshold'),
    { target: { value: 'DEBUG' } })
  fireEvent.click(getByText('Save Changes'));
  // fireEvent.click(getByText('Settings'));  // why is this unecessary?
  expect(getByText('DEBUG').selected).toBeTruthy();
})