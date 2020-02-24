import React from 'react';
import { fireEvent, render, wait, waitForElement } from '@testing-library/react'

import { InvestJob } from '../src/InvestJob';

import SAMPLE_SPEC from './data/carbon_args_spec.json';

import { getSpec, saveToPython, writeParametersToFile,
         fetchValidation } from '../src/server_requests';
jest.mock('../src/server_requests');
getSpec.mockResolvedValue({});
fetchValidation.mockResolvedValue({});


test('Clicking an invest button renders SetupTab', async () => {
  getSpec.mockResolvedValue(SAMPLE_SPEC);
  const spy = jest.spyOn(InvestJob.prototype, 'investGetSpec');

  const { getByText, debug } = render(
    <InvestJob 
      investList={{Carbon: {internal_name: 'carbon'}}}
      investSettings={null}
      recentSessions={[]}
      updateRecentSessions={() => {}}
      saveSettings={() => {}}
    />);
  const carbon = getByText('Carbon');
  fireEvent.click(carbon);  // Choosing a model from the list
  await wait(() => {
    const execute = getByText('Execute');
    // Expect a disabled Execute button:
    expect(execute).toBeTruthy();
    expect(execute.hasAttribute('disabled')).toBeFalsy();
  });
  
  expect(spy).toHaveBeenCalledTimes(1);  // the click handler
  expect(getSpec).toHaveBeenCalledTimes(1);  // the wrapper around fetch
})