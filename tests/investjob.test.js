import React from 'react';
import { fireEvent, render, wait, waitForElement } from '@testing-library/react'
import '@testing-library/jest-dom'

import { InvestJob } from '../src/InvestJob';
import SAMPLE_SPEC from './data/carbon_args_spec.json';
import { getSpec, saveToPython, writeParametersToFile,
         fetchValidation } from '../src/server_requests';
jest.mock('../src/server_requests');

test('Clicking an invest button renders SetupTab', async () => {
  getSpec.mockResolvedValue(SAMPLE_SPEC);
  fetchValidation.mockResolvedValue([[['workspace_dir'], 'invalid because']]);
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
    // Expect a disabled Execute button and a visible SetupTab
    expect(execute).toBeTruthy();
    expect(execute).toBeDisabled();  // depends on the mocked fetchValidation
    expect(getByText('Setup').classList.contains('active')).toBeTruthy();
  });
  
  expect(spy).toHaveBeenCalledTimes(1);  // the click handler
  expect(getSpec).toHaveBeenCalledTimes(1);  // the wrapper around fetch
})

test('Clicking a recent session renders SetupTab', async () => {
  fetchValidation.mockResolvedValue([[['workspace_dir'], 'invalid because']]);
  const spy = jest.spyOn(InvestJob.prototype, 'loadState');

  const { getByText, debug } = render(
    <InvestJob 
      investList={{}}
      investSettings={null}
      recentSessions={['carbon_setup']}
      updateRecentSessions={() => {}}
      saveSettings={() => {}}
    />);
  const recent = getByText('carbon_setup');
  fireEvent.click(recent);  // a recent session button
  await wait(() => {
    const execute = getByText('Execute');
    // Expect a disabled Execute button and a visible SetupTab
    expect(execute).toBeTruthy();
    expect(execute).toBeDisabled(); // depends on the mocked fetchValidation
    expect(getByText('Setup').classList.contains('active')).toBeTruthy();
    // TODO: toBeVisible doesn't work w/ the attributes on these nodes
    // expect(getByText('Setup')).toBeVisible();
    // expect(getByText('Resources')).not.toBeVisible();
  });
  
  expect(spy).toHaveBeenCalledTimes(1);  // called by the click handler
})

// test('Browsing for recent session renders SetupTab', async () => {
//   // TODO: This functionality might be dropped.
// })

test('Save Parameters/Python enable after model select ', async () => {

  const { getByText, debug } = render(
    <InvestJob 
      investList={{}}
      investSettings={null}
      recentSessions={['carbon_setup']}
      updateRecentSessions={() => {}}
      saveSettings={() => {}}
    />);

  // Check the dropdown before any model setup
  fireEvent.click(getByText('Save'));
  await wait(() => {
    expect(getByText('Save parameters to JSON')).toBeDisabled();
    expect(getByText('Save to Python script')).toBeDisabled();
  })

  // Now load a model setup using a recent session
  fireEvent.click(getByText('carbon_setup'));
  await wait(() => {
    expect(getByText('Save parameters to JSON')).toBeEnabled();
    expect(getByText('Save to Python script')).toBeEnabled();
  });
})


// Tests:
test('Populating arguments enables Execute button')
test('Execute launches a python subprocess and switches to Log display')