import React from 'react';
import { render, fireEvent, waitForElement } from '@testing-library/react'

import { InvestJob } from '../src/InvestJob';

import SAMPLE_SPEC from './data/carbon_args_spec.json';
import fetch from 'node-fetch';
jest.mock('node-fetch');


test('click on an invest button enables SetupTab', async () => {
  // This is the value I want returned by the first call to post,
  // from within getInvestSpec. But then there are other posts that
  // happen subsequently, which should return other things.
  // request.post.mockResolvedValue(SAMPLE_SPEC)
  fetch.mockResolvedValue(SAMPLE_SPEC);
  // const getSpecSpy = jest.spyOn(getSpec);
  const spy = jest.spyOn(InvestJob.prototype, 'investGetSpec');

  const { getByText, debug } = render(
    <InvestJob 
      investList={{Carbon: {internal_name: 'carbon'}}}
      investSettings={null}
      recentSessions={[]}
      updateRecentSessions={() => {}}
      saveSettings={() => {}}
    />);
  // debug();
  console.log('PRE-CLICK');
  fireEvent.click(getByText('Carbon'));
  console.log('POST-CLICK');
  // debug();
  const setupNode = await waitForElement(() => {
    getByText('Execute')
  })
  expect(getByText('Execute')).toHaveAttribute('disabled');
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(spy).toHaveBeenCalledTimes(1);
  // expect(getByText('Setup')).toHaveAttribute('disabled');
})