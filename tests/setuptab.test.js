import fs from 'fs';
import path from 'path';
import React from 'react';
import { fireEvent, render,
         wait, waitForElement } from '@testing-library/react'
import '@testing-library/jest-dom'
import { InvestJob } from '../src/InvestJob';
// import { SetupTab } from '../src/components/SetupTab';
import { getSpec, fetchValidation } from '../src/server_requests';
jest.mock('../src/server_requests');

// generated this file from `invest getspec carbon --json`
// import ARGS_SPEC from './data/carbon_args_spec.json';
// const ARG_TYPE_INPUT_MAP = {
//   csv: "text",
//   vector: "text",
//   raster: "text",
//   directory: "text",
//   freestyle_string: "text",
//   number: "text",
//   boolean: "radio",
//   option_string: "select"
// }

function renderSetupFromSpec(spec) {
  getSpec.mockResolvedValue(spec);
  const {getByText, getByLabelText, ...utils} = render(
    <InvestJob 
      investList={{Carbon: {internal_name: 'carbon'}}}
      investSettings={null}
      recentSessions={[]}
      updateRecentSessions={() => {}}
      saveSettings={() => {}}
    />);
  // fireEvent.click(getByText('Carbon'));
  fetchValidation.mockResolvedValue([[Object.keys(spec.args), 'invalid because']])
  return { utils, getByText, getByLabelText }
}

test('SetupTab: an input form for a directory', async () => {
  const spec = { args: { arg: { name: 'Workspace', type: 'directory' } } }
  const {utils, getByText, getByLabelText } = renderSetupFromSpec(spec)
  fireEvent.click(getByText('Carbon'));
  const input = await waitForElement(() => {
    return getByLabelText(spec.args.arg.name)
  })
  fireEvent.change(input, { target: { value: 'foo' } })
  await wait(() => {
    expect(input).toHaveValue('foo')
    expect(input).toHaveAttribute('type', 'text')
    expect(input.classList.contains('is-invalid')).toBeTruthy();
    expect(getByText('Browse'))
    expect(getByText('invalid because', { exact: false }))
  })
})

// test('SetupTab: an input form for a csv', async () => {
//   const spec = { arg: { name: 'foo', type: 'csv' } }
//   const {utils, getByText, getByLabelText } = renderSetupFromSpec(spec)
//   await wait(() => {
//     expect(getByLabelText(spec.arg.name)).toHaveAttribute('type', 'text')
//     expect(getByText('Browse'))
//   })
// })

// test('SetupTab: an input form for a vector', async () => {
//   const spec = { arg: { name: 'foo', type: 'vector' } }
//   const {utils, getByText, getByLabelText } = renderSetupFromSpec(spec)
//   await wait(() => {
//     expect(getByLabelText(spec.arg.name)).toHaveAttribute('type', 'text')
//     expect(getByText('Browse'))
//   })
// })

// test('SetupTab: an input form for a raster', async () => {
//   const spec = { arg: { name: 'foo', type: 'raster' } }
//   const {utils, getByText, getByLabelText } = renderSetupFromSpec(spec)
//   await wait(() => {
//     expect(getByLabelText(spec.arg.name)).toHaveAttribute('type', 'text')
//     expect(getByText('Browse'))
//   })
// })

// test('SetupTab: an input form for a freestyle_string', async () => {
//   const spec = { arg: { name: 'foo', type: 'freestyle_string' } }
//   const {utils, getByText, getByLabelText } = renderSetupFromSpec(spec)
//   await wait(() => {
//     expect(getByLabelText(spec.arg.name)).toHaveAttribute('type', 'text')
//   })
// })

// test('SetupTab: an input form for a number', async () => {
//   const spec = { arg: { name: 'foo', type: 'number' } }
//   const {utils, getByText, getByLabelText } = renderSetupFromSpec(spec)
//   await wait(() => {
//     expect(getByLabelText(spec.arg.name)).toHaveAttribute('type', 'text')
//   })
// })

// test('SetupTab: an input form for a boolean', async () => {
//   const spec = { arg: { name: 'foo', type: 'boolean' } }
//   const {utils, getByText, getByLabelText } = renderSetupFromSpec(spec)
//   await wait(() => {
//     expect(getByLabelText(spec.arg.name)).toHaveAttribute('type', 'radio')
//   })
// })

// test('SetupTab: an input form for an option_string', async () => {
//   const spec = { arg: { 
//     name: 'foo', 
//     type: 'option_string', 
//     validation_options: { options: ['a', 'b'] } } }
//   const {utils, getByText, getByLabelText } = renderSetupFromSpec(spec)
//   await wait(() => {
//     // utils.debug(getByLabelText(spec.arg.name))
//     expect(getByLabelText(spec.arg.name)).toHaveValue('a');
//     expect(getByLabelText(spec.arg.name)).not.toHaveValue('b');
//   })
// })
