import fs from 'fs';
import path from 'path';
import React from 'react';
import { remote } from 'electron';
import { createEvent, fireEvent, render,
         wait, waitForElement } from '@testing-library/react'
import '@testing-library/jest-dom'

import { InvestJob } from '../src/InvestJob';
// import { SetupTab } from '../src/components/SetupTab';
import { getSpec, fetchDatastackFromFile, fetchValidation } from '../src/server_requests';
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

beforeEach(() => {
  jest.resetAllMocks()
})

function renderSetupFromSpec(spec) {
  getSpec.mockResolvedValue(spec);
  const { getByText, getByLabelText, ...utils } = render(
    <InvestJob 
      investList={{Carbon: {internal_name: 'carbon'}}}
      investSettings={null}
      recentSessions={[]}
      updateRecentSessions={() => {}}
      saveSettings={() => {}}
    />);
  // fireEvent.click(getByText('Carbon'));
  return { getByText, getByLabelText, utils }
}

test('SetupTab: expect an input form for a directory', async () => {
  const spec = { args: { arg: { name: 'Workspace', type: 'directory' } } }
  fetchValidation.mockResolvedValue(
    [[Object.keys(spec.args), 'invalid because']])
  const { getByText, getByLabelText, utils } = renderSetupFromSpec(spec)
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

test('SetupTab: expect an input form for a csv', async () => {
  /** Also testing the browse button functionality */

  const spec = { args: { arg: { name: 'foo', type: 'csv' } } }
  fetchValidation.mockResolvedValue(
    [[Object.keys(spec.args), 'invalid because']])
  const { getByText, getByLabelText, utils } = renderSetupFromSpec(spec)
  fireEvent.click(getByText('Carbon'));
  const input = await waitForElement(() => {
    return getByLabelText(spec.args.arg.name)
  })
  // Typing in a value
  fireEvent.change(input, { target: { value: 'foo' } })
  await wait(() => {
    expect(input).toHaveValue('foo')
    expect(input).toHaveAttribute('type', 'text')
    expect(input.classList.contains('is-invalid')).toBeTruthy();
    expect(getByText('Browse'))
    expect(getByText('invalid because', { exact: false }))
  })

  // Browsing for a file
  const filepath = 'grilled_cheese.csv'
  let mockDialogData = { filePaths: [filepath] }
  remote.dialog.showOpenDialog.mockResolvedValue(mockDialogData)
  fireEvent.click(getByText('Browse'))
  await wait(() => {
    expect(input).toHaveValue(filepath)
    expect(input.classList.contains('is-invalid')).toBeTruthy();
    expect(getByText('invalid because', { exact: false }))
  })

  // Now browse again, but this time cancel it and expect the previous value
  mockDialogData = { filePaths: [] } // empty array is a mocked 'Cancel'
  remote.dialog.showOpenDialog.mockResolvedValue(mockDialogData)
  fireEvent.click(getByText('Browse'))
  await wait(() => {
    expect(input).toHaveValue(filepath)
    expect(input.classList.contains('is-invalid')).toBeTruthy();
    expect(getByText('invalid because', { exact: false }))
  })

})

test('SetupTab: expect an input form for a vector', async () => {
  const spec = { args: { arg: { name: 'foo', type: 'vector' } } }
  fetchValidation.mockResolvedValue(
    [[Object.keys(spec.args), 'invalid because']])
  const { getByText, getByLabelText, utils } = renderSetupFromSpec(spec)
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

test('SetupTab: expect an input form for a raster', async () => {
  const spec = { args: { arg: { name: 'foo', type: 'raster' } } }
  fetchValidation.mockResolvedValue(
    [[Object.keys(spec.args), 'invalid because']])
  const { getByText, getByLabelText, utils } = renderSetupFromSpec(spec)
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

test('SetupTab: expect an input form for a freestyle_string', async () => {
  const spec = { args: { arg: { name: 'foo', type: 'freestyle_string' } } }
  fetchValidation.mockResolvedValue([])
  const { getByText, getByLabelText, utils } = renderSetupFromSpec(spec)
  fireEvent.click(getByText('Carbon'));
  const input = await waitForElement(() => {
    return getByLabelText(spec.args.arg.name)
  })
  fireEvent.change(input, { target: { value: 'foo' } })
  await wait(() => {
    expect(input).toHaveValue('foo')
    expect(input).toHaveAttribute('type', 'text')
    // Not really possible to invalidate a freestyle_string
    expect(input.classList.contains('is-invalid')).toBeFalsy();
  })
})

test('SetupTab: expect an input form for a number', async () => {
  const spec = { args: { arg: { name: 'foo', type: 'number' } } }
  fetchValidation.mockResolvedValue(
    [[Object.keys(spec.args), 'invalid because']])
  const { getByText, getByLabelText, utils } = renderSetupFromSpec(spec)
  fireEvent.click(getByText('Carbon'));
  const input = await waitForElement(() => {
    return getByLabelText(spec.args.arg.name)
  })
  fireEvent.change(input, { target: { value: 'foo' } })
  await wait(() => {
    expect(input).toHaveValue('foo')
    expect(input).toHaveAttribute('type', 'text')
    expect(input.classList.contains('is-invalid')).toBeTruthy();
    expect(getByText('invalid because', { exact: false }))
  })
})

test('SetupTab: expect an input form for a boolean', async () => {
  const spec = { args: { arg: { name: 'foo', type: 'boolean' } } }
  fetchValidation.mockResolvedValue([])
  const { getByText, getByLabelText, utils } = renderSetupFromSpec(spec)
  fireEvent.click(getByText('Carbon'));
  const input = await waitForElement(() => {
    return getByLabelText(spec.args.arg.name)
  })
  await wait(() => {
    expect(input).toHaveAttribute('type', 'radio')
  })
})

test('SetupTab: expect an input form for an option_string', async () => {
  const spec = { args: { arg: { 
    name: 'foo', 
    type: 'option_string', 
    validation_options: { options: ['a', 'b'] } } } }
  fetchValidation.mockResolvedValue([])
  const { getByText, getByLabelText, utils } = renderSetupFromSpec(spec)
  fireEvent.click(getByText('Carbon'));
  const input = await waitForElement(() => {
    return getByLabelText(spec.args.arg.name)
  })
  await wait(() => {
    expect(input).toHaveValue('a');
    expect(input).not.toHaveValue('b');
  })
})

test('SetupTab: populating inputs to enable & disable Execute', async () => {
  /*
  This tests that changes to input values trigger validation. 
  The fetchValidation return value is always mocked, but then this
  also tests that validation results correctly enable/disable the 
  Execute button and display feedback messages on invalid inputs.
  */
  const spec = { args: {
    a: { 
      name: 'a', 
      type: 'freestyle_string'},
    b: {
      name: 'b', 
      type: 'number'},
    c: {
      name: 'c',
      type: 'csv'} } }

  const { getByText, getByLabelText, utils } = renderSetupFromSpec(spec)
  
  let invalidFeedback = 'is a required key'
  fetchValidation.mockResolvedValue([[['a', 'b'], invalidFeedback]])
  fireEvent.click(getByText('Carbon')); // triggers validation
  await wait(() => {
    expect(getByText('Execute')).toBeDisabled();
    // The inputs are invalid so the invalid feedback message is present.
    // But, the inputs have not yet been touched, so the message is hidden
    // by CSS 'display: none'. Unfortunately, the bootstrap stylesheet is
    // not loaded in this testing DOM, so cannot assert the message is not visible.
    utils.getAllByText(invalidFeedback, { exact: false }).forEach(element => {
      expect(element).toBeInTheDocument()
      // Would be nice if these worked, but they do not:
      // expect(element).not.toBeVisible()
      // expect(element).toHaveStyle('display: none')
    })
  })
  
  const [a, b, c] = await waitForElement(() => {
    return [
      getByLabelText(spec.args.a.name),
      getByLabelText(spec.args.b.name),
      getByLabelText(spec.args.c.name)]
  })

  // These new values will be valid - Execute should enable
  fetchValidation.mockResolvedValue([])
  fireEvent.change(a, { target: { value: 'foo' } })  // triggers validation
  fireEvent.change(b, { target: { value: 1 } })      // triggers validation
  await wait(() => {
    expect(getByText('Execute')).toBeEnabled();
  })
  // Now that inputs are valid, feedback message should be cleared:
  // Note: Can't put this inside wait - it will timeout waiting to be not null.
  // But it does rely on waiting for the change event to propogate. 
  // Putting it after the above `await` works.
  utils.queryAllByText(invalidFeedback, { exact: false }).forEach(element => {
    expect(element).toBeNull()
  })

  // This new value will be invalid - Execute should disable
  invalidFeedback = 'must be a number';
  fetchValidation.mockResolvedValue([[['b'], invalidFeedback]])
  fireEvent.change(b, { target: { value: 'one' } })  // triggers validation
  await wait(() => {
    expect(getByText('Execute')).toBeDisabled();
    expect(getByText(invalidFeedback, { exact: false })).toBeInTheDocument()
  })
})

test('SetupTab: test dragover of a datastack/logfile', async () => {
  /** Fire a drop event and mock the resolved datastack.
  * This expects batchUpdateArgs to update form values after the drop.
  */
  const spec = {
    args: {
      arg1: { name: 'Workspace', type: 'directory' },
      arg2: { name: 'AOI', type: 'vector' }
    },
    module: 'natcap.invest.carbon'
  }
  
  fetchValidation.mockResolvedValue(
    [[Object.keys(spec.args), 'invalid because']])
  
  const mock_datastack = {
    module_name: spec.module,
    args: { arg1: 'circle', arg2: 'square'}
  }
  fetchDatastackFromFile.mockResolvedValue(mock_datastack)

  const { getByText, getByLabelText, utils } = renderSetupFromSpec(spec)
  fireEvent.click(getByText('Carbon'));
  const setupForm = await waitForElement(() => {
    return utils.getByTestId('setup-form')
  })

  // This should work but doesn't due to lack of dataTransfer object in jsdom:
  // https://github.com/jsdom/jsdom/issues/1568
  // const dropEvent = new Event('drop', 
  //   { dataTransfer: { files: ['foo.txt'] } 
  // })
  // fireEvent.drop(setupForm, dropEvent)
  
  // Below is a patch similar to the one described here:
  // https://github.com/testing-library/react-testing-library/issues/339
  const fileDropEvent = createEvent.drop(setupForm)
  const fileArray = ['foo.txt']
  Object.defineProperty(fileDropEvent, 'dataTransfer', {
    value: { files : fileArray }
  })
  fireEvent(setupForm, fileDropEvent)

  // using `findBy...`, which returns a promise, is much better
  // than wrapping `getBy...` and the `expect` calls inside `await wait()`
  // because the latter will timeout instead of failing if the expect fails.
  // TODO: look for places to replace getBy...
  const arg1 = await utils.findByLabelText(spec.args.arg1.name)
  const arg2 = await utils.findByLabelText(spec.args.arg2.name)
  expect(arg1).toHaveValue(mock_datastack.args.arg1)
  expect(arg2).toHaveValue(mock_datastack.args.arg2)
})