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

const DIRECTORY_CONSTANTS = {
  CACHE_DIR: 'tests/data/testing-cache', //  for storing state snapshot files
  TEMP_DIR: 'tests/data/testing-tmp',  // for saving datastack json files prior to investExecute
  INVEST_UI_DATA: 'tests/data/testing-ui_data'
}

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

function cleanupDir(dir) {
  fs.readdirSync(dir).forEach(file => {
    fs.unlinkSync(path.join(dir, file))
  })
  fs.rmdirSync(dir)
}

beforeAll(() => {
  for (const dir in DIRECTORY_CONSTANTS) {
    fs.mkdirSync(DIRECTORY_CONSTANTS[dir])
  }
})

afterAll(() => {
  for (const dir in DIRECTORY_CONSTANTS) {
    cleanupDir(DIRECTORY_CONSTANTS[dir])
  }
})

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
      directoryConstants={DIRECTORY_CONSTANTS}
    />);
  // fireEvent.click(getByText('Carbon'));
  return { getByText, getByLabelText, utils }
}

test('SetupTab: expect an input form for a directory', async () => {
  const spec = { args: { arg: { 
    name: 'Workspace',
    type: 'directory',
    about: 'this is a workspace' } } }
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

  // Expect the info dialog contains the about text, when clicked
  fireEvent.click(getByText('i'))
  expect(getByText(spec.args.arg.about)).toBeTruthy()
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

test('SetupTab: test a UI spec with a boolean controller arg', async () => {
  
  const spec = { module: 'natcap.invest.dummy', args: { 
    controller: { 
      name: 'A', 
      type: 'boolean'}, 
    arg2: { 
      name: 'B', 
      type: 'number'}, 
    arg3: { 
      name: 'C', 
      type: 'number'}, 
    arg4: { 
      name: 'D', 
      type: 'number'},
    arg5: {
      name: 'E',
      type: 'number'
    } } }

  const ui_spec = {
    controller: { 
      ui_control: ['arg2', 'arg3', 'arg4'], },
    arg2: { 
      ui_option: 'disable', },
    arg3: { 
      ui_option: 'hide', },
    arg4: { 
      ui_option: 'foo', } }  // an invalid option should be ignored
    // arg5 is deliberately missing from ui_spec to demonstrate that that is okay. 

  fs.writeFileSync(path.join(
    DIRECTORY_CONSTANTS.INVEST_UI_DATA, spec.module + '.json'),
      JSON.stringify(ui_spec))
  
  fetchValidation.mockResolvedValue([])
  const { getByText, getByLabelText, utils } = renderSetupFromSpec(spec)
  fireEvent.click(getByText('Carbon'));

  const controller = await utils.findByLabelText(spec.args.controller.name)
  const arg2 = await utils.findByLabelText(spec.args.arg2.name)
  const arg3 = await utils.findByLabelText(spec.args.arg3.name)
  const arg4 = await utils.findByLabelText(spec.args.arg4.name)
  const arg5 = await utils.findByLabelText(spec.args.arg5.name)

  // Boolean Radios should default to "false" when a spec is loaded
  await wait(() => {
    expect(arg2).toBeDisabled();
    // The 'hide' style is applied to the whole Form.Group which 
    // includes the Label and Input. Right now, the only good way
    // to query the Form.Group node is using a data-testid property.
    expect(utils.getByTestId('group-arg3')).toHaveClass('arg-hide')

    // These inputs are not being controlled, but should exist:
    expect(arg4).toBeVisible();
    expect(arg4).toBeEnabled();
    expect(arg5).toBeVisible();
    expect(arg5).toBeEnabled();
  })
  // fireEvent.change doesn't trigger the change handler but .click does
  // even though React demands an onChange handler for controlled checkbox inputs.
  // https://github.com/testing-library/react-testing-library/issues/156
  fireEvent.click(controller, { target: { value: "true" } })

  // Now everything should be visible/enabled.
  await wait(() => {
    expect(arg2).toBeEnabled();
    expect(arg2).toBeVisible();
    expect(arg3).toBeEnabled();
    expect(arg3).toBeVisible();
    expect(arg4).toBeEnabled();
    expect(arg4).toBeVisible();
    expect(arg5).toBeEnabled();
    expect(arg5).toBeVisible();
  })
})

test('SetupTab: expect non-boolean input can disable/hide optional inputs', async () => {
  // Normally the UI options are loaded from a seperate spec on disk
  // that is merged with ARGS_SPEC. But for testing, it's convenient
  // to just use one spec. And it works just the same.
  const spec = { args: { 
    controller: { 
      name: 'A', 
      type: 'csv', 
      ui_control: ['arg2'], },
    arg2: { 
      name: 'B', 
      type: 'number', 
      ui_option: 'disable', } } }
  
  fetchValidation.mockResolvedValue([])
  const { getByText, getByLabelText, utils } = renderSetupFromSpec(spec)
  fireEvent.click(getByText('Carbon'));

  const controller = await utils.findByLabelText(spec.args.controller.name)
  const arg2 = await utils.findByLabelText(spec.args.arg2.name)

  // The optional input should be disabled while the controlling input
  // has a falsy value (undefined or '')
  await wait(() => {
    expect(arg2).toBeDisabled();
  })

  fireEvent.change(controller, { target: { value: "foo.csv" } })

  // Now everything should be enabled.
  await wait(() => {
    expect(arg2).toBeEnabled();
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