import fs from 'fs';
import path from 'path';
import React from 'react';
import { fireEvent, render,
         wait, waitForElement } from '@testing-library/react'
import '@testing-library/jest-dom'
import { SetupTab } from '../src/components/SetupTab';

// generated this file from `invest getspec carbon --json`
// import ARGS_SPEC from './data/carbon_args_spec.json';

const ARG_TYPE_INPUT_MAP = {
  csv: "text",
  vector: "text",
  raster: "text",
  directory: "text",
  freestyle_string: "text",
  number: "text",
  boolean: "radio",
  option_string: "select"
}

function renderSetupFromSpec(spec) {
  const { getByText, getByLabelText } = render(
    <SetupTab
      args={spec}
      argsValid={false}
      modulename={null}
      updateArg={() => {}}
      batchUpdateArgs={() => {}}
      investValidate={() => {}}
      argsValuesFromSpec={() => {}}
      investExecute={() => {}}
    />)
  return { getByText, getByLabelText }
}

test('SetupTab: an input form for a directory', async () => {
  const spec = { arg: { name: 'Workspace', type: 'directory' } }
  const { getByText, getByLabelText } = renderSetupFromSpec(spec)
  await wait(() => {
    expect(getByLabelText(spec.arg.name)).toHaveAttribute('type', 'text')
    expect(getByText('Browse'))
  })
})

test('SetupTab: an input form for a csv', async () => {
  const spec = { arg: { name: 'foo', type: 'csv' } }
  const { getByText, getByLabelText } = renderSetupFromSpec(spec)
  await wait(() => {
    expect(getByLabelText(spec.arg.name)).toHaveAttribute('type', 'text')
    expect(getByText('Browse'))
  })
})

test('SetupTab: an input form for a vector', async () => {
  const spec = { arg: { name: 'foo', type: 'vector' } }
  const { getByText, getByLabelText } = renderSetupFromSpec(spec)
  await wait(() => {
    expect(getByLabelText(spec.arg.name)).toHaveAttribute('type', 'text')
    expect(getByText('Browse'))
  })
})

test('SetupTab: an input form for a raster', async () => {
  const spec = { arg: { name: 'foo', type: 'raster' } }
  const { getByText, getByLabelText } = renderSetupFromSpec(spec)
  await wait(() => {
    expect(getByLabelText(spec.arg.name)).toHaveAttribute('type', 'text')
    expect(getByText('Browse'))
  })
})

test('SetupTab: an input form for a freestyle_string', async () => {
  const spec = { arg: { name: 'foo', type: 'freestyle_string' } }
  const { getByText, getByLabelText } = renderSetupFromSpec(spec)
  await wait(() => {
    expect(getByLabelText(spec.arg.name)).toHaveAttribute('type', 'text')
  })
})

test('SetupTab: an input form for a number', async () => {
  const spec = { arg: { name: 'foo', type: 'number' } }
  const { getByText, getByLabelText } = renderSetupFromSpec(spec)
  await wait(() => {
    expect(getByLabelText(spec.arg.name)).toHaveAttribute('type', 'text')
  })
})

test('SetupTab: an input form for a boolean', async () => {
  const spec = { arg: { name: 'foo', type: 'boolean' } }
  const { getByText, getByLabelText } = renderSetupFromSpec(spec)
  await wait(() => {
    expect(getByLabelText(spec.arg.name)).toHaveAttribute('type', 'radio')
  })
})

test('SetupTab: an input form for an option_string', async () => {
  const spec = { arg: { 
    name: 'foo', 
    type: 'option_string', 
    validation_options: { options: ['a', 'b'] } } }
  const { getByText, getByLabelText } = renderSetupFromSpec(spec)
  await wait(() => {
    expect(getByLabelText(spec.arg.name)).toHaveFormValues('a')
  })
})
