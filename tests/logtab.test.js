import path from 'path';
import React from 'react';
import { fireEvent, render,
         wait, waitForElement } from '@testing-library/react'
import '@testing-library/jest-dom'

import { LogTab } from '../src/components/LogTab';

test('LogTab: before there is a logfile', async () => {
  /*
  It is bad practice to call `rerender` in order to update props.
  It would be much better to manipulate the parent component
  and let the props get passed around naturally. But in this
  case it is the `investExecute` method of the parent 
  (child_process.spawn callbacks in particular) that update 
  these props. And we don't yet have a good way to mock that.
  */
  const { getByText, queryByText, rerender, debug } = render(
    <LogTab
      sessionProgress={null}
      logfile={null}
      logStdErr={null}
    />)
  // not much to expect yet but a div with an empty string
  expect(true)

  // After a props update, there should be placeholder text
  // until the invest logfile is present on disk
  rerender(
    <LogTab
      sessionProgress={null}
      logfile={null}
      logStdErr={null}
    />)
  await wait(() => {
    expect(getByText('Starting...')).toBeInTheDocument()
  })

  // Once the subprocess is running, there is a logfile
  // and we should see it's contents.
  rerender(
    <LogTab
      jobStatus={'running'}
      logfile={path.resolve('tests/data/invest_logfile.txt')}
      logStdErr={null}
    />)
  await wait(() => {
    expect(getByText(
      'This is a fake invest logfile',
      { exact: false })).toBeInTheDocument()
  })

  // When the subprocess exits without error:
  rerender(
    <LogTab
      jobStatus='success'
      logfile={path.resolve('tests/data/invest_logfile.txt')}
      logStdErr={null}
    />)
  await wait(() => {
    expect(getByText('Model Completed')).toBeInTheDocument()
  }) 

  // Then maybe a new invest subprocess starts
  rerender(
    <LogTab
      jobStatus='running'
      logfile={path.resolve('tests/data/invest_logfile.txt')}
      logStdErr={null}
    />)
  await wait(() => {
    expect(getByText(
      'This is a fake invest logfile',
      { exact: false })).toBeInTheDocument()
  }) 
  expect(queryByText('Model Completed')).toBeNull()

  // And that subprocess exits with an error
  rerender(
    <LogTab
      jobStatus='error'
      logfile={path.resolve('tests/data/invest_logfile.txt')}
      logStdErr={'ValueError: bad data'}
    />)
  await wait(() => {
    expect(getByText(
      'This is a fake invest logfile',
      { exact: false })).toBeInTheDocument()
    expect(getByText('ValueError: bad data')).toHaveClass('alert-danger')
  }) 
})