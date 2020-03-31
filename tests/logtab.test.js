import path from 'path';
import fs from 'fs';
import os from 'os';
import React from 'react';
import { fireEvent, render,
         wait, waitForElement } from '@testing-library/react'
import '@testing-library/jest-dom'

import { LogTab } from '../src/components/LogTab';

const TEMP_LOG_DIR = path.resolve('tests/data/tmp-log');

beforeAll(() => {
  fs.mkdirSync(TEMP_LOG_DIR)
})

afterAll(() => {
  fs.rmdirSync(TEMP_LOG_DIR)
})

describe('LogTab: integration testing', () => {

  const logFile = path.resolve(path.join(TEMP_LOG_DIR, 'log.txt'))
  const logContent = 'This is a fake invest logfile'

  beforeEach(() => {
    // It's critical to add this line-ending as the file is
    // read back in with `tail.on('line'...)`
    fs.writeFileSync(logFile, logContent + os.EOL)
  })

  afterEach(() => {
    fs.unlinkSync(logFile)
  })

  test('LogTab: integration testing', async () => {
    /* TODO:
    It is bad practice to call `rerender` in order to update props.
    It would be much better to manipulate the parent component
    and let the props get passed around naturally. But in this
    case it is the `investExecute` method of the parent 
    (child_process.spawn callbacks in particular) that update 
    these props. And we don't yet have a good way to mock that.
    */
    const { getByText, queryByText, rerender, debug } = render(
      <LogTab
        jobStatus={null}
        logfile={null}
        logStdErr={null}
      />)
    // not much to expect yet but a div with an empty string
    expect(true)

    // After a props update, there should be placeholder text
    // until the invest logfile is present on disk
    rerender(
      <LogTab
        jobStatus={null}
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
        logfile={logFile}
        logStdErr={null}
      />)
    await wait(() => {
      expect(getByText(
        logContent,
        { exact: false })).toBeInTheDocument()
    })

    // When the subprocess exits without error:
    rerender(
      <LogTab
        jobStatus='success'
        logfile={logFile}
        logStdErr={null}
      />)
    await wait(() => {
      expect(getByText('Model Completed')).toBeInTheDocument()
      expect(getByText('Open Workspace')).toBeEnabled()
    }) 

    // Then maybe a new invest subprocess starts
    rerender(
      <LogTab
        jobStatus='running'
        logfile={logFile}
        logStdErr={null}
      />)
    await wait(() => {
      expect(getByText(
        logContent,
        { exact: false })).toBeInTheDocument()
    }) 
    expect(queryByText('Model Completed')).toBeNull()
    expect(queryByText('Open Workspace')).toBeNull()

    // And that subprocess exits with an error
    rerender(
      <LogTab
        jobStatus='error'
        logfile={logFile}
        logStdErr={'ValueError: bad data'}
      />)
    await wait(() => {
      expect(getByText(
        logContent,
        { exact: false })).toBeInTheDocument()
      expect(getByText('ValueError: bad data')).toHaveClass('alert-danger')
      expect(getByText('Open Workspace')).toBeEnabled()
    }) 
  })
})

describe('LogTab: non-fatal stderr does not raise Alert', () => {
    
  const logfile = path.resolve(path.join(TEMP_LOG_DIR, 'log.txt'))
  const nonFatalStdErr = 'GDAL ERROR 4'

  beforeEach(() => {
    // It's critical to add this line-ending as the file is
    // read back in with `tail.on('line'...`
    fs.writeFileSync(logfile, nonFatalStdErr + os.EOL)
  })

  afterEach(() => {
    fs.unlinkSync(logfile)
  })

  test('LogTab: non-fatal stderr does not raise Alert', async () => {

    const { getByText, queryByText, rerender, debug } = render(
      <LogTab
        jobStatus={null}
        logfile={null}
        logStdErr={null}
      />)

    // This will start tailing the log
    rerender(<LogTab
        jobStatus={'running'}
        logfile={logfile}
        logStdErr={nonFatalStdErr}
      />)

    rerender(<LogTab
        jobStatus={'success'}
        logfile={logfile}
        logStdErr={nonFatalStdErr}
      />)

    await wait(() => {
      const content = getByText(nonFatalStdErr)
      expect(content).toBeInTheDocument()
      expect(content).not.toHaveClass('alert-danger')
      expect(getByText('Model Completed')).toBeInTheDocument()
    })
  })
})