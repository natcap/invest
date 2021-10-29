/* These tests cover how an existing logfile is rendered. Other tests in
app.test.js cover the integrations between the log components and others,
like how starting and stopping invest subprocesses trigger log updates.
*/
import path from 'path';
import fs from 'fs';
import React from 'react';
import { render } from '@testing-library/react';
import '@testing-library/jest-dom';

import LogTab from '../../src/renderer/components/LogTab';
import {
  setupInvestLogReaderHandler,
  markupMessage,
  LOG_PATTERNS,
} from '../../src/main/setupInvestHandlers';
import { removeIpcMainListeners } from '../../src/main/main';

function renderLogTab(logfilePath, primaryPythonLogger) {
  const { ...utils } = render(
    <LogTab
      executeClicked={false}
      jobID="foo"
      logfile={logfilePath}
      pyModuleName={primaryPythonLogger}
    />
  );
  return utils;
}

function makeLogFile(text) {
  const workspace = fs.mkdtempSync('tests/data-log-');
  const logfilePath = path.join(workspace, 'logfile.txt');
  fs.writeFileSync(logfilePath, text);
  return logfilePath;
}

function cleanupLogFile(logfilePath) {
  fs.unlink(logfilePath, (err) => {
    if (err) { throw err; }
    fs.rmdirSync(path.dirname(logfilePath));
  });
}

describe('LogTab displays log from a file', () => {
  const uniqueText = 'utils.prepare_workspace';
  const primaryPythonLogger = 'natcap.invest.annual_water_yield';

  const logText = `
2021-01-15 07:14:37,147 (natcap.invest.utils) ${uniqueText}(124) INFO Writing log ...
2021-01-15 07:14:37,147 (__main__) cli.main(521) Level 100 Starting model with parameters: 
Arguments for InVEST ${primaryPythonLogger} 3.9.0.post147+gcc5a7cfe:
biophysical_table_path        C:/Users/dmf/projects/invest/data/biophysical_table_gura.csv
workspace_dir                 C:/Users/dmf/projects/invest-workbench/runs/awy

2021-01-15 07:14:37,148 (${primaryPythonLogger}) annual_water_yield.execute(268) INFO Validating arguments
2021-01-15 07:14:37,148 (natcap.invest.validation) validation._wrapped_validate_func(915) INFO ...
2021-01-15 07:14:37,525 (taskgraph.Task) Task.__init__(333) WARNING the ...
2021-01-15 07:14:37,636 (pygeoprocessing.geoprocessing) geoprocessing.align_and_resize_raster_stack(795) INFO ...

2021-01-19 14:08:32,779 (taskgraph.Task) Task.add_task(781) ERROR Something went wrong when adding task ...
Traceback (most recent call last):
  File site-packages/natcap/invest/utils.py, line 860, in reclassify_raster
  File site-packages/pygeoprocessing/geoprocessing.py, line 1836, in reclassify_raster
  File site-packages/pygeoprocessing/geoprocessing.py, line 438, in raster_calculator
  File site-packages/pygeoprocessing/geoprocessing.py, line 1829, in _map_dataset_to_value_op
pygeoprocessing.geoprocessing.ReclassificationMissingValuesError: (The following 1 raster values [8] from ...

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File site-packages/taskgraph/Task.py, line 747, in add_task
  File site-packages/taskgraph/Task.py, line 1234, in _call
  File site-packages/natcap/invest/sdr/sdr.py, line 997, in _calculate_w
  File site-packages/natcap/invest/utils.py, line 868, in reclassify_raster
ValueError: Values in the LULC raster were found that are not represented under the lucode column of the Biophysical table....
2021-01-19 14:08:32,780 (natcap.invest.utils) utils.prepare_workspace(130) INFO Elapsed time: 3.5s
`;

  let logfilePath;
  beforeAll(() => {
    logfilePath = makeLogFile(logText);
    setupInvestLogReaderHandler();
  });

  afterAll(() => {
    removeIpcMainListeners();
    cleanupLogFile(logfilePath);
  });

  test('Text in logfile is rendered', async () => {
    const { findByText } = renderLogTab(
      logfilePath, primaryPythonLogger
    );

    const log = await findByText(new RegExp(uniqueText));
    expect(log).toBeInTheDocument();
  });

  test('message from non-primary invest logger is plain', async () => {
    const { findByText } = renderLogTab(
      logfilePath, primaryPythonLogger
    );

    const log = await findByText(new RegExp(uniqueText));
    expect(log).not.toHaveClass();
  });

  // Skip because https://github.com/natcap/invest-workbench/issues/169
  test.skip('messages from primary invest logger are highlighted', async () => {
    const { findAllByText } = renderLogTab(
      logfilePath, primaryPythonLogger
    );

    const messages = await findAllByText(new RegExp(primaryPythonLogger));
    messages.forEach((msg) => {
      expect(msg).toHaveClass('invest-log-primary');
    });
  });

  // Skip because https://github.com/natcap/invest-workbench/issues/169
  test.skip('error messages are highlighted', async () => {
    const { findAllByText } = renderLogTab(
      logfilePath, primaryPythonLogger
    );

    // The start of a python traceback
    let errorMessages = await findAllByText(/Traceback/);
    errorMessages.forEach((msg) => {
      expect(msg).toHaveClass('invest-log-error');
    });

    // The indented contents of a python traceback
    errorMessages = await findAllByText(/File site-packages/);
    errorMessages.forEach((msg) => {
      expect(msg).toHaveClass('invest-log-error');
    });

    // an ERROR-level message from a python logger
    errorMessages = await findAllByText(/ERROR Something went wrong/);
    errorMessages.forEach((msg) => {
      expect(msg).toHaveClass('invest-log-error');
    });

    // a message from a custom python exception class
    errorMessages = await findAllByText(/ReclassificationMissingValuesError/);
    errorMessages.forEach((msg) => {
      expect(msg).toHaveClass('invest-log-error');
    });

    // a message from a built-in python exception class
    errorMessages = await findAllByText(/ValueError:/);
    errorMessages.forEach((msg) => {
      expect(msg).toHaveClass('invest-log-error');
    });
  });
});

describe('Unit tests for invest logger message markup', () => {
  const pyModuleName = 'natcap.invest.carbon';

  test('Message from the invest model gets primary class attribute', () => {
    const logPatterns = { ...LOG_PATTERNS };
    logPatterns['invest-log-primary'] = new RegExp(pyModuleName);
    const message = `2021-01-15 07:14:37,148 (${pyModuleName}) ... INFO`;
    const markup = markupMessage(message, logPatterns);
    // Rendering and using DOM matchers adds confidence that we have valid html
    // Render the same way we do in LogDisplay component:
    const { getByText } = render(
      <div dangerouslySetInnerHTML={{ __html: markup }} />
    );
    expect(getByText(message)).toHaveClass('invest-log-primary');
  });

  test.each([
    '... (osgeo.gdal) ... ERROR',
    '... (foo.bar) ... SomeCustomError',
    `... (${pyModuleName}) ... ValueError`,
    'Traceback...',
  ])('Message "%s" gets error class attribute', (message) => {
    const logPatterns = { ...LOG_PATTERNS };
    logPatterns['invest-log-primary'] = new RegExp(pyModuleName);
    const markup = markupMessage(message, logPatterns);
    const { getByText } = render(
      <div dangerouslySetInnerHTML={{ __html: markup }} />
    );
    expect(getByText(message)).toHaveClass('invest-log-error');
  });

  test('All other messages do not get markup', () => {
    const logPatterns = { ...LOG_PATTERNS };
    logPatterns['invest-log-primary'] = new RegExp(pyModuleName);
    const message = '2021-01-15 07:14:37,148 (foo.bar) ... INFO';
    const markup = markupMessage(message, logPatterns);
    const { getByText } = render(
      <div dangerouslySetInnerHTML={{ __html: markup }} />
    );
    expect(getByText(message)).not.toHaveClass();
  });

  test('Messages pass through if pattern is not a RegExp', () => {
    const message = `2021-01-15 07:14:37,148 (${pyModuleName}) ... INFO`;
    expect(LOG_PATTERNS['invest-log-primary']).toBeNull();
    const markup = markupMessage(message, LOG_PATTERNS);
    const { getByText } = render(
      <div dangerouslySetInnerHTML={{ __html: markup }} />
    );
    expect(getByText(message)).not.toHaveClass();
  });
});
