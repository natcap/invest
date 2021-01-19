import path from 'path';
import fs from 'fs';
import React from 'react';
import {
  fireEvent, render, waitFor, within
} from '@testing-library/react';
import '@testing-library/jest-dom';

import LogTab from '../src/components/LogTab';
import { cleanupDir } from '../src/utils';

describe('LogTab', () => {
  let workspace;
  let logfilePath;
  const logfileName = 'logfile.txt';
  const uniqueText = 'utils.prepare_workspace';
  const primaryPythonLogger = 'natcap.invest.hydropower.hydropower_water_yield';
  const logText = `
2021-01-15 07:14:37,147 (natcap.invest.utils) ${uniqueText}(124) INFO Writing log ...
2021-01-15 07:14:37,147 (__main__) cli.main(521) Level 100 Starting model with parameters: 
Arguments for InVEST ${primaryPythonLogger} 3.9.0.post147+gcc5a7cfe:
biophysical_table_path        C:/Users/dmf/projects/invest/data/biophysical_table_gura.csv
workspace_dir                 C:/Users/dmf/projects/invest-workbench/runs/awy

2021-01-15 07:14:37,148 (${primaryPythonLogger}) hydropower_water_yield.execute(268) INFO Validating arguments
2021-01-15 07:14:37,148 (natcap.invest.validation) validation._wrapped_validate_func(915) INFO ...
2021-01-15 07:14:37,525 (taskgraph.Task) Task.__init__(333) WARNING the ...
2021-01-15 07:14:37,636 (pygeoprocessing.geoprocessing) geoprocessing.align_and_resize_raster_stack(795) INFO ...
`;

  beforeEach(() => {
    workspace = fs.mkdtempSync(path.join('tests/data', 'log-'));
    logfilePath = path.join(workspace, logfileName);
    fs.writeFileSync(logfilePath, logText);
  });

  afterEach(() => {
    cleanupDir(workspace);
  });

  test('Text in logfile is rendered', async () => {
    const { findByText } = render(
      <LogTab
        jobStatus="success"
        logfile={logfilePath}
        logStdErr=""
        procID={undefined}
        pyModuleName={primaryPythonLogger}
        terminateInvestProcess={() => {}}
        sidebarFooterElementId="divID"
      />
    );

    const log = await findByText(new RegExp(uniqueText));
    expect(log).toBeInTheDocument();
  });

  test('message from non-primary invest logger is plain', async () => {
    const { findByText } = render(
      <LogTab
        jobStatus="success"
        logfile={logfilePath}
        logStdErr=""
        procID={undefined}
        pyModuleName={primaryPythonLogger}
        terminateInvestProcess={() => {}}
        sidebarFooterElementId="divID"
      />
    );

    const log = await findByText(new RegExp(uniqueText));
    expect(log).not.toHaveClass();
  });

  test('messages from primary invest logger are highlighted', async () => {
    const { findAllByText } = render(
      <LogTab
        jobStatus="success"
        logfile={logfilePath}
        logStdErr=""
        procID={undefined}
        pyModuleName={primaryPythonLogger}
        terminateInvestProcess={() => {}}
        sidebarFooterElementId="divID"
      />
    );

    const messages = await findAllByText(new RegExp(primaryPythonLogger));
    messages.forEach((msg) => {
      expect(msg).toHaveClass('invest-log-primary');
    });
  });
});
