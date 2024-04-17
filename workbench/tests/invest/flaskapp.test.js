import fs from 'fs';
import https from 'https';
import os from 'os';
import path from 'path';
import readline from 'readline';
import url from 'url';

import React from 'react';
import { render } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

import * as server_requests from '../../src/renderer/server_requests';
import { argsDictFromObject } from '../../src/renderer/utils';
import SetupTab from '../../src/renderer/components/SetupTab';
import ResourcesLinks from '../../src/renderer/components/ResourcesLinks';
import {
  createPythonFlaskProcess,
  shutdownPythonProcess,
  getFlaskIsReady,
} from '../../src/main/createPythonFlaskProcess';
import findInvestBinaries from '../../src/main/findInvestBinaries';

// This test starts a python subprocess, which can be slow
jest.setTimeout(120000);

let flaskSubprocess;
beforeAll(async () => {
  const isDevMode = true; // otherwise need to mock process.resourcesPath
  const investExe = findInvestBinaries(isDevMode);
  flaskSubprocess = createPythonFlaskProcess(investExe);
  await getFlaskIsReady();
});

afterAll(async () => {
  await shutdownPythonProcess(flaskSubprocess);
});

describe('requests to flask endpoints', () => {
  let WORKSPACE;
  beforeAll(() => {
    WORKSPACE = fs.mkdtempSync(path.join(os.tmpdir(), 'data-'));
  });

  afterAll(() => {
    fs.rmSync(WORKSPACE, { recursive: true, force: true });
  });

  test('invest list items have expected properties', async () => {
    const investList = await server_requests.getInvestModelNames();
    Object.values(investList).forEach((item) => {
      expect(item.model_name).not.toBeUndefined();
    });
  });

  test('fetch invest model args spec', async () => {
    const spec = await server_requests.getSpec('carbon');
    const expectedKeys = ['model_name', 'pyname', 'userguide', 'args'];
    expectedKeys.forEach((key) => {
      expect(spec[key]).not.toBeUndefined();
    });
  });

  test('fetch invest validation', async () => {
    const spec = await server_requests.getSpec('carbon');
    // it's okay to validate even if none of the args have values yet
    const argsDict = argsDictFromObject(spec.args);
    const payload = {
      model_module: spec.pyname,
      args: JSON.stringify(argsDict),
    };

    const results = await server_requests.fetchValidation(payload);
    // There's always an array of arrays, where each child array has
    // two elements: 1) an array of invest arg keys, 2) string message
    expect(results[0]).toHaveLength(2);
  });

  test('write parameters to file and parse them from file', async () => {
    const spec = await server_requests.getSpec('carbon');
    const argsDict = argsDictFromObject(spec.args);
    const workspace = fs.mkdtempSync(WORKSPACE);
    const filepath = path.join(workspace, 'foo.json');
    const payload = {
      filepath: filepath,
      moduleName: spec.pyname,
      args: JSON.stringify(argsDict),
      relativePaths: true,
    };

    // First test the data is written
    await server_requests.writeParametersToFile(payload);
    const data = JSON.parse(fs.readFileSync(filepath));
    const expectedKeys = [
      'args',
      'invest_version',
      'model_name'
    ];
    expectedKeys.forEach((key) => {
      expect(data[key]).not.toBeUndefined();
    });

    // Second test the datastack is read and parsed
    const data2 = await server_requests.fetchDatastackFromFile(
      { filepath: filepath });
    const expectedKeys2 = [
      'type',
      'args',
      'invest_version',
      'module_name',
      'model_run_name',
      'model_human_name',
    ];
    expectedKeys2.forEach((key) => {
      expect(data2[key]).not.toBeUndefined();
    });
  });

  test('write parameters to python script', async () => {
    const modelName = 'carbon'; // as appearing in `invest list`
    const spec = await server_requests.getSpec(modelName);
    const argsDict = argsDictFromObject(spec.args);
    const workspace = fs.mkdtempSync(WORKSPACE);
    const filepath = path.join(workspace, 'foo.py');
    const payload = {
      filepath: filepath,
      modelname: modelName,
      args: JSON.stringify(argsDict),
    };
    await server_requests.saveToPython(payload);

    const file = readline.createInterface({
      input: fs.createReadStream(filepath),
      crlfDelay: Infinity,
    });
    // eslint-disable-next-line
    for await (const line of file) {
      expect(`${line}`).toBe('# coding=UTF-8');
      break;
    }
  });
});

/** Some tests make http requests to check that links are status 200.
 *
 * @param {object} options - as passed to https.request
 * @returns {Promise} - resolves status code of the request.
 */
function getUrlStatus(options) {
  return new Promise((resolve) => {
    const req = https.request(options, (response) => {
      resolve(response.statusCode);
    });
    req.end();
  });
}

// Need a custom matcher to get useful message back on test failure
expect.extend({
  toBeStatus200: (received, address) => {
    if (received === 200) {
      return { pass: true };
    }
    return {
      pass: false,
      message: () => `expected ${received} to be 200 for ${address}`,
    };
  },
});

describe('Test building model UIs and forum links', () => {

  const models =  [
    'annual_water_yield',
    'carbon',
    'coastal_blue_carbon',
    'coastal_blue_carbon_preprocessor',
    'coastal_vulnerability',
    'crop_production_percentile',
    'crop_production_regression',
    'delineateit',
    'forest_carbon_edge_effect',
    'habitat_quality',
    'habitat_risk_assessment',
    'ndr',
    'pollination',
    'recreation',
    'routedem',
    'scenario_generator_proximity',
    'scenic_quality',
    'sdr',
    'seasonal_water_yield',
    'stormwater',
    'urban_cooling_model',
    'urban_flood_risk_mitigation',
    'urban_nature_access',
    'wave_energy',
    'wind_energy'
  ];

  test.each(models)('test building each model setup tab', async (model) => {
    const argsSpec = await server_requests.getSpec(model);
    const { findByRole } = render(
      <SetupTab
        pyModuleName={argsSpec.pyname}
        modelName={argsSpec.model_name}
        argsSpec={argsSpec.args}
        userguide={argsSpec.userguide}
        uiSpec={argsSpec.ui_spec}
        argsInitValues={undefined}
        investExecute={() => {}}
        nWorkers="-1"
        sidebarSetupElementId="foo"
        sidebarFooterElementId="foo"
        executeClicked={false}
        switchTabs={() => {}}
      />
    );
    expect(await findByRole('textbox', { name: /workspace/i }))
      .toBeInTheDocument();
  });

  test.each(models)('test each forum link', async (model) => {
    const argsSpec = await server_requests.getSpec(model);
    const { findByRole } = render(
      <ResourcesLinks
        moduleName={model}
        docs={argsSpec.userguide}
      />
    );
    const link = await findByRole('link', { name: /frequently asked questions/i });
    const address = link.getAttribute('href');
    // If a model has no tag, the link defaults to the forum homepage,
    // This will pass the urlStatus check, but we want to know if that happened,
    // so check url text first,
    expect(address).toContain('/tag/');
    const options = {
      method: 'HEAD',
      host: url.parse(address).host,
      path: url.parse(address).pathname,
    };
    const status = await getUrlStatus(options);
    expect(status).toBeStatus200(address);
  });
});
