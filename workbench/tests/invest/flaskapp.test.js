import { app, ipcRenderer } from 'electron';
import fs from 'fs';
import https from 'https';
import os from 'os';
import path from 'path';
import readline from 'readline';
import url from 'url';

import React from 'react';
import { render } from '@testing-library/react';
import '@testing-library/jest-dom';

import * as serverRequests from '../../src/renderer/server_requests';
import { argsDictFromObject } from '../../src/renderer/utils';
import SetupTab from '../../src/renderer/components/SetupTab';
import ResourcesLinks from '../../src/renderer/components/ResourcesLinks';
import {
  createCoreServerProcess,
  shutdownPythonProcess
} from '../../src/main/createPythonFlaskProcess';
import { findInvestBinaries } from '../../src/main/findBinaries';
import { settingsStore } from '../../src/main/settingsStore';
import { checkFirstRun, APP_HAS_RUN_TOKEN } from '../../src/main/setupCheckFirstRun';
import { ipcMainChannels } from '../../src/main/ipcMainChannels';

// This test starts a python subprocess, which can be slow
jest.setTimeout(120000);

const CORE_PORT = 56789;
let flaskSubprocess;
beforeAll(async () => {
  jest.spyOn(ipcRenderer, 'invoke').mockImplementation((channel, arg) => {
    if (channel === ipcMainChannels.GET_SETTING) {
      if (arg === 'core.port') {
        return CORE_PORT;
      }
      if (arg.endsWith('.type')) {
        return 'core';
      }
    }
    return Promise.resolve();
  });
  const userDataPath = app.getPath('userData');
  const hasRunTokenPath = path.join(userDataPath, APP_HAS_RUN_TOKEN);
  if (fs.existsSync(hasRunTokenPath)) {
    fs.unlinkSync(hasRunTokenPath);
  }
  const isDevMode = true; // otherwise need to mock process.resourcesPath
  const investExe = findInvestBinaries(isDevMode);
  settingsStore.set('investExe', investExe);
  await createCoreServerProcess(CORE_PORT);
});

afterAll(async () => {
  await shutdownPythonProcess(settingsStore.get('core.pid'));
});

describe('requests to flask endpoints', () => {
  let WORKSPACE;
  beforeAll(() => {
    WORKSPACE = fs.mkdtempSync(path.join(os.tmpdir(), 'data-'));
  });

  afterAll(() => {
    fs.rmSync(WORKSPACE, { recursive: true, force: true });
  });

  test('fetch invest model args spec', async () => {
    const spec = await serverRequests.getSpec('carbon');
    const expectedKeys = ['model_id', 'model_title', 'userguide', 'args'];
    expectedKeys.forEach((key) => {
      expect(spec[key]).toBeDefined();
    });
  });

  test('fetch invest validation', async () => {
    const spec = await serverRequests.getSpec('carbon');
    // it's okay to validate even if none of the args have values yet
    const argsDict = argsDictFromObject(spec.args);
    const payload = {
      model_id: 'carbon',
      args: JSON.stringify(argsDict),
    };

    const results = await serverRequests.fetchValidation(payload);
    // There's always an array of arrays, where each child array has
    // two elements: 1) an array of invest arg keys, 2) string message
    expect(results[0]).toHaveLength(2);
  });

  test('write parameters to file and parse them from file', async () => {
    const spec = await serverRequests.getSpec('carbon');
    const argsDict = argsDictFromObject(spec.args);
    const workspace = fs.mkdtempSync(WORKSPACE);
    const filepath = path.join(workspace, 'foo.json');
    const payload = {
      filepath: filepath,
      model_id: spec.model_id,
      args: JSON.stringify(argsDict),
      relativePaths: true,
    };

    // First test the data is written
    await serverRequests.writeParametersToFile(payload);
    const data = JSON.parse(fs.readFileSync(filepath));
    const expectedKeys = [
      'args',
      'invest_version',
      'model_id',
    ];
    expectedKeys.forEach((key) => {
      expect(data[key]).toBeDefined();
    });

    // Second test the datastack is read and parsed
    const data2 = await serverRequests.fetchDatastackFromFile(
      { filepath: filepath });
    const expectedKeys2 = [
      'type',
      'args',
      'invest_version',
      'model_id',
    ];
    expectedKeys2.forEach((key) => {
      expect(data2[key]).toBeDefined();
    });
  });

  test('write parameters to python script', async () => {
    const modelID = 'carbon'; // as appearing in `invest list`
    const spec = await serverRequests.getSpec(modelID);
    const argsDict = argsDictFromObject(spec.args);
    const workspace = fs.mkdtempSync(WORKSPACE);
    const filepath = path.join(workspace, 'foo.py');
    const payload = {
      filepath: filepath,
      model_id: modelID,
      args: JSON.stringify(argsDict),
    };
    await serverRequests.saveToPython(payload);

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

  test('get geometamaker profile', async () => {
    const profile = await serverRequests.getGeoMetaMakerProfile();
    expect(profile).toHaveProperty('contact');
    expect(profile).toHaveProperty('license');
  });

  test('set geometamaker profile', async () => {
    const result = {
      message: 'Metadata profile saved',
      error: false,
    };
    jest.spyOn(window, 'fetch')
      .mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(result),
      });

    const payload = {
      contact: {},
      license: {},
    };
    const response = await serverRequests.setGeoMetaMakerProfile(payload);
    expect(response).toStrictEqual(result);
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
  const modelIDs = [
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
    'wind_energy',
  ];

  test.each(modelIDs)('test building each model setup tab', async (modelID) => {
    const argsSpec = await serverRequests.getSpec(modelID);
    const { findByRole } = render(
      <SetupTab
        modelID={modelID}
        argsSpec={argsSpec.args}
        userguide={argsSpec.userguide}
        isCoreModel={true}
        inputFieldOrder={argsSpec.input_field_order}
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

  test.each(modelIDs)('test each forum link', async (modelID) => {
    const argsSpec = await serverRequests.getSpec(modelID);
    const { findByRole } = render(
      <ResourcesLinks
        modelID={modelID}
        isCoreModel={true}
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
