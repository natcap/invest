import { spawn, spawnSync } from 'child_process';
import fetch from 'node-fetch';
import * as server_requests from '../src/server_requests';
import { findInvestBinaries, createPythonFlaskProcess } from '../src/main_helpers';
import dotenv from 'dotenv';
dotenv.config();

// These won't run in the same CI as the rest of our tests
// because we don't have the python env available.
// we can have the python env available during the build & dist CI,
// at which point we want to run these tests in production mode, not devMode.
// Or we can fetch prebuilt binaries in the Actions test-runner
// and setup a .env config in order to run in devMode

beforeAll(async () => {
	const binaries = await findInvestBinaries(true); // trues force devMode
  createPythonFlaskProcess(binaries.server, true);
  await server_requests.getFlaskIsReady()
})

afterAll(() => {
  server_requests.shutdownPythonProcess()
})

test('invest list items have expected properties', async () => {
  const investList = await server_requests.getInvestList();
  Object.values(investList).forEach((item) => {
    expect(item.internal_name !== undefined).toBe(true)
  })
})

test('fetch invest model args spec', async () => {
  const spec = await server_requests.getSpec('carbon');
  const expectedKeys = ['model_name', 'module', 'userguide_html', 'args']
  expectedKeys.forEach((key) => {
    expect(spec[key] !== undefined).toBe(true)
  })
})