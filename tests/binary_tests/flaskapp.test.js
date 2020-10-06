import fs from 'fs';
import readline from 'readline';
import { spawn, spawnSync } from 'child_process';
import fetch from 'node-fetch';
import * as server_requests from '../../src/server_requests';
import { findInvestBinaries, createPythonFlaskProcess } from '../../src/main_helpers';
import { argsDictFromObject } from '../../src/utils';
import dotenv from 'dotenv';
dotenv.config();

jest.setTimeout(250000)

beforeAll(async () => {
	const binaries = await findInvestBinaries(true); // trues force devMode
  createPythonFlaskProcess(binaries.server, true);
  await server_requests.getFlaskIsReady()
})

afterAll(async () => {
  await server_requests.shutdownPythonProcess()
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

test('fetch invest validation', async () => {
  const spec = await server_requests.getSpec('carbon');
   // it's okay to validate even if none of the args have values yet
  const argsDict = argsDictFromObject(spec.args)
  const payload = { 
    model_module: spec.module,
    args: JSON.stringify(argsDict)
  };
  
  const results = await server_requests.fetchValidation(payload);
  // There's always an array of arrays, where each child array has
  // two elements: 1) an array of invest arg keys, 2) string message
  expect(results[0].length).toBe(2)
})

test('write parameters to file and parse them from file', async () => {
  const spec = await server_requests.getSpec('carbon');
  const argsDict = argsDictFromObject(spec.args);
  const filepath = 'tests/data/foo.json';
  const payload = { 
    parameterSetPath: filepath,
    moduleName: spec.module,
    args: JSON.stringify(argsDict),
    relativePaths: true
  };
  
  // First test the data is written
  const _ = await server_requests.writeParametersToFile(payload);
  const data = JSON.parse(fs.readFileSync(filepath));
  const expectedKeys = [
    'args',
    'invest_version',
    'model_name'
  ];
  expectedKeys.forEach((key) => {
    expect(data[key] !== undefined).toBe(true)
  })

  // Second test the datastack is read and parsed
  const data2 = await server_requests.fetchDatastackFromFile(filepath);
  const expectedKeys2 = [
    'type',
    'args',
    'invest_version',
    'module_name',
    'model_run_name',
  ];
  expectedKeys2.forEach((key) => {
    expect(data2[key] !== undefined).toBe(true)
  })

  fs.unlinkSync(filepath)
})

test('write parameters to python script', async () => {
  const modelName = 'carbon' // as appearing in `invest list`
  const spec = await server_requests.getSpec(modelName);
  const argsDict = argsDictFromObject(spec.args);
  const filepath = 'tests/data/foo.py';
  const payload = { 
    filepath: filepath,
    modelname: modelName,
    pyname: spec.module,
    args: JSON.stringify(argsDict),
  };
  const _ = await server_requests.saveToPython(payload);
  
  const file = readline.createInterface({
    input: fs.createReadStream(filepath),
    crlfDelay: Infinity
  })
  for await (const line of file) {
    expect(`${line}`).toBe('# coding=UTF-8')
    break
  }
  fs.unlinkSync(filepath)
})
