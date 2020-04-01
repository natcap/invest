import { spawn, spawnSync } from 'child_process';
import fetch from 'node-fetch';
import { getFlaskIsReady } from '../src/server_requests';

jest.setTimeout(15000)

// TODO: this stuff never worked reliably, but should
// definitely be revisited. The main goal is to test that 
// the flask server reliably starts and stops when the 
// electron application is opened and closed.

// jest shows a failed test suite if there are no tests,
// so a placeholder until we revisit the rest of tests.
test('placeholder', () => {
  expect(true).toBeTruthy()
})

// test('start and stop python server', async () => {

//   const main = spawn(
//     'npx', ['electron', '-r', '@babel/register', '.'], {
//     shell: true,
//     // stdio: 'ignore',
//     detatched: false,
//   });

//   let readydata;
//   readydata = await getFlaskIsReady()
//   console.log(readydata)
//   expect(readydata).toBe('Flask ready');
  
//   const mainKiller = spawnSync(
//     'taskkill /PID ' + main.pid + ' /T /F', {shell: true});

//   setTimeout(async () => {
//     readydata = await getFlaskIsReady()
//     expect(readyData).toBe('ECONNREFUSED');
//     done();
//   }, 10000)
// })