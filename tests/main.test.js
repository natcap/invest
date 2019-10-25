import { spawn, spawnSync } from 'child_process';
import request from 'request';

test('flask starts and stops with app', async () => {
	const main = spawn(
      'npx', ['electron', '-r', '@babel/register', '.'], {
      shell: true,
      stdio: 'ignore',
      detatched: false,
    });

  console.log('Started main process as PID ' + main.pid);

  // verify flask server is running
  let data;
  let statusCode;
  await request.get('http://localhost:5000/ready', 
  	(error, response, body) => {
      // console.log(error);
      console.log(body);
  		data = JSON.parse(body);
      statusCode = response.statusCode;
  	})
	expect(data === 'flask ready').toBe(true);
	expect(statusCode === 200).toBe(true);

  const mainKiller = spawnSync(
      'taskkill /PID ' + main.pid + ' /T /F', {shell: true});
  console.log('Killed main process as PID' + main.pid);

  // verify flask server is not running
  let errorCode;
  await request.get('http://localhost:5000/ready', 
    (error, response, body) => {
      errorCode = error.code
    }
  )
  expect(errorCode === 'ECONNREFUSED').toBe(true);
});

