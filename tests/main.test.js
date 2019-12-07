import { spawn, spawnSync } from 'child_process';
import request from 'request';

test('flask starts and stops with app', () => {
	const main = spawn(
      'npx', ['electron', '-r', '@babel/register', '.'], {
      shell: true,
      // stdio: 'ignore',
      detatched: false,
    });

  console.log('Started main process as PID ' + main.pid);

  // verify flask server is running
  let data;
  let statusCode;
  // setTimeout(() => {
  request.get('http://localhost:5000/ready', 
  (error, response, body) => {
    // console.log(error);
    data = JSON.parse(body);
    statusCode = response.statusCode;
    console.log(data);
  	expect(data === 'flask ready').toBe(true);
  	expect(statusCode === 200).toBe(true);
  });
  // }, 2000);

  const mainKiller = spawnSync(
      'taskkill /PID ' + main.pid + ' /T /F', {shell: true});
  console.log('Killed main process as PID' + main.pid);

  // verify flask server is not running
  setTimeout(() => {
    console.log('after timeout');
    let errorCode;
    request.get('http://localhost:5000/ready', 
      (error, response, body) => {
        errorCode = error.code
        console.log(errorCode);
        expect(errorCode === 'ECONNREFUSED').toBe(true);
      }
    )  
  }, 2000)
});

