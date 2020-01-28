import { spawn, spawnSync } from 'child_process';
import request from 'request';

const main = spawn(
    'npx', ['electron', '-r', '@babel/register', '.'], {
    shell: true,
    // stdio: 'ignore',
    detatched: false,
  });

// let readydata;
// let statusCode;
// request.get('http://localhost:5000/ready', 
//     (error, response, body) => {
//       readydata = JSON.parse(body);
//       statusCode = response.statusCode;
//       console.assert(readydata === 'Flask ready');
//       console.assert(statusCode === 200);
//       // const mainKiller = spawnSync(
//       //   'taskkill /PID ' + main.pid + ' /T /F', {shell: true});
//     }
//   );

// verify flask server is running after electron app is ready
let readydata;
let statusCode;
main.stderr.on('data', (data) => {
  console.log(data.toString());
  request.get('http://localhost:5000/ready', 
      (error, response, body) => {
        readydata = JSON.parse(body);
        statusCode = response.statusCode;
        console.assert(readydata === 'Flask ready');
        console.assert(statusCode === 200);
        // const mainKiller = spawnSync(
        //   'taskkill /PID ' + main.pid + ' /T /F', {shell: true});
      }
    );
});

  //   setTimeout(() => {
  //     let errorCode;
  //     request.get('http://localhost:5000/ready', 
  //       (error, response, body) => {
  //         console.log('Killed main process as PID' + main.pid);
  //         errorCode = error.code
  //         console.log(errorCode);
  //         expect(errorCode === 'ECONNREFUSED').toBe(true);
  //         done();
  //       }
  //     )  
  //   }, 10000)
  // });
  // setTimeout(() => {
  //   request.get('http://localhost:5000/ready', 
  //     (error, response, body) => {
  //       console.log('flask is running');
  //       data = JSON.parse(body);
  //       statusCode = response.statusCode;
  //     	expect(data === 'flask ready').toBe(true);
  //     	expect(statusCode === 200).toBe(true);
  //     }
  //   );
  // }, 5000);

  // const mainKiller = spawnSync(
  //     'taskkill /PID ' + main.pid + ' /T /F', {shell: true});

  // verify flask server is not running
  // setTimeout(() => {
  //   console.log('after timeout');
  //   let errorCode;
  //   request.get('http://localhost:5000/ready', 
  //     (error, response, body) => {
  //       console.log('Killed main process as PID' + main.pid);
  //       errorCode = error.code
  //       console.log(errorCode);
  //       expect(errorCode === 'ECONNREFUSED').toBe(true);
  //       done();
  //     }
  //   )  
  // }, 10000)

