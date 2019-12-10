import request from 'request';

export function makeInvestList() {
  return new Promise(function(resolve, reject) {
    setTimeout(() => {
      request.get(
        'http://localhost:5000/models',
        (error, response, body) => {
          if (!error && response.statusCode == 200) {
            const models = JSON.parse(body);
            resolve(models);
          } else if (error) {
            console.error(error);
          } else {
            try {
              console.log('Status: ' + response.statusCode);
            }
            catch (e) {
              console.error(e);
            }
          }
        }
      );
    }, 500)  // wait, the server only just launced in a subprocess.
  });
}