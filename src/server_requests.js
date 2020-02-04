import request from 'request';

export function getSpec(payload) {
  return new Promise(function(resolve, reject) {
    request.post(
      'http://localhost:5000/getspec',
      payload,
      (error, response, body) => {
        if (!error && response.statusCode == 200) {
          resolve(body);
        } else {
          console.log('Status: ' + response.statusCode)
          console.log('Error: ' + error.message)
          return null;
        }
      }
    );
  });
}