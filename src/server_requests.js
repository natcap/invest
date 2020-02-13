import fetch from 'node-fetch';

export function getSpec(payload) {
  return (
    fetch('http://localhost:5000/getspec', {
      method: 'post',
      body: payload,
    })
    .then((response) => { return response })
    .catch((error) => { console.log(error) }))
}