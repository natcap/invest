import fetch from 'node-fetch';

export function shutdownPythonProcess() {
  fetch('http://localhost:5000/shutdown', {
    method: 'get',
  })
  .then((response) => { return response.text() })
  .then((text) => { console.log(text) })
  .catch((error) => { console.log(error) })
}

export function investList() {
  return(
    fetch('http://localhost:5000/models', {
      method: 'get',
    })
    .then((response) => { return response.json() })
    .catch((error) => { console.log(error) })
  )
}

export function getSpec(payload) {
  return (
    fetch('http://localhost:5000/getspec', {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
    .then((response) => { return response.json() })
    .catch((error) => { console.log(error) })
  )
}

export function fetchValidation(payload) {
  return (
    fetch('http://localhost:5000/validate', {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
    .then((response) => { return response.json() })
    .catch((error) => { console.log(error) })
  )
}

export function fetchLogfilename(payload) {
  return (
    fetch('http://localhost:5000/get_invest_logfilename', {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
    .then((response) => { return response.text() })
    .catch((error) => { console.log(error) })
    )
}

export function fetchDatastackFromFile(payload) {
  return (
    fetch('http://localhost:5000/post_datastack_file', {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
    .then((response) => { return response.json() })
    .catch((error) => { console.log(error) })
  )
}

export function saveToPython(payload) {
  fetch('http://localhost:5000/save_to_python', {
    method: 'post',
    body: JSON.stringify(payload),
    headers: { 'Content-Type': 'application/json' },
  })
  .then((response) => { return response.text() })
  .then((text) => { console.log(text) })
  .catch((error) => { console.log(error) })
}

export function writeParametersToFile(payload) {
  fetch('http://localhost:5000/write_parameter_set_file', {
    method: 'post',
    body: JSON.stringify(payload),
    headers: { 'Content-Type': 'application/json' },
  })
  .then((response) => { return response.text() })
  .then((text) => { console.log(text) })
  .catch((error) => { console.log(error) })
}