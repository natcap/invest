import fetch from 'node-fetch';

// TODO: elsewhere I've used async/await instead of 
// .then chaining of callbacks. Consider refactoring
// everything here with async/await.

export function getFlaskIsReady() {
  return(
    fetch('http://localhost:5000/ready', {
      method: 'get',
    })
    .then((response) => { return response.text() })
    // .then((text) => { console.log(text) })
    .catch(async (error) => {
      console.log(error)
      if (error.code === 'ECONNREFUSED') {
        // try again after a short pause
        await new Promise(resolve => setTimeout(resolve, 50));
        console.log('recursive call coming')
        return await getFlaskIsReady()
      } else {
        console.log(error)
        return error 
      }
   })
  )
}

// TODO: sometimes this fetch doesn't complete or error,
// hence the logging
export function getInvestList() {
  console.log('pre-fetch models')
  return(
    fetch('http://localhost:5000/models', {
      method: 'get',
    })
    .then((response) => { 
      console.log(response)
      return response
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
  // even though we don't need a response sent back
  // from this fetch, we must ``return`` a Promise 
  // in order to ``await writeParametersToFile``.
  return (
    fetch('http://localhost:5000/write_parameter_set_file', {
      method: 'post',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
    .then((response) => { return response.text() })
    .then((text) => { console.log(text) })
    .catch((error) => { console.log(error) })
  );
}