/**
 * @jest-environment node
 */

/**
 * Validate URLs to deployed sampledata zip files. These URLs were constructed
 * during the invest build process (`make sampledata`). This test will
 * run in GHA, post `make deploy`.
 * */

import http from 'http';
import url from 'url';

import sampledataRegistry from '../../src/renderer/sampledata_registry.json';

function getUrlStatus(options) {
  return new Promise((resolve) => {
    const request = http.request(options, (response) => {
      resolve(response.statusCode);
    });
    request.end();
  });
}

test.each(
  Object.values(sampledataRegistry).map((item) => item.url)
)('check url: %s', async (address) => {
  const options = {
    method: 'HEAD',
    host: url.parse(address).host,
    path: url.parse(address).pathname,
  };
  const status = await getUrlStatus(options);
  expect(status).toBe(200);
});
