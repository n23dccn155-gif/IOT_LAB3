// Run: node LAB3/frontend/tests/api.test.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const window = { LAB3Config: { healthUrl: () => '/health' } };
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../api.js'), 'utf8'), { window });

(async () => {
  const client = new window.LAB3Api.ApiClient({});
  client._request = async () => ({ status: 200, data: { ready: false, status: 'loading' }, timing: {} });
  const loading = await client.health();
  assert.equal(loading.ok, false, 'HTTP 200 alone does not mean the model is ready');
  assert.equal(loading.error.kind, 'server');
  client._request = async () => ({ status: 200, data: { ready: true, status: 'ok' }, timing: {} });
  assert.equal((await client.health()).ok, true);
  client._request = async () => { throw new Error('offline'); };
  assert.equal((await client.health()).ok, false);
  console.log('PASS: health reflects model readiness and connection errors');
})().catch(error => { console.error(error); process.exitCode = 1; });
