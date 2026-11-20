// Run against an isolated monitor release after integrate_local_training.py.
'use strict';

const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const fs = require('node:fs/promises');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { setTimeout: delay } = require('node:timers/promises');

async function port() {
  const server = net.createServer();
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const chosen = server.address().port;
  await new Promise((resolve) => server.close(resolve));
  return chosen;
}

async function withMonitor(checkout, fixture, protectAll, check) {
  const chosen = await port();
  const base = `http://127.0.0.1:${chosen}`;
  const child = spawn(process.execPath, ['server.js'], {
    cwd: checkout, windowsHide: true, stdio: ['ignore', 'ignore', 'pipe'],
    env: { ...process.env, HOST: '127.0.0.1', PORT: String(chosen),
      SSHBROWSE_CONFIG: path.join(fixture, 'config.json'), INSTINCT_LOCAL_TRAINING_DIR: fixture,
      SSHBROWSE_PROTECT_LOGS: protectAll ? '1' : '0', SSHBROWSE_REQUIRE_AUTH: '1',
      SSHBROWSE_ADMIN_USER: 'test-admin', SSHBROWSE_ADMIN_PASSWORD: 'fixture-only-secret' },
  });
  let errors = '';
  child.stderr.on('data', (value) => { errors += value; });
  try {
    for (let attempt = 0; attempt < 50; attempt += 1) {
      if (child.exitCode != null) throw new Error(errors || 'Monitor exited');
      try {
        if ((await fetch(`${base}/api/health`)).ok) return await check(base);
      } catch (error) {
        if (error.cause?.code !== 'ECONNREFUSED') throw error;
      }
      await delay(100);
    }
    throw new Error(errors || 'Monitor failed to start');
  } finally {
    if (child.exitCode == null) {
      const exited = new Promise((resolve) => child.once('exit', resolve));
      child.kill();
      await exited;
    }
  }
}

async function main() {
  const checkout = path.resolve(process.argv[2]);
  const fixture = await fs.mkdtemp(path.join(os.tmpdir(), 'instinct-status-test-'));
  const now = Date.now() / 1000;
  const status = { status: { status: 'running', steps: 1762480, updates: 218455,
    processed_valid_tokens: 21100745424, updated_at: now, process_created: now - 900 },
  phase: 'running', alive: true, plannedTokens: 23999989035, tokensPerSec: 18500,
  handoff: { step: 1762000, tokens: 21094989234, cloudPoints: 1, localPoints: 1 },
  gpu: '0, RTX 5070 Ti, 99, 58, 15100, 16303, 295, 300, 62, 2715, 3090', system: {} };
  const log = ['Epoch:[1/1](2058430/2013531), loss: 0.796944, learning_rate: 0.00003964, epoch_time: 116.870min, tokens/s: 37913.0',
    '--- Local continuation at checkpoint step 1,762,000 (21,094,989,234 tokens) ---',
    'Epoch:[1/1](2058547/2013531), loss: 1.239377, learning_rate: 0.00003964, epoch_time: 0.239min, tokens/s: 11932.0', ''].join('\n');
  await fs.writeFile(path.join(fixture, 'config.json'), JSON.stringify({ autoConnect: false, streamlit: { enabled: false } }));
  await fs.writeFile(path.join(fixture, 'status.json'), JSON.stringify(status));
  await fs.writeFile(path.join(fixture, 'training.log'), log);
  await withMonitor(checkout, fixture, false, async (base) => {
    const endpoint = `${base}/api/training/status`;
    const data = await (await fetch(endpoint)).json();
    assert.equal(data.source, 'local24b');
    assert.equal(data.handoff.step, 1762000);
    assert.equal(data.progress.steps, 1762480);
    assert.equal(data.progress.tokens, 21100745424);
    assert.equal(data.metrics.tokensPerSec, 18500);
    assert.equal(data.metrics.epochTimeMin, null);
    assert.equal(data.gpu.devices[0].name, 'RTX 5070 Ti');
    assert.deepEqual(data.history.steps, [2058430, 2058547]);
    assert.deepEqual(data.history.loss, [0.796944, 1.239377]);
    assert.deepEqual(data.recent.tokensPerSec, [11932]);
    assert.equal(await (await fetch(`${base}${data.downloadUrl}`)).text(), log);
    assert.equal((await fetch(`${endpoint}?source=ssh`)).status, 401);
    const oldPage = await fetch(`${base}/instinct-24b/`, { redirect: 'manual' });
    assert.equal(oldPage.status, 302);
    assert.equal(oldPage.headers.get('location'), '/#training-status');

    if (process.env.INSTINCT_TEST_PLAYWRIGHT) {
      const { chromium } = require(process.env.INSTINCT_TEST_PLAYWRIGHT);
      const browser = await chromium.launch({ executablePath: process.env.INSTINCT_TEST_CHROME,
        args: ['--disable-gpu', '--no-proxy-server'] });
      try {
        const page = await browser.newPage();
        await page.goto(`${base}/instinct-24b/`);
        await page.locator('#status-content').waitFor();
        assert.equal(new URL(page.url()).pathname, '/');
        assert.equal(await page.locator('.view-tab.active').getAttribute('data-view'), 'status');
        assert.match(await page.locator('#status-loss-hint').innerText(), /AutoDL 1 点 → 本地 1 点/);
        assert.match(await page.locator('#status-gpu-grid').innerText(), /RTX 5070 Ti/);
        assert.match(await page.locator('#status-progress-text').innerText(), /1,762,480/);
        assert.equal(await page.locator('#status-download').getAttribute('href'), data.downloadUrl);
        status.status.steps = 1762560;
        await fs.writeFile(path.join(fixture, 'status.json'), JSON.stringify(status));
        await page.locator('#btn-status-refresh').click();
        await page.getByText(/1,762,560 微步/).waitFor();
      } finally {
        await browser.close();
      }
    }

    await fs.rename(path.join(fixture, 'status.json'), path.join(fixture, 'status.saved.json'));
    assert.equal((await fetch(`${endpoint}?source=local24b`)).status, 404);
    assert.equal((await fetch(endpoint)).status, 401);
    await fs.rename(path.join(fixture, 'status.saved.json'), path.join(fixture, 'status.json'));
  });
  await withMonitor(checkout, fixture, true, async (base) => {
    for (const query of ['?source=local24b', '?source=local24b&download=1']) {
      assert.equal((await fetch(`${base}/api/training/status${query}`)).status, 401);
    }
  });
  console.log('Verified merged training status, history, download, SSH fallback and strict auth.');
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
