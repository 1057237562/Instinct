'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const { parseTrainingLog } = require('./log-parser');
const training = require('./training-status');

const directory = process.env.INSTINCT_LOCAL_TRAINING_DIR || '/var/lib/ssh-log-monitor/instinct-24b';
const statusPath = path.join(directory, 'status.json');
const logPath = path.join(directory, 'training.log');
const downloadUrl = '/api/training/status?source=local24b&download=1';

async function read(options = {}) {
  let report;
  try {
    report = JSON.parse(await fs.readFile(statusPath, 'utf8'));
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw error;
  }
  const status = training.parseRunStatus(report.status);
  if (!status || !Number.isFinite(status.tokens) || !Number.isFinite(report.plannedTokens)) return null;

  const file = await fs.stat(logPath);
  const meta = { size: file.size, mtimeMs: file.mtimeMs };
  const history = await training.readLogHistory({ id: 'local24b' }, logPath, meta, {
    maxPoints: options.historyMaxPoints,
    refreshMs: options.historyRefreshMs,
    parseLog: async (_session, filename, settings) => parseTrainingLog(await fs.readFile(filename, 'utf8'), settings),
  });
  // Recent throughput should describe local training only, even just after handoff.
  const handle = await fs.open(logPath, 'r');
  let tail;
  try {
    const buffer = Buffer.alloc(Math.min(16 * 1024, file.size));
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, file.size - buffer.length);
    tail = buffer.subarray(0, bytesRead).toString('utf8');
  } finally {
    await handle.close();
  }
  const log = training.parseTrainingLogTail(tail);
  log.points = log.points.slice(Math.max(0, log.points.length - (report.handoff.localPoints || 0)));
  log.last = log.points.at(-1) || null;
  const rate = Number.isFinite(report.tokensPerSec) && report.tokensPerSec > 0 ? report.tokensPerSec : null;
  if (log.last) log.last = { ...log.last, tokensPerSec: rate, epochTimeMin: null };

  const data = training.summariseTrainingState({
    manifest: { name: 'Instinct 24B · 本地续训', backend: 'cublas', plannedTokens: report.plannedTokens },
    status, log, history, gpus: training.parseNvidiaSmi(report.gpu), system: report.system,
    alive: report.alive,
  });
  data.phase = report.phase;
  return {
    ...data, source: 'local24b', handoff: report.handoff, historyAxis: 'cursor',
    statusPath: '本地训练上报', logPath: 'AutoDL → 本地合并日志', downloadUrl,
  };
}

module.exports = { read, logPath };
