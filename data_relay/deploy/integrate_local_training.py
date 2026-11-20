"""Wire the local Instinct publisher into the existing monitor's training status view."""
from pathlib import Path
import shutil
import sys


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Monitor source anchor changed: {old[:90]}')
    return text.replace(old, new, 1)


def integrate(root):
    server = root / 'server.js'
    code = server.read_text(encoding='utf-8')
    if 'const localTraining = ' not in code:
        anchor = "const trainingStatus = require('./src/training-status');"
        code = replace_once(code, anchor, anchor + "\nconst localTraining = require('./src/local-training.cjs');")
        anchor = "app.get('/api/training/status', requireSession, requireFeature('training'),"
        code = replace_once(code, anchor, '''app.get('/api/training/status', requireFeature('training'), asyncRoute(async (req, res, next) => {
  if (req.query.source === 'ssh') return next();
  const snapshot = await localTraining.read(loadConfig().training);
  if (!snapshot) {
    if (req.query.source !== 'local24b') return next();
    return res.status(404).json({ error: '本地 24B 训练状态尚未上报', code: 'NO_LOCAL_TRAINING' });
  }
  res.setHeader('Cache-Control', 'no-store');
  if (req.query.download === '1') {
    return res.download(localTraining.logPath, 'instinct-24b-training.log');
  }
  res.json({ ok: true, ...snapshot });
}));

app.get(/^\\/instinct-24b(?:\\/index\\.html|\\/)?$/, (_req, res) => res.redirect(302, '/#training-status'));
app.get('/instinct-24b/training.log', (_req, res) => res.redirect(302, '/api/training/status?source=local24b&download=1'));

''' + anchor)
        server.write_text(code, encoding='utf-8')

    page = root / 'public/index.html'
    html = page.read_text(encoding='utf-8')
    if 'id="status-source"' not in html:
        anchor = '            <button class="btn ghost" id="btn-status-refresh">刷新</button>'
        html = replace_once(html, anchor, '''            <select class="btn ghost" id="status-source" aria-label="训练状态来源" hidden>
              <option value="local24b">AutoDL → 本地 24B</option>
              <option value="ssh">SSH 目标</option>
            </select>
            <a class="btn ghost" id="status-download" hidden download>下载合并日志</a>
''' + anchor)
        for anchor, key in (('id="boot-settings" hidden>设置服务器</button>', 'boot'),
                            ('id="connect-settings" data-feature="settings">服务器设置</button>', 'connect')):
            html = replace_once(html, anchor, anchor + f'\n        <a class="btn ghost" id="{key}-training-status" href="#training-status" data-feature="training">训练状态</a>')
        page.write_text(html, encoding='utf-8')

    app = root / 'public/app.js'
    js = app.read_text(encoding='utf-8')
    if 'function enterLocalTrainingStatus' not in js:
        js = replace_once(js, 'const statusState = {', "const statusState = {\n  source: '',")
        js = replace_once(js, "const name = (data.manifest && data.manifest.name) || '原生 cuBLAS 训练';", '''const name = (data.manifest && data.manifest.name) || '原生 cuBLAS 训练';
  const source = $('status-source');
  const download = $('status-download');
  if (data.source === 'local24b') source.hidden = false;
  source.value = data.source === 'local24b' ? 'local24b' : 'ssh';
  download.hidden = !data.downloadUrl;
  if (data.downloadUrl) download.href = data.downloadUrl;''')
        js = replace_once(js, "if (lossSeries.full) parts.push(`step ${formatStep(from)}–${formatStep(to)}`);", '''const axis = data.historyAxis === 'cursor' ? '日志游标' : 'step';
      if (lossSeries.full) parts.push(`${axis} ${formatStep(from)}–${formatStep(to)}`);
      if (data.handoff) {
        parts.push(`AutoDL ${formatStep(data.handoff.cloudPoints)} 点 → 本地 ${formatStep(data.handoff.localPoints)} 点`);
        parts.push(`第 ${formatStep(data.handoff.step)} 微步续训`);
      }''')
        js = replace_once(js, 'if (lossSeries.full && from > 0) {', "if (lossSeries.full && from > 0 && !data.handoff) {")
        js = replace_once(js, "if (lossSeries.sampled) parts.push('已按步数降采样');", "if (lossSeries.sampled) parts.push('已降采样');")
        js = replace_once(js, 'tooltip.innerHTML = `<div class="tt-step">step ${escapeHtml(formatStep(best.x))}</div>`', '''const axis = statusState.data?.historyAxis === 'cursor' ? '日志游标' : 'step';
  tooltip.innerHTML = `<div class="tt-step">${axis} ${escapeHtml(formatStep(best.x))}</div>`''')
        js = replace_once(js, "`日志步 ${metrics.logStep.toLocaleString('en-US')}`", "`${data.historyAxis === 'cursor' ? '日志游标' : '日志步'} ${metrics.logStep.toLocaleString('en-US')}`")
        js = replace_once(js, "    const query = options.force ? '?refresh=1' : '';\n    const data = await api(`/api/training/status${query}`);", '''    const query = new URLSearchParams();
    if (options.force) query.set('refresh', '1');
    if (statusState.source) query.set('source', statusState.source);
    const data = await api(`/api/training/status?${query}`);''')
        for old, new in (
            ('if (!state.session || el.statusView.hidden || document.hidden) return;',
             "if ((!state.session && statusState.source !== 'local24b') || el.statusView.hidden || document.hidden) return;"),
            ('if (!el.statusView.hidden && state.session) {',
             "if (!el.statusView.hidden && (state.session || statusState.source === 'local24b')) {"),
            ('if (el.statusView && !el.statusView.hidden && state.session) {',
             "if (el.statusView && !el.statusView.hidden && (state.session || statusState.source === 'local24b')) {"),
        ):
            js = replace_once(js, old, new)
        js = replace_once(js, 'function switchView(name) {', '''function enterLocalTrainingStatus(data) {
  statusState.source = 'local24b';
  statusState.data = data;
  statusState.lastLoadedAt = Date.now();
  el.bootView.hidden = true;
  el.connectView.hidden = true;
  el.appView.hidden = false;
  if (!state.session) el.connLabel.textContent = '本地 24B 上报';
  renderTrainingStatus(data);
  switchView('status');
}

async function tryLocalTrainingStatus() {
  if (!featureOn('training')) return false;
  try {
    const data = await api('/api/training/status?source=local24b');
    if (data.source !== 'local24b' || !data.available) return false;
    enterLocalTrainingStatus(data);
    return true;
  } catch {
    return false;
  }
}

$('status-source').addEventListener('change', (event) => {
  statusState.source = event.target.value;
  if (statusState.source === 'ssh' && !state.session) {
    boot({ sshView: 'status' });
    return;
  }
  loadTrainingStatus({ force: true });
  startStatusPolling();
});
for (const id of ['boot-training-status', 'connect-training-status']) {
  $(id).addEventListener('click', (event) => {
    event.preventDefault();
    history.replaceState(null, '', '#training-status');
    boot();
  });
}

function switchView(name) {''')
        js = replace_once(js, "  const view = requested === 'status' && !featureOn('training') ? 'logs' : requested;", '''  const view = requested === 'status' && !featureOn('training') ? 'logs' : requested;
  if (view !== 'status' && !state.session && statusState.source === 'local24b') {
    history.replaceState(null, '', location.pathname + location.search);
    boot({ sshView: view });
    return;
  }''')
        js = replace_once(js, 'async function boot() {', 'async function boot({ sshView = null } = {}) {')
        js = replace_once(js, '  try {\n    if (cfg.autoConnect) {', "  if (!sshView && await tryLocalTrainingStatus()) return;\n\n  try {\n    if (cfg.autoConnect) {")
        js = replace_once(js, 'enterApp(out.session, out.defaultView || cfg.defaultView);', 'enterApp(out.session, sshView || out.defaultView || cfg.defaultView);')
        js = replace_once(js, 'enterApp(data.session, cfg.defaultView);', 'enterApp(data.session, sshView || cfg.defaultView);')
        app.write_text(js, encoding='utf-8')

    shutil.copy2(Path(__file__).with_name('monitor-local-training.cjs'), root / 'src/local-training.cjs')


if __name__ == '__main__':
    integrate(Path(sys.argv[1]))
