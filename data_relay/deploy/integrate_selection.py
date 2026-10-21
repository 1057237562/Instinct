"""Idempotent shared server selector migration for the deployed SSH monitor."""
from pathlib import Path
import shutil

def integrate(root, assets):
    def edit(relative, transform):
        path=root/relative
        text=path.read_text(encoding='utf-8')
        path.write_text(transform(text),encoding='utf-8')

    for source,target in [('monitor-server-selection.cjs','src/server-selection.cjs'),('monitor-server-selection.js','public/server-selection.js')]:
        shutil.copy2(assets/source,root/target)

    def server(text):
        if "require('./src/server-selection.cjs')" in text:return text
        text=text.replace('const relayProxy = createRelayProxy();',"""const { createServerSelection } = require('./src/server-selection.cjs');
const relaySelection = createServerSelection({loadConfig,saveServerConfig,ssh,streamlitPanel});
const relayProxy = createRelayProxy({onConfig:patch=>relaySelection.patch(patch)});""")
        text=text.replace("app.get('/api/settings',", "app.get('/api/relay/servers', asyncRoute(async (req,res)=>{res.json(await relaySelection.list());}));\n\napp.get('/api/settings',",1)
        for anchor in ["app.post('/api/connect', requireFeature('manualConnect'), asyncRoute(async (req, res) => {", "app.post('/api/auto-connect', asyncRoute(async (req, res) => {"]:
            text=text.replace(anchor,anchor+'\n  await relaySelection.ready();\n  const targetGeneration=relaySelection.generation();',1)
        text=text.replace('  setSessionCookie(res, session);','  relaySelection.guard(session,targetGeneration);\n  setSessionCookie(res, session);')
        start=text.index('  saveServerConfig(patch);',text.index("app.post('/api/settings',"))
        end=text.index("  res.clearCookie('sshsid'",start)
        text=text[:start]+'  await relaySelection.manual(patch);\n'+text[end:]
        return text
    edit('server.js',server)

    def config(text):
        if 'server.relayInstanceUuid' in text:return text
        text=text.replace('      // 密码来源：',"      relayInstanceUuid: String(server.relayInstanceUuid || ''),\n      relayInstanceName: String(server.relayInstanceName || ''),\n      // 密码来源：",1)
        anchor='  // 配置文件里绝不保留明文凭据。'
        text=text.replace(anchor,"""  for (const field of ['relayInstanceUuid','relayInstanceName']) {
    if (patch[field] !== undefined) server[field] = String(patch[field] || '');
  }
  if (patch.autoConnect !== undefined) result.autoConnect = Boolean(patch.autoConnect);

"""+anchor,1)
        # Expose only the UUID/name metadata with the public target summary.
        text=text.replace('      host: config.server.host,',"      relayInstanceUuid: config.server.relayInstanceUuid,\n      relayInstanceName: config.server.relayInstanceName,\n      host: config.server.host,",1)
        return text
    edit('src/config.js',config)

    def page(text):
        if 'id="server-picker"' in text:return text
        text=text.replace('id="boot-relay">数据与训练</button>', 'id="boot-relay">数据与训练</button>\n        <button class="btn ghost" data-server-picker>切换服务器</button>',1)
        text=text.replace('id="connect-relay">数据与训练</button>', 'id="connect-relay">数据与训练</button>\n      <button type="button" class="btn ghost block" data-server-picker>切换服务器</button>',1)
        text=text.replace('<div class="topbar-actions">','<div class="topbar-actions">\n        <button class="btn ghost" data-server-picker>切换服务器</button>',1)
        text=text.replace('<form id="settings-form"', '<button class="btn ghost" data-server-picker>从 AutoDL 实例切换（同步数据与训练）</button>\n        <form id="settings-form"',1)
        text=text.replace('  <div class="toast-stack"',"""  <div class="modal-backdrop" id="server-picker" hidden>
    <div class="modal"><div class="modal-head"><strong>切换服务器</strong><button class="btn ghost" id="server-picker-close">关闭</button></div>
      <div class="modal-body"><p class="settings-hint">数据、训练、SSH 文件与日志共用此选择。原服务器离线也可切换，已提交任务仍使用原目标。</p>
        <label class="field"><span>AutoDL 实例</span><select id="shared-server-choice" style="max-width:100%;width:100%"><option value="">正在加载…</option></select></label>
        <p class="settings-hint" id="server-picker-status" role="status"></p>
        <div class="settings-actions"><button class="btn ghost" id="server-picker-refresh">刷新实例</button><button class="btn primary" id="server-picker-apply">切换并连接</button></div>
      </div></div>
  </div>
  <div class="toast-stack" """,1)
        text=text.replace('<script src="app.js"></script>','<script src="app.js"></script>\n  <script src="server-selection.js"></script>',1)
        return text
    edit('public/index.html',page)

    def app(text):
        if 'let serverViewEpoch=' in text:return text
        text=text.replace('const state = {','let serverViewEpoch=0;\nconst state = {',1)
        text=text.replace('async function api(path, options = {}) {','async function api(path, options = {}) {\n  const requestEpoch=serverViewEpoch;',1)
        text=text.replace('    return await apiRequest(path, options, true);',"    if(requestEpoch!==serverViewEpoch)throw Error('服务器已切换，请重新操作');\n    return await apiRequest(path, options, true);",1)
        text=text.replace('async function apiRequest(path, options, allowRetry) {','async function apiRequest(path, options, allowRetry) {\n  const requestEpoch=serverViewEpoch;',1)
        text=text.replace('    // 网络层直接失败',"    if(requestEpoch!==serverViewEpoch)throw Error('服务器已切换，请重新操作');\n    // 网络层直接失败",1)
        text=text.replace('  const text = await res.text();',"  const text = await res.text();\n  if(requestEpoch!==serverViewEpoch)throw Error('服务器已切换，请重新操作');",1)
        text=text.replace("  if(intent==='relay'){", "  if(intent==='server-selection'){if(el.loginView)el.loginView.hidden=true;openServerPicker();return;}\n  if(intent==='relay'){",1)
        text=text.replace('function showRelayView() {','function showRelayView() {\n  serverViewEpoch++;',1)
        # Opening a management page must not be undone by a slow connection to the previous host.
        text=text.replace('async function boot() {','async function boot() {\n  const bootEpoch=serverViewEpoch;',1)
        prefix,boot=text.split('async function boot() {',1)
        for anchor in ['    state.adminAuth = await fetchAuthStatus();','    cfg = await loadPublicConfig();','          const out = await requestAutoConnect();',"    const res = await fetch(apiUrl('/api/session'), requestOptions());"]:
            boot=boot.replace(anchor,anchor+'\n    if(bootEpoch!==serverViewEpoch)return;',1)
        boot=boot.replace('  } catch (err) {','  } catch (err) {\n    if(bootEpoch!==serverViewEpoch)return;')
        boot=boot.replace('      for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {','      for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {\n        if(bootEpoch!==serverViewEpoch)return;',1)
        # Limit cancellation edits to boot(), not the login handlers below it.
        end=boot.index('// 管理面登录 / 登出 / 改口令')
        boot=boot[:end]+boot[end:].replace('    if(bootEpoch!==serverViewEpoch)return;\n','')
        text=prefix+'async function boot() {'+boot
        text=text.replace('      applySession(data.session);', '      if(reconnectEpoch!==serverViewEpoch)return null;\n      applySession(data.session);',1)
        text=text.replace('  reconnectInFlight = (async () => {','  const reconnectEpoch=serverViewEpoch;\n  reconnectInFlight = (async () => {',1)
        text=text.replace("  const view = name === 'logs' ? 'logs' : 'files';", "  if(!state.session){sharedConnectPromise=sharedServerChanged();return;}\n  const view = name === 'logs' ? 'logs' : 'files';",1)
        return text
    edit('public/app.js',app)

    def tunnel(text):
        if 'let targetGeneration=0;' in text:return text
        text=text.replace('  let handle = null;','  let targetGeneration=0;\n  let handle = null;',1)
        text=text.replace('    starting = startForwardFn(', '    const generation=targetGeneration;\n    const attemptPromise = startForwardFn(',1)
        text=text.replace('      .then((h) => {','      .then(async (h) => {\n        if(generation!==targetGeneration || stopped){await h.stop();return null;}',1)
        text=text.replace('      .catch((err) => {','      .catch((err) => {\n        if(generation!==targetGeneration || stopped)return null;',1)
        text=text.replace('        starting = null;','        if(starting===attemptPromise)starting = null;',1)
        text=text.replace('    return starting;','    starting=attemptPromise;\n    return starting;',1)
        text=text.replace('    stopped = true;','    stopped = true;\n    targetGeneration++;\n    starting=null;',1)
        return text
    edit('src/streamlit-proxy.js',tunnel)

if __name__=='__main__':
    import sys
    integrate(Path(sys.argv[1]).resolve(),Path(__file__).resolve().parent)
