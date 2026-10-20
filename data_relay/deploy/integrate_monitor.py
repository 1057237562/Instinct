"""Apply a minimal, repeatable integration to the existing SSH monitor checkout."""
from pathlib import Path
import shutil
import sys

root=Path(sys.argv[1]).resolve()
assets=Path(__file__).resolve().parent
server=root/'server.js'; code=server.read_text(encoding='utf-8')
if 'createRelayProxy' not in code:
    anchor="const app = express();"
    code=code.replace(anchor,"const { createRelayProxy, isPull } = require('./src/relay-proxy.cjs');\n"+anchor,1)
    anchor='// 其余请求才解析请求体。'
    block=r"""// Instinct relay shares the existing admin session and CSRF guard.
const relayProxy = createRelayProxy();
app.get(/^\/relay$/, (req,res)=>res.redirect(308,'/relay/'));
app.use('/relay', csrfOriginGuard, (req,res,next)=> {
  if(isPull(req))return next();
  adminAuthGuard(req,res,next);
},relayProxy);

"""
    if anchor not in code:raise SystemExit('Monitor proxy anchor changed; review server.js')
    code=code.replace(anchor,block+anchor,1)
code=code.replace("app.get('/relay', (req,res)=>res.redirect(308,'/relay/'));", "app.get(/^\\/relay$/, (req,res)=>res.redirect(308,'/relay/'));")
server.write_text(code,encoding='utf-8')
shutil.copy2(assets/'monitor-relay-proxy.cjs',root/'src/relay-proxy.cjs')

page=root/'public/index.html'; code=page.read_text(encoding='utf-8')
if 'data-view="relay"' not in code:
    code=code.replace('<button type="button" class="view-tab" data-view="logs">训练日志</button>',
        '<button type="button" class="view-tab" data-view="logs">训练日志</button>\n          <button type="button" class="view-tab" data-view="relay">数据与训练</button>',1)
    code=code.replace('        <button class="btn ghost" id="boot-settings" hidden>设置服务器</button>',
        '        <button class="btn ghost" id="boot-settings" hidden>设置服务器</button>\n        <button class="btn primary" id="boot-relay">数据与训练</button>',1)
    code=code.replace('  <script src="config.js"></script>',
        '  <style>#relay-view{height:calc(100dvh - 90px);min-height:500px}#relay-frame{width:100%;height:100%;border:0;background:#0d1117}.topbar-actions .view-switch{flex-shrink:0}@media(max-width:720px){#relay-view{height:calc(100dvh - 130px)}.topbar-actions{flex-wrap:wrap}}</style>\n  <script src="config.js"></script>',1)
    # Keep the panel in the existing app container, beside the file/log views.
    marker='    <main class="layout log-layout" id="log-view" hidden>'
    if marker not in code:raise SystemExit('Monitor view anchor changed')
    code=code.replace(marker,'    <main id="relay-view" hidden><iframe id="relay-frame" title="Instinct 数据与训练" allow="clipboard-write"></iframe></main>\n\n'+marker,1)
if 'id="connect-relay"' not in code:
    code=code.replace('<div class="default-note" id="default-note" hidden></div>',
        '<div class="default-note" id="default-note" hidden></div>\n      <button type="button" class="btn primary block" id="connect-relay">数据与训练</button>',1)
page.write_text(code,encoding='utf-8')

app=root/'public/app.js';code=app.read_text(encoding='utf-8')
if 'function showRelayView()' not in code:
    marker='function switchView(name) {'
    addition="""function showRelayView() {
  if (!ensureAdminAuth('relay', '数据与训练使用现有管理员登录。')) return;
  el.bootView.hidden=true;el.connectView.hidden=true;el.appView.hidden=false;
  el.fileView.hidden=true;el.logView.hidden=true;
  document.getElementById('relay-view').hidden=false;
  el.viewTabs.forEach(tab=>tab.classList.toggle('active',tab.dataset.view==='relay'));
  const frame=document.getElementById('relay-frame');
  if(!frame.getAttribute('src'))frame.src='/relay/?managed=1';
  stopLiveTail();
}
document.getElementById('boot-relay').addEventListener('click',showRelayView);
window.addEventListener('message',event=>{
  const frame=document.getElementById('relay-frame');
  if(event.origin!==location.origin || event.source!==frame.contentWindow || event.data?.type!=='relay-auth-required')return;
  state.adminAuth={required:true,authenticated:false};
  ensureAdminAuth('relay','登录已过期，请重新登录。');
});

"""
    if marker not in code:raise SystemExit('Monitor switchView anchor changed')
    code=code.replace(marker,addition+marker+"\n  if(name==='relay'){showRelayView();return;}\n  document.getElementById('relay-view').hidden=true;",1)
    marker='  const intent = state.loginIntent;\n  state.loginIntent = null;'
    if marker not in code:raise SystemExit('Monitor login continuation anchor changed')
    code=code.replace(marker,marker+"\n  if(intent==='relay'){if(el.loginView)el.loginView.hidden=true;showRelayView();return;}",1)
if "getElementById('connect-relay').addEventListener" not in code:
    code+="\ndocument.getElementById('connect-relay').addEventListener('click',showRelayView);\n"
app.write_text(code,encoding='utf-8')
from integrate_selection import integrate
integrate(root, assets)
print('Integrated monitor relay proxy, shared login and data/training view')
