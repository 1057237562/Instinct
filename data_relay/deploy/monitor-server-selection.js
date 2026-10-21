'use strict';
// Shares app.js state/functions; credentials never enter this module.
const serverPicker=document.getElementById('server-picker');
const serverChoices=document.getElementById('shared-server-choice');
const serverPickerStatus=document.getElementById('server-picker-status');
let sharedConnectPromise=null;
async function refreshServerChoices() {
  serverChoices.disabled=true;document.getElementById('server-picker-apply').disabled=true;
  serverChoices.replaceChildren(new Option('正在加载实例…',''));
  serverPickerStatus.textContent='正在获取实例…';
  try {
    const response=await fetch('/api/relay/servers');
    if(response.status===401){state.adminAuth={required:true,authenticated:false};serverPicker.hidden=true;ensureAdminAuth('server-selection','请使用现有管理员账号继续切换服务器。');return;}
    const data=await response.json();if(!response.ok)throw Error(data.error || '实例获取失败');
    serverChoices.replaceChildren();
    const placeholder=new Option('请选择服务器','');serverChoices.add(placeholder);
    for(const instance of data.instances || [])serverChoices.add(new Option(`${instance.name || instance.uuid} · ${instance.status}`,instance.uuid));
    serverChoices.value=data.ssh_instance_uuid || data.selected || '';
    serverChoices.disabled=false;document.getElementById('server-picker-apply').disabled=false;
    serverPickerStatus.textContent=`当前 SSH：${data.ssh.host?`${data.ssh.username}@${data.ssh.host}:${data.ssh.port}`:'等待实例提供 SSH 地址'}${data.ssh_instance_uuid?'':'（尚未联动，请选择实例并切换）'}`;
  }catch(error){serverPickerStatus.textContent=error.message;}
}
function openServerPicker() {
  serverViewEpoch++; // Do not let an old startup/reconnect overwrite this view.
  if(!ensureAdminAuth('server-selection','切换服务器沿用当前管理员登录。'))return;
  closeSettings();
  document.getElementById('toast-stack').replaceChildren();
  serverPicker.hidden=false;refreshServerChoices();
}
document.querySelectorAll('[data-server-picker]').forEach(button=>button.addEventListener('click',openServerPicker));
function closeServerPicker(){
  serverPicker.hidden=true;
  if(!state.session && document.getElementById('relay-view').hidden)boot();
}
document.getElementById('server-picker-close').onclick=closeServerPicker;
document.getElementById('server-picker-refresh').onclick=refreshServerChoices;
serverPicker.addEventListener('click',event=>{if(event.target===serverPicker)closeServerPicker();});

async function sharedServerChanged({preserveRelay=false,broadcast=true}={}) {
  const epoch=++serverViewEpoch;
  stopSessionKeepalive();stopLiveTail();lastReconnectAt=0;reconnectInFlight=null;
  state.session=null;state.cwd=null;state.entries=[];state.selectedPath=null;state.dirSeq++;
  logState.root=null;logState.entries=[];logState.selected=[];logState.runs.clear();resetViewer();
  const dot=el.connLabel.parentElement.querySelector('.status-dot');if(dot)dot.classList.add('off');
  el.connLabel.textContent='正在连接所选服务器…';
  if(broadcast)serverSelectionChannel?.postMessage({type:'changed'});
  const frame=document.getElementById('relay-frame');
  frame.contentWindow?.postMessage({type:'monitor-server-changed'},location.origin);
  try {
    const cfg=await loadPublicConfig();if(epoch!==serverViewEpoch)return;
    const response=await fetch('/api/auto-connect?force=1',{method:'POST',signal:AbortSignal.timeout(25000)});
    const data=await response.json();if(epoch!==serverViewEpoch)return;
    if(!response.ok || !data.session)throw Error(data.error || 'SSH 暂不可用');
    applySession(data.session);
    if(!preserveRelay)enterApp(data.session,'files');
    else toast('服务器已切换，SSH 已连接');
  }catch(error){
    if(epoch!==serverViewEpoch)return;
    el.connLabel.textContent='所选服务器暂未连接';
    if(!preserveRelay)showBootError(`服务器选择已保存，SSH 暂未连接：${error.message}。可继续切换服务器。`);
    else toast('服务器选择已保存，SSH 暂不可用；可稍后重试或切换服务器');
  }
}
document.getElementById('server-picker-apply').onclick=async()=>{
  const uuid=serverChoices.value;if(!uuid){serverPickerStatus.textContent='请先选择实例';return;}
  const button=document.getElementById('server-picker-apply');button.disabled=true;
  serverPickerStatus.textContent='正在同步数据、训练和 SSH 连接目标…';
  try {
    const response=await fetch('/relay/api/config',{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify({instance_uuid:uuid})});
    if(response.status===401){state.adminAuth={required:true,authenticated:false};serverPicker.hidden=true;ensureAdminAuth('server-selection','登录已过期，请重新登录后切换服务器。');return;}
    const data=await response.json();if(!response.ok)throw Error(data.error || '切换失败');
    serverPicker.hidden=true;
    sharedConnectPromise=sharedServerChanged({preserveRelay:!document.getElementById('relay-view').hidden});
    await sharedConnectPromise;
  }catch(error){serverPickerStatus.textContent=error.message;}
  finally{button.disabled=false;}
};
const serverSelectionChannel=typeof BroadcastChannel==='function'?new BroadcastChannel('instinct-server-selection'):null;
serverSelectionChannel?.addEventListener('message',()=>{
  sharedConnectPromise=sharedServerChanged({preserveRelay:!document.getElementById('relay-view').hidden,broadcast:false});
});
window.addEventListener('message',event=>{
  const frame=document.getElementById('relay-frame');
  if(event.origin!==location.origin || event.source!==frame.contentWindow || event.data?.type!=='relay-server-changed')return;
  sharedConnectPromise=sharedServerChanged({preserveRelay:true});
});
