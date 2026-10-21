let codeSyncPlan=null,codeSyncBusy=false,codeSyncPolling=false,codeSyncAction=null;
const codeSyncStates={comparing:'正在比较',ready:'待确认',syncing:'正在同步',completed:'同步完成',failed:'操作失败',interrupted:'操作中断',rolling_back:'正在回滚',rolled_back:'已回滚'};

async function compareTrainingCode(){
  if(codeSyncBusy)return;
  if(!trainingSettingsLoaded){toast('训练设置正在加载，请稍后比较');return;}
  const instance=$('#instSelect').value;if(!instance){toast('请先选择 AutoDL 实例');return;}
  codeSyncBusy=true;codeSyncPlan=null;renderCodeSyncPlan();$('#codeSyncMessage').textContent='正在创建本地快照并比较远端 SHA-256…';
  try{
    await api('/api/config',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({training:{local_repo_root:$('#trainingLocalRepo').value.trim(),repo_root:$('#trainingRepo').value.trim()}})});
    const plan=await api('/api/code-sync',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({instance_uuid:instance,include_assets:$('#codeSyncAssets').checked})});
    codeSyncPlan=plan;$('#codeSyncDetails').open=true;await loadCodeSyncHistory();
  }catch(error){$('#codeSyncMessage').textContent='比较失败：'+error.message;}
  finally{codeSyncBusy=false;renderCodeSyncPlan();}
}
function renderCodeSyncPlan(){
  const p=codeSyncPlan,active=p&&['syncing','rolling_back','comparing'].includes(p.status);
  $('#codeSyncCompare').disabled=codeSyncBusy||active;
  const changes=p?.files.filter(f=>f.status!=='same')||[];
  $('#codeSyncApply').disabled=codeSyncBusy||!p||p.status!=='ready'||!changes.length||p.instance_uuid!==$('#instSelect').value;
  if(!p){$('#codeSyncSummary').textContent='';$('#codeSyncFiles').replaceChildren();$('#codeSyncProgress').style.width='0%';return;}
  $('#codeSyncMessage').textContent=`${codeSyncStates[p.status]||p.status}：${p.message}`;
  $('#codeSyncSummary').textContent=`${p.local_root} → ${p.instance_uuid}:${p.remote_root} · ${p.files.length} 个文件 / ${changes.length} 个变更 / ${fmtB(p.changed_bytes)} · 快照 ${fmtT(p.created_at)}`;
  $('#codeSyncProgress').style.width=(p.status==='completed'?100:p.changed_bytes?Math.min(100,p.copied_bytes/p.changed_bytes*100):0)+'%';
  const rows=$('#codeSyncShowSame').checked?p.files:changes;
  $('#codeSyncFiles').innerHTML=rows.map(f=>`<tr><td class="mono">${escapeHtml(f.path)}</td><td>${({same:'一致',new:'新增',changed:'替换'})[f.status]||escapeHtml(f.status)}</td><td>${fmtB(f.size)}</td><td class="mono" title="${escapeHtml(f.sha256+' / '+(f.remote_sha256||'不存在'))}">${escapeHtml(f.sha256.slice(0,12))}<br>${escapeHtml(f.remote_sha256?.slice(0,12)||'不存在')}</td></tr>`).join('')||'<tr><td colspan="4" class="empty">没有需要同步的文件</td></tr>';
}
function confirmCodeSync(action){
  const p=codeSyncPlan;if(!p)return;
  if(p.instance_uuid!==$('#instSelect').value){toast('当前实例与快照不一致，请重新比较或选择原实例');return;}
  codeSyncAction={id:p.id,action,instance:p.instance_uuid};
  const changed=p.files.filter(f=>f.status!=='same');
  $('#codeSyncDialogTitle').textContent=action==='apply'?'确认同步训练代码':'确认恢复远端备份';
  $('#codeSyncReview').textContent=`本机快照：${p.local_root}\n目标实例：${p.instance_uuid}\n远端仓库：${p.remote_root}\n备份目录：${p.backup_path}\n\n${changed.map(f=>`${f.status==='new'?'新增':'替换'}  ${f.path}`).join('\n')}`;
  $('#codeSyncWarning').textContent=action==='apply'?'将安装比较时的本机快照，替换文件保留备份；有训练运行时会被拒绝。':'将恢复该次同步前的远端内容，并移除该次新增的文件。如果文件又被修改，回滚会被拒绝。';
  $('#codeSyncDialogMessage').textContent='';$('#codeSyncConfirm').disabled=false;$('#codeSyncDialog').showModal();
}
async function executeCodeSync(){
  const action=codeSyncAction;if(!action)return;
  if(action.instance!==$('#instSelect').value){$('#codeSyncDialog').close();toast('目标实例已变化，请重新确认');return;}
  $('#codeSyncConfirm').disabled=true;
  try{
    codeSyncPlan=await api(`/api/code-sync/${encodeURIComponent(action.id)}/${action.action}`,{method:'POST'});
    $('#codeSyncDialog').close();renderCodeSyncPlan();await loadCodeSyncHistory();
  }catch(error){$('#codeSyncDialogMessage').textContent=error.message;$('#codeSyncConfirm').disabled=false;}
}
async function loadCodeSyncHistory(){
  try{
    const response=await api('/api/code-sync');
    $('#codeSyncHistory').innerHTML=response.plans.map(p=>`<div class="file-item"><span class="filename">${escapeHtml(p.instance_uuid)} · ${escapeHtml(codeSyncStates[p.status]||p.status)}<br><span class="muted">${escapeHtml(fmtT(p.created_at))} · ${escapeHtml(p.message)}</span></span><button data-code-plan="${escapeHtml(p.id)}">查看</button>${['completed','failed','interrupted'].includes(p.status)?`<button data-code-plan="${escapeHtml(p.id)}" data-rollback="1">回滚</button>`:''}</div>`).join('')||'<div class="empty">暂无同步记录</div>';
  }catch(error){$('#codeSyncHistory').textContent=error.message;}
}
$('#codeSyncHistory').addEventListener('click',async event=>{
  const button=event.target.closest('button[data-code-plan]');if(!button)return;
  try{codeSyncPlan=await api('/api/code-sync/'+encodeURIComponent(button.dataset.codePlan));renderCodeSyncPlan();$('#codeSyncDetails').open=true;if(button.dataset.rollback)confirmCodeSync('rollback');}
  catch(error){toast(error.message);}
});
async function pollCodeSync(){
  if(codeSyncPolling||!codeSyncPlan||!['syncing','rolling_back'].includes(codeSyncPlan.status))return;
  codeSyncPolling=true;const id=codeSyncPlan.id;
  try{const p=await api('/api/code-sync/'+encodeURIComponent(id));if(codeSyncPlan?.id!==id)return;codeSyncPlan=p;renderCodeSyncPlan();if(!['syncing','rolling_back'].includes(p.status))await loadCodeSyncHistory();}
  catch(error){$('#codeSyncMessage').textContent='状态查询失败：'+error.message;}
  finally{codeSyncPolling=false;}
}
setInterval(()=>{if(getToken())pollCodeSync();},3000);
