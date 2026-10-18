let datasetPath='',datasetSnapshot=null,datasetEpoch=0,datasetPage=0,datasetFiltered=[],datasetDeletion=null,datasetInstanceId='',datasetPreviewEpoch=0;
const DATASET_PAGE_SIZE=50;

function syncDatasetInstanceChoices(){
  const select=$('#datasetInstance'),old=datasetInstanceId;
  select.innerHTML=$('#instSelect').innerHTML;select.value=$('#instSelect').value;
  datasetInstanceId=select.value;
  if(old!==select.value){
    datasetEpoch++;datasetPath='';datasetSnapshot=null;datasetDeletion=null;
    $('#datasetDeleteDialog').close();$('#datasetPreviewDialog').close();
    renderRemoteDatasets();
    if(!$('#datasetsPanel').hidden)loadRemoteDatasets('');
  }
}

async function loadRemoteDatasets(path=''){
  const instance=$('#datasetInstance').value,epoch=++datasetEpoch;
  datasetSnapshot=null;datasetPath=path;datasetPage=0;renderRemoteDatasets();
  if(!getToken()||!instance){$('#datasetMessage').textContent='请先保存访问令牌并选择目标实例。';return;}
  $('#datasetRefresh').disabled=true;$('#datasetMessage').textContent='正在读取远端目录和数据盘容量…';
  try{
    const response=await api('/api/datasets?'+new URLSearchParams({instance_uuid:instance,path}));
    if(epoch!==datasetEpoch||instance!==$('#datasetInstance').value)return;
    datasetSnapshot=response;
    $('#datasetMessage').textContent=!response.reachable?'实例未运行，请在“数据上传”页无卡开机后刷新。':!response.exists?'dataset 目录尚未创建，可先上传数据。':
      `已连接 ${response.instance_uuid} · 更新于 ${new Date().toLocaleTimeString()}`+(response.truncated?' · 目录超过 10,000 项，仅显示已加载部分':'');
    renderRemoteDatasets();
  }catch(error){if(epoch===datasetEpoch)$('#datasetMessage').textContent='读取失败：'+error.message;}
  finally{if(epoch===datasetEpoch)$('#datasetRefresh').disabled=false;}
}

function renderRemoteDatasets(resetPage=true){
  if(resetPage)datasetPage=0;
  const snapshot=datasetSnapshot,entries=snapshot?.entries||[],files=entries.filter(e=>e.kind==='file');
  $('#datasetFree').textContent=fmtB(snapshot?.disk?.free);
  $('#datasetCount').textContent=snapshot?.reachable?String(files.length):'—';
  $('#datasetSize').textContent=snapshot?.reachable?fmtB(files.reduce((sum,e)=>sum+e.size,0)):'—';
  const crumb=$('#datasetBreadcrumb');crumb.replaceChildren();
  const addCrumb=(label,path)=>{const button=document.createElement('button');button.textContent=label;button.onclick=()=>loadRemoteDatasets(path);crumb.append(button);};
  addCrumb('dataset','');let current='';
  for(const name of datasetPath.split('/').filter(Boolean)){current=current?current+'/'+name:name;crumb.append(document.createTextNode(' / '));addCrumb(name,current);}
  const search=$('#datasetSearch').value.trim().toLocaleLowerCase(),type=$('#datasetType').value,sort=$('#datasetSort').value;
  datasetFiltered=entries.filter(e=>e.name.toLocaleLowerCase().includes(search)&&(type==='all'||type==='corpus'&&e.trainable||type==='report'&&e.kind==='file'&&e.name.endsWith('.json')||type==='directory'&&e.kind==='directory'));
  datasetFiltered.sort((a,b)=>sort==='largest'?(b.size||0)-(a.size||0):sort==='newest'?b.modified_at-a.modified_at:(a.kind!=='directory')-(b.kind!=='directory')||a.name.localeCompare(b.name,'zh-CN'));
  datasetPage=Math.min(datasetPage,Math.max(0,Math.ceil(datasetFiltered.length/DATASET_PAGE_SIZE)-1));
  const start=datasetPage*DATASET_PAGE_SIZE;
  $('#datasetRows').innerHTML=datasetFiltered.slice(start,start+DATASET_PAGE_SIZE).map((entry,i)=>{
    const idx=start+i,directory=entry.kind==='directory',file=entry.kind==='file';
    const action=(label,act,css='')=>`<button class="${css}" data-dataset-action="${act}" data-index="${idx}">${label}</button>`;
    return `<tr><td>${directory?`<button class="dataset-name" data-dataset-action="open" data-index="${idx}">▸ ${escapeHtml(entry.name)}</button>`:escapeHtml(entry.name)}</td>
      <td class="muted">${directory?'目录':entry.trainable?'语料':file?'文件':'链接 / 特殊文件'}</td><td>${fmtB(entry.size)}</td><td class="muted">${escapeHtml(fmtT(entry.modified_at))}</td>
      <td class="acts">${directory?action('打开','open'):file?action('复制路径','copy')+(entry.previewable?action('预览','preview'):'')+(entry.trainable?action('用于训练','train'):'')+action('删除','delete','danger'):''}</td></tr>`;
  }).join('')||`<tr><td colspan="5" class="empty">${!snapshot?'等待加载远端文件':!snapshot.reachable?'实例离线':entries.length?'没有匹配的文件':'当前目录为空'}</td></tr>`;
  $('#datasetResultCount').textContent=`${datasetFiltered.length} 项${datasetFiltered.length?' · 第 '+(datasetPage+1)+' / '+Math.ceil(datasetFiltered.length/DATASET_PAGE_SIZE)+' 页':''}`;
  $('#datasetPrev').disabled=datasetPage===0;$('#datasetNext').disabled=start+DATASET_PAGE_SIZE>=datasetFiltered.length;
}
function changeDatasetPage(delta){datasetPage+=delta;renderRemoteDatasets(false);}

$('#datasetRows').addEventListener('click',async event=>{
  const button=event.target.closest('button[data-dataset-action]');if(!button||!datasetSnapshot)return;
  const entry=datasetFiltered[Number(button.dataset.index)],snapshot=datasetSnapshot,epoch=datasetEpoch;
  if(!entry||snapshot.instance_uuid!==$('#datasetInstance').value)return;
  const action=button.dataset.datasetAction,fullPath=snapshot.root+'/'+entry.path;
  if(action==='open')return loadRemoteDatasets(entry.path);
  if(action==='delete'){
    datasetDeletion={instance_uuid:snapshot.instance_uuid,path:entry.path,version:entry.version};
    $('#datasetDeleteInstance').textContent=`实例 ${snapshot.instance_uuid} · ${fmtB(entry.size)}`;
    $('#datasetDeletePath').textContent=fullPath;$('#datasetDeleteMessage').textContent='';
    $('#datasetDeleteConfirm').disabled=false;$('#datasetDeleteDialog').showModal();return;
  }
  button.disabled=true;
  try{
    if(action==='copy'){
      if(navigator.clipboard?.writeText)await navigator.clipboard.writeText(fullPath);
      else{const area=document.createElement('textarea');area.value=fullPath;document.body.append(area);area.select();const copied=document.execCommand('copy');area.remove();if(!copied)throw new Error('浏览器无法复制，请使用文件预览中的路径');}
      toast('已复制：'+fullPath);
    }else if(action==='preview'){
      const previewEpoch=++datasetPreviewEpoch;
      $('#datasetPreviewTitle').textContent=entry.name;$('#datasetPreviewHint').textContent=fullPath;$('#datasetPreviewText').textContent='正在读取…';$('#datasetPreviewDialog').showModal();
      const data=await api('/api/datasets/preview?'+new URLSearchParams({instance_uuid:snapshot.instance_uuid,path:entry.path}));
      if(epoch!==datasetEpoch||previewEpoch!==datasetPreviewEpoch)return;
      let text=data.text;if(entry.name.endsWith('.json')&&!data.truncated){try{text=JSON.stringify(JSON.parse(text),null,2);}catch{}}
      $('#datasetPreviewText').textContent=text;$('#datasetPreviewHint').textContent=fullPath+(data.truncated?' · 仅显示前 128 KiB':'');
    }else if(action==='train'){
      switchView('training',{skipRefresh:true});await refreshTraining();
      if(epoch!==datasetEpoch||snapshot.instance_uuid!==$('#instSelect').value)throw new Error('目标实例已变化，请重新选择文件');
      $('#trainingData').value=fullPath;
      panelChanged();$('#trainingMessage').textContent='已选择远端语料；配置训练参数后即可下发任务。';
    }
  }catch(error){if(action==='preview')$('#datasetPreviewText').textContent=error.message;else toast(error.message);}
  finally{button.disabled=false;}
});

async function confirmDatasetDelete(){
  const deletion=datasetDeletion;if(!deletion)return;
  if(deletion.instance_uuid!==$('#datasetInstance').value){$('#datasetDeleteDialog').close();return;}
  $('#datasetDeleteConfirm').disabled=true;$('#datasetDeleteMessage').textContent='正在检查训练/交付状态和文件版本…';
  try{
    await api('/api/datasets',{method:'DELETE',headers:{'Content-Type':'application/json'},body:JSON.stringify(deletion)});
    $('#datasetDeleteDialog').close();datasetDeletion=null;toast('已删除远端文件：'+deletion.path);await loadRemoteDatasets(datasetPath);
  }catch(error){$('#datasetDeleteMessage').textContent=error.message;$('#datasetDeleteConfirm').disabled=false;}
}
syncDatasetInstanceChoices();renderRemoteDatasets();
