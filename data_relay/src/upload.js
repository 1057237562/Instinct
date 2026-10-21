// Browser upload controller. Hashes run off the UI thread, including on plain HTTP.
const DATA_CHUNK = 16 * 1024 * 1024;
let selectedData = [], dataRecords = [], resumeDataId = null, dataBusy = false, dataPaused = false;
let uploadSettingsLoaded = false;
const liveRequests = new Set();

function switchView(view, options={}) {
  document.querySelectorAll('[data-view]').forEach(el => el.hidden = el.dataset.view !== view);
  document.querySelectorAll('[data-view-button]').forEach(el => {
    if (el.dataset.viewButton === view) el.setAttribute('aria-current', 'page');
    else el.removeAttribute('aria-current');
  });
  if(view==='training' && getToken()&&!options.skipRefresh)refreshTraining();
  if(view==='settings' && getToken())loadUploadSettings();
  if(view==='datasets' && getToken())loadRemoteDatasets(datasetPath);
}

function toast(message) {
  $('#toast').textContent=message; $('#toast').hidden=false;
  clearTimeout(toast.timer); toast.timer=setTimeout(()=>$('#toast').hidden=true,6000);
}

async function loadUploadSettings() {
  const cfg=await api('/api/config');
  if(document.activeElement!==$('#pullBaseUrl'))$('#pullBaseUrl').value=cfg.pull_base_url||'';
  if(!uploadSettingsLoaded)$('#dataMode').value=cfg.pull_base_url?'pull':'sftp';
  uploadSettingsLoaded=true; updateChannelHint();
}
function updateChannelHint() {
  $('#channelHint').textContent=$('#dataMode').value==='pull'
    ?'AutoDL 从中继多连接拉取；请在连接设置中填写实例可达的中继地址。'
    :'通过实例 SSH 通道续传，不需要公开中继地址；速度取决于 SSH 通道带宽。';
}
async function savePullAddress() {
  try {
    await api('/api/config',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({pull_base_url:$('#pullBaseUrl').value.trim()})});
    $('#pullMessage').textContent='已保存，新任务可选择 HTTP 高速拉取。';
    await loadUploadSettings();
  }catch(error){$('#pullMessage').textContent=error.message;}
}

function addDataFiles(files) {
  if(dataBusy)return;
  if(resumeDataId)selectedData=[];
  for(const file of files)if(!selectedData.some(f=>f.name===file.name))selectedData.push(file);
  renderSelection();
}
function renderSelection() {
  $('#selectedFiles').replaceChildren();
  selectedData.forEach((file,index)=>{
    const row=document.createElement('div');row.className='file-item';
    const name=document.createElement('span');name.className='filename';name.textContent=file.name;
    const size=document.createElement('span');size.className='muted';size.textContent=fmtB(file.size);
    const remove=document.createElement('button');remove.textContent='移除';remove.disabled=dataBusy;
    remove.onclick=()=>{selectedData.splice(index,1);renderSelection();};
    row.append(name,size,remove);$('#selectedFiles').append(row);
  });
  $('#selectionSummary').textContent=selectedData.length?`${selectedData.length} 个文件 · ${fmtB(selectedData.reduce((n,f)=>n+f.size,0))}`:'尚未选择文件';
  $('#dataStart').textContent=resumeDataId?'继续上传并交付':'上传并交付到 dataset';
  $('#dataReset').hidden=!resumeDataId;$('#dataReset').disabled=dataBusy;
}
function resetDataUpload(){if(dataBusy)return;resumeDataId=null;selectedData=[];$('#dataFiles').value='';$('#dataMessage').textContent='已切换到新上传，之前的断点仍在传输记录中。';$('#ingestBar').style.width='0%';renderSelection();}

// SHA-256 fallback for non-secure HTTP origins where SubtleCrypto is unavailable.
function sha256Fallback(buffer) {
  const k=new Uint32Array([0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]);
  const input=new Uint8Array(buffer), n=input.length, padded=new Uint8Array(Math.ceil((n+9)/64)*64);
  padded.set(input);padded[n]=128;
  const data=new DataView(padded.buffer);data.setUint32(padded.length-8,Math.floor(n/0x20000000));data.setUint32(padded.length-4,(n*8)>>>0);
  const h=new Uint32Array([0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]), w=new Uint32Array(64);
  const r=(x,n)=>(x>>>n)|(x<<(32-n));
  for(let offset=0;offset<padded.length;offset+=64){
    for(let i=0;i<16;i++)w[i]=data.getUint32(offset+i*4);
    for(let i=16;i<64;i++){const a=w[i-15],b=w[i-2];w[i]=(w[i-16]+(r(a,7)^r(a,18)^(a>>>3))+w[i-7]+(r(b,17)^r(b,19)^(b>>>10)))>>>0;}
    let [a,b,c,d,e,f,g,j]=h;
    for(let i=0;i<64;i++){const t1=(j+(r(e,6)^r(e,11)^r(e,25))+((e&f)^(~e&g))+k[i]+w[i])>>>0;
      const t2=((r(a,2)^r(a,13)^r(a,22))+((a&b)^(a&c)^(b&c)))>>>0;
      j=g;g=f;f=e;e=(d+t1)>>>0;d=c;c=b;b=a;a=(t1+t2)>>>0;}
    [a,b,c,d,e,f,g,j].forEach((v,i)=>h[i]=(h[i]+v)>>>0);
  }
  return Array.from(h,v=>v.toString(16).padStart(8,'0')).join('');
}

const hashWorkerUrl=URL.createObjectURL(new Blob([
  sha256Fallback.toString(),
  `;self.onmessage=async({data:{id,buffer}})=>{try{const hash=self.crypto?.subtle?Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),v=>v.toString(16).padStart(2,'0')).join(''):sha256Fallback(buffer);self.postMessage({id,hash});}catch(error){self.postMessage({id,error:error.message});}};`
],{type:'text/javascript'}));
const hashWorker=new Worker(hashWorkerUrl);URL.revokeObjectURL(hashWorkerUrl);
const hashWaiters=new Map();let nextHashId=0;
hashWorker.onmessage=({data})=>{const pending=hashWaiters.get(data.id);if(!pending)return;hashWaiters.delete(data.id);data.error?pending.reject(new Error(data.error)):pending.resolve(data.hash);};
hashWorker.onerror=()=>{hashWaiters.forEach(p=>p.reject(new Error('文件校验线程失败，请刷新页面后续传')));hashWaiters.clear();};
async function hashBlob(blob){const buffer=await blob.arrayBuffer();return new Promise((resolve,reject)=>{const id=++nextHashId;hashWaiters.set(id,{resolve,reject});hashWorker.postMessage({id,buffer},[buffer]);});}
async function fingerprint(file){return hashBlob(new Blob([`${file.name}\n${file.size}\n`,file.slice(0,65536),file.slice(Math.max(0,file.size-65536))]));}

function pauseDataUpload(){dataPaused=true;liveRequests.forEach(xhr=>xhr.abort());$('#dataMessage').textContent='正在暂停，已确认分块会保留…';}
function sendChunk(taskId,fi,ci,blob,hash,onProgress){
  return new Promise((resolve,reject)=>{
    const xhr=new XMLHttpRequest();liveRequests.add(xhr);
    const finish=(error)=>{liveRequests.delete(xhr);error?reject(error):resolve();};
    xhr.open('PUT',relayUrl(`/api/data/uploads/${encodeURIComponent(taskId)}/files/${fi}/chunks/${ci}`));
    xhr.setRequestHeader('Authorization',getToken());xhr.setRequestHeader('x-chunk-sha256',hash);
    xhr.setRequestHeader('Content-Type','application/octet-stream');xhr.timeout=180000;
    xhr.upload.onprogress=e=>onProgress(e.loaded);
    xhr.onload=()=>{if(xhr.status>=200&&xhr.status<300)return finish();let message=xhr.responseText;try{message=JSON.parse(message).error||message;}catch{}
      finish(Object.assign(new Error(message||`HTTP ${xhr.status}`),{retryable:xhr.status>=500}));};
    xhr.onerror=xhr.ontimeout=()=>finish(Object.assign(new Error('网络中断或超时'),{retryable:true}));
    xhr.onabort=()=>finish(new Error('上传已暂停'));xhr.send(blob);
  });
}

async function startDataUpload(){
  if(dataBusy)return;
  if(!getToken()){switchView('data');$('#token').focus();toast('请先保存中继访问令牌');return;}
  if(!selectedData.length){$('#dataFiles').click();return;}
  const instance=$('#instSelect').value;
  if(!resumeDataId&&!instance){toast('请先选择目标 AutoDL 实例');return;}
  dataBusy=true;dataPaused=false;$('#dataStart').disabled=true;$('#dataPause').hidden=false;
  for(const id of ['dataFiles','dataName','dataMode'])$('#'+id).disabled=true;
  renderSelection();$('#dataMessage').textContent='正在读取文件指纹…';
  try {
    const files=[];for(const file of selectedData)files.push({name:file.name,size:file.size,fingerprint:await fingerprint(file)});
    if(dataPaused)throw new Error('上传已暂停');
    let task;
    if(resumeDataId){
      task=await api('/api/tasks/'+encodeURIComponent(resumeDataId));
      if(task.status!=='receiving')throw new Error('该任务已提交交付，请在传输记录查看状态');
      const expected=task.data_upload.files;
      if(expected.length!==files.length||expected.some(f=>!files.some(g=>g.name===f.name&&g.size===f.size&&g.fingerprint===f.fingerprint)))throw new Error('所选文件与任务不一致，请重新选择原始文件（包括审计报告）');
      selectedData=expected.map(f=>selectedData.find(g=>g.name===f.name));
    }else{
      task=await api('/api/data/uploads',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:$('#dataName').value,instance_uuid:instance,upload_mode:$('#dataMode').value,files})});
      resumeDataId=task.id;
    }
    const work=[];let acknowledged=0;
    task.data_upload.files.forEach((f,fi)=>{for(let ci=0;ci<f.chunks.length;ci++){const size=Math.min(DATA_CHUNK,f.size-ci*DATA_CHUNK);if(f.chunks[ci])acknowledged+=size;work.push({fi,ci,size,hash:f.chunks[ci]});}});
    const total=task.upload_total_bytes, started=performance.now(), inflight=new Map();let sent=0,next=0;
    const progress=()=>{
      const pending=[...inflight.values()].reduce((a,b)=>a+b,0), done=Math.min(total,acknowledged+pending);
      const seconds=(performance.now()-started)/1000,rate=(sent+pending)/Math.max(seconds,.1),remaining=rate>0?Math.ceil((total-done)/rate):0;
      $('#ingestBar').style.width=(total?100*done/total:100)+'%';
      $('#dataMessage').textContent=`本地 → 中继：${fmtB(done)} / ${fmtB(total)} · ${fmtB(rate)}/s${remaining?' · 约 '+(remaining<60?remaining+' 秒':Math.ceil(remaining/60)+' 分钟'):''} · 已有分块也会重新核对内容`;
    };
    const worker=async()=>{
      while(next<work.length){
        if(dataPaused)throw new Error('上传已暂停');
        const {fi,ci,size,hash:saved}=work[next++],key=`${fi}:${ci}`,blob=selectedData[fi].slice(ci*DATA_CHUNK,ci*DATA_CHUNK+size);
        const hash=await hashBlob(blob);
        if(dataPaused)throw new Error('上传已暂停');
        if(saved){if(saved!==hash)throw new Error(`原文件内容已变化：${selectedData[fi].name}。请新建任务上传。`);progress();continue;}
        for(let attempt=0;;attempt++){
          try{await sendChunk(task.id,fi,ci,blob,hash,loaded=>{inflight.set(key,loaded);progress();});break;}
          catch(error){inflight.delete(key);if(dataPaused||!error.retryable||attempt>=3)throw error;await new Promise(r=>setTimeout(r,1000*2**attempt));if(dataPaused)throw new Error('上传已暂停');}
        }
        inflight.delete(key);acknowledged+=size;sent+=size;progress();
      }
    };
    const workers=Array.from({length:Math.min(4,Math.max(1,work.length))},()=>worker().catch(error=>{dataPaused=true;liveRequests.forEach(xhr=>xhr.abort());throw error;}));
    const outcomes=await Promise.allSettled(workers),failure=outcomes.find(r=>r.status==='rejected');
    if(failure)throw failure.reason;if(dataPaused)throw new Error('上传已暂停');
    await api(`/api/data/uploads/${encodeURIComponent(task.id)}/complete`,{method:'POST'});
    $('#dataMessage').textContent='文件已接收，后台正在校验并交付到 dataset。现在可以关闭页面。';
    $('#ingestBar').style.width='100%';resumeDataId=null;selectedData=[];$('#dataFiles').value='';
    await loadTasks();
  }catch(error){$('#dataMessage').textContent=error.message+'；已确认的分块保留，可继续上传。';loadTasks().catch(()=>{});}
  finally{dataBusy=false;$('#dataStart').disabled=false;$('#dataPause').hidden=true;for(const id of ['dataFiles','dataName','dataMode'])$('#'+id).disabled=false;renderSelection();}
}

function updateDataTasks(tasks){dataRecords=tasks.sort((a,b)=>b.created_at-a.created_at);renderDataTasks();}
function renderDataTasks(){
  const filter=$('#dataFilter').value,labels={receiving:'等待 / 正在接收',verifying:'校验 / 等待交付',uploading:'交付到 AutoDL',failed:'需要重试',done:'数据就绪',cancelled:'已暂停'};
  const tasks=dataRecords.filter(t=>filter==='all'||(filter==='active'?!['done','failed','cancelled'].includes(t.status):t.status===filter));
  $('#dataTasks').innerHTML=tasks.map(t=>{
    const receiving=t.status==='receiving',done=t.status==='done',busy=['verifying','uploading'].includes(t.status);
    const bytes=receiving?t.received_bytes:t.upload_done_bytes,pct=t.upload_total_bytes?Math.min(100,100*bytes/t.upload_total_bytes):(done?100:0);
    return `<article class="transfer-card"><div class="row"><h3 style="flex:1">${escapeHtml(t.name)}</h3><span class="badge b-${escapeHtml(t.status)}">${labels[t.status]||escapeHtml(t.status)}</span></div>
      <div class="transfer-meta">${t.data_upload.files.length} 个文件 · ${escapeHtml(t.data_upload.instance_uuid)} · ${escapeHtml(fmtT(t.created_at))} · ${escapeHtml(t.upload_mode.toUpperCase())}</div>
      <div class="steps"><span class="${receiving?'current':''}">① 本地 → 中继</span><span class="${busy?'current':''}">② 校验与交付</span><span class="${done?'current':''}">③ dataset 就绪</span></div>
      <div class="bar"><i style="width:${pct}%"></i></div><div class="transfer-meta">${fmtB(bytes)} / ${fmtB(t.upload_total_bytes)}${t.status==='uploading'?' · 传输后仍需校验 SHA-256':''}</div>
      ${t.status==='uploading'?`<div class="transfer-meta" data-transfer-rate="${escapeHtml(t.id)}" style="color:var(--acc)">${escapeHtml(transferRateText(t))}</div>`:''}
      ${t.error?`<div class="status-text" style="color:var(--warn)">${escapeHtml(t.error)}</div>`:''}
      <details><summary class="muted">文件与目标路径</summary><div class="transfer-meta">${t.data_upload.files.map(f=>escapeHtml('/root/autodl-tmp/dataset/'+f.name)).join('<br>')}</div></details>
      <div class="row" style="margin-top:12px">${receiving?`<button data-data-action="resume" data-id="${escapeHtml(t.id)}">选择原文件继续上传</button>`:''}${!receiving&&!busy&&!done?`<button data-data-action="retry" data-id="${escapeHtml(t.id)}">继续交付</button>`:''}
      ${done?`<button data-data-action="train" data-id="${escapeHtml(t.id)}">前往训练</button>`:''}${!busy?`<button data-data-action="delete" data-id="${escapeHtml(t.id)}">删除中继记录与副本</button>`:''}</div></article>`;
  }).join('')||'<div class="empty">暂无符合条件的传输记录。选择文件后即可开始。</div>';
}
$('#dataTasks').addEventListener('click',async event=>{
  const button=event.target.closest('button[data-data-action]');if(!button)return;
  const {id,dataAction:action}=button.dataset;
  if(action==='resume'){
    if(dataBusy){toast('请先暂停当前本地上传');return;}
    resumeDataId=id;selectedData=[];renderSelection();$('#dataMessage').textContent='请选择此任务的全部原始文件，已完成分块会自动跳过。';$('#dataFiles').click();return;
  }
  if(action==='train'){switchView('training');return;}
  if(action==='delete'){
    if(dataBusy&&resumeDataId===id){toast('请先暂停本地上传');return;}
    if(!confirm('删除中继上的任务与数据副本？AutoDL dataset 中已交付文件会保留。'))return;
  }
  button.disabled=true;
  try{
    await api(action==='delete'?`/api/tasks/${encodeURIComponent(id)}`:`/api/data/uploads/${encodeURIComponent(id)}/complete`,{method:action==='delete'?'DELETE':'POST'});
    if(action==='delete'&&resumeDataId===id){resumeDataId=null;renderSelection();}
    await loadTasks();
  }catch(error){toast(error.message);button.disabled=false;}
});

$('#dataFiles').addEventListener('change',event=>addDataFiles(event.target.files));
$('#dataMode').addEventListener('change',updateChannelHint);
const drop=$('#dataDrop');
for(const type of ['dragenter','dragover'])drop.addEventListener(type,event=>{event.preventDefault();drop.classList.add('drag');});
for(const type of ['dragleave','drop'])drop.addEventListener(type,event=>{event.preventDefault();drop.classList.remove('drag');});
drop.addEventListener('drop',event=>addDataFiles(event.dataTransfer.files));
window.addEventListener('beforeunload',event=>{if(dataBusy){event.preventDefault();event.returnValue='';}});
$('#dataInstanceSlot').append($('#instanceCard'));
updateChannelHint();
if(getToken())refreshAll();else{$('#dataMessage').textContent='先在顶部保存中继访问令牌，再选择目标实例与文件。';}
