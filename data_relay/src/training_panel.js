// Interactive state uses Config WebUI keys; only the server compiles CLI arguments.
let trainingPanelDocument=null, trainingSchema=null, trainingPanelStage='', trainingSaveTimer=null;
let trainingPreviewTimer=null, trainingPreviewSequence=0, trainingSaveQueue=Promise.resolve();
const trainingLaunchFields={name:'#trainingName',config_path:'#trainingConfig',data_path:'#trainingData',
  nproc:'#trainingNproc',from_weight:'#trainingBase',existing_resume:'#trainingExisting',gpu_devices:'#trainingDevices'};

function panelStatus(text){$('#trainingPanelStatus').textContent=text;}
function panelKey(field){return field.key.replace('{stage}',trainingPanelStage);}
function panelDefault(field){
  if(field.key==='factor'&&trainingPanelDocument.state.rope_scaling_type==='longrope')return (trainingPanelDocument.state.max_position_embeddings||32768)/(trainingPanelDocument.state.original_max_position_embeddings||4096);
  if(field.key==='original_max_position_embeddings'&&trainingPanelDocument.state.rope_scaling_type==='longrope')return 4096;
  return field.key.includes('{stage}')?trainingSchema.stage_defaults[trainingPanelStage][field.key.replace('_{stage}','')]:field.default;
}
function panelValue(input,field){
  if(field.type==='bool')return input.value===''?undefined:input.value==='true';
  if(input.value.trim()==='')return undefined;
  if(field.type==='json')return JSON.parse(input.value);
  if(['int','float'].includes(field.type)){
    const n=Number(input.value);if(!Number.isFinite(n)||(field.type==='int'&&!Number.isInteger(n)))throw new Error(field.label+' 必须是有效数值');
    return n;
  }
  return input.value;
}
function panelControl(field,value,extra=false){
  const label=document.createElement('label');label.className='f';
  const title=document.createElement('span');title.textContent=field.label;label.append(title);
  const select=field.choices||field.type==='bool',input=document.createElement(select?'select':'input');
  input.dataset.panelKey=extra?field.key:panelKey(field);input.dataset.panelExtra=String(extra);
  input.id='tp_'+(extra?'extra_':'')+input.dataset.panelKey;
  if(select){
    const choices=field.choices||(field.type==='bool'?[true,false]:[]);
    if(extra){const option=new Option('默认'+(field.default==null?'':`（${field.default}）`),'');input.append(option);}
    for(const value of choices){
      const text=typeof value==='boolean'?(value?'开启':'关闭'):
        field.key==='model_architecture'&&value==='recurrsive_instinct'?'RecurrsiveInstinct':String(value);
      input.append(new Option(text,String(value)));
    }
    if(field.type==='bool'&&(value===0||value===1))value=Boolean(value);
    if(value!==undefined&&!choices.some(v=>String(v)===String(value)))input.append(new Option(String(value),String(value)));
    input.value=value===undefined?'':String(value);
  }else{
    input.type=['int','float'].includes(field.type)?'number':'text';
    if(input.type==='number'){input.step=field.type==='int'?'1':'any';if(field.min!=null)input.min=field.min;if(field.max!=null)input.max=field.max;}
    input.value=value===undefined||value===null?'':typeof value==='object'?JSON.stringify(value):String(value);
    if(extra&&field.default!=null)input.placeholder='默认：'+field.default;
  }
  input.title=field.key;
  input.addEventListener('change',()=>{
    try{
      const value=panelValue(input,field),target=extra?trainingPanelDocument.extras[trainingPanelStage]:trainingPanelDocument.state;
      if(value===undefined)delete target[input.dataset.panelKey];else target[input.dataset.panelKey]=value;
      input.setCustomValidity('');if(extra)$('#trainingArgs').value=JSON.stringify(target,null,2);
      panelChanged();
    }catch(error){input.setCustomValidity(error.message);input.reportValidity();panelStatus(error.message);}
  });
  label.append(input);return label;
}
function renderPanelControls(){
  const area=$('#trainingControls');area.replaceChildren();
  trainingPanelDocument.extras[trainingPanelStage]??={};
  for(const [index,group] of trainingSchema.groups.entries()){
    const details=document.createElement('details');details.className='training-control-group';details.open=index<2;
    const summary=document.createElement('summary');summary.textContent=group.title;
    const grid=document.createElement('div');grid.className='training-grid';
    for(const field of group.fields){const key=panelKey(field);grid.append(panelControl(field,trainingPanelDocument.state[key]??panelDefault(field)));}
    details.append(summary,grid);area.append(details);
  }
  const details=document.createElement('details');details.className='training-control-group';
  const summary=document.createElement('summary');summary.textContent='阶段专用参数与日志 / 保存间隔（空值沿用训练脚本默认）';
  const grid=document.createElement('div');grid.className='training-grid';
  for(const field of Object.values(trainingSchema.arguments[trainingPanelStage]))grid.append(panelControl(field,trainingPanelDocument.extras[trainingPanelStage][field.key],true));
  details.append(summary,grid);area.append(details);
  $('#trainingArgs').value=JSON.stringify(trainingPanelDocument.extras[trainingPanelStage],null,2);
}
function rememberTrainingLaunch(){
  if(!trainingPanelDocument||!trainingPanelStage)return;
  const launch={};for(const [key,selector] of Object.entries(trainingLaunchFields))launch[key]=$(selector).value;
  trainingPanelDocument.launch[trainingPanelStage]=launch;
  trainingPanelDocument.state['data_path_'+trainingPanelStage]=launch.data_path;
  trainingPanelDocument.state.train_type=trainingPanelStage;
}
function restoreTrainingLaunch(){
  const launch=trainingPanelDocument.launch[trainingPanelStage]||{};
  for(const [key,selector] of Object.entries(trainingLaunchFields)){
    const el=$(selector);let value=launch[key]??(key==='config_path'?'__panel_cache__':key==='nproc'?'1':'');
    if(key==='data_path')value=launch.data_path??trainingPanelDocument.state['data_path_'+trainingPanelStage]??'';
    if(el.tagName==='SELECT'&&value&&![...el.options].some(o=>o.value===String(value)))el.append(new Option(String(value),String(value)));
    el.value=value;
  }
}
async function loadEditableTrainingPanel(){
  if(trainingPanelDocument)return;
  trainingPanelBusy=true;
  try{
    const result=await api('/api/training/panel');trainingSchema=result.schema;trainingPanelDocument=result.document;
    trainingPanelDocument.extras??={};trainingPanelDocument.launch??={};
    trainingPanelStage=trainingPanelDocument.state.train_type||'pretrain';$('#trainingTrainer').value=trainingPanelStage;
    $('#trainingPreset').replaceChildren(new Option('选择模型预设…',''),...Object.keys(trainingSchema.presets).map(s=>new Option(s,s)));
    restoreTrainingLaunch();renderPanelControls();$('#trainingCacheSource').textContent=result.source;
    panelStatus('配置已载入；修改后自动保存到中继');await previewTrainingPanel();
  }catch(error){panelStatus('面板读取失败：'+error.message);}finally{trainingPanelBusy=false;}
}
async function saveTrainingPanel(){
  if(!trainingPanelDocument)return;
  clearTimeout(trainingSaveTimer);rememberTrainingLaunch();
  const body=JSON.stringify(trainingPanelDocument);panelStatus('正在保存配置…');
  const saving=trainingSaveQueue.catch(()=>{}).then(()=>api('/api/training/panel',{method:'PUT',headers:{'Content-Type':'application/json'},body}));
  trainingSaveQueue=saving;
  try{await saving;if(body===JSON.stringify(trainingPanelDocument))panelStatus('已保存到中继 · 刷新页面可继续编辑');}
  catch(error){panelStatus('保存失败：'+error.message);throw error;}
}
function panelChanged(){
  rememberTrainingLaunch();panelStatus('尚未保存…');
  clearTimeout(trainingSaveTimer);trainingSaveTimer=setTimeout(()=>saveTrainingPanel().catch(()=>{}),600);
  clearTimeout(trainingPreviewTimer);trainingPreviewTimer=setTimeout(previewTrainingPanel,350);
}
function changeTrainingStage(){
  if(!trainingPanelDocument)return;rememberTrainingLaunch();trainingPanelStage=$('#trainingTrainer').value;
  restoreTrainingLaunch();renderPanelControls();panelChanged();
}
async function loadTrainingPanelCache(keepStage){
  if(!trainingSchema)await loadEditableTrainingPanel();if(!trainingSchema)return false;
  trainingPanelBusy=true;
  try{
    const stage=keepStage?trainingPanelStage:'';
    const cached=await api('/api/training/panel_cache'+(stage?'?trainer='+encodeURIComponent(stage):''));
    trainingPanelDocument.state=cached.panel_state;trainingPanelStage=cached.trainer;$('#trainingTrainer').value=cached.trainer;
    trainingPanelDocument.state.train_type=cached.trainer;
    trainingPanelDocument.state['data_path_'+cached.trainer]=cached.args.data_path||'';
    trainingPanelDocument.extras={};trainingPanelDocument.launch={};
    restoreTrainingLaunch();renderPanelControls();$('#trainingCacheSource').textContent='Config WebUI：'+cached.source;
    await saveTrainingPanel();await previewTrainingPanel();return true;
  }catch(error){panelStatus('导入失败：'+error.message);return false;}finally{trainingPanelBusy=false;}
}
function trainingPanelRequest(){
  if(!trainingPanelDocument)throw new Error('请等待面板配置载入');rememberTrainingLaunch();
  const args=JSON.parse($('#trainingArgs').value||'{}');
  if(!args||Array.isArray(args)||typeof args!=='object')throw new Error('额外参数必须是 JSON 对象');
  args.data_path=$('#trainingData').value.trim();
  const trainer=trainingPanelStage;
  if(trainer!=='pretrain'&&!$('#trainingExisting').value)args.from_weight=$('#trainingBase').value;
  const state={};
  // Runtime/UI history from a webui_state import is never sent as launch input.
  for(const group of trainingSchema.groups)for(const field of group.fields){
    const key=panelKey(field);if(trainingPanelDocument.state[key]!=null)state[key]=trainingPanelDocument.state[key];
  }
  if(trainingPanelDocument.state.inference_rope_scaling!=null)state.inference_rope_scaling=trainingPanelDocument.state.inference_rope_scaling;
  state.from_resume=Boolean($('#trainingExisting').value);
  return {name:$('#trainingName').value,trainer,args,panel_state:state,config_path:$('#trainingConfig').value,
    nproc:Number($('#trainingNproc').value),gpu_devices:$('#trainingDevices').value.trim(),existing_resume:$('#trainingExisting').value,auto_gpu:true,
    shutdown_after_train:true,shutdown_delay_seconds:60};
}
async function previewTrainingPanel(){
  const sequence=++trainingPreviewSequence;
  try{
    const result=await api('/api/training/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(trainingPanelRequest())});
    if(sequence!==trainingPreviewSequence)return;
    $('#trainingCommand').textContent=result.command;$('#trainingCommandNote').textContent=result.note;
    const model=result.request.model_config;
    trainingPanelCache=model?{model_config:model,preset:trainingPanelDocument.state.preset,trainer:trainingPanelStage}:null;
    if(model){renderTrainingCachedModel(trainingPanelCache);$('#trainingModelTitle').textContent='当前模型配置';}
    else{$('#trainingModelTitle').textContent='服务器配置文件';$('#trainingModelSummary').replaceChildren();$('#trainingModelJson').textContent=$('#trainingConfig').value;}
    updateTrainingModelNotice();
  }catch(error){if(sequence===trainingPreviewSequence){$('#trainingCommand').textContent='配置错误：'+error.message;$('#trainingCommandNote').textContent='';}}
}
async function startTraining(event){
  event.preventDefault();$('#trainingStart').disabled=true;
  try{
    if(trainingPanelBusy)throw new Error('配置正在载入，请稍后启动');
    const request=trainingPanelRequest();await saveTrainingPanel();
    $('#trainingMessage').textContent='正在检查训练配置、GPU 和磁盘并下发任务…';
    const job=await api('/api/training/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});
    trainingOperationId=job.operation_id;trainingCanStart=false;$('#trainingMessage').textContent=job.message;await loadTrainingOperations();
  }catch(error){$('#trainingMessage').textContent='启动失败：'+error.message;}
  finally{$('#trainingStart').disabled=!trainingCanStart;}return false;
}
function inferTrainingStage(){
  const file=$('#trainingExisting').value.split('/').pop();let stage=file.startsWith('full_dist_')?'distillation':trainingSchema?.stages.find(s=>file.startsWith(s+'_'));
  if(stage&&stage!==trainingPanelStage){
    const checkpoint=$('#trainingExisting').value;$('#trainingTrainer').value=stage;changeTrainingStage();$('#trainingExisting').value=checkpoint;
  }
  panelChanged();updateTrainingModelNotice();
}
$('#trainingPreset').addEventListener('change',()=>{
  const preset=trainingSchema.presets[$('#trainingPreset').value];if(!preset)return;
  Object.assign(trainingPanelDocument.state,structuredClone(preset));renderPanelControls();panelChanged();
});
$('#trainingArgs').addEventListener('change',()=>{
  try{const extra=JSON.parse($('#trainingArgs').value);if(!extra||Array.isArray(extra)||typeof extra!=='object')throw new Error('请填写 JSON 对象');
    trainingPanelDocument.extras[trainingPanelStage]=extra;renderPanelControls();panelChanged();
  }catch(error){panelStatus(error.message);}
});
for(const selector of Object.values(trainingLaunchFields))$(selector).addEventListener('change',panelChanged);
$('#trainingImport').addEventListener('change',async event=>{
  const file=event.target.files[0];if(!file)return;
  try{
    if(file.size>524288)throw new Error('配置文件超过 512 KiB');const value=JSON.parse(await file.text());
    const next=value.state?value:{state:value,extras:{},launch:{}};next.extras??={};next.launch??={};
    // Validate before replacing the working editor.
    await api('/api/training/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({trainer:next.state.train_type||'pretrain',panel_state:next.state,args:{},config_path:'__panel_cache__'})});
    trainingPanelDocument=next;trainingPanelStage=next.state.train_type||'pretrain';$('#trainingTrainer').value=trainingPanelStage;
    restoreTrainingLaunch();renderPanelControls();await saveTrainingPanel();await previewTrainingPanel();
  }catch(error){panelStatus('导入失败：'+error.message);}finally{event.target.value='';}
});
function exportTrainingPanel(){
  rememberTrainingLaunch();const url=URL.createObjectURL(new Blob([JSON.stringify(trainingPanelDocument,null,2)],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download='relay-training-config.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
