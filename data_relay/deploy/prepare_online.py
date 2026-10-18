"""Prepare a private deployment bundle. Dataset/weight/checkpoint files are excluded."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tomllib

repo=Path(__file__).resolve().parents[2]
destination=Path(sys.argv[1]).resolve(); destination.mkdir(parents=True,exist_ok=True)
stage=destination/'relay-release';stage.mkdir(exist_ok=True)
source_config=repo/'data_relay/config.toml'
raw=source_config.read_text(encoding='utf-8-sig'); config=tomllib.loads(raw)

def set_value(section,key,value):
    global raw
    match=re.search(r'^\['+re.escape(section)+r'\][^\n]*\n(.*?)(?=^\[|\Z)',raw,re.M|re.S)
    if not match:
        raw+='\n['+section+']\n';return set_value(section,key,value)
    block=match.group(1);line=key+' = '+json.dumps(value,ensure_ascii=False)
    pattern=r'^'+re.escape(key)+r'\s*=.*$'
    block=re.sub(pattern,lambda _:line,block,flags=re.M) if re.search(pattern,block,re.M) else block+line+'\n'
    raw=raw[:match.start(1)]+block+raw[match.end(1):]

set_value('server','listen','127.0.0.1:7800')
set_value('server','data_dir','/var/lib/instinct-relay')
set_value('runner','python','python3')
set_value('autodl','pull_base_url','http://81.68.133.63:8080/relay')
set_value('training','local_repo_root','/opt/instinct-relay/workspace')
set_value('training','panel_state_path','/opt/instinct-relay/workspace/trainer/webui_state.json')
(stage/'config.toml').write_text(raw,encoding='utf-8')
(stage/'proxy.json').write_text(json.dumps({'port':7800,'token':config['server']['api_token']}),encoding='utf-8')
binary=repo/'data_relay/target/x86_64-unknown-linux-gnu/release/data_relay'
shutil.copy2(binary,stage/'data_relay')
workspace=stage/'workspace'
count=0
for folder in ['trainer','model','scripts']:
    for directory, dirs, files in os.walk(repo/folder):
        relative=Path(directory).relative_to(repo)
        dirs[:]=[d for d in dirs if not d.startswith('.') and d!='__pycache__' and
                 (folder!='scripts' or relative.as_posix().startswith('scripts/data_loader') or d=='data_loader')]
        for name in files:
            src=Path(directory)/name;rel=src.relative_to(repo)
            allowed=name.endswith('.py') or (folder=='trainer' and name.startswith('config_') and name.endswith('.json')) or rel.as_posix() in {
                'trainer/autotune_pretrain.json','model/tokenizer.json','model/tokenizer_config.json',
                'model/special_tokens_map.json','model/vocab.json','model/merges.txt','scripts/monitor_training_shutdown.ps1'}
            if not allowed:continue
            if src.is_symlink() or src.stat().st_size>32*1024**2:raise ValueError('Unsafe code asset: '+str(rel))
            out=workspace/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,out);count+=1
for name in ['run_loop.py','run_linear.py','eval_llm.py','requirements.txt','requirements-fp8.txt','start_config_webui.bat']:
    src=repo/name
    if src.is_file():shutil.copy2(src,workspace/name);count+=1
def portable(value):
    if isinstance(value,dict):return {k:portable(v) for k,v in value.items()}
    if isinstance(value,list):return [portable(v) for v in value]
    if isinstance(value,str):
        normalized=value.replace('\\','/');prefix=repo.as_posix().rstrip('/')+'/'
        if normalized.lower().startswith(prefix.lower()):return normalized[len(prefix):]
    return value
panel=repo/'trainer/webui_state.json'
if panel.is_file():
    (workspace/'trainer/webui_state.json').write_text(json.dumps(portable(json.loads(panel.read_text(encoding='utf-8'))),ensure_ascii=False,indent=2),encoding='utf-8')
runtime=stage/'runtime';runtime.mkdir(exist_ok=True)
local_data=Path(config['server']['data_dir'])
if not local_data.is_absolute():local_data=repo/'data_relay'/local_data
for src in (local_data/'tasks').glob('*/task.json'):
    record=json.loads(src.read_text(encoding='utf-8'))
    if record.get('status') not in ['done','failed','cancelled','pending']:
        raise ValueError('A local task is active; do not migrate two workers')
    if record.get('status')=='pending':continue
    out=runtime/src.relative_to(local_data);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,out)
    if src.with_name('task.log').is_file():shutil.copy2(src.with_name('task.log'),out.with_name('task.log'))
for src in (local_data/'training').rglob('*.json'):
    record=json.loads(src.read_text(encoding='utf-8'))
    if 'operations' in src.parts and record.get('status') not in ['completed','failed','interrupted','cancelled']:
        raise ValueError('A local GPU queue is active; refusing duplicate migration')
    out=runtime/src.relative_to(local_data);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(portable(record),ensure_ascii=False,indent=2),encoding='utf-8')
for name in ['known_hosts.json']:
    src=local_data/name
    if src.is_file():shutil.copy2(src,runtime/name)
shutil.copy2(repo/'data_relay/deploy/instinct-relay.service',stage/'instinct-relay.service')
archive=destination/'relay-release.tgz'
with tarfile.open(archive,'w:gz') as tf:
    for src in stage.rglob('*'):
        if src.is_file():tf.add(src,arcname=src.relative_to(stage))
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'code_files':count,
                  'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'datasets_included':False}))
