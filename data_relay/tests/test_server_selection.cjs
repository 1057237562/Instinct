'use strict';
const assert=require('node:assert/strict');
const {createServerSelection}=require('../deploy/monitor-server-selection.cjs');
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};
(async()=>{
 let cfg={autoConnect:true,server:{host:'old.invalid',port:22,username:'root',password:'old-secret',privateKey:'old-key',passphrase:'old-pass',relayInstanceUuid:'old'}};
 let selected='old',disconnected=0,restarted=0,fail=false,failAck=false;const calls=[];
 const wait=deferred();let hold=false;
 const manager=createServerSelection({loadConfig:()=>structuredClone(cfg),saveServerConfig:patch=>{const {autoConnect,...server}=patch;cfg={autoConnect,server};},
  ssh:{disconnectAll(){disconnected++;},disconnect(id){calls.push(['disconnect',id]);}},streamlitPanel:{async restart(){restarted++;}},
  request:async(method,path,body)=>{
   calls.push([method,path,body]);
   if(path==='/api/integration/ssh-target'){
    if(body.instance_uuid==='invalid')throw Error('fixture unavailable');
    if(hold){hold=false;await wait.promise;}
    return {instance_uuid:body.instance_uuid,selected_instance_uuid:selected,name:body.instance_uuid,target:body.instance_uuid==='off'?null:
      {host:body.instance_uuid+'.invalid',port:2022,username:'root',password:'new-secret'}};
   }
   if(method==='PUT'){if(fail)throw Error('fixture persist failed');selected=body.instance_uuid;if(failAck)throw Error('lost response');return {instance_uuid:selected};}
   if(path==='/api/config')return {instance_uuid:selected};
   return {selected,instances:[]};
  }});
 const epoch=manager.generation();
 await manager.patch({instance_uuid:'new'});
 assert.equal(cfg.server.host,'new.invalid');assert.equal(cfg.server.password,'new-secret');assert.equal(cfg.server.privateKey,'');assert.equal(cfg.server.passphrase,'');
 assert.equal(selected,'new');assert.equal(disconnected,1);assert.equal(restarted,1);
 assert.equal(calls.find(c=>c[0]==='PUT')[2].ssh_password,'');
 assert.throws(()=>manager.guard({id:'stale'},epoch),/服务器已切换/);
 assert.equal(JSON.stringify(await manager.list()).includes('secret'),false);
 const before=structuredClone(cfg);fail=true;
 await assert.rejects(manager.patch({instance_uuid:'bad'}),/persist failed/);assert.deepEqual(cfg,before);assert.equal(selected,'new');fail=false;
 await assert.rejects(manager.patch({instance_uuid:'invalid'}),/unavailable/);assert.deepEqual(cfg,before);
 failAck=true;await manager.patch({instance_uuid:'ack-lost'});assert.equal(cfg.server.relayInstanceUuid,selected);failAck=false;
 // Selection never probes the old SSH target; off instances clear it rather than reusing credentials.
 await manager.patch({instance_uuid:'off'});assert.equal(cfg.server.host,'');assert.equal(cfg.server.password,'');assert.equal(selected,'off');
 hold=true;const a=manager.patch({instance_uuid:'a'}),b=manager.patch({instance_uuid:'b'});wait.resolve();await Promise.all([a,b]);
 assert.equal(selected,'b');assert.equal(cfg.server.host,'b.invalid');
 await manager.manual({host:'custom.invalid',port:22,username:'me',password:'custom'});assert.equal(cfg.server.relayInstanceUuid,'');
 console.log('PASS: shared selection, offline old target, credential replacement/redaction, rollback, queued switches and stale connection rejection.');
})().catch(e=>{console.error(e);process.exitCode=1;});
