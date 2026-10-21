'use strict';
// One backend selection for relay operations and the SSH file/log browser.
const http = require('node:http');
const fs = require('node:fs');
function createServerSelection({loadConfig,saveServerConfig,ssh,streamlitPanel,resolve,request:injectedRequest}) {
  resolve ||= ()=>JSON.parse(fs.readFileSync(process.env.SSHBROWSE_RELAY_CONFIG || '/etc/instinct-relay/proxy.json','utf8'));
  const request = injectedRequest || ((method,path,body)=>new Promise((accept,reject)=>{
    let config;try {config=resolve();if(!config.token)throw Error();}catch {reject(Error('数据与训练服务尚未就绪'));return;}
    const bytes=body===undefined?null:Buffer.from(JSON.stringify(body));
    const req=http.request({hostname:'127.0.0.1',port:config.port || 7800,path,method,
      headers:{authorization:config.token,...(bytes?{'content-type':'application/json','content-length':bytes.length}:{})}},res=>{
      let size=0;const chunks=[];
      res.on('data',chunk=>{size+=chunk.length;if(size>2*1024*1024)res.destroy();else chunks.push(chunk);});
      res.on('error',()=>reject(Error('数据与训练服务响应中断')));
      res.on('end',()=>{
        let value;try {value=JSON.parse(Buffer.concat(chunks));}catch {reject(Error('数据与训练服务响应无效'));return;}
        if(res.statusCode<200 || res.statusCode>=300){
          const expired=/AuthorizeFailed|登录超时|重新登录/.test(String(value.error || ''));
          reject(Error(expired?'AutoDL 登录已过期，请在“数据与训练”刷新登录后重试':'无法读取或保存实例，请检查 AutoDL 登录和实例状态'));return;
        }
        accept(value);
      });
    });
    req.setTimeout(65000,()=>req.destroy());
    req.on('error',()=>reject(Error('数据与训练服务连接失败')));
    req.end(bytes);
  }));
  let tail=Promise.resolve(),version=0,busy=false;
  function exclusive(fn) {
    const result=tail.then(async()=>{busy=true;version++;try{return await fn();}finally{busy=false;}});
    tail=result.catch(()=>{});return result;
  }
  function invalidate() {
    ssh.disconnectAll();
    // Restart drops the old tunnel immediately; do not wait for SSH readiness.
    streamlitPanel.restart().catch(()=>{});
  }
  async function patch(input) {
    return exclusive(async()=>{
      const patch={...input};
      if(!Object.hasOwn(patch,'instance_uuid'))return request('PUT','/api/config',patch);
      const uuid=String(patch.instance_uuid || '').trim();
      if(!uuid)throw Error('请选择明确的 AutoDL 实例');
      if(['SSH_HOST','SSH_PORT','SSH_USER','SSH_PASSWORD','SSHBROWSE_AUTO_CONNECT'].some(key=>process.env[key]))
        throw Error('SSH 目标被服务环境变量固定，请先移除对应覆盖配置');
      const info=await request('POST','/api/integration/ssh-target',{instance_uuid:uuid});
      if(info.instance_uuid!==uuid)throw Error('实例信息与选择不符');
      const previous=loadConfig(),old={...previous.server,autoConnect:previous.autoConnect};
      const target=info.target;
      if(target && (!target.host || !target.username || !Number.isInteger(target.port) || target.port<1 || target.port>65535 || !target.password))
        throw Error('所选实例缺少可用的 SSH 地址或凭据，请刷新 AutoDL 登录后重试');
      if(patch.ssh_password)throw Error('切换实例时请使用实例自身的 SSH 凭据');
      const next={...(target || {host:'',port:22,username:'root',password:''}),authType:'password',privateKey:'',passphrase:'',
        relayInstanceUuid:uuid,relayInstanceName:info.name || uuid,autoConnect:true};
      // Never carry a password override from the old AutoDL instance to the new one.
      if(uuid!==info.selected_instance_uuid)patch.ssh_password='';
      try {
        saveServerConfig(next);
        let result;
        try {result=await request('PUT','/api/config',patch);}
        catch(error) {
          // A lost HTTP response may follow a successful durable save. Re-read
          // before rolling back, so the two services cannot silently disagree.
          if(uuid!==info.selected_instance_uuid) {
            try {const saved=await request('GET','/api/config');if(saved.instance_uuid===uuid)result=saved;}catch {}
          }
          if(!result)throw error;
        }
        invalidate();
        return {...result,ssh_selection:{instance_uuid:uuid,pending:!target}};
      } catch(error) {
        try {saveServerConfig(old);}catch {invalidate();throw Error('切换失败且恢复连接配置失败，请检查服务器设置');}
        throw error;
      }
    });
  }
  return {
    patch,
    list:async()=>{await tail;const result=await request('GET','/api/autodl/instances');const s=loadConfig().server;
      return {...result,ssh_instance_uuid:s.relayInstanceUuid || '',ssh:{host:s.host,port:s.port,username:s.username}};},
    ready:()=>tail,
    generation:()=>version,
    guard(session,expected){if(busy || version!==expected){ssh.disconnect(session.id);const error=Error('服务器已切换，请重新连接');error.status=409;error.code='TARGET_CHANGED';throw error;}},
    manual:patch=>exclusive(async()=>{saveServerConfig({...patch,relayInstanceUuid:'',relayInstanceName:''});invalidate();}),
  };
}
module.exports={createServerSelection};
