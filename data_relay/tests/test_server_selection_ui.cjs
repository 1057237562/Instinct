'use strict';
// Real monitor/auth/encrypted storage and managed relay UI, fake AutoDL backend only.
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),http=require('node:http'),net=require('node:net'),cp=require('node:child_process'),assert=require('node:assert/strict');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'..'),monitor=path.resolve(process.argv[2]);
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'server-selection-'));
const listen=s=>new Promise(r=>s.listen(0,'127.0.0.1',()=>r(s.address().port)));
const pause=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
 let selected='old',privateRequests=0,expired=false;const secret='backend-fixture-password';
 const relay=http.createServer(async(req,res)=>{
  const url=new URL(req.url,'http://localhost'),name=url.pathname;
  if(name==='/' || /^\/[a-z_]+\.js$/.test(name)){
   const file=path.join(root,'src',name==='/'?'index.html':name.slice(1));
   res.setHeader('content-type',name==='/'?'text/html; charset=utf-8':'text/javascript');res.end(fs.readFileSync(file));return;
  }
  assert.equal(req.headers.authorization,'backend-fixture-token');
  let body='';for await(const chunk of req)body+=chunk;const input=body?JSON.parse(body):{};
  let value={};
  if(name==='/api/integration/ssh-target'){privateRequests++;value={instance_uuid:input.instance_uuid,selected_instance_uuid:selected,name:input.instance_uuid,target:{host:`${input.instance_uuid}.invalid`,port:2222,username:'root',password:secret}};}
  else if(name==='/api/config'){
   if(req.method==='PUT')selected=input.instance_uuid || selected;
   value={instance_uuid:selected,data_chunk_size:16777216,training:{},disk:{}};
  }else if(name==='/api/autodl/instances'){
   if(expired){res.statusCode=400;value={error:'AutoDL API AuthorizeFailed: 登录超时，请重新登录'};}
   else value={selected,instances:['old','new'].map(uuid=>({uuid,name:uuid,status:uuid==='old'?'shutdown':'running',selected:uuid===selected}))};
  }
  else if(name==='/api/autodl/status')value={uuid:selected,status:'running'};
  else if(name==='/api/tasks')value={tasks:[]};
  else if(name==='/api/data/uploads')value={uploads:[]};
  res.setHeader('content-type','application/json');res.end(JSON.stringify(value));
 });
 const relayPort=await listen(relay),probe=net.createServer(),monitorPort=await listen(probe);await new Promise(r=>probe.close(r));
 const base=`http://127.0.0.1:${monitorPort}`;
 fs.writeFileSync(path.join(temp,'config.json'),JSON.stringify({autoConnect:true,streamlit:{enabled:false},server:{host:'old.invalid',port:22,username:'root'},features:{manualConnect:false}}));
 fs.writeFileSync(path.join(temp,'proxy.json'),JSON.stringify({port:relayPort,token:'backend-fixture-token'}));
 const child=cp.spawn(process.execPath,['server.js'],{cwd:monitor,windowsHide:true,stdio:['ignore','pipe','pipe'],env:{...process.env,HOST:'127.0.0.1',PORT:String(monitorPort),SSHBROWSE_CONFIG:path.join(temp,'config.json'),SSHBROWSE_RELAY_CONFIG:path.join(temp,'proxy.json'),SSHBROWSE_ADMIN_USER:'fixture',SSHBROWSE_ADMIN_PASSWORD:'fixture-password-only'}});
 let logs='';child.stderr.on('data',d=>logs+=d);let browser;
 try{
  for(let i=0;i<100;i++){try{await fetch(base+'/api/health');break;}catch{await pause(100);}}
  assert.equal((await fetch(base+'/api/relay/servers')).status,401);
  browser=await chromium.launch({channel:'msedge',headless:true});const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  // Old target remains pending; switching must not wait for it.
  let releaseOld;const oldWait=new Promise(r=>releaseOld=r);let first=true;
  await page.route('**/api/auto-connect**',async route=>{
   if(first){first=false;await oldWait;await route.fulfill({status:504,json:{error:'old target offline',code:'SSH_CONNECTION_LOST'}});return;}
   await route.fulfill({json:{session:{id:'fixture-session',host:`${selected}.invalid`,port:2222,username:'root',home:'/root'},defaultView:'files'}});
  });
  await page.route('**/api/list?**',route=>route.fulfill({json:{path:'/root',entries:[]}}));
  await page.goto(base);await page.locator('#boot-view [data-server-picker]').click();
  await page.locator('#login-form [name=username]').fill('fixture');await page.locator('#login-form [name=password]').fill('fixture-password-only');await page.locator('#login-btn').click();
  await page.locator('#shared-server-choice option[value=new]').waitFor({state:'attached'});
  await page.locator('#shared-server-choice').selectOption('new');await page.locator('#server-picker-apply').click();
  await page.locator('#conn-label').filter({hasText:'new.invalid'}).waitFor();
  releaseOld();await pause(100);assert(await page.locator('#boot-view').isHidden(),'late old failure must not hide new view');
  assert.equal(selected,'new');assert.equal(privateRequests,1);
  const pub=await (await page.request.get(base+'/api/public-config')).json();assert.equal(pub.server.relayInstanceUuid,'new');assert.equal(pub.server.host,'new.invalid');assert(!JSON.stringify(pub).includes(secret));
  for(const file of ['config.json','credentials.enc.json'])assert(!fs.readFileSync(path.join(temp,file),'utf8').includes(secret),'password must only be encrypted on disk');
  assert.equal((await page.request.post(base+'/relay/api/integration/ssh-target',{data:{instance_uuid:'old'}})).status(),404);
  assert.equal((await page.request.put(base+'/relay/api/config',{headers:{origin:'http://untrusted.invalid'},data:{instance_uuid:'old'}})).status(),403);assert.equal(selected,'new');
  await page.locator('[data-view=relay]').click();const frame=page.frameLocator('#relay-frame');
  await frame.locator('#instSelect option[value=old]').waitFor({state:'attached'});assert.equal(await frame.locator('#instSelect').inputValue(),'new');
  await frame.locator('#instSelect').selectOption('old');
  await page.locator('#conn-label').filter({hasText:'old.invalid'}).waitFor();
  assert.equal((await (await page.request.get(base+'/api/public-config')).json()).server.relayInstanceUuid,'old');
  assert(await page.locator('#relay-view').isVisible(),'switch from relay must preserve relay tab');
  await page.locator('.topbar-actions [data-server-picker]').click();await page.locator('#shared-server-choice').selectOption('new');await page.locator('#server-picker-apply').click();
  await page.locator('#conn-label').filter({hasText:'new.invalid'}).waitFor();
  await page.waitForFunction(()=>document.querySelector('#relay-frame').contentDocument.querySelector('#instSelect').value==='new');
  await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'mobile overflow');
  await page.locator('.topbar-actions [data-server-picker]').click();
  await page.locator('#server-picker-status').filter({hasText:'当前 SSH'}).waitFor();
  await page.screenshot({path:path.join(temp,'selection-mobile.png'),fullPage:true});
  expired=true;await page.locator('#server-picker-refresh').click();
  await page.locator('#server-picker-status').filter({hasText:'AutoDL 登录已过期'}).waitFor();
  assert(await page.locator('#server-picker-apply').isDisabled());assert.equal(selected,'new');
  assert.deepEqual(errors,[]);console.log('PASS: offline switch during old connection, existing login, encrypted backend credentials, both selectors synchronized, late-response rejection, CSRF and mobile layout.');
 }catch(error){console.error(logs);if(browser)for(const context of browser.contexts())for(const page of context.pages()){await page.screenshot({path:path.join(temp,'failure.png'),fullPage:true});console.error((await page.locator('body').innerText()).slice(0,1200));}throw error;}
 finally{if(browser)await browser.close();child.kill();relay.closeAllConnections();relay.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
