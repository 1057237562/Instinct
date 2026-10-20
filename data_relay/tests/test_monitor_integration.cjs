'use strict';
// Full local monitor + relay, real admin session, mocked remote inventory only.
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),net=require('node:net'),cp=require('node:child_process'),assert=require('node:assert/strict');
const {chromium}=require('playwright');
const monitor=path.resolve(process.argv[2]),repo=path.resolve(__dirname,'../..');
const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'relay-integration-'));
const port=()=>new Promise(resolve=>{const s=net.createServer();s.listen(0,'127.0.0.1',()=>{const p=s.address().port;s.close(()=>resolve(p));});});
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function ready(url){for(let i=0;i<100;i++){try{await fetch(url);return;}catch{await pause(100);}}throw new Error('service startup timed out');}
(async()=>{
 const relayPort=await port(),monitorPort=await port(),base=`http://127.0.0.1:${monitorPort}`;
 const config=path.join(temporary,'relay.toml');fs.writeFileSync(config,`[server]\nlisten="127.0.0.1:${relayPort}"\napi_token="private-test-token"\ndata_dir="${temporary.replaceAll('\\','/')}/runtime"\n`);
 const monitorConfig=path.join(temporary,'monitor.json');fs.writeFileSync(monitorConfig,JSON.stringify({autoConnect:false,streamlit:{enabled:false},server:{},features:{manualConnect:true}}));
 const proxy=path.join(temporary,'proxy.json');fs.writeFileSync(proxy,JSON.stringify({port:relayPort,token:'private-test-token'}));
 const relay=cp.spawn(path.join(repo,'data_relay/target/release/data_relay.exe'),['--config',config,'serve'],{cwd:temporary,windowsHide:true,stdio:'ignore'});
 const node=cp.spawn(process.execPath,['server.js'],{cwd:monitor,windowsHide:true,stdio:['ignore','pipe','pipe'],env:{...process.env,HOST:'127.0.0.1',PORT:String(monitorPort),SSHBROWSE_CONFIG:monitorConfig,SSHBROWSE_RELAY_CONFIG:proxy,SSHBROWSE_ADMIN_USER:'fixture-admin',SSHBROWSE_ADMIN_PASSWORD:'fixture-password-only'}});
 let log='';node.stderr.on('data',d=>{log+=d});
 let browser;
 try{
  await ready(`http://127.0.0.1:${relayPort}/`);await ready(base+'/api/health');
  assert.equal((await fetch(base+'/relay/api/config')).status,401);
  browser=await chromium.launch({channel:'msedge',headless:true});const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/relay/api/autodl/**',route=>route.fulfill({json:route.request().url().endsWith('/instances')?{selected:'fixture',instances:[{uuid:'fixture',selected:true,status:'shutdown'}]}:{uuid:'fixture',status:'shutdown'}}));
  await page.route('**/relay/api/training/environment',route=>route.fulfill({json:{reachable:false,instance:{uuid:'fixture',status:'shutdown'}}}));
  await page.route('**/relay/api/training/jobs',route=>route.fulfill({json:{jobs:[],reachable:false}}));
  await page.goto(base);await page.locator('#connect-relay').waitFor();await page.locator('#connect-relay').click();
  await page.locator('#login-form [name=username]').fill('fixture-admin');await page.locator('#login-form [name=password]').fill('fixture-password-only');await page.locator('#login-btn').click();
  const frame=page.frameLocator('#relay-frame');await frame.locator('#connState').filter({hasText:'已连接'}).waitFor();
  assert(await frame.locator('#token').isHidden());assert.equal(await frame.locator('body').textContent().then(t=>t.includes('private-test-token')),false);
  await frame.getByRole('button',{name:'服务器训练',exact:true}).click();await frame.locator('#tp_batch_size').waitFor();
  await frame.locator('#trainingCommand').filter({hasText:'train_pretrain.py'}).waitFor();
  await frame.locator('body').evaluate(()=>window.hashBlob(new Blob(['test-worker-fixture']))).catch(e=>{throw new Error('hash worker failed: '+e.message)});
  const response=await page.request.post(base+'/relay/api/training/preview',{data:{trainer:'pretrain',panel_state:{hidden_size:512,num_hidden_layers:32},args:{}}});assert.equal(response.status(),200);
  const rejected=await page.request.put(base+'/relay/api/config',{headers:{origin:'http://untrusted.invalid'},data:{}});assert.equal(rejected.status(),403);
  await page.screenshot({path:path.join(os.tmpdir(),'instinct-relay-online/integrated-monitor.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'mobile overflow');
  assert.deepEqual(errors,[]);console.log('PASS: offline entry, existing admin login, same-origin iframe, prefixed APIs/assets, hash worker, preview and CSRF boundary.');
 }catch(error){console.error(log);if(browser)for(const context of browser.contexts())for(const page of context.pages()){
  console.error('Frames:',page.frames().map(f=>f.url()));
  for(const f of page.frames())try{console.error((await f.locator('body').innerText()).slice(0,700));}catch{}
  await page.screenshot({path:path.join(os.tmpdir(),'instinct-relay-online/integration-failure.png'),fullPage:true});
 }throw error;}
 finally{if(browser)await browser.close();relay.kill();node.kill();await pause(300);}
})().catch(e=>{console.error(e);process.exitCode=1;});
