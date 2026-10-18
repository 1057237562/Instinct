// Fixture-only UI test: no remote training file is changed.
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),path=require('node:path'),os=require('node:os');
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});
  try{
    const page=await browser.newPage({viewport:{width:1400,height:1100}}),errors=[],actions=[];
    page.on('pageerror',e=>errors.push(e.message));
    const base=process.env.RELAY_TEST_URL||'http://127.0.0.1:17800';assert.match(base,/^http:\/\/(127\.0\.0\.1|localhost):/);
    const plan={id:'cs_'+'a'.repeat(32),instance_uuid:'test-a',local_root:'D:/AI/Instinct',remote_root:'/root/autodl-tmp',
      created_at:1792074000,updated_at:1792074000,status:'ready',message:'比较完成',total_bytes:1200,changed_bytes:900,copied_bytes:0,
      backup_path:'/root/autodl-tmp/.relay_code_sync/cs_fixture/backup',files:[
        {path:'scripts/config_webui.py',status:'changed',size:500,sha256:'a'.repeat(64),remote_sha256:'b'.repeat(64)},
        {path:'trainer/new.py',status:'new',size:400,sha256:'c'.repeat(64),remote_sha256:null},
        {path:'model/model_instinct.py',status:'same',size:300,sha256:'d'.repeat(64),remote_sha256:'d'.repeat(64)}]};
    await page.route('**/api/autodl/**',route=>route.fulfill({json:route.request().url().endsWith('/instances')?{selected:'test-a',instances:[{uuid:'test-a',selected:true,status:'running'}]}:{uuid:'test-a',status:'running'}}));
    await page.route('**/api/training/**',route=>{if(['/api/training/panel','/api/training/preview'].includes(new URL(route.request().url()).pathname))return route.continue();return route.fulfill({json:route.request().url().includes('panel_cache')?
      {trainer:'pretrain',args:{data_path:'dataset/test.parquet'},model_config:{hidden_size:512,num_hidden_layers:32},sha256:'fixture'}:
      {reachable:true,instance:{uuid:'test-a',status:'running'},jobs:[],datasets:[],reports:[],configs:[],weights:[],resume_checkpoints:[],operations:[]}});});
    await page.route('**/api/code-sync**',route=>{
      const r=route.request(),url=new URL(r.url());
      if(r.method()==='POST'){
        if(url.pathname==='/api/code-sync'){assert.equal(r.postDataJSON().instance_uuid,'test-a');return route.fulfill({json:plan});}
        actions.push(url.pathname);plan.status=url.pathname.endsWith('/apply')?'syncing':'rolling_back';
        return route.fulfill({status:202,json:plan});
      }
      if(url.pathname==='/api/code-sync')return route.fulfill({json:{plans:[plan]}});
      if(plan.status==='syncing'){plan.status='completed';plan.message='同步完成';plan.copied_bytes=900;}
      if(plan.status==='rolling_back'){plan.status='rolled_back';plan.message='备份已恢复';}
      return route.fulfill({json:plan});
    });
    await page.goto(base);await page.locator('#token').fill('local-ui-test-only');await page.getByRole('button',{name:'保存令牌',exact:true}).click();
    await page.locator('#connState').filter({hasText:'已连接'}).waitFor();
    await page.getByRole('button',{name:'服务器训练',exact:true}).click();
    await page.locator('#tp_batch_size').waitFor();
    await page.getByRole('button',{name:'比较本地与远端代码',exact:true}).click();
    await page.locator('#codeSyncMessage').filter({hasText:'比较完成'}).waitFor();
    assert.equal(await page.locator('#codeSyncFiles tr').count(),2);
    await page.locator('#codeSyncShowSame').check();assert.equal(await page.locator('#codeSyncFiles tr').count(),3);
    await page.getByRole('button',{name:'同步变更文件',exact:true}).click();
    assert.equal(actions.length,0);assert.match(await page.locator('#codeSyncReview').textContent(),/test-a/);
    await page.locator('#codeSyncDialog').getByRole('button',{name:'取消',exact:true}).click();assert.equal(actions.length,0);
    await page.getByRole('button',{name:'同步变更文件',exact:true}).click();await page.getByRole('button',{name:'确认执行',exact:true}).click();
    await page.locator('#codeSyncMessage').filter({hasText:'同步完成'}).waitFor({timeout:10000});assert.equal(actions.length,1);
    await page.getByText('同步记录与回滚',{exact:true}).click();await page.locator('#codeSyncHistory').getByRole('button',{name:'回滚',exact:true}).click();
    await page.getByRole('button',{name:'确认执行',exact:true}).click();
    await page.locator('#codeSyncMessage').filter({hasText:'已回滚'}).waitFor({timeout:10000});assert.equal(actions.length,2);
    await page.locator('#codeSyncTitle').scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(os.tmpdir(),'instinct-relay-ui-preview/code-sync.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'mobile overflow');
    assert.deepEqual(errors,[]);console.log('PASS: scope comparison, explicit confirmation, background progress, rollback history and mobile layout.');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
