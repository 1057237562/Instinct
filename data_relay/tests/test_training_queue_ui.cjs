// Mock remote capacity/startup state; never boots an AutoDL instance.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});
  try{
    const page=await browser.newPage(),errors=[],cancellations=[];
    page.on('pageerror',e=>errors.push(e.message));
    const base=process.env.RELAY_TEST_URL||'http://127.0.0.1:17800';assert.match(base,/^http:\/\/(127\.0\.0\.1|localhost):/);
    const operation={id:'op_'+'a'.repeat(32),instance_uuid:'pinned-fixture',action:'start',status:'waiting_gpu',
      message:'GPU 暂无空闲资源，已进入训练队列；每 60 秒重试',next_retry_at:Math.floor(Date.now()/1000)+60,gpu_attempts:3};
    await page.route('**/api/autodl/**',route=>route.fulfill({json:route.request().url().endsWith('/instances')?
      {selected:'fixture',instances:[{uuid:'fixture',selected:true,status:'shutdown'}]}:{uuid:'fixture',status:'shutdown'}}));
    await page.route('**/api/training/**',route=>{
      const r=route.request(),url=new URL(r.url());
      if(['/api/training/panel','/api/training/preview'].includes(url.pathname))return route.continue();
      if(url.pathname.endsWith('/cancel')){
        cancellations.push(url.pathname);operation.status='cancelled';operation.message='训练排队已取消';
        return route.fulfill({json:{ok:true,status:'cancelled',message:operation.message}});
      }
      if(url.pathname.endsWith('/operations'))return route.fulfill({json:{operations:[operation]}});
      return route.fulfill({json:{reachable:false,instance:{uuid:'fixture',status:'shutdown'},jobs:[]}});
    });
    await page.goto(base);await page.locator('#token').fill('local-ui-test-only');await page.getByRole('button',{name:'保存令牌',exact:true}).click();
    await page.locator('#connState').filter({hasText:'已连接'}).waitFor();await page.getByRole('button',{name:'服务器训练',exact:true}).click();
    await page.locator('#trainingQueue').filter({hasText:'pinned-fixture'}).waitFor();
    assert.match(await page.locator('#trainingQueue').textContent(),/资源不足 3 次/);assert(await page.locator('#trainingStart').isDisabled());
    await page.reload();await page.getByRole('button',{name:'服务器训练',exact:true}).click();
    await page.getByRole('button',{name:'取消排队',exact:true}).waitFor();
    await page.getByRole('button',{name:'取消排队',exact:true}).click();await page.locator('#trainingQueue').filter({hasText:'已取消'}).waitFor();
    assert.equal(cancellations.length,1);assert(await page.locator('#trainingStart').isEnabled());
    assert.deepEqual(errors,[]);console.log('PASS: pending queue survives page reload, shows pinned instance/retry time, cancels and unlocks start.');
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
