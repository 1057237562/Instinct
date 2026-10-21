// Mock task progress to check the actual 3-second polling UI without cloud writes.
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),path=require('node:path'),os=require('node:os');
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});
  try{
    const page=await browser.newPage({viewport:{width:1400,height:1100}}),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    const base=process.env.RELAY_TEST_URL||'http://127.0.0.1:17800';
    assert.match(base,/^http:\/\/(127\.0\.0\.1|localhost):/);
    let sample=0,phase='transferring',offline=false;
    await page.route('**/api/autodl/**',route=>route.fulfill({json:route.request().url().endsWith('/instances')?{selected:'test',instances:[{uuid:'test',status:'running',selected:true}]}:{uuid:'test',status:'running'}}));
    await page.route('**/api/tasks',route=>{
      if(offline)return route.fulfill({status:503,json:{error:'simulated outage'}});
      sample++;
      const task={id:'transfer-test',name:'续传速率测试',status:'uploading',created_at:1792074000,upload_mode:'sftp',upload_phase:phase,upload_attempt:1,
        upload_total_bytes:1024**3,upload_done_bytes:800*1024**2+sample*6*1024**2,upload_transferred_bytes:sample*6*1024**2,
        data_upload:{instance_uuid:'test',files:[{name:'clean.parquet',size:1024**3}]}};
      return route.fulfill({json:{tasks:[task,{...task,id:'script-transfer',name:'脚本交付',data_upload:undefined}]}});
    });
    await page.goto(base);await page.locator('#token').fill('local-ui-test-only');await page.getByRole('button',{name:'保存令牌',exact:true}).click();
    const dataRate=page.locator('[data-transfer-rate="transfer-test"]');
    await dataRate.filter({hasText:'采样中'}).waitFor();
    await dataRate.filter({hasText:/MB\/s.*预计剩余 \d/}).waitFor({timeout:12000});
    assert(!/800.*MB\/s/.test(await dataRate.textContent()));
    await page.getByRole('button',{name:'脚本任务',exact:true}).click();
    await page.locator('[data-transfer-rate="script-transfer"]').filter({hasText:'MB/s'}).waitFor();
    await page.getByRole('button',{name:'数据上传',exact:true}).click();
    await page.screenshot({path:path.join(os.tmpdir(),'instinct-relay-ui-preview/transfer-speed.png'),fullPage:true});
    phase='verifying';await dataRate.filter({hasText:'正在校验'}).waitFor({timeout:10000});
    assert(!(await dataRate.textContent()).includes('/s'));
    phase='preparing';await dataRate.filter({hasText:'正在连接'}).waitFor({timeout:10000});
    offline=true;await dataRate.filter({hasText:'等待重新连接'}).waitFor({timeout:10000});
    offline=false;phase='transferring';await dataRate.filter({hasText:'采样中'}).waitFor({timeout:10000});
    assert.deepEqual(errors,[]);
    console.log('PASS: data/script transfer rates, ETA, preparation/verification states and stale network recovery.');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
