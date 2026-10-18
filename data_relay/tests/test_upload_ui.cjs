// Run against an isolated relay with token local-ui-test-only; never a production relay.
// NODE_PATH must include playwright. AutoDL and completion are mocked: no cloud writes.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const os=require('node:os');

(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});
  try {
    const page=await browser.newPage({viewport:{width:1400,height:1100}}),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    const base=process.env.RELAY_TEST_URL||'http://127.0.0.1:17800';
    assert.match(base,/^http:\/\/(127\.0\.0\.1|localhost):/);
    await page.route('**/api/autodl/**',route=>{
      const endpoint=new URL(route.request().url()).pathname;
      const instance={uuid:'test-instance',name:'测试实例（不连接云端）',status:'running',start_mode:'non_gpu',data_disk_used:40*1073741824,data_disk_total:200*1073741824};
      return route.fulfill({json:endpoint.endsWith('/instances')?{selected:instance.uuid,instances:[{...instance,selected:true}]}:instance});
    });
    await page.route('**/api/data/uploads/*/complete',route=>route.fulfill({status:202,json:{ok:true}}));
    await page.route('**/api/data/uploads/*/files/*/chunks/*',async route=>{
      // Leave enough time to pause a real local upload between acknowledged chunks.
      await new Promise(r=>setTimeout(r,route.request().url().endsWith('/chunks/0')?500:2000));
      await route.continue();
    });
    await page.goto(base);
    await page.locator('#token').fill('local-ui-test-only');
    await page.getByRole('button',{name:'保存令牌',exact:true}).click();
    await page.locator('#connState').filter({hasText:'已连接'}).waitFor();
    const corpus={name:'ui-test-'+Date.now()+'.parquet',mimeType:'application/octet-stream',buffer:Buffer.alloc(33*1024*1024+3,37)};
    const audit={name:corpus.name+'.report.json',mimeType:'application/json',buffer:Buffer.from('{"test_fixture":true}')};
    await page.locator('#dataFiles').setInputFiles([corpus,audit]);
    await page.locator('#dataName').fill('分块续传浏览器测试');
    const firstChunk=page.waitForResponse(r=>r.url().includes('/chunks/')&&r.status()===200);
    await page.locator('#dataStart').click();
    await firstChunk;
    await page.locator('#dataPause').click();
    await page.waitForFunction(()=>!document.querySelector('#dataStart').disabled);
    const tasks=await (await page.request.get(base+'/api/tasks',{headers:{Authorization:'local-ui-test-only'}})).json();
    const task=tasks.tasks.find(t=>t.data_upload?.files.some(f=>f.name===corpus.name));
    assert(task&&task.received_bytes>0&&task.received_bytes<task.upload_total_bytes);
    await page.reload();
    const record=page.locator('.transfer-card').filter({hasText:'分块续传浏览器测试'}).first();
    const chooserEvent=page.waitForEvent('filechooser');
    await record.getByRole('button',{name:'选择原文件继续上传'}).click();
    const chooser=await chooserEvent;await chooser.setFiles([corpus,audit]);
    await page.locator('#dataStart').click();
    await page.locator('#dataMessage').filter({hasText:'现在可以关闭页面'}).waitFor({timeout:60000});
    const saved=await (await page.request.get(base+'/api/tasks/'+task.id,{headers:{Authorization:'local-ui-test-only'}})).json();
    assert(saved.data_upload.files.every(f=>f.chunks.every(Boolean)));
    await page.getByRole('button',{name:'连接设置',exact:true}).click();
    await page.locator('#pullBaseUrl').fill('http://relay-test.example:7800');
    await page.getByRole('button',{name:'保存地址',exact:true}).click();
    await page.locator('#pullMessage').filter({hasText:'已保存'}).waitFor();
    await page.getByRole('button',{name:'脚本任务',exact:true}).click();
    assert(await page.locator('#taskForm').isVisible());
    await page.getByRole('button',{name:'数据上传',exact:true}).click();
    const output=path.join(os.tmpdir(),'instinct-relay-ui-preview');fs.mkdirSync(output,{recursive:true});
    await page.screenshot({path:path.join(output,'desktop-connected.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Mobile horizontal overflow');
    await page.screenshot({path:path.join(output,'mobile.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    console.log('PASS: real chunk upload, pause, reload, resume, settings, navigation, desktop/mobile layout; no AutoDL writes.');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
