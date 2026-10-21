// Local browser fixtures only; no remote file is modified.
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),path=require('node:path'),os=require('node:os');
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});
  try{
    const page=await browser.newPage({viewport:{width:1400,height:1000}}),errors=[],deletions=[];
    page.on('pageerror',e=>errors.push(e.message));
    const base=process.env.RELAY_TEST_URL||'http://127.0.0.1:17800';
    assert.match(base,/^http:\/\/(127\.0\.0\.1|localhost):/);
    const file=(name,size=4096)=>({name,path:name,size,modified_at:1792074000,kind:'file',version:'a'.repeat(64),trainable:name.endsWith('.parquet'),previewable:name.endsWith('.json')});
    let entries=[{name:'shards',path:'shards',kind:'directory',size:null,modified_at:1792074000},file('general_clean.parquet',11*1024**3),file('general_clean.parquet.report.json'),file('<img src=x onerror=alert(1)>.json')];
    await page.route('**/api/autodl/**',route=>route.fulfill({json:route.request().url().endsWith('/instances')?
      {selected:'test-a',instances:[{uuid:'test-a',name:'测试实例 A',status:'running',selected:true},{uuid:'test-b',name:'离线实例 B',status:'shutdown'}]}:{uuid:'test-a',status:'running'}}));
    await page.route('**/api/training/**',route=>{
      const url=route.request().url();
      if(['/api/training/panel','/api/training/preview'].includes(new URL(url).pathname))return route.continue();
      return route.fulfill({json:url.includes('panel_cache')?{trainer:'pretrain',args:{data_path:'old.parquet'},model_config:{hidden_size:512,num_hidden_layers:32},sha256:'fixture'}:
        {instance:{uuid:'test-a',status:'running'},reachable:true,jobs:[],datasets:[],reports:[],configs:[],weights:[],resume_checkpoints:[],operations:[]}});
    });
    await page.route('**/api/datasets**',async route=>{
      const request=route.request(),url=new URL(request.url()),instance=url.searchParams.get('instance_uuid');
      if(request.method()==='DELETE'){
        const data=request.postDataJSON();deletions.push(data);entries=entries.filter(e=>e.path!==data.path);
        return route.fulfill({json:{ok:true,deleted:data.path}});
      }
      if(url.pathname.endsWith('/preview'))return route.fulfill({json:{text:'{"identity_reviewed":true}',truncated:false}});
      const folder=url.searchParams.get('path');
      return route.fulfill({json:{root:'/root/autodl-tmp/dataset',instance_uuid:instance,path:folder,reachable:instance==='test-a',exists:true,
        entries:instance!=='test-a'?[]:folder?[{...file('part.parquet'),path:'shards/part.parquet'}]:entries,disk:{free:25*1024**3,total:50*1024**3}}});
    });
    await page.goto(base);await page.locator('#token').fill('local-ui-test-only');await page.getByRole('button',{name:'保存令牌',exact:true}).click();
    await page.locator('#connState').filter({hasText:'已连接'}).waitFor();
    await page.getByRole('button',{name:'远端数据集',exact:true}).click();
    await page.locator('#datasetRows').getByText('general_clean.parquet',{exact:true}).waitFor();
    assert.equal(await page.locator('#datasetRows img').count(),0);
    await page.locator('#datasetSearch').fill('report');assert.equal(await page.locator('#datasetRows tr').count(),1);
    await page.getByRole('button',{name:'预览',exact:true}).click();await page.locator('#datasetPreviewText').filter({hasText:'identity_reviewed'}).waitFor();
    await page.getByRole('button',{name:'关闭预览',exact:true}).click();await page.locator('#datasetSearch').fill('');
    await page.locator('#datasetRows').getByRole('button',{name:'▸ shards'}).click();
    await page.locator('#datasetRows').getByText('part.parquet',{exact:true}).waitFor();
    await page.locator('#datasetBreadcrumb').getByRole('button',{name:'dataset',exact:true}).click();
    const corpus=page.locator('#datasetRows tr').filter({hasText:'general_clean.parquet'}).filter({has:page.getByRole('button',{name:'用于训练'})});
    await corpus.getByRole('button',{name:'用于训练'}).click();
    await page.waitForFunction(()=>document.querySelector('#trainingData').value==='/root/autodl-tmp/dataset/general_clean.parquet');
    assert.equal(await page.locator('#trainingAudit').count(),0);
    assert.equal(await page.locator('#trainingReviewed').count(),0);
    await page.getByRole('button',{name:'远端数据集',exact:true}).click();
    await page.locator('#datasetRows').getByText('general_clean.parquet',{exact:true}).waitFor();
    await corpus.getByRole('button',{name:'删除',exact:true}).click();
    assert.equal(deletions.length,0);assert.match(await page.locator('#datasetDeletePath').textContent(),/general_clean.parquet$/);
    await page.getByRole('button',{name:'取消',exact:true}).click();assert.equal(deletions.length,0);
    await corpus.getByRole('button',{name:'删除',exact:true}).click();await page.getByRole('button',{name:'确认删除',exact:true}).click();
    await page.waitForFunction(()=>!document.querySelector('#datasetDeleteDialog').open);
    assert.equal(deletions.length,1);assert.equal(deletions[0].instance_uuid,'test-a');assert.equal(deletions[0].version,'a'.repeat(64));
    await page.locator('#datasetRows').getByText('general_clean.parquet.report.json',{exact:true}).waitFor();
    await page.screenshot({path:path.join(os.tmpdir(),'instinct-relay-ui-preview/datasets-desktop.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'mobile overflow');
    await page.screenshot({path:path.join(os.tmpdir(),'instinct-relay-ui-preview/datasets-mobile.png'),fullPage:true});
    await page.locator('#datasetInstance').selectOption('test-b');
    await page.locator('#datasetMessage').filter({hasText:'实例未运行'}).waitFor();
    assert.equal(await page.locator('#datasetRows button').count(),0);
    assert.deepEqual(errors,[]);console.log('PASS: navigation, filtering, preview, training handoff, confirmed deletion, instance switch, mobile layout, XSS escaping.');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
