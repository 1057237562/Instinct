// Real local panel API + mocked remote lifecycle: never starts AutoDL.
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),path=require('node:path'),os=require('node:os');
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],starts=[];
    page.on('pageerror',e=>errors.push(e.message));
    const base=process.env.RELAY_TEST_URL||'http://127.0.0.1:17800';assert.match(base,/^http:\/\/(127\.0\.0\.1|localhost):/);
    const fixtureHeaders={Authorization:'local-ui-test-only'};
    const schema=(await (await page.request.get(base+'/api/training/panel',{headers:fixtureHeaders})).json()).schema;
    const reset=await page.request.put(base+'/api/training/panel',{headers:fixtureHeaders,data:{state:{...schema.presets['Instinct V1 MoE'],train_type:'pretrain'},extras:{},launch:{}}});
    assert.equal(reset.status(),200);
    await page.route('**/api/autodl/**',route=>route.fulfill({json:route.request().url().endsWith('/instances')?
      {selected:'fixture',instances:[{uuid:'fixture',selected:true,status:'shutdown'}]}:{uuid:'fixture',status:'shutdown'}}));
    await page.route('**/api/training/**',route=>{
      const r=route.request(),url=new URL(r.url());
      if(['/api/training/panel','/api/training/preview','/api/training/panel_cache'].includes(url.pathname))return route.continue();
      if(r.method()==='POST'&&url.pathname==='/api/training/jobs'){
        starts.push(r.postDataJSON());return route.fulfill({status:202,json:{operation_id:'fixture',message:'模拟下发成功'}});
      }
      return route.fulfill({json:{instance:{uuid:'fixture',status:'shutdown'},reachable:false,jobs:[],operations:[]}});
    });
    await page.goto(base);await page.locator('#token').fill('local-ui-test-only');await page.getByRole('button',{name:'保存令牌',exact:true}).click();
    await page.locator('#connState').filter({hasText:'已连接'}).waitFor();await page.getByRole('button',{name:'服务器训练',exact:true}).click();
    await page.locator('#tp_batch_size').waitFor();
    await page.locator('#trainingTrainer').selectOption('pretrain');await page.locator('#trainingPreset').selectOption('Instinct V1 MoE');
    await page.locator('#trainingData').fill('dataset/example.parquet');await page.locator('#trainingData').press('Tab');
    await page.locator('#tp_batch_size').fill('3');await page.locator('#tp_batch_size').press('Tab');
    await page.locator('#tp_learning_rate_pretrain').fill('0.0004');await page.locator('#tp_learning_rate_pretrain').press('Tab');
    await page.locator('#tp_use_compile').selectOption('false');
    await page.locator('#trainingCommand').filter({hasText:"'--use_compile' '0'"}).waitFor();
    assert.match(await page.locator('#trainingCommand').textContent(),/train_pretrain.py/);
    await page.locator('#trainingTrainer').selectOption('cpt');
    assert.equal(await page.locator('#tp_learning_rate_cpt').inputValue(),'0.00003');
    await page.locator('#tp_learning_rate_cpt').fill('0.00002');await page.locator('#tp_learning_rate_cpt').press('Tab');
    await page.locator('#trainingTrainer').selectOption('grpo');
    await page.getByText('阶段专用参数与日志 / 保存间隔（空值沿用训练脚本默认）',{exact:true}).click();
    await page.locator('#tp_extra_loss_type').selectOption('grpo');await page.locator('#tp_extra_num_generations').fill('8');await page.locator('#tp_extra_num_generations').press('Tab');
    await page.locator('#trainingCommand').filter({hasText:"'--loss_type' 'grpo'"}).waitFor();
    await page.locator('#trainingTrainer').selectOption('pretrain');
    assert.equal(await page.locator('#tp_learning_rate_pretrain').inputValue(),'0.0004');assert.equal(await page.locator('#tp_batch_size').inputValue(),'3');
    assert.equal(await page.locator('#trainingData').inputValue(),'dataset/example.parquet');
    await page.getByRole('button',{name:'保存配置',exact:true}).click();await page.locator('#trainingPanelStatus').filter({hasText:'已保存'}).waitFor();
    await page.reload();await page.getByRole('button',{name:'服务器训练',exact:true}).click();await page.locator('#tp_batch_size').waitFor();
    assert.equal(await page.locator('#tp_batch_size').inputValue(),'3');assert.equal(await page.locator('#tp_use_compile').inputValue(),'false');
    assert.equal(await page.locator('#trainingReviewed').count(),0);assert.equal(await page.locator('#trainingAudit').count(),0);
    await page.locator('#trainingStart').click();await page.locator('#trainingMessage').filter({hasText:'模拟下发成功'}).waitFor();
    assert.equal(starts.length,1);assert.equal(starts[0].panel_state.batch_size,3);assert.equal(starts[0].panel_state.use_compile,false);
    assert.equal(starts[0].args.data_path,'dataset/example.parquet');assert.equal(starts[0].identity_reviewed,undefined);assert.equal(starts[0].audit_path,undefined);
    assert.equal(starts[0].shutdown_after_train,true);assert.equal(starts[0].shutdown_delay_seconds,60);
    await page.locator('#tp_batch_size').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(os.tmpdir(),'instinct-relay-ui-preview/training-panel.png')});
    await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'mobile overflow');
    assert.deepEqual(errors,[]);console.log('PASS: editable presets/parameters, stage isolation, CLI preview, persistence and dispatch without review gate.');
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
