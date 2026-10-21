'use strict';
const assert=require('node:assert/strict'),http=require('node:http'),crypto=require('node:crypto');
const {createRequire}=require('node:module'),path=require('node:path');
const requireMonitor=createRequire(path.join(path.resolve(process.argv[2]),'package.json'));
const express=requireMonitor('express');
const {createRelayProxy,isPull}=require('../deploy/monitor-relay-proxy.cjs');
const listen=server=>new Promise(resolve=>server.listen(0,'127.0.0.1',()=>resolve(server.address().port)));
(async()=>{
 const secret='proxy-fixture-secret';let chunkCount=0;
 const upstream=http.createServer((req,res)=>{
  const p=new URL(req.url,'http://localhost').pathname;
  if(p.startsWith('/pull/')){
   if(!req.url.includes('t=fixture')){res.writeHead(403);res.end();return;}
   assert.equal(req.headers.authorization,undefined);
   assert.equal(req.headers.range,'bytes=2-5');res.writeHead(206,{'content-range':'bytes 2-5/8'});res.end('2345');return;
  }
  if(req.headers.authorization!==secret){res.writeHead(401);res.end();return;}
  assert.equal(req.headers.cookie,undefined);
  const hash=crypto.createHash('sha256');req.on('data',data=>{chunkCount++;hash.update(data);});
  req.on('end',()=>{res.setHeader('content-type','application/json');res.end(JSON.stringify({path:req.url,sha256:hash.digest('hex')}));});
 });
 const upstreamPort=await listen(upstream),app=express();
 app.use('/relay',(req,res,next)=>{if(isPull(req)||req.headers.cookie==='fixture-admin')next();else res.status(401).end();},createRelayProxy({resolve:()=>({port:upstreamPort,token:secret}),onConfig:async patch=>({instance_uuid:patch.instance_uuid,shared:true})}));
 app.use(express.json({limit:'1kb'}));const server=http.createServer(app);const port=await listen(server);
 const base=`http://127.0.0.1:${port}`;
 try{
  assert.equal((await fetch(base+'/relay/api/config')).status,401);
  for(const name of ['integration','%69ntegration'])assert.equal((await fetch(base+`/relay/api/${name}/ssh-target`,{method:'POST',headers:{cookie:'fixture-admin'}})).status,404);
  const config=await fetch(base+'/relay/api/config',{method:'PUT',headers:{cookie:'fixture-admin','content-type':'application/json'},body:JSON.stringify({instance_uuid:'fixture'})});
  assert.deepEqual(await config.json(),{instance_uuid:'fixture',shared:true});
  const bytes=Buffer.alloc(20*1024*1024,17);
  const response=await fetch(base+'/relay/api/chunks/0',{method:'PUT',headers:{cookie:'fixture-admin',authorization:'attacker-token','content-type':'application/octet-stream'},body:bytes});
  assert.equal(response.status,200);const data=await response.json();assert.equal(data.path,'/api/chunks/0');
  assert.equal(data.sha256,crypto.createHash('sha256').update(bytes).digest('hex'));assert(chunkCount>1,'upload must stream');
  const pull=await fetch(base+'/relay/pull/task/data?t=fixture',{headers:{range:'bytes=2-5',authorization:secret}});
  assert.equal(pull.status,206);assert.equal(pull.headers.get('content-range'),'bytes 2-5/8');assert.equal(await pull.text(),'2345');
  assert.equal((await fetch(base+'/relay/pull/task/data?t=wrong')).status,403);
  const bad=await new Promise((resolve,reject)=>{const r=http.get({host:'127.0.0.1',port,path:'/relay/pull/task/../../api/config'},s=>{s.resume();resolve(s.statusCode);});r.on('error',reject);});
  assert.equal(bad,401);
  console.log('PASS: shared access boundary, private upstream token, 20 MiB streaming, Range and pull token enforcement.');
 }finally{server.closeAllConnections();upstream.closeAllConnections();await Promise.all([new Promise(r=>server.close(r)),new Promise(r=>upstream.close(r))]);}
})().catch(error=>{console.error(error);process.exitCode=1;});
