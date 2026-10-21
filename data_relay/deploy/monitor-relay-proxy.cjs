'use strict';
// Mount before body parsers. Stream chunks/Range responses; never buffer corpora.
const http = require('node:http');
const fs = require('node:fs');
const HOP = new Set(['connection','keep-alive','proxy-authenticate','proxy-authorization','te','trailer','transfer-encoding','upgrade']);
function headersFor(headers) {
  const omit = new Set([...HOP, ...String(headers.connection || '').toLowerCase().split(',').map(s=>s.trim())]);
  return Object.fromEntries(Object.entries(headers).filter(([key])=>!omit.has(key.toLowerCase())));
}
function isPull(req) {
  return ['GET','HEAD'].includes(req.method) && /^\/pull\/[a-zA-Z0-9_-]+\//.test(req.url || '');
}
function createRelayProxy(options={}) {
  const resolve = options.resolve || (()=>JSON.parse(fs.readFileSync(process.env.SSHBROWSE_RELAY_CONFIG || '/etc/instinct-relay/proxy.json','utf8')));
  return function relayProxy(req,res) {
    let pathname;
    try {pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);}catch {res.status(400).end();return;}
    // Credentials are only consumed by the monitor backend, never by a browser.
    if(pathname.toLowerCase().startsWith('/api/integration/')) {res.status(404).end();return;}
    if(options.onConfig && req.method==='PUT' && pathname.replace(/\/+$/,'')==='/api/config') {
      let size=0;const chunks=[];
      req.on('data',chunk=>{size+=chunk.length;if(size<=65536)chunks.push(chunk);});
      req.on('end',async()=>{
        if(size>65536){res.status(413).json({error:'配置请求过大'});return;}
        try {
          const patch=JSON.parse(Buffer.concat(chunks));
          if(!patch || typeof patch!=='object' || Array.isArray(patch))throw Error('配置格式无效');
          res.json(await options.onConfig(patch));
        }catch(error){res.status(400).json({error:error.message});}
      });
      return;
    }
    let config;
    try {config=resolve();if(!config.token)throw new Error('not configured');}
    catch {res.status(503).json({error:'数据中继尚未就绪，请稍后刷新'});return;}
    if(!req.url.startsWith('/') || req.url.startsWith('//')) {res.status(400).end();return;}
    const headers=headersFor(req.headers);
    delete headers.cookie;delete headers.authorization;
    headers.host=`127.0.0.1:${config.port || 7800}`;
    headers['x-forwarded-prefix']='/relay';
    // HTTP pulls authenticate with their per-task query token. Never grant
    // service-level access to an anonymous pull, even for a malformed URL.
    if(!isPull(req))headers.authorization=config.token;
    const upstream=http.request({hostname:'127.0.0.1',port:config.port || 7800,method:req.method,path:req.url,headers},incoming=>{
      const outgoing=headersFor(incoming.headers);
      outgoing['cache-control']='no-store';
      outgoing['x-content-type-options']='nosniff';
      outgoing['x-frame-options']='SAMEORIGIN';
      outgoing['referrer-policy']='no-referrer';
      outgoing['content-security-policy']="default-src 'self'; script-src 'self' 'unsafe-inline'; worker-src 'self' blob:; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; frame-ancestors 'self'; object-src 'none'; base-uri 'self'";
      res.writeHead(incoming.statusCode,outgoing);incoming.pipe(res);
    });
    upstream.setTimeout(600000,()=>upstream.destroy(new Error('timeout')));
    upstream.on('error',()=>{if(!res.headersSent)res.status(502).json({error:'数据中继连接中断，请稍后重试'});else res.destroy();});
    req.on('aborted',()=>upstream.destroy());
    res.on('close',()=>{if(!res.writableEnded)upstream.destroy();});
    req.pipe(upstream);
  };
}
module.exports={createRelayProxy,isPull};
