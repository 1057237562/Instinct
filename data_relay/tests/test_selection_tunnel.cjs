'use strict';
const assert=require('node:assert/strict'),path=require('node:path');
const {createStreamlitPanel}=require(path.join(path.resolve(process.argv[2]),'src/streamlit-proxy.js'));
(async()=>{
 const pending=[];let oldStopped=false;
 const panel=createStreamlitPanel({resolveOptions:()=>({enabled:true,remoteHost:'127.0.0.1',remotePort:8501,localPort:0}),
  startForward:()=>new Promise(resolve=>pending.push(resolve)),log:()=>{},warn:()=>{}});
 const old=panel.start();assert.equal(pending.length,1);
 const switched=panel.restart();await Promise.resolve();assert.equal(pending.length,2);
 pending[1]({port:2222,connected:true,stop:async()=>{}});await switched;
 pending[0]({port:1111,connected:true,stop:async()=>{oldStopped=true;}});await old;
 assert(oldStopped);assert.equal(panel.status().port,2222);assert(panel.status().connected);await panel.stop();
 console.log('PASS: a late tunnel to the old server is closed and cannot replace the new tunnel.');
})().catch(error=>{console.error(error);process.exitCode=1;});
