// Rate samples belong to one browser view and one delivery attempt. Never count
// resume credits as network throughput, or reuse a rate after a long tab pause.
const RelayTransferRates = (() => {
  class Tracker {
    constructor(){this.samples=new Map();}
    observe(task, now=performance.now()){
      const done=Math.max(0,Number(task.upload_done_bytes)||0),total=Math.max(0,Number(task.upload_total_bytes)||0);
      const phase=task.upload_phase||(total>0&&done>=total?'verifying':'transferring');
      const result={phase,rate:null,eta:null};
      if(task.status!=='uploading'||phase!=='transferring'){
        this.samples.delete(task.id);return result;
      }
      const bytes=Number(task.upload_transferred_bytes??task.upload_done_bytes)||0,attempt=task.upload_attempt||0;
      let state=this.samples.get(task.id),last=state?.points.at(-1);
      if(!state||state.attempt!==attempt||bytes<last.bytes||now<=last.time||now-last.time>30000){
        this.samples.set(task.id,{attempt,points:[{time:now,bytes}]});return result;
      }
      if(now-last.time>=1000)state.points.push({time:now,bytes});
      while(state.points.length>2&&state.points[1].time<=now-10000)state.points.shift();
      const first=state.points[0];last=state.points.at(-1);
      if(last.time-first.time<1000)return result;
      result.rate=Math.max(0,(last.bytes-first.bytes)*1000/(last.time-first.time));
      result.eta=result.rate>0&&done<total?(total-done)/result.rate:null;
      return result;
    }
    retain(ids){for(const id of this.samples.keys())if(!ids.has(id))this.samples.delete(id);}
  }
  function duration(seconds){
    if(!Number.isFinite(seconds)||seconds<0)return '—';
    const n=Math.ceil(seconds);
    if(n<60)return `${n} 秒`;
    if(n<3600)return `${Math.floor(n/60)} 分 ${n%60} 秒`;
    const minutes=Math.ceil(n/60);
    return `${Math.floor(minutes/60)} 小时 ${minutes%60} 分`;
  }
  function describe(task, metrics, formatBytes){
    if(task.status!=='uploading')return '';
    if(metrics.stale)return '进度暂未更新 · 等待重新连接';
    if(metrics.phase==='preparing')return '中继 → AutoDL · 正在连接 / 检查空间';
    if(metrics.phase==='verifying')return '传输完成 · 正在校验远端文件';
    const label=task.upload_mode==='pull'?'落盘估算速率':task.upload_mode==='jupyter'?'完成速率（估算）':'传输速率';
    if(metrics.rate===null)return `中继 → AutoDL · ${label}：采样中…`;
    return `${label}：${formatBytes(Math.round(metrics.rate))}/s · 预计剩余 ${duration(metrics.eta)}`+
      (metrics.rate===0?' · 暂无新增传输':'');
  }
  return {Tracker,duration,describe};
})();
if(typeof module!=='undefined'&&module.exports)module.exports=RelayTransferRates;
