"""Autoregressive sampling and multi-turn trajectories with exact policy masks."""
import json
import re
import numpy as np
from .agent_tools import parse_tool_calls,execute_tool,CHECK_ARGS,rep_penalty


def sample(logits,rng,history=(),temperature=.8,top_k=0,top_p=.9,repetition_penalty=1.):
    raw=np.asarray(logits,np.float64).copy()
    if not np.isfinite(raw).all():raise FloatingPointError('nonfinite generation logits')
    adjusted=raw.copy()
    if repetition_penalty<=0:raise ValueError('repetition penalty must be positive')
    for token in set(history):adjusted[token]=adjusted[token]/repetition_penalty if adjusted[token]>0 else adjusted[token]*repetition_penalty
    if temperature<=0:chosen=int(np.argmax(adjusted))
    else:
        adjusted/=temperature
        if top_k>0 and top_k<len(adjusted):adjusted[adjusted<np.partition(adjusted,-top_k)[-top_k]]=-np.inf
        order=np.argsort(adjusted)[::-1];probs=np.exp(adjusted[order]-np.max(adjusted));probs/=probs.sum()
        if 0<top_p<1:probs[np.cumsum(probs)-probs>=top_p]=0.;probs/=probs.sum()
        chosen=int(rng.choice(order,p=probs))
    maximum=raw.max();logp=raw[chosen]-maximum-np.log(np.exp(raw-maximum).sum())
    return chosen,float(logp)


def generate(model,prompt,rng,eos,max_new_tokens=256,temperature=.8,top_k=0,top_p=.9,repetition_penalty=1.):
    if not prompt:raise ValueError('generation prompt cannot be empty')
    e=model.e;mark=e.backend.mark();prior=model.training;model.training=False
    tokens,old_logps=[],[];cache=None
    # Loop depth-dependent caches need all body steps to remain available.
    try:
        with e.no_grad():
            current=list(prompt)
            for _ in range(min(max_new_tokens,model.config['max_position_embeddings']-len(prompt))):
                output=model(np.asarray([current],np.int64),cache=cache,use_cache=True)
                logits=e.numpy(output.logits[:,-1,:]).reshape(-1)
                token,logp=sample(logits,rng,[*prompt,*tokens],temperature,top_k,top_p,repetition_penalty)
                tokens.append(token);old_logps.append(logp);cache=output.cache
                e.backend.release_since(mark,keep=model.cache_tensors(cache))
                if token==eos:break
                current=[token]
    finally:model.training=prior;e.backend.release_since(mark)
    return tokens,old_logps


def trajectory(model,tokenizer,row,rng,max_new_tokens=256,max_total_len=2500,max_turns=3,thinking_ratio=.5,agent=False,**sampling):
    thinking=bool(rng.random()<thinking_ratio)
    messages=[dict(m) for m in row['messages']];tools=row.get('tools')
    prompt=tokenizer.encode(tokenizer.chat(messages,tools,generation=True,thinking=thinking))
    # Keep the newest prompt tokens while reserving a generation slot.
    prompt=prompt[-max(1,min(max_total_len-1,model.config['max_position_embeddings']-1)):]
    tokens=list(prompt);mask=[0]*len(prompt);old=[0.]*len(prompt);outputs=[];unfinished=False
    for turn in range(max_turns if agent else 1):
        budget=min(max_new_tokens,max_total_len-len(tokens),model.config['max_position_embeddings']-len(tokens))
        if budget<=0:unfinished=True;break
        generated,logps=generate(model,tokens,rng,tokenizer.eos,budget,**sampling)
        tokens.extend(generated);mask.extend([1]*len(generated));old.extend(logps)
        text=tokenizer.decode([t for t in generated if t!=tokenizer.eos]);outputs.append(text)
        calls=parse_tool_calls(text) if agent else []
        if not calls:break
        unfinished=turn==max_turns-1
        # Exact continuation tokens: model-produced spans get policy gradients;
        # simulated tool observations and role headers have zero policy mask.
        observation='' if generated and generated[-1]==tokenizer.eos else tokenizer.eos_token
        observation+='\n'+tokenizer.bos_token+'user'
        for call in calls:
            args=call.get('arguments',{})
            if isinstance(args,str):
                try:args=json.loads(args)
                except json.JSONDecodeError:args={}
            valid_names={t['function']['name'] for t in (tools or [])}
            result=execute_tool(call.get('name',''),args) if call.get('name') in valid_names else None
            payload=json.dumps(result if result is not None else {'error':'tool not found'},ensure_ascii=False)[:2048]
            observation+='\n<tool_response>\n'+payload+'\n</tool_response>'
        observation+=tokenizer.eos_token+'\n'
        if not unfinished:
            observation+=tokenizer.bos_token+'assistant\n'+('<think>\n' if thinking else '<think>\n\n</think>\n\n')
        obs=tokenizer.encode(observation)[:max(0,max_total_len-len(tokens))]
        tokens.extend(obs);mask.extend([0]*len(obs));old.extend([0.]*len(obs))
        if unfinished:break
    return dict(tokens=tokens,mask=mask,old_logps=old,outputs=outputs,unfinished=unfinished,row=row)


def reward(trajectory,reward_model,agent=False):
    row=trajectory['row'];outputs=trajectory['outputs'];response=''.join(outputs)
    answer=(outputs[-1] if outputs else '').split('</think>')[-1].strip()
    calls=[call for text in outputs for call in parse_tool_calls(text)]
    score=0.
    if agent:score-=.5*sum(abs(t.count('<tool_call>')-t.count('</tool_call>')) for t in outputs)
    if not agent or not calls:
        score+=.5 if (5 if agent else 20)<=len(response.strip())<=800 else -.5
        if '</think>' in response:
            thinking,answer=response.split('</think>',1);answer=answer.strip()
            score+=1. if 20<=len(thinking.strip())<=300 else -.5
            score+=.25 if response.count('</think>')==1 else -.25
        if reward_model is not None:score+=reward_model.get_score(row['messages'],answer)
        score-=rep_penalty(answer)
    else:
        valid_names={t['function']['name'] for t in (row.get('tools') or [])};valid=0
        for call in calls:
            name,args=call.get('name',''),call.get('arguments',{})
            if isinstance(args,str):
                try:args=json.loads(args)
                except json.JSONDecodeError:args={}
            valid+=int(bool(name in valid_names and isinstance(args,dict) and CHECK_ARGS.get(name,lambda a:False)(args)))
        gt=row.get('gt',[]);gap=abs(valid-len(gt))+max(0,len(calls)-valid)
        score+=.5 if gap==0 else -.5*gap
        final='' if trajectory['unfinished'] else answer.split('</tool_call>')[-1]
        numbers=[float(n) for n in re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])',final.replace(',',''))]
        hits=set()
        for item in gt:
            s=str(item).strip()
            if s and s.lower() in final.lower():hits.add(s)
            elif re.fullmatch(r'[-+]?\d+(?:\.\d+)?',s.replace(',','')) and any(abs(float(s.replace(',',''))-n)<1.e-6 for n in numbers):hits.add(s)
        if gt:score+=2.5*len(hits)/len(gt)
        if trajectory['unfinished']:score-=.5
        score-=rep_penalty(final or answer)
    return float(np.clip(score,-3.,3.)) if agent else float(score)


def collate(trajectories,pad):
    length=max(len(t['tokens']) for t in trajectories)
    ids=np.full((len(trajectories),length),pad,np.int64);attention=np.zeros_like(ids)
    mask=np.zeros(ids.shape,np.float32);old=np.zeros(ids.shape,np.float32)
    for i,t in enumerate(trajectories):
        n=len(t['tokens']);ids[i,:n]=t['tokens'];attention[i,:n]=1;mask[i,:n]=t['mask'];old[i,:n]=t['old_logps']
    return ids,attention,mask[:,1:],old[:,1:]
