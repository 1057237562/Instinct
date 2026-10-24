"""Native InternLM2 reward inference from local safetensors; no remote code execution.

Architecture/weight layout reference:
https://huggingface.co/internlm/internlm2-1_8b-reward/blob/main/modeling_internlm2.py
The reward model is evaluated in FP16/BF16, independently of actor FP8 training.
"""
import json
import re
from pathlib import Path
import numpy as np
from jinja2.sandbox import ImmutableSandboxedEnvironment
from .autograd import Engine
from .model import rope_tables


class RewardTokenizer:
    def __init__(self,path):
        import sentencepiece as spm
        self.sp=spm.SentencePieceProcessor(model_file=str(path/'tokenizer.model'))
        cfg=json.loads((path/'tokenizer_config.json').read_text(encoding='utf-8'))
        self.special={v['content']:int(k) for k,v in cfg['added_tokens_decoder'].items()}
        # BOS/EOS can be native SentencePiece pieces rather than added tokens.
        for name in ('bos_token','eos_token','unk_token'):
            token=cfg.get(name)
            if token and token not in self.special:self.special[token]=self.sp.piece_to_id(token)
        self.pattern=re.compile('('+'|'.join(re.escape(k) for k in sorted(self.special,key=len,reverse=True))+')')
        self.template=ImmutableSandboxedEnvironment().from_string(cfg['chat_template']);self.bos=cfg['bos_token']
    def encode(self,messages):
        text=self.template.render(messages=messages,add_generation_prompt=False,bos_token=self.bos)
        out=[]
        for part in self.pattern.split(text):
            if not part:continue
            out.extend([self.special[part]] if part in self.special else self.sp.encode(part,out_type=int))
        return out


class InternLMReward:
    def __init__(self,backend,path,dtype='fp16'):
        from safetensors import safe_open
        path=Path(path);self.e=Engine(backend,fp8=False);self.dtype=dtype
        self.config=json.loads((path/'config.json').read_text(encoding='utf-8'))
        c=self.config
        if c.get('model_type')!='internlm2' or c.get('bias',False) or c.get('rope_scaling'):
            raise ValueError('native reward loader supports the released bias-free, unscaled InternLM2 reward architecture')
        self.tokenizer=RewardTokenizer(path);self.weights={}
        files=sorted(path.glob('*.safetensors'))
        if not files:raise FileNotFoundError('No safetensors weights in reward_model_path')
        for file in files:
            with safe_open(str(file),framework='numpy') as f:
                for name in f.keys():
                    mark=backend.mark()
                    value=self.e.array(f.get_tensor(name),dtype)
                    self.weights[name]=value;backend.release_since(mark,keep=(value.data,))
        c=dict(c,head_dim=c['hidden_size']//c['num_attention_heads'])
        self.cos,self.sin=(self.e.array(t) for t in rope_tables(c))
        required={'model.tok_embeddings.weight','model.norm.weight','v_head.weight'}
        for i in range(c['num_hidden_layers']):
            required.update(f'model.layers.{i}.{n}.weight' for n in ('attention_norm','ffn_norm','attention.wqkv','attention.wo','feed_forward.w1','feed_forward.w2','feed_forward.w3'))
        missing=required-self.weights.keys()
        if missing:raise ValueError('missing reward weights: '+', '.join(sorted(missing)))

    def score_ids(self,ids):
        e,c=self.e,self.config;mark=e.backend.mark()
        if not ids or len(ids)>c['max_position_embeddings']:raise ValueError('reward sequence is empty or exceeds position limit')
        try:
            with e.no_grad():
                x=e.embedding(self.weights['model.tok_embeddings.weight'],e.array(np.asarray([ids],np.int64)))
                h,k,d=c['num_attention_heads'],c['num_key_value_heads'],c['hidden_size']//c['num_attention_heads'];group=h//k;s=len(ids)
                def linear(z,name):return e.linear(z,self.weights[name+'.weight'])
                def norm(z,name):return e.norm(z,self.weights[name+'.weight'],c['rms_norm_eps'])
                for i in range(c['num_hidden_layers']):
                    local=e.backend.mark();p=f'model.layers.{i}.'
                    z=norm(x,p+'attention_norm');fused=linear(z,p+'attention.wqkv').reshape(1,s,k,group+2,d)
                    q=fused[:,:,:,:group,:].reshape(1,s,h,d);key=fused[:,:,:,group,:];v=fused[:,:,:,group+1,:]
                    q,key=e.rotary(q,self.cos,self.sin),e.rotary(key,self.cos,self.sin)
                    a=e.attention(q,key,v,flash=True).reshape(1,s,h*d)
                    x=x+linear(a,p+'attention.wo');z=norm(x,p+'ffn_norm')
                    x=x+linear(e.unary(linear(z,p+'feed_forward.w1'),'silu')*linear(z,p+'feed_forward.w3'),p+'feed_forward.w2')
                    # All previous layer activations are dead; retain only x.
                    e.backend.release_since(mark,keep=(x.data,))
                return e.item(linear(norm(x,'model.norm')[:,-1:],'v_head'))
        finally:e.backend.release_since(mark)

    def get_score(self,messages,response):
        history='\n'.join(f"{m['role']}: {m.get('content','')}" for m in messages[:-1])
        query=messages[-1].get('content','') if messages else ''
        context=f'{history}\n以上是对话历史。我的新问题是：\n{query}' if history else query
        ids=self.tokenizer.encode([{'role':'user','content':context},{'role':'assistant','content':response}])
        if not ids or ids[-1]!=self.config['reward_token_id']:ids.append(self.config['reward_token_id'])
        return float(np.clip(self.score_ids(ids),-3.,3.))
