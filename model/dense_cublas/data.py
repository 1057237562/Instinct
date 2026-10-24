"""JSONL indexing, Rust tokenization and Jinja chat templates without Torch."""
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer as RustTokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment

SYSTEM_PROMPTS=[
    '你是一个知识丰富的AI，尽力为用户提供准确的信息。','你是Instinct，一个小巧但有用的语言模型。',
    '你是一个专业的AI助手，请提供有价值的回答。','你是Instinct，请尽力帮助用户解决问题。','你是一个可靠的AI，请给出准确的回答。',
    'You are a helpful AI assistant.','You are Instinct, a lightweight intelligent assistant.',
    "You are a friendly chatbot. Please answer the user's questions carefully.",
    'You are a knowledgeable AI. Try your best to provide accurate information.',
    'You are Instinct, a small but useful language model.']


class Tokenizer:
    def __init__(self,path='./model'):
        path=Path(path);self.raw=RustTokenizer.from_file(str(path/'tokenizer.json'))
        self.config=json.loads((path/'tokenizer_config.json').read_text(encoding='utf-8'))
        def token(key):
            t=self.config[key];return t['content'] if isinstance(t,dict) else t
        self.bos_token,self.eos_token,self.pad_token=(token(k) for k in ('bos_token','eos_token','pad_token'))
        self.bos,self.eos,self.pad=(self.raw.token_to_id(t) for t in (self.bos_token,self.eos_token,self.pad_token))
        env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
        env.filters['tojson']=lambda value,**kw:json.dumps(value,ensure_ascii=False,**kw)
        env.globals['raise_exception']=lambda message:(_ for _ in ()).throw(ValueError(message))
        self.template=env.from_string(self.config['chat_template'])
        self.assistant=self.encode(self.bos_token+'assistant\n');self.terminator=self.encode(self.eos_token+'\n')

    def encode(self,text):return self.raw.encode(text,add_special_tokens=False).ids
    def decode(self,ids,skip_special_tokens=False):return self.raw.decode([int(x) for x in ids],skip_special_tokens=skip_special_tokens)

    def chat(self,messages,tools=None,generation=False,thinking=False):
        messages=[dict(m) for m in messages]
        for m in messages:
            if m.get('tools') and m.get('role')=='system':tools=json.loads(m['tools']) if isinstance(m['tools'],str) else m['tools']
            if isinstance(m.get('tool_calls'),str):m['tool_calls']=json.loads(m['tool_calls'])
        return self.template.render(messages=messages,tools=tools,add_generation_prompt=generation,open_thinking=thinking,
            bos_token=self.bos_token,eos_token=self.eos_token,pad_token=self.pad_token)

    def labels(self,ids):
        labels=np.full(len(ids),-100,dtype=np.int64);ids=list(ids);i=0
        while i<len(ids):
            if ids[i:i+len(self.assistant)]==self.assistant:
                start=i+len(self.assistant);end=start
                while end<len(ids) and ids[end:end+len(self.terminator)]!=self.terminator:end+=1
                end=min(end+len(self.terminator),len(ids));labels[start:end]=ids[start:end];i=end
            else:i+=1
        labels[np.asarray(ids)==self.pad]=-100
        return labels


class JsonlDataset:
    def __init__(self,path,tokenizer,mode,max_length,rng=None):
        self.path,self.tokenizer,self.mode,self.max_length=Path(path),tokenizer,mode,max_length
        self.rng=np.random.default_rng(42) if rng is None else rng
        self.offsets=[]
        with self.path.open('rb') as f:
            while True:
                offset=f.tell();line=f.readline()
                if not line:break
                if line.strip():self.offsets.append(offset)
        if not self.offsets:raise ValueError('empty dataset: '+str(path))
    def __len__(self):return len(self.offsets)
    def row(self,index):
        with self.path.open('rb') as f:f.seek(self.offsets[index]);return json.loads(f.readline())
    def _pad(self,ids):return np.asarray((ids[:self.max_length]+[self.tokenizer.pad]*self.max_length)[:self.max_length],np.int64)
    def _sft(self,messages,augment=True):
        tk=self.tokenizer
        if augment and not any(m.get('tools') for m in messages) and messages[0]['role']!='system' and self.rng.random()<.2:
            messages=[{'role':'system','content':self.rng.choice(SYSTEM_PROMPTS)}]+messages
        text=tk.chat(messages)
        if augment and self.rng.random()>.2:text=text.replace('<think>\n\n</think>\n\n','')
        ids=self._pad(tk.encode(text));return ids,tk.labels(ids)
    def __getitem__(self,index):
        row=self.row(index);tk=self.tokenizer
        if self.mode=='pretrain':
            ids=self._pad([tk.bos]+tk.encode(str(row['text']))[:self.max_length-2]+[tk.eos])
            labels=ids.copy();labels[ids==tk.pad]=-100;return ids,labels
        if self.mode=='dpo':return self._sft(row['chosen'],False),self._sft(row['rejected'],False)
        if self.mode in ('ppo','grpo','agent'):
            messages=row['conversations']
            if messages and messages[-1]['role']=='assistant':messages=messages[:-1]
            tools=row.get('tools')
            for m in messages:
                if m.get('tools'):tools=json.loads(m['tools']) if isinstance(m['tools'],str) else m['tools']
            return dict(messages=messages,tools=tools,gt=row.get('gt',[]))
        return self._sft(row['conversations'])
    def batch(self,indices):
        rows=[self[int(i)] for i in indices]
        if self.mode in ('ppo','grpo','agent'):return rows
        if self.mode=='dpo':rows=[x[0] for x in rows]+[x[1] for x in rows]
        return tuple(np.stack(x) for x in zip(*rows))
