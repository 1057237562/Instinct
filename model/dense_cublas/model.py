"""Torch-free Instinct models. cuBLASLt projections and Triton VJPs.

The math attention and recurrent paths prioritize a readable correctness baseline;
shape-specific scheduling is deliberately left to the GPU benchmark harness.
"""
from dataclasses import dataclass
import math
import numpy as np
from .autograd import Value


def configuration(values):
    cfg = dict(hidden_size=768, num_hidden_layers=8, vocab_size=6400,
        num_attention_heads=8, num_key_value_heads=4, hidden_act='silu', dropout=0.,
        rms_norm_eps=1e-6, rope_theta=1e6, max_position_embeddings=32768,
        tie_word_embeddings=True, model_architecture='standard', use_moe=False,
        num_experts=4, num_experts_per_tok=1, norm_topk_prob=True, router_aux_loss_coef=5e-4,
        flash_attn=True, use_grad_checkpoint=0, early_exit_layers=[], early_exit_loss_weight=.3,
        full_attention_interval=4, linear_conv_kernel_dim=4, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_num_key_heads=4, linear_num_value_heads=8,
        loop_max_steps=8, q_threshold=.9, n_supervision=6, depth_reward=.01,
        exit_in_training=False, beta=.5, distill_weight=0., distill_temperature=2.,
        teacher_stop_grad=True, depth_gain_reward=False)
    cfg.update(values)
    if cfg['model_architecture'] == 'looped_legacy':
        for key, value in dict(prelude_layers=1, loop_iters=8, coda_layers=1, use_input_injection=True).items():
            cfg.setdefault(key, value)
    cfg.setdefault('head_dim', cfg['hidden_size'] // cfg['num_attention_heads'])
    cfg.setdefault('intermediate_size', math.ceil(math.pi * cfg['hidden_size'] / 64) * 64)
    cfg.setdefault('moe_intermediate_size', cfg['intermediate_size'])
    if cfg['model_architecture'] == 'moe': cfg['use_moe'] = True
    n = cfg['num_hidden_layers']
    cfg.setdefault('loop_encoder_layers', [0] if n > 2 else [])
    cfg.setdefault('loop_body_layers', list(range(1, n - 1)) if n > 2 else list(range(n)))
    cfg.setdefault('loop_output_layers', [n - 1] if n > 2 else [])
    if cfg.get('inference_rope_scaling') and not cfg.get('rope_scaling'):
        cfg['rope_scaling'] = dict(factor=16., original_max_position_embeddings=2048, beta_fast=32., beta_slow=1., attention_factor=1.)
    return cfg


def rope_tables(cfg):
    dim, end, base = cfg['head_dim'], cfg['max_position_embeddings'], cfg['rope_theta']
    freqs = 1. / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    scaling, amplitude = cfg.get('rope_scaling'), 1.
    if scaling:
        original, factor = scaling.get('original_max_position_embeddings', 2048), scaling.get('factor', 16.)
        amplitude = scaling.get('attention_factor', 1.)
        if end > original:
            inv = lambda b: dim * math.log(original / (b * 2 * math.pi)) / (2 * math.log(base))
            low, high = max(math.floor(inv(scaling.get('beta_fast', 32.))), 0), min(math.ceil(inv(scaling.get('beta_slow', 1.))), dim // 2 - 1)
            ramp = np.clip((np.arange(dim // 2) - low) / max(high - low, .001), 0., 1.)
            freqs = freqs * (1. - ramp + ramp / factor)
    phase = np.outer(np.arange(end), freqs).astype(np.float32)
    return tuple(np.tile(fn(phase), (1, 2)).astype(np.float32) * amplitude for fn in (np.cos, np.sin))


@dataclass
class Output:
    logits: Value
    hidden: Value
    aux_loss: Value
    loss: Value | None = None
    cache: dict | None = None
    depth: int = 1


class Model:
    def __init__(self, engine, config, param_dtype='fp32', activation_dtype='bf16', kv_cache_dtype='fp32', seed=42, muon_mantissa=False):
        self.e, self.config = engine, configuration(config)
        if self.config['use_grad_checkpoint'] in (3, 4) and (
                self.config['model_architecture'] != 'standard' or self.config['use_moe']):
            raise ValueError('native ckpt3/4 require standard Dense')
        self.param_dtype, self.activation_dtype, self.kv_cache_dtype = param_dtype, activation_dtype, kv_cache_dtype
        self.muon_mantissa=bool(muon_mantissa)
        if self.muon_mantissa and param_dtype != 'bf16':
            raise ValueError('Muon mantissa requires BF16 hidden parameters')
        self.params, self.adapters, self.training = {}, {}, True
        self.fuse_projections=False
        self.loss_chunk_size=0
        self.rng = np.random.default_rng(seed)
        c, h = self.config, self.config['hidden_size']
        self._parameter('model.embed_tokens.weight', (c['vocab_size'], h))
        self.layer_prefixes = [f'model.layers.{i}.' for i in range(c['num_hidden_layers'])]
        if c['model_architecture'] == 'looped_legacy':
            self.layer_prefixes = ([f'model.prelude.{i}.' for i in range(c['prelude_layers'])]
                + (['model.loop_block.'] if c['loop_iters'] else [])
                + [f'model.coda.{i}.' for i in range(c['coda_layers'])])
        for i, p in enumerate(self.layer_prefixes):
            for name in ('input_layernorm.weight', 'post_attention_layernorm.weight'): self._parameter(p + name, (h,), ones=True)
            if self.is_linear(i):
                kh, vh, dk, dv = (c[k] for k in ('linear_num_key_heads', 'linear_num_value_heads', 'linear_key_head_dim', 'linear_value_head_dim'))
                if vh % kh: raise ValueError('linear value heads must be divisible by key heads')
                width = 2 * kh * dk + vh * dv
                for name, outdim in [('in_proj_qkv', width), ('in_proj_z', vh * dv), ('in_proj_b', vh), ('in_proj_a', vh)]:
                    self._parameter(p + 'linear_attn.' + name + '.weight', (outdim, h))
                self._parameter(p + 'linear_attn.out_proj.weight', (h, vh * dv))
                self._parameter(p + 'linear_attn.conv1d.weight', (width, 1, c['linear_conv_kernel_dim']))
                self._parameter(p + 'linear_attn.norm.weight', (dv,), ones=True)
                self._parameter(p + 'linear_attn.dt_bias', (vh,), ones=True)
                self._parameter(p + 'linear_attn.A_log', (vh,), array=np.log(self.rng.uniform(.01, 16, vh)))
            else:
                d, q, k = c['head_dim'], c['num_attention_heads'], c['num_key_value_heads']
                for name, shape in [('q_proj', (q*d,h)), ('k_proj',(k*d,h)), ('v_proj',(k*d,h)), ('o_proj',(h,q*d))]:
                    self._parameter(p + 'self_attn.' + name + '.weight', shape)
                for name in ('q_norm', 'k_norm'): self._parameter(p + 'self_attn.' + name + '.weight', (d,), ones=True)
            if c['use_moe']:
                self._parameter(p + 'mlp.gate.weight', (c['num_experts'], h))
                for j in range(c['num_experts']): self._ffn_parameters(p + f'mlp.experts.{j}.', c['moe_intermediate_size'])
            else: self._ffn_parameters(p + 'mlp.', c['intermediate_size'])
        self._parameter('model.norm.weight', (h,), ones=True)
        if c['tie_word_embeddings']: self.params['lm_head.weight'] = self.params['model.embed_tokens.weight']
        else: self._parameter('lm_head.weight', (c['vocab_size'], h))
        if c['model_architecture'] == 'looped':
            parts = sum((c[name] for name in ('loop_encoder_layers', 'loop_body_layers', 'loop_output_layers')), [])
            if sorted(parts) != list(range(c['num_hidden_layers'])) or not c['loop_body_layers']:
                raise ValueError('loop partitions must cover each layer exactly once; body cannot be empty')
            self._parameter('model.exit_head.weight', (1,h), zeros=True)
            self._parameter('model.exit_head.bias', (1,), zeros=True)
            self._parameter('model.loop_gate.weight', (h,h), zeros=True)
            self._parameter('model.loop_gate.dt_bias', (h,), ones=True)
            self._parameter('model.loop_gate.A_log', (h,), zeros=True)
        if c['model_architecture'] == 'looped_legacy' and c['loop_iters']:
            self._parameter('model.loop_pos_embed.weight', (c['loop_iters'], h), zeros=True)
        self.cos, self.sin = (engine.array(x) for x in rope_tables(c))
        self._zero_aux = engine.full((1,), 0.)

    def _parameter(self, name, shape, ones=False, zeros=False, array=None):
        if array is None: array = np.ones(shape) if ones else np.zeros(shape) if zeros else self.rng.normal(0., .02, shape)
        mark = self.e.backend.mark()
        from .muon import hidden_matrix
        dtype = 'fp32' if self.muon_mantissa and not hidden_matrix(name,shape) else self.param_dtype
        self.params[name] = self.e.array(np.asarray(array, np.float32), dtype, True, name)
        retained=(self.params[name].data,)
        if getattr(self.e.backend,'cache_weight_scales',False) and len(shape)==2 and dtype=='bf16':
            from .quantization import reserve_weight_scale
            retained+= (reserve_weight_scale(self.e.ops,self.params[name].data),)
        self.e.backend.release_since(mark, keep=retained)
        return self.params[name]

    def _ffn_parameters(self, prefix, width):
        h = self.config['hidden_size']
        for name, shape in [('gate_proj', (width,h)), ('up_proj',(width,h)), ('down_proj',(h,width))]: self._parameter(prefix + name + '.weight', shape)

    def is_linear(self, index):
        return self.config['model_architecture'] == 'linear' and (index + 1) % self.config['full_attention_interval'] != 0

    def parameters(self, trainable=True):
        return list(dict.fromkeys(p for p in self.params.values() if p.requires_grad or not trainable))

    def freeze(self):
        for p in self.parameters(False): p.requires_grad = False
        self.training = False
        return self

    def add_lora(self, rank=16):
        originals = list(self.params.items())
        for p in self.parameters(False): p.requires_grad = False
        for name, weight in originals:
            if len(weight.shape) == 2 and weight.shape[0] == weight.shape[1] and '.embed_tokens.' not in name:
                stem = name[:-len('weight')]
                a = self._parameter(stem + 'lora.A.weight', (rank, weight.shape[1]))
                b = self._parameter(stem + 'lora.B.weight', (weight.shape[0], rank), zeros=True)
                self.adapters[name] = (a, b)
        if not self.adapters: raise ValueError('LoRA requires at least one square projection')

    def merge_lora(self):
        with self.e.no_grad():
            for name, (a, b) in self.adapters.items(): self.e.copy(self.params[name], self.params[name] + b.cast('fp32') @ a.cast('fp32'))
        self.params = {k:v for k,v in self.params.items() if '.lora.' not in k}
        self.adapters.clear()

    def linear(self, x, name):
        if self.muon_mantissa:x=x.cast(self.activation_dtype)
        out = self.e.linear(x, self.params[name])
        if name in self.adapters:
            a, b = self.adapters[name]
            out = out + self.e.linear(self.e.linear(x, a), b)
        return out

    def projections(self,x,names):
        if self.muon_mantissa:x=x.cast(self.activation_dtype)
        if any(name in self.adapters for name in names):
            return [self.linear(x,name) for name in names]
        weights=[self.params[name] for name in names]
        if not self.fuse_projections:return self.e.linears(x,weights)
        joined=self.e.linear(x,self.e.cat(weights,0))
        result=[];begin=0
        for weight in weights:
            result.append(joined[...,begin:begin+weight.shape[0]]);begin+=weight.shape[0]
        return result

    def norm(self, x, name):
        target=self.activation_dtype if (self.muon_mantissa and name.endswith('layernorm.weight')
            and getattr(self.e.backend,'fused_norm_cast',False)) else None
        return self.e.norm(x, self.params[name], self.config['rms_norm_eps'],out_dtype=target)
    def head(self, x): return self.linear(self.norm(x, 'model.norm.weight'), 'lm_head.weight')

    def _pack_cache(self, x):
        dtype = {'fp8_e4m3':'fp8e4nv', 'fp8_e5m2':'fp8e5'}.get(self.kv_cache_dtype, self.kv_cache_dtype)
        if dtype not in ('fp8e4nv','fp8e5'): return (x.cast(dtype).detach(), None)
        maximum = self.e.reduce(self.e.unary(x.cast('fp32'), 'abs'), (1,3), True, 'max')
        scale = self.e.binary(maximum, 1.e-12, 'max') / (448. if dtype == 'fp8e4nv' else 57344.)
        return ((x / scale).cast(dtype).detach(), scale.detach())

    def _unpack_cache(self, cache):
        x, scale = cache
        return (x.cast('fp32') * scale if scale is not None else x).cast(self.activation_dtype)

    def reorder_cache(self, cache, indices):
        """Reorder/duplicate batch rows for beam or grouped decoding."""
        index=self.e.array(np.asarray(indices,np.int64))
        def reorder(value):
            if isinstance(value,Value):
                return self.e.embedding(value.reshape(value.shape[0],-1),index).reshape(len(indices),*value.shape[1:]).detach()
            if isinstance(value,dict):return {k:reorder(v) for k,v in value.items()}
            if isinstance(value,tuple):return tuple(reorder(v) for v in value)
            return value
        with self.e.no_grad():return reorder(cache)

    @staticmethod
    def cache_tensors(cache):
        out = []
        def visit(x):
            if isinstance(x, Value): out.append(x.data)
            elif isinstance(x, dict):
                for v in x.values(): visit(v)
            elif isinstance(x, (list,tuple)):
                for v in x: visit(v)
        visit(cache)
        return out

    def attention(self, x, i, mask, cache, new_cache, start, cache_key, packed=None):
        if self.is_linear(i): return self.linear_attention(x, i, cache, new_cache, cache_key)
        e, c, p = self.e, self.config, self.layer_prefixes[i]+'self_attn.'
        b,s,_ = x.shape; d, hq, hkv = c['head_dim'], c['num_attention_heads'], c['num_key_value_heads']
        with e.ops.reuse_scope():
            q,k,v=self.projections(x,[p+name+'_proj.weight' for name in ('q','k','v')])
        cos, sin = packed if packed is not None else (self.cos, self.sin)
        rope_start = -1 if packed is not None else start
        q=e.norm_rotary(q.reshape(b,s,hq,d),self.params[p+'q_norm.weight'],cos,sin,c['rms_norm_eps'],rope_start)
        k=e.norm_rotary(k.reshape(b,s,hkv,d),self.params[p+'k_norm.weight'],cos,sin,c['rms_norm_eps'],rope_start)
        v=v.reshape(b,s,hkv,d)
        if cache and cache_key in cache:
            pk,pv = cache[cache_key]
            k, v = e.cat([self._unpack_cache(pk), k],1), e.cat([self._unpack_cache(pv), v],1)
        if new_cache is not None: new_cache[cache_key] = (self._pack_cache(k), self._pack_cache(v))
        out = e.attention(q,k,v,mask,c['dropout'],self.training,c['flash_attn'],start,packed=packed is not None).reshape(b,s,hq*d)
        return e.dropout(self.linear(out,p+'o_proj.weight'),c['dropout'],self.training)

    def ffn(self, x, prefix):
        from . import ffn
        if ffn.eligible(self,x,prefix):return ffn.apply(self,x,prefix)
        return self.linear(self.ffn_hidden(x,prefix),prefix+'down_proj.weight')

    def ffn_hidden(self,x,prefix):
        with self.e.ops.reuse_scope():
            gate, up = self.projections(x,[prefix+'gate_proj.weight',prefix+'up_proj.weight'])
        activated=self.e.swiglu(gate,up,round_gate=not self.fuse_projections,
            up_grad_dtype=up.dtype if getattr(self.e.backend,'compact_gradients',False) else None) if self.config['hidden_act']=='silu' else self.e.unary(gate,self.config['hidden_act'])*up
        return activated

    def feed_forward(self, x, i):
        e,c,p = self.e,self.config,self.layer_prefixes[i]+'mlp.'
        if not c['use_moe']: return self.ffn(x,p), self._zero_aux
        flat = x.reshape(-1,x.shape[-1])
        scores = e.softmax(self.linear(flat,p+'gate.weight').cast('fp32'))
        selected = e.topk_mask(scores,c['num_experts_per_tok'])
        weights = scores * selected
        if c['norm_topk_prob']: weights = weights / (weights.sum(-1,True)+1.e-20)
        from .grouped import feed_forward
        out = feed_forward(self,flat,weights,selected,p)
        aux = (selected.mean(0)*scores.mean(0)).sum()*c['num_experts']*c['router_aux_loss_coef'] if self.training else e.full((1,),0.)
        return out.reshape(*x.shape).cast(x.dtype),aux

    def block(self, x, i, mask=None, cache=None, new_cache=None, start=0, cache_key=None, packed=None):
        e,c,p = self.e,self.config,self.layer_prefixes[i]
        cache_key = str(i) if cache_key is None else cache_key
        checkpoint = c['use_grad_checkpoint'] if self.training and new_cache is None else 0
        ps = [v for k,v in self.params.items() if k.startswith(p) and v.requires_grad]
        def attention(z): return self.attention(self.norm(z,p+'input_layernorm.weight'),i,mask,cache,new_cache,start,cache_key,packed)
        if not c['use_moe']:
            # Dense has no router auxiliary output. Do not pack an extra zero
            # into every checkpoint's activation, then copy/slice it back out.
            def dense_mlp(z): return self.ffn(self.norm(z,p+'post_attention_layernorm.weight'),p+'mlp.')
            def attention_residual(z):
                a=attention(z)
                if getattr(e.backend,'fused_residual_norm',False):
                    target=self.activation_dtype if self.muon_mantissa and getattr(e.backend,'fused_norm_cast',False) else None
                    return e.residual_norm(z,a,self.params[p+'post_attention_layernorm.weight'],c['rms_norm_eps'],out_dtype=target)
                z=z+a
                return z,self.norm(z,p+'post_attention_layernorm.weight')
            def dense_whole(z):
                z,normalized=attention_residual(z)
                return z+self.ffn(normalized,p+'mlp.')
            def dense_replay(z,output):
                z,normalized=attention_residual(z)
                from . import ffn
                if ffn.eligible(self,normalized,p+'mlp.'):
                    return ffn.apply(self,normalized,p+'mlp.',residual=z,output=output)
                hidden=self.ffn_hidden(normalized,p+'mlp.')
                return e.linear_residual_replay(hidden,self.params[p+'mlp.down_proj.weight'],z,output)
            if checkpoint in (3, 4):
                if self.fuse_projections or self.adapters:
                    raise ValueError('native ckpt3/4 require separate Dense projections without LoRA')
                from .selective import ReplayCache, recipe_plan
                selected = (recipe_plan(c, x.shape, self.activation_dtype, i,
                    auxiliary_kv=getattr(e.backend,'selective_aux_kv',False),
                    attention_min_seq=getattr(e.backend,'selective_attention_min_seq',4096)) if checkpoint == 4
                            else ReplayCache({v.name for v in ps if len(v.shape) == 2}, c['dropout'] == 0.))
                replay=dense_replay if getattr(e.backend,'terminal_replay',False) else None
                return e.checkpoint(dense_whole, x, ps, selected,replay=replay), self._zero_aux
            if checkpoint==2:return e.checkpoint(dense_whole,x,ps),self._zero_aux
            x=x+(e.checkpoint(attention,x,ps) if checkpoint==1 else attention(x))
            return x+(e.checkpoint(dense_mlp,x,ps) if checkpoint==1 else dense_mlp(x)),self._zero_aux
        def mlp(z):
            value,aux = self.feed_forward(self.norm(z,p+'post_attention_layernorm.weight'),i)
            return e.cat([value.reshape(-1),aux.reshape(1)],0)
        def unpack(z): return z[:-1].reshape(*x.shape),z[-1].reshape(1)
        def whole(z):
            z = z + attention(z)
            f,a = unpack(mlp(z))
            return e.cat([(z+f).reshape(-1),a],0)
        if checkpoint == 2: return unpack(e.checkpoint(whole,x,ps))
        x = x + (e.checkpoint(attention,x,ps) if checkpoint == 1 else attention(x))
        f,a = unpack(e.checkpoint(mlp,x,ps) if checkpoint == 1 else mlp(x))
        return x+f,a

    def __call__(self, ids, labels=None, attention_mask=None, cache=None, use_cache=False, return_logits=True, sequence_ids=None):
        e,c = self.e,self.config
        native=getattr(e.backend,'native_attention',None)
        native_context=None
        if native is not None:
            host_ids=getattr(sequence_ids,'host_array',None) if isinstance(sequence_ids,Value) else sequence_ids
            # Masked/opaque device-only metadata keeps the existing kernel path.
            native.prepare(ids.shape,host_ids if attention_mask is None else None,
                           c['num_attention_heads'],c['num_key_value_heads'],c['head_dim'])
            native_context=native.context
        if not isinstance(ids, Value): ids = e.array(np.asarray(ids,np.int64))
        if labels is not None and not isinstance(labels,Value): labels=e.array(np.asarray(labels,np.int64))
        if attention_mask is not None and not isinstance(attention_mask,Value): attention_mask=e.array(np.asarray(attention_mask,np.int64))
        start = 0 if cache is None else cache.get('position',0)
        if start+ids.shape[1]>c['max_position_embeddings']: raise ValueError('sequence exceeds configured RoPE table')
        if use_cache and self.training: raise ValueError('KV cache is an evaluation-only operation')
        new_cache = {'position':start+ids.shape[1]} if use_cache else None
        packed = None
        if sequence_ids is not None:
            if c['model_architecture'] != 'standard' or use_cache or cache is not None or not c['flash_attn']:
                raise ValueError('packed training requires standard architecture, fused attention and no KV cache')
            if not isinstance(sequence_ids, Value): sequence_ids=e.array(np.asarray(sequence_ids,np.int64))
            if sequence_ids.shape != ids.shape: raise ValueError('sequence_ids shape differs from input IDs')
            from .packing import metadata
            attention_mask, positions, labels = metadata(e, sequence_ids, labels, attention_mask)
            if native is not None:attention_mask.data.native_attention_context=native_context
            packed = tuple(e.embedding(t, positions).reshape(-1, c['head_dim']) for t in (self.cos, self.sin))
        embedded=e.embedding(self.params['model.embed_tokens.weight'],ids)
        x=e.dropout(embedded if self.muon_mantissa else embedded.cast(self.activation_dtype),c['dropout'],self.training)
        aux,early,depth=self._zero_aux,[],1
        def run(layers, value, suffix=''):
            nonlocal aux
            for i in layers:
                inference_mark=e.backend.mark() if not e.recording else None
                value,a=self.block(value,i,attention_mask,cache,new_cache,start,str(i)+suffix,packed)
                if a is not self._zero_aux:aux=aux+a
                if labels is not None and self.training and i+1 in c['early_exit_layers']:
                    early.append(e.ce(self.head(value),labels,shift=True))
                if inference_mark is not None:
                    e.backend.release_since(inference_mark,keep=(value.data,aux.data,*self.cache_tensors(new_cache),*(v.data for v in early)))
            return value
        if c['model_architecture']=='looped':
            x,aux_loop,depth=self.looped(x,run,labels,use_cache); aux=aux+aux_loop
        elif c['model_architecture']=='looped_legacy':
            # Original input injection shares the prelude graph (it is not detached).
            x=run(range(c['prelude_layers']),x); injected=x
            loop_aux=aux
            for step in range(c['loop_iters']):
                signal=self.params['model.loop_pos_embed.weight'][step]
                x=run([c['prelude_layers']],x+signal+(injected if c['use_input_injection'] else 0.),f':loop{step}')
                # The original collects shared block auxiliary loss once, from its last invocation.
                if step+1<c['loop_iters']:aux=loop_aux
            begin=c['prelude_layers']+bool(c['loop_iters'])
            x=run(range(begin,len(self.layer_prefixes)),x);depth=c['loop_iters']
        else: x=run(range(c['num_hidden_layers']),x)
        hidden=self.norm(x,'model.norm.weight')
        fused=labels is not None and self.loss_chunk_size>0 and not return_logits and 'lm_head.weight' not in self.adapters
        logits=self.linear(hidden,'lm_head.weight') if return_logits or (labels is not None and not fused) else None
        loss=e.linear_ce(hidden[:,:-1],self.params['lm_head.weight'],labels[:,1:],self.loss_chunk_size) if fused else (None if labels is None else e.ce(logits,labels,shift=True))
        if loss is not None and early: loss=loss+sum(early)*c['early_exit_loss_weight']/len(early)
        return Output(logits,hidden,aux,loss,new_cache,depth)

    def looped(self,x,run,labels,use_cache=False):
        """LoopUS-style gated recurrence, supervised confidence and survival cost.

        Halting is tokenwise (causal training/decode parity), using strict q >
        threshold. Every depth is evaluated when populating a decode cache, but
        halted tokens keep their state. K random depths receive quality losses.
        """
        e,c=self.e,self.config
        x=run(c['loop_encoder_layers'],x)
        active=e.full((*x.shape[:2],1),1.);survival=active
        outputs,qualities,confidences=[],[],[];expected_depth=e.full((1,),0.)
        for step in range(c['loop_max_steps']):
            expected_depth=expected_depth+survival.mean()
            candidate=run(c['loop_body_layers'],x,f':loop{step}')
            decay=-self.params['model.loop_gate.A_log'].cast('fp32').exp()*(self.linear(x,'model.loop_gate.weight').cast('fp32')+self.params['model.loop_gate.dt_bias']).softplus()
            gate=1.-decay.exp()
            updated=(candidate*gate+x*(1.-gate)).cast(x.dtype)
            x=(updated*active+x*(1.-active)).cast(x.dtype)
            out=run(c['loop_output_layers'],x,f':loop{step}')
            q_logits=self.linear(self.norm(out,'model.norm.weight'),'model.exit_head.weight')+self.params['model.exit_head.bias']
            q=q_logits.sigmoid();survival=survival*(1.-q)
            outputs.append(out);confidences.append(q_logits)
            if labels is not None:qualities.append(self.head(out))
            if not self.training or c['exit_in_training']:
                active=active*(1.-e.binary(q,c['q_threshold'],'gt'))
                if not use_cache and not np.any(e.numpy(active)):break
        penalty=expected_depth*c['depth_reward']
        if qualities:
            # Deterministic depth sampling shares the saved engine RNG stream.
            rng=np.random.default_rng(np.random.SeedSequence([e.seed,e.counter]));e.counter+=1
            selected=sorted(rng.choice(len(qualities),min(c['n_supervision'],len(qualities)),replace=False).tolist())
            target=labels[:,1:];valid=e.binary(target,0.,'ge');denom=e.binary(valid.sum(),1.,'max')
            losses=[e.ce(z[:,:-1],target) for z in qualities]
            quality=e.full((1,),0.)
            for depth in selected:
                term=losses[depth]
                logits=qualities[depth][:,:-1]
                # The argmax correctness target is nondifferentiable routing data.
                maxima=e.reduce(logits,-1,True,'max')
                correct=e.gather_last(e.binary(logits,maxima,'eq'),target).detach()
                confidence=confidences[depth][:,:-1].reshape(*target.shape)
                term=term+((confidence.softplus()-correct*confidence)*valid).sum()/denom
                if depth:term=term+e.unary(losses[depth]-losses[depth-1].detach(),'relu')*c['beta']
                if depth and c['depth_gain_reward']:
                    term=term-e.unary(losses[0].detach()-losses[depth],'relu')*float(c['depth_gain_reward'])
                quality=quality+term
            penalty=penalty+quality/len(selected)
            if c['distill_weight'] and len(outputs)>1:
                temperature=c['distill_temperature']
                teacher_log=e.softmax(qualities[-1][:,:-1].cast('fp32')/temperature,log=True)
                if c['teacher_stop_grad']:teacher_log=teacher_log.detach()
                teacher=teacher_log.exp()
                for depth in selected:
                    if depth==len(outputs)-1:continue
                    student=e.softmax(qualities[depth][:,:-1].cast('fp32')/temperature,log=True)
                    kl=(teacher*(teacher_log-student)).sum(-1)
                    penalty=penalty+(kl*valid).sum()/denom*temperature*temperature*c['distill_weight']/len(selected)
        return outputs[-1],penalty,len(outputs)

    def linear_attention(self,x,i,cache,new_cache,key):
        from .recurrent import linear_attention
        return linear_attention(self,x,i,cache,new_cache,key)
