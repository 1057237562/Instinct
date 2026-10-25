"""WebUI compatibility contract and honest readiness reporting (stdlib only).

All config_webui features are acceptance requirements except torch.compile.
Implementation availability and real-GPU acceptance are reported separately.
The native runner is available; target-GPU validation remains required.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math


TRAIN_TYPES = ('pretrain', 'full_sft', 'lora', 'dpo', 'ppo', 'grpo', 'agent', 'distillation')
OPTIMIZERS = ('adamw', 'adafactor', 'muon')
ARCHITECTURES = ('standard', 'moe', 'linear', 'looped', 'looped_legacy')
EXCLUDED = frozenset(('use_compile', 'compile_mode'))

# Every exported model field must have an owner. Tests check this against the
# real WebUI's build/export functions, without importing Streamlit or Torch.
MODEL_FIELDS = {
    'model': ('hidden_size', 'num_hidden_layers', 'vocab_size', 'intermediate_size',
              'tie_word_embeddings', 'model_architecture'),
    'dense_block': ('num_attention_heads', 'num_key_value_heads', 'head_dim', 'rms_norm_eps'),
    'activations': ('hidden_act',),
    'dropout': ('dropout',),
    'rope': ('max_position_embeddings', 'rope_theta', 'inference_rope_scaling', 'rope_scaling'),
    'attention': ('flash_attn',),
    'recompute': ('use_grad_checkpoint','selective_cache_mib'),
    'moe': ('use_moe', 'num_experts', 'num_experts_per_tok', 'moe_intermediate_size',
            'norm_topk_prob', 'router_aux_loss_coef'),
    'linear': ('full_attention_interval', 'linear_conv_kernel_dim', 'linear_key_head_dim',
               'linear_value_head_dim', 'linear_num_key_heads', 'linear_num_value_heads'),
    'looped': ('loop_max_steps', 'q_threshold', 'n_supervision', 'depth_reward',
               'exit_in_training', 'beta', 'distill_weight', 'distill_temperature',
               'teacher_stop_grad', 'depth_gain_reward', 'loop_encoder_layers',
               'loop_body_layers', 'loop_output_layers', 'prelude_layers', 'loop_iters', 'coda_layers', 'use_input_injection'),
    'early_exit': ('early_exit_layers', 'early_exit_loss_weight'),
}
TRAINING_FIELDS = {
    'training': ('train_type', 'dataset_size', 'batch_size', 'max_seq_len', 'num_workers', 'prefetch_factor', 'pin_memory'),
    'optimizer': ('optimizer','muon_mantissa','muon_momentum_dtype','muon_batch_size','muon_workspace_mb'),
    'precision': ('param_dtype', 'activation_dtype'),
    'kv_cache': ('kv_cache_dtype',),
    'checkpoint_io': ('from_resume', 'base_weight'),
    'recompute': ('use_grad_checkpoint',),
}
ROPE_FIELDS = frozenset(('type', 'beta_fast', 'beta_slow', 'factor',
                        'original_max_position_embeddings', 'attention_factor'))


@dataclass(frozen=True)
class Feature:
    status: str
    depends_on: tuple
    acceptance: str


FEATURES = {
    'dense_block': Feature('implemented_gpu_unverified', (),
        'Forward/dX/dW of all seven projections and RMSNorm/RoPE/GQA/SwiGLU/residuals match reference; native GPU tests pass.'),
    'scheduler': Feature('implemented_gpu_unverified', ('dense_block',),
        'Stable buffers, plan/workspace cache, single/dual stream event DAG, optional tuning/graph replay pass on target GPU.'),
    'model': Feature('implemented_gpu_unverified', ('dense_block',),
        'Embedding + arbitrary layer stack + final norm/LM head + shifted masked CE; tied and untied weights accumulate correctly.'),
    'activations': Feature('implemented_gpu_unverified', ('dense_block',),
        'Silu, exact project GELU and ReLU forward/backward agree for all Dense/expert FFNs.'),
    'dropout': Feature('implemented_gpu_unverified', ('model',),
        'Match attention/residual dropout; save RNG seed/counter and reproduce masks during recompute/resume; disable in eval.'),
    'rope': Feature('implemented_gpu_unverified', ('dense_block',),
        'Generate project-compatible RoPE/YaRN tables including all five scaling controls and decode offsets.'),
    'attention': Feature('implemented_gpu_unverified', ('dense_block',),
        'Fused causal GQA supports dropout and head dimensions through 256 with deterministic recomputation; larger heads/decode retain the general path.'),
    'precision': Feature('implemented_gpu_unverified', ('dense_block',),
        'FP32/BF16/FP16 activation/parameter combinations, FP8 E4M3 forward/E5M2 gradients, FP32 accumulation; optimizer state/loss scaling and convergence validated.'),
    'recompute': Feature('implemented_gpu_unverified', ('model', 'dropout'),
        'Modes 0/1/2: none/selective attention+FFN/full layer; fewer live activations, correct replay RNG and FP8 scales.'),
    'moe': Feature('implemented_gpu_unverified', ('model', 'activations'),
        'Device-only stable compact top-k routing, ragged grouped GEMMs with per-expert FP8, optional top-k renormalization and router auxiliary loss/gradients; Dense/MoE combinations supported.'),
    'linear': Feature('implemented_gpu_unverified', ('model',),
        'GatedDeltaNet convolution + state recurrence forward/backward, configurable heads/dimensions and full-attention intervals; streaming state parity.'),
    'looped': Feature('implemented_gpu_unverified', ('model',),
        'Legacy prelude/shared-loop/coda checkpoint equivalence plus LoopUS encoder/body/output partitions, repeated shared-weight gradient accumulation, q threshold/cap, supervision, depth/monotonicity rewards, self-distillation and teacher gradients.'),
    'early_exit': Feature('implemented_gpu_unverified', ('model',),
        'Intermediate shared-head CE at configured layers and weighted losses; accumulate shared head and backbone gradients.'),
    'kv_cache': Feature('implemented_gpu_unverified', ('model', 'precision'),
        'FP32/BF16/FP16 and per-(batch,head) scaled E4M3/E5M2 decode cache, growth/reorder/reset and full-prefix parity.'),
    'optimizer': Feature('implemented_gpu_unverified', ('model', 'precision'),
        'AdamW/Adafactor/Muon updates and states match project references; gradient accumulation, clipping, schedules and finite/overflow handling.'),
    'checkpoint_io': Feature('implemented_gpu_unverified', ('optimizer',),
        'Base weight selection, atomic save/resume of parameters, optimizer, RNG, sampler and scaler; legacy .pth model/optimizer/scaler/scheduler/progress conversion outside CUDA runtime, with explicit reporting of unsaved RNG/partial-gradient limits.'),
    'training': Feature('implemented_gpu_unverified', ('model', 'optimizer', 'checkpoint_io'),
        'Spawn-worker prefetch, bounded pinned upload staging, deterministic batch augmentation; Pretrain/SFT tokenization, mini/normal data, padding/loss masks, accumulation, logging and complete train/eval steps without Torch.'),
    'lora': Feature('implemented_gpu_unverified', ('training',),
        'Low-rank projections/gradients, frozen base weights, adapter optimizer/save/load/merge; same target modules as train_lora.'),
    'dpo': Feature('implemented_gpu_unverified', ('training',),
        'Chosen/rejected log probabilities, frozen reference, DPO loss and gradients match train_dpo.'),
    'ppo': Feature('implemented_gpu_unverified', ('training', 'kv_cache'),
        'Rollout sampling, reward/reference/value models, GAE, clipped objectives and value updates match train_ppo.'),
    'grpo': Feature('implemented_gpu_unverified', ('training', 'kv_cache'),
        'Group rollouts/rewards, normalized advantages, masks, importance ratios and GRPO/CISPO loss variants match train_grpo.'),
    'agent': Feature('implemented_gpu_unverified', ('grpo',),
        'Multi-turn tool rollout boundaries, rewards, masks and updates match train_agent; tool orchestration remains host-side.'),
    'distillation': Feature('implemented_gpu_unverified', ('training',),
        'Separate teacher/student weights, temperature-scaled KD and label CE, teacher stop-gradient and checkpoint behavior.'),
    'webui': Feature('implemented_gpu_unverified', ('training',),
        'Backend-aware start/status/logs/history/metrics and weight discovery, presets, import/export, parameter estimates and diagrams; preserve existing PyTorch workflows.'),
}


def _keys(groups):
    return {key for fields in groups.values() for key in fields}


def _positive(values, names):
    for name in names:
        if name in values and (isinstance(values[name], bool) or not isinstance(values[name], (int, float))
                               or not math.isfinite(values[name]) or values[name] <= 0):
            raise ValueError(f'{name} must be a positive finite number')


def _choice(values, key, choices):
    if key in values and values[key] not in choices:
        raise ValueError(f'{key} must be one of {choices}')


def _layer_list(model, key):
    if key in model:
        layers = model[key]
        if not isinstance(layers, list) or any(type(i) is not int or i < (1 if key == 'early_exit_layers' else 0) or i >= model['num_hidden_layers'] + (1 if key == 'early_exit_layers' else 0) for i in layers):
            raise ValueError(f'{key} must contain valid layer indices')
        if len(layers) != len(set(layers)):
            raise ValueError(f'{key} contains duplicate layers')


@dataclass
class WebUIRequest:
    model: dict
    training: dict
    excluded: dict

    @classmethod
    def from_dicts(cls, model, training=None):
        """Read exported model JSON plus WebUI training panel settings.

        Unknown keys are errors; torch.compile settings are explicitly listed
        as excluded. Values are preserved rather than silently falling back to
        Dense, AdamW or FP32 when another feature has not been implemented.
        """
        if not isinstance(model, dict) or (training is not None and not isinstance(training, dict)):
            raise ValueError('model and training must be JSON objects')
        model, training = deepcopy(model), deepcopy(training or {})
        unknown = (set(model) - _keys(MODEL_FIELDS) - EXCLUDED) | (set(training) - _keys(TRAINING_FIELDS) - EXCLUDED)
        if unknown:
            raise ValueError('unknown WebUI fields: ' + ', '.join(sorted(unknown)))
        excluded = {}
        for section, values in (('model', model), ('training', training)):
            for key in EXCLUDED & values.keys():
                excluded[f'{section}.{key}'] = values.pop(key)
        defaults = dict(hidden_size=768, num_hidden_layers=8, vocab_size=6400,
                        num_attention_heads=8, num_key_value_heads=4, model_architecture='standard',
                        hidden_act='silu', dropout=0.0, use_moe=False, use_grad_checkpoint=0,
                        inference_rope_scaling=False, flash_attn=True)
        model = defaults | model
        _positive(model, ('hidden_size', 'num_hidden_layers', 'vocab_size', 'num_attention_heads', 'num_key_value_heads'))
        for key in ('hidden_size', 'num_hidden_layers', 'vocab_size', 'num_attention_heads', 'num_key_value_heads'):
            if type(model[key]) is not int:
                raise ValueError(f'{key} must be an integer')
        model.setdefault('head_dim', model['hidden_size'] // model['num_attention_heads'])
        # Instinct uses pi * hidden_size, rounded to a multiple of 64.
        model.setdefault('intermediate_size', math.ceil(math.pi * model['hidden_size'] / 64) * 64)
        _positive(model, ('head_dim', 'intermediate_size', 'rms_norm_eps', 'max_position_embeddings', 'rope_theta'))
        if model['num_attention_heads'] % model['num_key_value_heads']:
            raise ValueError('num_attention_heads must be divisible by num_key_value_heads')
        if model['head_dim'] % 2:
            raise ValueError('RoPE requires an even head_dim')
        _choice(model, 'model_architecture', ARCHITECTURES)
        for key in ('prelude_layers','loop_iters','coda_layers'):
            if key in model and (type(model[key]) is not int or model[key]<0):raise ValueError(key+' must be a nonnegative integer')
        _choice(model, 'hidden_act', ('silu', 'gelu', 'relu'))
        _choice(model, 'use_grad_checkpoint', (0, 1, 2, 3, 4))
        if 'selective_cache_mib' in model and (type(model['selective_cache_mib']) is not int or model['selective_cache_mib']<0):
            raise ValueError('selective_cache_mib must be a nonnegative integer')
        if model['use_grad_checkpoint'] in (3, 4) and (model['model_architecture'] != 'standard' or model.get('use_moe')):
            raise ValueError('native ckpt3/4 require standard Dense')
        if not isinstance(model['dropout'], (int, float)) or not 0 <= model['dropout'] < 1:
            raise ValueError('dropout must be in [0, 1)')
        rope = model.get('rope_scaling')
        if rope is not None:
            if not isinstance(rope, dict) or set(rope) - ROPE_FIELDS:
                raise ValueError('invalid rope_scaling fields')
            _choice(rope, 'type', ('yarn',))
            _positive(rope, ROPE_FIELDS - {'type'})
        for key in ('early_exit_layers', 'loop_encoder_layers', 'loop_body_layers', 'loop_output_layers'):
            _layer_list(model, key)
        if model['use_moe'] or model['model_architecture'] == 'moe':
            _positive(model, ('num_experts', 'num_experts_per_tok', 'moe_intermediate_size'))
            if model.get('num_experts_per_tok', 1) > model.get('num_experts', 4):
                raise ValueError('num_experts_per_tok exceeds num_experts')
        if model['model_architecture'] == 'moe': model['use_moe'] = True
        if 'q_threshold' in model and not 0 <= model['q_threshold'] <= 1:
            raise ValueError('q_threshold must be in [0, 1]')
        _positive(model, ('distill_temperature',))
        for key in ('early_exit_loss_weight', 'router_aux_loss_coef', 'depth_reward', 'depth_gain_reward', 'distill_weight', 'beta'):
            if key in model and (not isinstance(model[key], (int,float)) or not math.isfinite(model[key]) or model[key] < 0):
                raise ValueError(key + ' must be finite and nonnegative')
        training = dict(train_type='pretrain', dataset_size='mini', batch_size=32, max_seq_len=768,
                        optimizer='adamw', param_dtype='fp32', activation_dtype='bfloat16',
                        kv_cache_dtype='fp32', from_resume=False, base_weight='auto', num_workers=8, prefetch_factor=2, pin_memory=True) | training
        _choice(training, 'train_type', TRAIN_TYPES)
        _choice(training, 'dataset_size', ('mini', 'normal'))
        _choice(training, 'optimizer', OPTIMIZERS)
        if training.get('muon_mantissa') and (training['optimizer']!='muon' or training['param_dtype']!='bf16'):
            raise ValueError('Muon mantissa requires optimizer=muon and BF16 parameters')
        _choice(training,'muon_momentum_dtype',('fp32','bf16'))
        if training.get('muon_momentum_dtype')=='bf16' and not training.get('muon_mantissa'):
            raise ValueError('BF16 Muon momentum requires mantissa')
        _positive(training,('muon_batch_size','muon_workspace_mb'))
        _choice(training, 'param_dtype', ('fp32', 'bf16', 'fp16'))
        _choice(training, 'activation_dtype', ('fp32', 'bfloat16', 'float16'))
        _choice(training, 'kv_cache_dtype', ('fp32', 'bf16', 'fp16', 'fp8_e4m3', 'fp8_e5m2'))
        _positive(training, ('batch_size', 'max_seq_len','prefetch_factor'))
        if type(training['num_workers']) is not int or training['num_workers']<0:raise ValueError('num_workers must be a nonnegative integer')
        if type(training['prefetch_factor']) is not int:raise ValueError('prefetch_factor must be an integer')
        if not isinstance(training['pin_memory'],bool):raise ValueError('pin_memory must be boolean')
        for values, names in ((model, ('head_dim', 'intermediate_size', 'max_position_embeddings',
                                      'num_experts', 'num_experts_per_tok', 'moe_intermediate_size',
                                      'full_attention_interval', 'linear_conv_kernel_dim',
                                      'linear_key_head_dim', 'linear_value_head_dim',
                                      'linear_num_key_heads', 'linear_num_value_heads', 'loop_max_steps', 'n_supervision')),
                              (training, ('batch_size', 'max_seq_len'))):
            for key in names:
                if key in values and (type(values[key]) is not int or values[key] <= 0):
                    raise ValueError(f'{key} must be a positive integer')
        if 'use_grad_checkpoint' in training:
            _choice(training, 'use_grad_checkpoint', (0, 1, 2, 3, 4))
            model['use_grad_checkpoint'] = training['use_grad_checkpoint']
        if model['use_grad_checkpoint'] in (3,4) and (model['model_architecture']!='standard' or model['use_moe']):
            raise ValueError('native ckpt3/4 require standard Dense')
        return cls(model, training, excluded)

    def required_features(self):
        needed = {'model', 'scheduler', 'precision', 'activations', 'rope', 'attention', 'training', 'webui'}
        for feature, enabled in (
            ('dropout', self.model['dropout'] > 0),
            ('moe', self.model['use_moe'] or self.model['model_architecture'] == 'moe'),
            ('linear', self.model['model_architecture'] == 'linear'),
            ('looped', self.model['model_architecture'] in ('looped','looped_legacy')),
            ('early_exit', bool(self.model.get('early_exit_layers'))),
            ('recompute', self.model['use_grad_checkpoint'] != 0),
            ('kv_cache', self.training['kv_cache_dtype'] != 'fp32'),
        ):
            if enabled:
                needed.add(feature)
        if self.training['train_type'] not in ('pretrain', 'full_sft'):
            needed.add(self.training['train_type'])
        def include(key):
            for dependency in FEATURES[key].depends_on:
                if dependency not in needed:
                    needed.add(dependency)
                    include(dependency)
        for key in tuple(needed):
            include(key)
        return sorted(needed)

    def report(self):
        needed = self.required_features()
        blockers = [key for key in needed if FEATURES[key].status != 'validated']
        return {'model': self.model, 'training': self.training,
                'excluded_torch_compile': self.excluded, 'training_ready': not blockers,
                'blockers': blockers, 'required_features': needed,
                'all_webui_requirements': {key: asdict(value) for key, value in FEATURES.items()},
                'implementation_ready': all(FEATURES[key].status.startswith('implemented') or FEATURES[key].status == 'validated' for key in needed),
                'native_runner': 'trainer/train_cublas.py',
                'note': 'Native training is implemented. training_ready remains false until target-GPU acceptance and convergence checks pass; see docs/dense_cublas.md.'}

    def require_training_ready(self):
        report = self.report()
        if not report['training_ready']:
            raise RuntimeError('cuBLAS target-GPU acceptance is not complete: ' + ', '.join(report['blockers']))
