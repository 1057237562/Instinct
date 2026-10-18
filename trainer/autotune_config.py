"""Validated settings and portable fingerprints for offline training autotuning.

Importing this module does not initialize CUDA. Profiles contain data only;
they cannot name executable code or change the model/optimizer mathematics.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
ARG_CHOICES = {
    'use_grad_checkpoint': (1, 2),
    'use_compile': (0, 1),
    'compile_mode': ('default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'),
}
ARG_POSITIVE = {'bucket_token_budget', 'bucket_gpu_memory_gb'}
ENV_CHOICES = {
    'INSTINCT_MOE_INNER_CHECKPOINT': ('0', '1', '2'),
    'INSTINCT_MOE_SELECTIVE_CHECKPOINT': ('0', 'down', 'gate_down'),
    'INSTINCT_MUON_PREALLOCATE': ('0', '1'),
}
ENV_POSITIVE = {'INSTINCT_MUON_WORKSPACE_MB', 'INSTINCT_MUON_BATCH_SIZE',
                'INSTINCT_TRAINING_CPU_THREADS', 'INSTINCT_GROUPED_MM_PROGRAMS_PER_SM'}
ENV_TILES = {'INSTINCT_GROUPED_MM_FWD_TILE', 'INSTINCT_GROUPED_MM_DW_TILE'}
PERF_ENV = set(ENV_CHOICES) | ENV_POSITIVE | ENV_TILES
# These switches affect compatibility but are not automatically changed.
CONTEXT_ENV = {
    'INSTINCT_MUON_NS_BACKEND', 'INSTINCT_MUON_BACKEND', 'INSTINCT_MUON_FUSED_UPDATES',
    'INSTINCT_MOE_FUSED_PACK', 'INSTINCT_MOE_FUSED_SWIGLU', 'INSTINCT_MOE_FUSED_ROUTING',
    'INSTINCT_MOE_TRANSPOSED_DW', 'INSTINCT_MOE_DIRECT_WEIGHTS', 'INSTINCT_GROUPED_MM_BACKEND',
    'INSTINCT_PACKED_ATTENTION_BACKEND', 'INSTINCT_CAUSAL_ATTENTION_BACKEND',
    'INSTINCT_FLEX_AUTOTUNE', 'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF',
}
CONTRACT_ARGS = (
    'dtype', 'param_dtype', 'fp8_training', 'fp8_filter', 'optimizer', 'accumulation_steps',
    'grad_clip', 'learning_rate', 'epochs', 'warmup_steps', 'warmup_ratio', 'min_lr_ratio',
    'sequence_packing', 'sequence_packing_mode', 'seq_bucket', 'bucket_max_seq_len',
    'bucket_large_threshold', 'packing_batch_size', 'max_seq_len', 'dataset_streaming',
    'streaming_chunk_mb', 'streaming_prefetch_chunks', 'cache_build_mode',
    'packing_num_proc', 'num_workers', 'bucket_loader_workers', 'data_cache_max_gb',
    'moe_router_top_k', 'moe_router_norm_topk_prob',
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def tile(value):
    parts = tuple(int(x) for x in str(value).split(','))
    if (len(parts) != 5 or any(x not in (16, 32, 64, 128) for x in parts[:3])
            or parts[3] not in (4, 8) or parts[4] not in (2, 3, 4)):
        raise ValueError('tile must be BM,BN,BK,warps,stages from the supported search space')
    return parts


def validate_settings(settings):
    if set(settings) - {'args', 'env'}:
        raise ValueError('settings only accept args and env')
    for key, value in settings.get('args', {}).items():
        if key in ARG_CHOICES:
            if value not in ARG_CHOICES[key]:
                raise ValueError(f'invalid tuning value for {key}: {value}')
        elif key in ARG_POSITIVE:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
            if key == 'bucket_token_budget' and int(value) != value:
                raise ValueError('token budget must be an integer')
        else:
            raise ValueError(f'not a tunable training argument: {key}')
    for key, value in settings.get('env', {}).items():
        if not isinstance(value, str):
            raise ValueError('environment tuning values must be strings')
        if key in ENV_CHOICES:
            if value not in ENV_CHOICES[key]:
                raise ValueError(f'invalid tuning value for {key}')
        elif key in ENV_POSITIVE:
            number = float(value)
            if not math.isfinite(number) or number <= 0:
                raise ValueError(f'{key} must be positive and finite')
            if key != 'INSTINCT_MUON_WORKSPACE_MB' and not number.is_integer():
                raise ValueError(f'{key} must be an integer')
        elif key in ENV_TILES:
            tile(value)
        else:
            raise ValueError(f'not a tunable environment variable: {key}')
    return settings


def changed(settings, section, key, value):
    result = {s: dict(settings.get(s, {})) for s in ('args', 'env')}
    result[section][key] = value
    return validate_settings(result)


def source_fingerprint():
    hashes = {}
    for directory in ('model', 'trainer', 'scripts/data_loader'):
        for path in sorted((ROOT / directory).rglob('*.py')):
            hashes[path.relative_to(ROOT).as_posix()] = file_digest(path)
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                 'vocab.json', 'merges.txt', 'added_tokens.json'):
        path = ROOT / 'model' / name
        if path.exists():
            hashes['model/' + name] = file_digest(path)
    return digest(hashes)


def hardware_fingerprint(device='cuda:0'):
    import torch
    if not torch.cuda.is_available() or not str(device).startswith('cuda'):
        raise RuntimeError('autotuning currently requires one CUDA GPU')
    props = torch.cuda.get_device_properties(device)
    def version(package):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None
    try:
        driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
            '--format=csv,noheader'], text=True, timeout=10).strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        driver = None
    return dict(gpu=props.name, capability=list(torch.cuda.get_device_capability(device)),
        vram_bytes=props.total_memory, sm_count=props.multi_processor_count,
        os=platform.system(), os_release=platform.release(), cpu=platform.processor(),
        logical_cpus=os.cpu_count(), python=platform.python_version(), torch=torch.__version__,
        cuda=torch.version.cuda, driver=driver, triton=version('triton') or version('triton-windows'),
        torchao=version('torchao'), source=source_fingerprint())


def training_contract(args):
    from trainer.trainer_utils import config_from_args
    config = config_from_args(args).to_dict()
    config.pop('use_grad_checkpoint', None)
    contract = dict(model=config, arguments={k: getattr(args, k, None) for k in CONTRACT_ARGS},
                    context_env={k: os.environ.get(k) for k in sorted(CONTEXT_ENV)})
    # PretrainedConfig contains integer id2label keys. Compare the persisted
    # JSON representation consistently in parent, worker and production.
    return json.loads(json.dumps(contract, allow_nan=False))


def verify_profile_model(config, args):
    """A resume can restore a different topology after argument validation."""
    if not getattr(args, 'autotune_profile', None):
        return
    actual = config.to_dict()
    actual.pop('use_grad_checkpoint', None)
    actual = json.loads(json.dumps(actual, allow_nan=False))
    expected = read_json(args.autotune_profile)['contract']['model']
    if actual != expected:
        raise ValueError('restored model differs from the autotune calibration; retune this topology')


def audit_source(path, audit_path, provenance_path=None):
    """Bind an existing reviewed audit to the exact corpus; does not clean data."""
    path = Path(path)
    if path.suffix.lower() not in ('.jsonl', '.parquet') or not path.is_file():
        raise ValueError('autotune currently accepts one audited JSONL or Parquet file')
    audit = read_json(audit_path)
    expected = audit.get('output_sha256') or (audit.get('output') or {}).get('sha256')
    actual = file_digest(path)
    if not expected or actual != expected:
        raise ValueError('corpus audit SHA-256 mismatch or missing output hash')
    # Compiled Parquet needs its cleaning/provenance sidecar as well as the
    # container hash. Human review remains a prerequisite, not a keyword test.
    provenance = read_json(provenance_path or audit_path)
    if not provenance.get('output_sha256'):
        raise ValueError('provide the reviewed cleaning/provenance report with output_sha256')
    return dict(sha256=actual, size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                audit_sha256=file_digest(audit_path), provenance_sha256=file_digest(provenance_path or audit_path))


def apply_profile(args):
    """Apply a validated pretrain profile before CUDA allocator/model setup."""
    path = getattr(args, 'autotune_profile', None)
    if not path:
        return
    if Path(sys.argv[0]).name != 'train_pretrain.py' or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('this autotune profile supports single-GPU train_pretrain.py only')
    if getattr(args, 'autotune_trial', None):
        raise ValueError('cannot combine a production profile and an autotune trial')
    if not args.from_resume and args.from_weight != 'none':
        raise ValueError('this calibration is for from-scratch pretraining; use from_weight=none')
    profile = read_json(path)
    if profile.get('schema') != SCHEMA or profile.get('status') != 'validated':
        raise ValueError('autotune profile is not validated')
    settings = validate_settings(profile['settings'])
    if hardware_fingerprint(args.device) != profile['hardware']:
        raise ValueError('autotune hardware/software/source fingerprint changed; retune this machine')
    if training_contract(args) != profile['contract']:
        raise ValueError('autotune training/model/data-pipeline configuration changed; retune')
    if file_digest(args.data_path) != profile['source']['sha256']:
        raise ValueError('autotune corpus changed; retune')
    supplied = {token.split('=', 1)[0] for token in sys.argv[1:] if token.startswith('--')}
    for key, value in settings['args'].items():
        if key == 'bucket_token_budget' and args.from_resume:
            continue  # configure_bucket_memory_budget inherits the saved cursor's budget.
        if '--' + key in supplied and getattr(args, key) != value:
            raise ValueError(f'explicit --{key} conflicts with validated autotune profile')
        setattr(args, key, value)
    for key, value in settings['env'].items():
        if key in os.environ and os.environ[key] != value:
            raise ValueError(f'{key} conflicts with validated autotune profile')
        os.environ[key] = value
    print(f'[Autotune] loaded {path}; resume keeps its saved batch budget', flush=True)


def quality_gate(reference, candidate, *, loss_limit=.02, validation_limit=.01):
    if candidate.get('status') != 'ok':
        return False, candidate.get('status', 'missing result')
    for key in ('initial_weights_sha256', 'validation_sha256', 'contract'):
        if reference.get(key) != candidate.get(key):
            return False, f'{key} differs'
    a, b = reference['losses'], candidate['losses']
    if len(a) != len(b) or not a or not all(math.isfinite(x) for x in a + b):
        return False, 'invalid training losses'
    for start in range(0, len(a), 20):
        aa, bb = a[start:start+20], b[start:start+20]
        if (sum(bb)-sum(aa)) / max(abs(sum(aa)), 1e-12) > loss_limit:
            return False, 'training loss gate failed'
    va, vb = reference['validation_final'], candidate['validation_final']
    if not math.isfinite(vb) or (vb-va) / max(abs(va), 1e-12) > validation_limit:
        return False, 'validation NLL gate failed'
    if not math.isfinite(candidate['useful_tokens_per_second']) or candidate['useful_tokens_per_second'] <= 0:
        return False, 'invalid throughput'
    return True, 'passed; equal updates, not equal tokens or a convergence proof'
