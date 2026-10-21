"""CPU-only Muon settings shared by the panel, launchers and trainer setup."""
import math
import os


PERFORMANCE_OPTIONS = {
    'muon_batch_size': 'INSTINCT_MUON_BATCH_SIZE',
    'muon_workspace_mb': 'INSTINCT_MUON_WORKSPACE_MB',
}


def precision_environment(arguments=None, environment=None):
    """Canonical effective precision for fingerprints and early validation."""
    arguments = arguments or {}
    environment = os.environ if environment is None else environment
    split = arguments.get('muon_mantissa')
    raw = str(int(split)) if split is not None else str(environment.get('INSTINCT_MUON_MANTISSA') or '0')
    raw = raw.strip().lower()
    if raw not in ('0', 'false', 'no', '1', 'true', 'yes'):
        raise ValueError('INSTINCT_MUON_MANTISSA must be 0 or 1')
    momentum = (arguments.get('muon_momentum_dtype')
                or environment.get('INSTINCT_MUON_MOMENTUM_DTYPE') or 'fp32').strip().lower()
    if momentum not in ('fp32', 'bf16'):
        raise ValueError('Muon momentum dtype must be fp32 or bf16')
    return {'INSTINCT_MUON_MANTISSA': '1' if raw in ('1', 'true', 'yes') else '0',
            'INSTINCT_MUON_MOMENTUM_DTYPE': momentum}


def performance_value(key, value):
    number = float(value)
    if not math.isfinite(number) or number < 0 or (key == 'muon_batch_size' and not number.is_integer()):
        raise ValueError(f'{key} 必须是非负有限数值，矩阵数必须为整数')
    return format(number, 'g')


def panel_arguments(state):
    """Freeze explicit precision choices; disabled controls retain their UI values."""
    muon = state.get('optimizer', 'adamw') == 'muon'
    split = muon and bool(state.get('muon_mantissa', False))
    if split and state.get('param_dtype', 'fp32') != 'bf16':
        raise ValueError('Muon 补偿权重需要将参数精度设为 bf16。')
    momentum = state.get('muon_momentum_dtype', 'fp32') if split else 'fp32'
    if momentum not in ('fp32', 'bf16'):
        raise ValueError('Muon 动量精度必须为 fp32 或 bf16。')
    result = dict(muon_mantissa=int(split), muon_momentum_dtype=momentum)
    for key in PERFORMANCE_OPTIONS:
        value = state.get(key, 0) if muon else 0
        performance_value(key, value)
        result[key] = int(value) if key == 'muon_batch_size' else float(value)
    return result


def use_chunk16(state):
    """Explicit button callback; does not enable low precision."""
    state.update(muon_batch_size=16, muon_workspace_mb=272.0)


def extract_performance_options(arguments, environment):
    """Move CLI aliases into autotune's existing environment search axes."""
    for key, variable in PERFORMANCE_OPTIONS.items():
        value = arguments.pop(key, None)
        if value is None:
            continue
        encoded = performance_value(key, value)
        if float(encoded) > 0:
            if variable in environment and float(environment[variable]) != float(encoded):
                raise ValueError(f'{key} 与 {variable} 的调优设置冲突')
            environment[variable] = encoded


def apply_cli_options(args):
    """CLI overrides env only when supplied; zero performance values mean auto."""
    environment = dict(os.environ)
    split = getattr(args, 'muon_mantissa', None)
    momentum = getattr(args, 'muon_momentum_dtype', None)
    if split is not None:
        environment['INSTINCT_MUON_MANTISSA'] = str(int(split))
    if momentum is not None:
        environment['INSTINCT_MUON_MOMENTUM_DTYPE'] = momentum
    for key, variable in PERFORMANCE_OPTIONS.items():
        value = getattr(args, key, None)
        if value is not None:
            encoded = performance_value(key, value)
            if float(encoded) > 0:
                environment[variable] = encoded
    precision = precision_environment(environment=environment)
    enabled = precision['INSTINCT_MUON_MANTISSA'] == '1'
    momentum = precision['INSTINCT_MUON_MOMENTUM_DTYPE']
    if enabled and (args.optimizer != 'muon' or args.param_dtype != 'bf16'):
        raise ValueError('Muon mantissa requires --optimizer muon --param_dtype bf16')
    if momentum == 'bf16' and not enabled:
        raise ValueError('BF16 momentum requires Muon mantissa')
    if enabled and environment.get('INSTINCT_MUON_BACKEND', 'batched').strip().lower() != 'batched':
        raise ValueError('Muon mantissa requires INSTINCT_MUON_BACKEND=batched')
    for variable in ('INSTINCT_MUON_MANTISSA', 'INSTINCT_MUON_MOMENTUM_DTYPE', *PERFORMANCE_OPTIONS.values()):
        if variable in environment:
            os.environ[variable] = environment[variable]
