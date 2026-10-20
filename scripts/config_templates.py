"""Complete Config WebUI snapshots and an atomic, local template library."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


TEMPLATES_FILE = Path(__file__).resolve().parents[1] / 'trainer' / 'webui_templates.json'
DEFAULT_TEMPLATES_FILE = Path(__file__).with_name('config_template_defaults.json')
PRIVATE_CONFIG_KEYS = {
    '_override_moe_int', '_override_lin_kd', '_override_lin_vd',
    '_sft_active_profile', '_sft_last_train_type', '_panel_synced_base',
}
RUNTIME_KEYS = {
    'train_proc', 'train_status', 'train_log_path', 'train_executed_cmd',
    'train_triggered', 'shutdown_deadline', 'pipeline_run_dir',
    'autotune_run_dir', 'autotune_result', 'history_log_select',
    'clear_train_log', 'pipeline_upload', 'preset_fingerprint', 'config_template',
}


def is_config_key(key):
    return (
        isinstance(key, str)
        and (not key.startswith('_') or key in PRIVATE_CONFIG_KEYS)
        and not key.startswith(('btn_', 'save_prefix_'))
        and key not in RUNTIME_KEYS
    )


def snapshot_config(state):
    """Copy all JSON configuration, including inactive stages and nested plans."""
    result = {}
    for key, value in state.items():
        if not is_config_key(key):
            continue
        try:
            result[key] = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError):
            continue
    return result


def restore_config(state, snapshot):
    """Replace configuration before rendering widgets; retain live process state."""
    config = snapshot_config(snapshot)
    # Remove absent settings and widget proxies so the previous template cannot
    # leak optional dimensions, topology radios or early-exit text into this one.
    for key in list(state):
        if is_config_key(key) or key in {
            '_arch_radio', '_residual_radio', '_early_exit_layers_str', '_preset_radio',
            '_sync_arch_radio', '_rerun_after_sft_switch',
        } or key.startswith('_pending_widget_'):
            del state[key]
    state.update(config)
    state.setdefault('preset', 'Custom')


def _read_library(path):
    path = Path(path) if path is not None else TEMPLATES_FILE
    if not path.exists():
        return {'version': 1, 'templates': {}}
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('templates'), dict):
        raise ValueError('模板文件格式无效')
    templates = data['templates']
    if any(not isinstance(name, str) or not name.strip() or not isinstance(config, dict)
           for name, config in templates.items()):
        raise ValueError('模板名称或配置格式无效')
    return data


def load_templates(path=None):
    return _read_library(path)['templates']


def load_model_presets():
    """Read model definitions from the same catalog used by template recipes."""
    return _read_library(DEFAULT_TEMPLATES_FILE)['presets']


def load_legacy_preset_names():
    return _read_library(DEFAULT_TEMPLATES_FILE).get('legacy_preset_names', {})


def _expand_default_templates(catalog):
    presets = catalog.get('presets', {})
    defaults = catalog.get('defaults', {})
    return {
        name: defaults | presets.get(name, {})
        | ({'preset': name} if name in presets else {}) | overrides
        for name, overrides in catalog['templates'].items()
    }


def load_default_templates():
    """Expand shared defaults, model parameters and per-template overrides."""
    return _expand_default_templates(_read_library(DEFAULT_TEMPLATES_FILE))


def preset_initial_state(name):
    """CPU-only preset initialization shared by Config WebUI and relay export."""
    catalog = _read_library(DEFAULT_TEMPLATES_FILE)
    name = catalog.get('legacy_preset_names', {}).get(name, name)
    model = catalog.get('presets', {}).get(name)
    if model is None:
        return None
    intermediate = math.ceil(model['hidden_size'] * math.pi / 64) * 64
    recurrent = model.get('model_architecture') == 'recurrsive_instinct'
    state = dict(
        residual_type='standard',
        intermediate_size=model.get('intermediate_size', intermediate),
        qk_bias=model.get('qk_bias', recurrent),
        qk_norm=model.get('qk_norm', not recurrent),
        _override_moe_int=bool(model.get('use_moe')
                              and model.get('moe_intermediate_size', intermediate) != intermediate),
    )
    state.update(model)
    if model.get('rope_scaling_type') == 'longrope':
        state.update(
            longrope_short_factors=json.dumps(model['short_factor']),
            longrope_long_factors=json.dumps(model['long_factor']),
            longrope_short_retained_tokens=model.get('short_retained_start_tokens', 0),
            longrope_long_retained_tokens=model.get('long_retained_start_tokens', 0),
            longrope_short_attention_factor=float(model.get('short_attention_factor', model.get('attention_factor', 1.0))),
            longrope_long_attention_factor=float(model.get('long_attention_factor', model.get('attention_factor', 0.0))),
        )
    training_keys = catalog.get('preset_training_keys', {}).get(name, ())
    if training_keys:
        recipe = _expand_default_templates(catalog)[name]
        state.update({key: recipe[key] for key in training_keys})
    return state


def write_templates(templates, path=None, *, defaults_initialized=None,
                    default_template_fingerprints=None):
    path = Path(path) if path is not None else TEMPLATES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    library = _read_library(path)
    library['templates'] = templates
    if defaults_initialized is not None:
        library['defaults_initialized'] = defaults_initialized
    if default_template_fingerprints is not None:
        library['default_template_fingerprints'] = default_template_fingerprints
    payload = json.dumps(library, ensure_ascii=False,
                         indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = handle.name
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def initialize_default_templates(path=None):
    """Add new defaults and refresh untouched ones; retain edits and deletions."""
    library = _read_library(path)
    catalog = _read_library(DEFAULT_TEMPLATES_FILE)
    defaults = _expand_default_templates(catalog)
    templates = dict(library['templates'])
    fingerprints = library.get('default_template_fingerprints')
    if fingerprints is None:
        fingerprints = (
            # Missing entries from the pre-fingerprint catalog are intentional deletions.
            dict.fromkeys(catalog.get('legacy_default_template_names', ()))
            if library.get('defaults_initialized') else {}
        )
    else:
        fingerprints = dict(fingerprints)

    def fingerprint(config):
        payload = json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    repo_root = Path(__file__).resolve().parents[1]
    for name, config in defaults.items():
        config = snapshot_config(config)
        # Shipped snapshots use portable dataset paths. The selector scans with
        # absolute paths, so resolve them when creating this machine's library.
        for key, value in config.items():
            if key.startswith('data_path_') and isinstance(value, str) and value:
                dataset_path = Path(value)
                if not dataset_path.is_absolute():
                    config[key] = str(repo_root / dataset_path)
        if name not in fingerprints:
            templates.setdefault(name, config)
        elif name in templates and fingerprint(templates[name]) == fingerprints[name]:
            templates[name] = config
        fingerprints[name] = fingerprint(config)
    if (templates != library['templates']
            or fingerprints != library.get('default_template_fingerprints')
            or not library.get('defaults_initialized')):
        write_templates(templates, path, defaults_initialized=True,
                        default_template_fingerprints=fingerprints)


def save_template(name, state, *, overwrite=False, path=None):
    name = name.strip()
    if not name:
        raise ValueError('请输入模板名称')
    templates = load_templates(path)
    if name in templates and not overwrite:
        raise ValueError('同名模板已存在，请使用其他名称或点击更新所选模板')
    templates[name] = snapshot_config(state)
    write_templates(templates, path)
    return name


def delete_template(name, path=None):
    templates = load_templates(path)
    templates.pop(name, None)
    write_templates(templates, path)
