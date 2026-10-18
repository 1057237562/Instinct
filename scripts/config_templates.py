"""Complete Config WebUI snapshots and an atomic, local template library."""

import json
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


def write_templates(templates, path=None, *, defaults_initialized=None):
    path = Path(path) if path is not None else TEMPLATES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    library = _read_library(path)
    library['templates'] = templates
    if defaults_initialized is not None:
        library['defaults_initialized'] = defaults_initialized
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
    """Import the former presets once; retain edits, names and later deletions."""
    library = _read_library(path)
    if library.get('defaults_initialized'):
        return
    defaults = load_templates(DEFAULT_TEMPLATES_FILE)
    templates = library['templates']
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
        templates.setdefault(name, config)
    write_templates(templates, path, defaults_initialized=True)


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
