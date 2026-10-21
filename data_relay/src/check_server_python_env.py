"""Check and optionally repair the instance Python env (numpy/pyarrow ABI mismatch).

Run through data_relay remote-script. Inspection is the default and read-only:
it probes actual imports plus dist metadata, prints one ENV_REPORT JSON line,
and exits 0 when healthy / 1 when the env is broken. Flip APPLY_REPAIR (or run
manually with --apply) to reinstall the broken pair; repair refuses to run
while a trainer process is active. Stdlib only, so the probes work even when
importing numpy or pyarrow itself crashes.
"""
import datetime
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig

ROOT = Path('/root/autodl-tmp')
APPLY_REPAIR = False
NUMPY_SPEC = None  # explicit constraint, e.g. 'numpy<2' or 'numpy==1.26.4'; auto-derived when None
PROBED = ('numpy', 'pyarrow', 'pandas')
META_ONLY = ('torch', 'datasets', 'streamlit')


def import_probe(name):
    try:
        module = importlib.import_module(name)
    except Exception as error:
        return {'import_ok': False, 'imported_version': None,
                'error': type(error).__name__ + ': ' + str(error)}
    return {'import_ok': True, 'imported_version': getattr(module, '__version__', None), 'error': None}


def numpy_core_probe():
    for module in ('numpy._core.multiarray', 'numpy.core.multiarray'):
        try:
            importlib.import_module(module)
        except Exception:
            continue
        return {'core_ok': True, 'core_module': module}
    return {'core_ok': False, 'core_module': None}


def dist_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def scan_duplicates(site_packages):
    found = {name: [] for name in PROBED}
    site_packages = Path(site_packages)
    if not site_packages.is_dir():
        return found
    for item in sorted(site_packages.iterdir()):
        lowered = item.name.lower()
        for package in PROBED:
            if lowered.startswith(package + '-') and lowered.endswith(('.dist-info', '.egg-info', '.egg-link')):
                found[package].append(item.name)
    return found


def conda_meta_records(env_prefix):
    directory = Path(env_prefix) / 'conda-meta'
    if not directory.is_dir():
        return {name: [] for name in PROBED}
    return {name: sorted(item.name for item in directory.glob(name + '-*.json')) for name in PROBED}


def collect_state():
    numpy_state = import_probe('numpy')
    numpy_state.update(dist_version=dist_version('numpy'), **numpy_core_probe())
    return {
        'python': {'executable': sys.executable, 'version': '.'.join(map(str, sys.version_info[:3]))},
        'numpy': numpy_state,
        'pyarrow': {**import_probe('pyarrow'), 'dist_version': dist_version('pyarrow')},
        'pandas': {**import_probe('pandas'), 'dist_version': dist_version('pandas')},
        'others': {name: dist_version(name) for name in META_ONLY},
        'install': {'site_packages': str(Path(sysconfig.get_path('purelib') or '.')),
                    'duplicate_dists': scan_duplicates(sysconfig.get_path('purelib') or '.'),
                    'conda_meta': conda_meta_records(sys.prefix)},
    }


def fresh_state():
    for key in [name for name in list(sys.modules) if name.split('.')[0] in PROBED]:
        del sys.modules[key]
    return collect_state()


def default_numpy_spec(state):
    records = state['install']['duplicate_dists'].get('numpy', [])
    version = state['numpy'].get('dist_version')
    if version and len(records) <= 1:
        return 'numpy==' + version
    return 'numpy<2'


def plan_repair(state, numpy_spec=None):
    """Pure decision over a collect_state() dict; unit-tested without touching the env."""
    numpy_ok = state['numpy']['import_ok'] and state['numpy']['core_ok']
    pyarrow_ok = state['pyarrow']['import_ok']
    duplicates = state['install']['duplicate_dists']
    spec = None
    if numpy_ok and pyarrow_ok and state['pandas']['import_ok']:
        diagnosis = '环境正常：numpy {0} 与 pyarrow {1} 均可导入，无需处理。'.format(
            state['numpy'].get('imported_version'), state['pyarrow'].get('imported_version'))
        return {'status': 'healthy', 'actions': [], 'numpy_spec': None, 'diagnosis': diagnosis}
    if numpy_ok:
        if pyarrow_ok:
            status = 'pandas_mismatch'
            actions = []
            if len(duplicates.get('pandas', [])) > 1:
                actions.append(('uninstall', 'pandas'))
            if state['pandas'].get('dist_version'):
                actions.append(('upgrade', 'pandas'))
            else:
                actions.append(('install', 'pandas'))
            diagnosis = 'pandas 无法导入（{0}）而 numpy/pyarrow 正常，单独重装 pandas 即可。'.format(
                state['pandas'].get('error'))
        else:
            status = 'pyarrow_mismatch'
            actions = []
            if len(duplicates.get('pyarrow', [])) > 1:
                actions.append(('uninstall', 'pyarrow'))
            if state['pyarrow'].get('dist_version'):
                actions.append(('upgrade', 'pyarrow'))
            else:
                actions.append(('install', 'pyarrow'))
            diagnosis = ('pyarrow {0} 无法导入（{1}），而 numpy {2} 本身正常：典型的 numpy/pyarrow 二进制'
                         '不匹配（pip 升降级过 numpy，或 pip/conda 混装）。升级 pyarrow 即可，不要动 numpy。'.format(
                    state['pyarrow'].get('dist_version'), state['pyarrow'].get('error'),
                    state['numpy'].get('imported_version')))
    else:
        spec = numpy_spec or default_numpy_spec(state)
        status = 'numpy_broken'
        actions = [('uninstall', 'pyarrow', 'numpy'), ('install', spec, 'pyarrow')]
        diagnosis = ('numpy {0} 自身损坏（{1}；常见于 pip/conda 混装或升级到一半），连带 pyarrow 无法导入。'
                     '将卸载两者，按 {2} 重装 numpy 并安装最新 pyarrow。'.format(
                state['numpy'].get('dist_version'), state['numpy'].get('error'), spec))
    if state['install']['conda_meta']['numpy']:
        diagnosis += ' 注意：conda 也管理过 numpy，之后请固定只用 pip（或只用 conda）安装，避免再次混装。'
    return {'status': status, 'actions': actions, 'numpy_spec': spec, 'diagnosis': diagnosis}


def render_commands(actions):
    commands = []
    for action in actions:
        pip = [sys.executable, '-m', 'pip']
        if action[0] == 'uninstall':
            commands.append(pip + ['uninstall', '-y', *action[1:]])
        elif action[0] == 'upgrade':
            commands.append(pip + ['install', '--no-cache-dir', '--upgrade', *action[1:]])
        elif action[0] == 'install':
            commands.append(pip + ['install', '--no-cache-dir', *action[1:]])
        else:
            raise ValueError('Unknown action: ' + repr(action))
    return commands


def apply_commands(commands):
    log = []
    for command in commands:
        rendered = ' '.join(shlex.quote(part) for part in command)
        print('RUN ' + rendered, flush=True)
        completed = subprocess.run(command, capture_output=True, text=True, errors='replace')
        entry = {'command': rendered, 'returncode': completed.returncode,
                 'stdout_tail': completed.stdout[-2000:], 'stderr_tail': completed.stderr[-2000:]}
        if completed.returncode != 0:
            entry['failed'] = True
        log.append(entry)
        print('PIP exit {0}: {1}'.format(completed.returncode,
              (completed.stderr or completed.stdout).strip()[-500:]), flush=True)
        if completed.returncode != 0:
            break
    return log


def training_active():
    if os.name != 'posix' or not Path('/proc').is_dir():
        return False
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (OSError, PermissionError):
            continue
        if any(Path(arg).name.startswith('train_') and arg.endswith('.py') for arg in args):
            return True
    return False


def write_audit(report):
    if not ROOT.is_dir():
        return
    audit = ROOT / '.cache' / 'relay_python_env'
    audit.mkdir(parents=True, exist_ok=True)
    output = audit / (datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('AUDIT_PATH ' + str(output), flush=True)


def main():
    args = [item for item in sys.argv[1:] if item]
    apply_flag = APPLY_REPAIR or '--apply' in args
    numpy_spec = NUMPY_SPEC
    if '--numpy-spec' in args:
        numpy_spec = args[args.index('--numpy-spec') + 1]

    report = {'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              **collect_state(), 'applied': False, 'repair_log': [], 'post': None}
    plan = plan_repair(report, numpy_spec)
    report['status'] = plan['status']
    report['diagnosis'] = plan['diagnosis']
    report['numpy_spec'] = plan['numpy_spec']
    report['actions'] = [list(action) for action in plan['actions']]
    report['commands'] = [' '.join(shlex.quote(part) for part in command)
                          for command in render_commands(plan['actions'])]
    print(plan['diagnosis'], flush=True)
    for command in report['commands']:
        print('PLAN ' + command, flush=True)

    if plan['status'] != 'healthy' and apply_flag:
        if training_active():
            raise ValueError('Training is active; skip environment repair')
        report['applied'] = True
        report['repair_log'] = apply_commands(render_commands(plan['actions']))
        report['post'] = fresh_state()
        post_plan = plan_repair(report['post'], numpy_spec)
        report['status'] = post_plan['status']
        report['diagnosis'] = post_plan['diagnosis']
        print('修复后状态: {0} — {1}'.format(post_plan['status'], post_plan['diagnosis']), flush=True)

    report['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print('ENV_REPORT ' + json.dumps(report, ensure_ascii=False), flush=True)
    write_audit(report)
    return 0 if report['status'] == 'healthy' else 1


if __name__ == '__main__':
    sys.exit(main())
