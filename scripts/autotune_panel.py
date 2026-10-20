"""Config WebUI integration for the existing isolated pretraining autotuner.

This module and its worker stay CPU-only. CUDA, corpus hashing and the full
profile compatibility checks run in the existing trainer child processes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
try:
    from scripts.autotune_bootstrap import prepare_imports
except ModuleNotFoundError:
    from autotune_bootstrap import prepare_imports

# CPU-only; the Windows lock currently affects this local checkout. Ordinary
# accessible checkouts keep Python's standard FileFinder unchanged.
prepare_imports()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def resolve_path(value, root=ROOT):
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(root) / path


def cli_values(command):
    """Read the shared WebUI builder's argv without a second training builder."""
    result = {}
    i = 0
    while i < len(command):
        token = command[i]
        if token.startswith('--'):
            name = token[2:]
            if i + 1 == len(command) or command[i + 1].startswith('--'):
                result[name] = True
            else:
                value = command[i + 1]
                try:
                    value = json.loads(value)
                except (ValueError, TypeError):
                    pass
                result[name] = value
                i += 1
        i += 1
    return result


def requirements(state, model):
    """Show all actionable prerequisites, using the same checks as launch."""
    return [
        (state.get('train_type', 'pretrain') == 'pretrain', '从零 Pretrain',
         '目前自动调优仅支持单卡从零预训练；CPT、SFT 和 RL 请使用各自训练配置。'),
        (model.get('model_architecture', 'standard') == 'standard', '标准 Dense / MoE',
         '目前自动调优支持标准 Dense / MoE 架构。'),
        (not state.get('from_resume', False), '未启用断点续训',
         '请关闭 Resume 后校准；实验使用独立随机初始化，不读取或覆盖续训状态。'),
        (bool(state.get('sequence_packing')) and state.get('sequence_packing_mode') == 'bucket', 'Bucket packing',
         '请开启 sequence packing 并选择 bucket 模式。'),
        (state.get('use_grad_checkpoint', 0) in (1, 2), 'Gradient checkpointing',
         '请开启 gradient checkpointing（模式 1 或 2）。'),
        (state.get('dataset_streaming', 'auto') == 'on' and state.get('cache_build_mode', 'inline') == 'inline', '流式加载 + Inline 缓存',
         '请使用 dataset streaming=on、cache build=inline，以匹配调优实验和生产数据管线。'),
        (bool(state.get('data_path_pretrain')), '已选择预训练数据',
         '请先选择预训练数据集。'),
    ]


def eligibility(state, model):
    return next((reason for ready, _, reason in requirements(state, model) if not ready), None)


def prepare_settings(state):
    """Widget callback: only set execution prerequisites, before the next render."""
    state.update(sequence_packing=True, sequence_packing_mode='bucket',
                 dataset_streaming='on', cache_build_mode='inline')
    if state.get('use_grad_checkpoint', 0) not in (1, 2):
        state['use_grad_checkpoint'] = 1


def make_spec(command, model_path, audit='', provenance='', environment=None):
    if environment is not None and not isinstance(environment, dict):
        raise ValueError('性能环境参数 JSON 必须是对象')
    args = cli_values(command)
    for key in ('autotune_profile', 'autotune_trial', 'save_weight', 'save_dir', 'pause_file', 'save_interval'):
        args.pop(key, None)
    if args.get('from_resume') or args.get('from_weight') != 'none':
        raise ValueError('自动调优需要从零预训练配置')
    args.pop('from_resume', None)
    args['config_path'] = str(model_path)
    args['profile'] = 'off'
    args['bucket_gpu_memory_gb'] = 'auto'
    args.setdefault('bucket_token_budget', 0)
    try:
        from scripts.muon_controls import extract_performance_options
    except ModuleNotFoundError:
        from muon_controls import extract_performance_options
    environment = dict(environment or {})
    extract_performance_options(args, environment)
    result = dict(args=args, environment=environment)
    if audit:
        result['audit'] = audit
    if provenance:
        result['provenance'] = provenance
    return result


def tuning_command(spec_path, output, state, *, plan_only=False):
    command = [sys.executable, '-u', str(ROOT / 'trainer' / 'autotune.py'),
               '--config', str(spec_path), '--output', str(output)]
    for key, default in (('max_trials', 24), ('warmup', 20), ('steps', 60),
                         ('validation_examples', 16)):
        command.extend(['--' + key.replace('_', '-'), str(state.get('autotune_' + key, default))])
    for key in ('peak_tflops', 'target_tokens_per_second'):
        value = state.get('autotune_' + key, 0)
        if value:
            command.extend(['--' + key.replace('_', '-'), str(value)])
    if state.get('autotune_refresh', False):
        command.append('--refresh')
    if plan_only:
        command.append('--plan-only')
    if prepare_imports():
        bootstrap = str(ROOT / 'scripts' / 'autotune_bootstrap.py')
        command.insert(2, bootstrap)
        command.extend(['--worker-launcher', bootstrap])
    return command


def profile_command(command, profile_path, train_type, model):
    """Leave performance settings to the profile; keep all math/data settings."""
    if not profile_path:
        return command
    if train_type != 'pretrain' or model.get('model_architecture', 'standard') != 'standard':
        raise ValueError('调优 profile 仅适用于标准架构单卡预训练')
    profile = read_json(resolve_path(profile_path))
    if profile.get('schema') != 1 or profile.get('status') not in ('validated', 'manual'):
        raise ValueError('profile 尚未通过训练与验证验收，也未标记为手动固定配置')
    # Lightweight validation only. The trainer checks hardware, complete model,
    # corpus SHA-256 and the training contract before allocating the model.
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from trainer.autotune_config import source_fingerprint, validate_settings
    settings = validate_settings(profile['settings'])
    if profile['hardware']['source'] != source_fingerprint():
        raise ValueError('训练代码已变化，请重新调优后再应用 profile')
    values = cli_values(command)
    contract = profile.get('contract', {})
    try:
        from scripts.muon_controls import PERFORMANCE_OPTIONS, precision_environment
    except ModuleNotFoundError:
        from muon_controls import PERFORMANCE_OPTIONS, precision_environment
    if precision_environment(values) != precision_environment(environment=contract.get('context_env', {})):
        raise ValueError('Muon 权重或动量精度与 profile 不一致；请恢复实验配置或重新调优')
    mismatches = [key for key, value in contract.get('arguments', {}).items()
                  if key in values and values[key] != value]
    mismatches += [key for key, value in model.items()
                   if key != 'use_grad_checkpoint' and key in contract.get('model', {})
                   and contract['model'][key] != value]
    if mismatches:
        raise ValueError('面板与 profile 配置不一致：' + ', '.join(sorted(set(mismatches))) + '；请恢复实验配置或重新调优')
    for key, value in settings['env'].items():
        if key in os.environ and os.environ[key] != value:
            raise ValueError(f'{key} 环境变量与已验收 profile 冲突')
    remove = {'--' + key for key in settings['args']} | {'--autotune_profile'}
    remove.update('--' + key for key, variable in PERFORMANCE_OPTIONS.items() if variable in settings['env'])
    result = []
    i = 0
    while i < len(command):
        if command[i] in remove:
            i += 2
        else:
            result.append(command[i])
            i += 1
    result.extend(['--autotune_profile', str(resolve_path(profile_path).resolve())])
    if prepare_imports():
        # Place the bootstrap before the trainer, after interpreter options.
        for index, argument in enumerate(result[1:], 1):
            if str(argument).endswith('train_pretrain.py'):
                result.insert(index, str(ROOT / 'scripts' / 'autotune_bootstrap.py'))
                break
    return result


def run_status(run_dir):
    """Recover a job after browser reload, verifying PID identity against reuse."""
    if not run_dir:
        return {}
    try:
        status = read_json(Path(run_dir) / 'run.json')
    except (OSError, ValueError):
        try:
            status = dict(read_json(Path(run_dir) / 'owner.json'), status='running')
        except (OSError, ValueError):
            return {}
    if status.get('status') == 'running':
        import psutil
        try:
            process = psutil.Process(status['pid'])
            alive = (abs(process.create_time() - status['create_time']) < .01
                     and any(Path(arg).name == 'autotune_webui_worker.py' for arg in process.cmdline()))
        except (psutil.Error, KeyError):
            alive = False
        if not alive:
            status['status'] = 'interrupted'
    return status


def active(state):
    return run_status(state.get('autotune_run_dir')).get('status') == 'running'


def launch(run_dir, command):
    """Detached supervisor retains logs/status and owns cancellation cleanup."""
    run_dir = Path(run_dir)
    write_json(run_dir / 'request.json', dict(command=command))
    options = dict(cwd=ROOT, env=dict(os.environ, PYTHONUTF8='1', INSTINCT_MANAGED_DATA_CACHE='1'))
    if os.name == 'nt':
        options['creationflags'] = subprocess.CREATE_NO_WINDOW
    else:
        options['start_new_session'] = True
    with (run_dir / 'worker.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scripts' / 'autotune_webui_worker.py'),
                                    '--run-dir', str(run_dir)], stdout=log, stderr=subprocess.STDOUT, **options)
    # Separate ownership and status files avoid a startup race overwriting the
    # supervisor's already completed result with the parent's running record.
    import psutil
    try:
        identity = psutil.Process(process.pid).create_time()
    except psutil.NoSuchProcess:
        return process
    write_json(run_dir / 'owner.json', dict(pid=process.pid, create_time=identity))
    return process


def profiles(output):
    paths = list(Path(output).glob('*/profile.json'))
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def read_log_tail(path, limit=64000):
    """Read a bounded tail even while a compiler is appending a large log."""
    with Path(path).open('rb') as handle:
        size = handle.seek(0, os.SEEK_END)
        offset = max(0, size - limit)
        handle.seek(offset)
        text = handle.read(limit).decode('utf-8', errors='replace')
    if offset and '\n' in text:
        text = text.split('\n', 1)[1]
    return text


def current_trial_log(controller_log):
    """Follow this controller's last launch, including already-running jobs."""
    matches = re.findall(r'^\[Autotune\].*; log=([^\r\n]+)', controller_log, re.MULTILINE)
    return resolve_path(matches[-1].strip()) if matches else None


def render_progress(st, run_dir):
    status = run_status(run_dir)
    if not status:
        st.caption('还没有调优任务。先预览搜索计划，再开始实测；进度与日志会显示在这里。')
        return
    labels = dict(running='运行中', success='本次任务完成', failed='失败',
                  interrupted='已中断', cancelled='已停止')
    st.write('调优状态：' + labels.get(status['status'], status['status']))
    st.caption('计划预览完成只表示配置检查结束；只有实测并通过复测才会产生下方 profile。')
    if status.get('error'):
        st.error(status['error'])
    if status['status'] == 'running':
        st.caption('每 2 秒自动刷新已写入磁盘的日志；环境探测 probe 不测吞吐，tokens/s=None 表示尚无吞吐数据。')
        if st.button('停止自动调优', key='btn_autotune_stop'):
            (Path(run_dir) / 'stop').touch()
            st.rerun()
    log = Path(run_dir) / 'controller.log'
    if not log.exists():
        log = Path(run_dir) / 'worker.log'
    try:
        controller = read_log_tail(log, limit=256000)
    except FileNotFoundError:
        controller = ''
    except OSError as exc:
        controller = ''
        st.warning(f'暂时无法读取调度日志：{exc}')
    trial_log = current_trial_log(controller)
    if trial_log is not None:
        st.markdown('**当前 / 最近试跑日志**')
        st.caption(f'试跑：{trial_log.parent.name} · {trial_log}')
        try:
            trial_status = read_json(trial_log.parent / 'result.json')
            trial_labels = dict(running='运行中', probe='环境探测完成', ok='测量完成',
                                oom='显存不足', driver_budget_exceeded='超过驱动显存预算',
                                process_error='进程异常退出')
            result_status = trial_status.get('status', '未记录')
            if result_status == 'running' and status['status'] != 'running':
                result_status = labels.get(status['status'], status['status'])
            st.write('试跑状态：' + trial_labels.get(result_status, result_status))
            if trial_status.get('error'):
                st.error(trial_status['error'])
        except (OSError, ValueError):
            pass
        try:
            memory = read_json(trial_log.parent / 'memory.json')
            samples = memory.get('samples', [])
            if samples:
                latest = samples[-1]
                metrics = st.columns(4)
                for column, label, value in zip(metrics,
                        ('实际分配', 'PyTorch 预留', '预留减分配', '全程预留峰值'),
                        (latest['allocated_gib'], latest['reserved_gib'],
                         latest['unallocated_reserved_gib'], memory['trial_peak_reserved_gib'])):
                    column.metric(label, f'{value:.2f} GiB')
                st.caption(f'显存阶段：{latest["phase"]} · '
                           f'不可释放的拆分块：{latest["inactive_split_gib"]:.2f} GiB。'
                           '预留减分配不代表全部可回收；全程峰值包含首次编译/预热。')
                with st.expander('显存阶段记录'):
                    st.dataframe([{key: row.get(key) for key in (
                        'phase', 'allocated_gib', 'reserved_gib', 'inactive_split_gib',
                        'peak_allocated_gib', 'peak_reserved_gib', 'released_reserved_gib',
                        'allocation_retries', 'oom_count')} for row in samples], width='stretch')
        except (OSError, ValueError, KeyError, TypeError):
            pass
        try:
            tail = read_log_tail(trial_log)
            info = trial_log.stat()
            age = max(0, int(time.time() - info.st_mtime))
            st.caption(f'最近写入：{age} 秒前 · 文件大小：{info.st_size:,} 字节 · 下方显示末尾最多 64 KB')
            if tail:
                st.code(tail, language='text', height=420)
            else:
                st.info('试跑已创建，等待子进程写入日志。')
        except FileNotFoundError:
            st.info('试跑已创建，等待子进程写入日志。')
        except OSError as exc:
            st.warning(f'暂时无法读取试跑日志：{exc}')
    elif status['status'] == 'running':
        st.info('正在准备调优，等待首个试跑；调度输出可在下方查看。')
    with st.expander('调度日志', expanded=trial_log is None):
        if controller:
            st.code(controller[-12000:], language='text')
        else:
            st.caption('等待调度器写入日志。')
    if st.button('刷新调优进度', key='btn_autotune_progress'):
        st.rerun()


def _widget(st, kind, label, *, key, value, **kwargs):
    """Saved Session State wins without also supplying a widget default."""
    if key not in st.session_state:
        kwargs['value'] = value
    return getattr(st, kind)(label, key=key, **kwargs)


def render(st, model, build_command, training_active):
    state = st.session_state
    busy = active(state)
    pipeline = state.get('_pipeline_proc')
    occupied = training_active or (pipeline is not None and pipeline.poll() is None)
    st.subheader('找到这台训练机的合适配置')
    estimate, measured = st.columns(2, gap='large')
    with estimate, st.container(border=True):
        st.markdown('**动态显存估算** · 随参数更新')
        st.caption('在「训练配置 → 数据加载与动态批量」启用 Bucket packing，token budget 设为 0。启动时结合 GPU 和模型配置校准每桶 batch；续训沿用保存预算。')
    with measured, st.container(border=True):
        st.markdown('**自动调优** · 用真实数据实测')
        st.caption('短跑候选并比较有效 tokens/s，通过训练 loss、验证 NLL 和独立复测后生成 profile。结果是本次搜索范围内的最佳配置。')
    st.caption('支持单卡、标准 Dense / MoE、从零预训练。先检查条件 → 预览搜索计划 → 开始实测 → 应用结果 → 回到「训练配置」启动。')
    with st.expander('01 · 调优准备与二分搜索', expanded=True):
        checks = requirements(state, model)
        reason = eligibility(state, model)
        st.markdown('**当前配置检查**')
        check_columns = st.columns(2)
        for index, (ready, label, _) in enumerate(checks):
            check_columns[index % 2].markdown(f'{":material/check_circle:" if ready else ":material/radio_button_unchecked:"} {label}')
        if reason:
            st.info(reason)
        else:
            st.success('面板条件已满足，可以先预览搜索计划。')
        missing_execution_settings = any(not ready for ready, _, _ in checks[3:6])
        st.button('一键准备调优设置', key='btn_autotune_prepare',
                  disabled=busy or occupied or not all(ready for ready, _, _ in checks[:3]) or not missing_execution_settings,
                  on_click=prepare_settings, args=(state,), width='stretch',
                  help='启用 Bucket packing、streaming=on、inline 缓存与 gradient checkpointing。保留模型、数据、精度、优化器和学习率；不会启动实验。')
        if occupied:
            st.warning('训练或流水线正在运行；结束后才能开始调优。')
        st.caption('此处只检查面板条件；数据 SHA-256、GPU 与完整训练合约由调优子进程校验。')
        # Discard obsolete inputs restored from older panel states/templates.
        for key in ('autotune_audit', 'autotune_audit_source', 'autotune_provenance',
                    'autotune_time_limit', 'autotune_trial_timeout', 'autotune_baseline_timeout'):
            state.pop(key, None)
        st.markdown('**搜索预算**')
        budget_columns = st.columns(3)
        for index, (key, label, default, minimum) in enumerate((
            ('max_trials', '最多候选（含基线与复测）', 24, 2),
            ('warmup', '每组预热 steps', 20, 2), ('steps', '每组计时 steps', 60, 1),
            ('validation_examples', '固定验证样本数', 16, 1))):
            with budget_columns[index % 3]:
                _widget(st, 'number_input', label, min_value=minimum, value=default, key='autotune_' + key)
        st.caption('自动调优不设总时限或单组超时，等待数据准备、编译、训练与验证完成；可随时点击「停止自动调优」。')
        with st.expander('高级调优选项'):
            _widget(st, 'text_input', '调优缓存目录', key='autotune_output', value='.cache/autotune_webui')
            _widget(st, 'number_input', 'GPU 峰值 TFLOPS（0=不计算 MFU）', min_value=0., value=0., key='autotune_peak_tflops', help='填写当前计算精度的非稀疏峰值，换机器时重新确认。')
            _widget(st, 'number_input', '验收吞吐 token/s（0=自动选最高）', min_value=0., value=0., key='autotune_target_tokens_per_second')
            _widget(st, 'checkbox', '忽略旧实验结果，重新测量', key='autotune_refresh', value=False)
            _widget(st, 'text_area', '搜索空间 JSON（留空使用自动搜索空间）', key='autotune_search', value='', help='与 trainer/autotune.py 的 search 字段相同；例如 {"args.bucket_token_budget": [8192, 12288, 16384, 20480]}')
            _widget(st, 'text_area', '性能环境参数 JSON（可选）', key='autotune_environment', value='', help='仅允许自动调优器的性能参数白名单。')
        output = resolve_path(state.get('autotune_output', '.cache/autotune_webui'))
        preview = tuning_command(output / 'runs' / '<run>' / 'spec.json', output, state)
        with st.expander('查看调优命令'):
            st.code(subprocess.list2cmdline(preview) if os.name == 'nt' else __import__('shlex').join(preview), language='bash')
        st.caption(f'最多 {int(state.get("autotune_max_trials", 24)) * (int(state.get("autotune_warmup", 20)) + int(state.get("autotune_steps", 60))):,} 次实验更新。实验不保存正式权重，成功结果与缓存可复用。')
        actions = st.columns(2)
        plan = actions[0].button('检查并预览搜索计划', key='btn_autotune_plan', disabled=bool(reason) or busy or occupied, width='stretch', help='只检查配置并列出搜索范围，不初始化 CUDA、不启动训练。')
        start = actions[1].button('开始自动调优', key='btn_autotune_start', disabled=bool(reason) or busy or occupied, type='primary', width='stretch', help='会占用 GPU，使用真实数据执行短训练和验证；浏览器关闭后仍继续。')
        if plan or start:
            try:
                run_dir = output / 'runs' / uuid.uuid4().hex[:12]
                run_dir.mkdir(parents=True)
                model_path = run_dir / 'model.json'
                write_json(model_path, model)
                built = build_command('pretrain', model, 'autotune_disposable', False)
                spec = make_spec(built['cmd'], model_path,
                                 environment=json.loads(state.get('autotune_environment') or '{}'))
                if state.get('autotune_search', '').strip():
                    spec['search'] = json.loads(state['autotune_search'])
                    if not isinstance(spec['search'], dict):
                        raise ValueError('搜索空间 JSON 必须是对象')
                # Validate the same search whitelist before creating a process.
                if str(ROOT) not in sys.path:
                    sys.path.append(str(ROOT))
                from trainer.autotune_config import validate_settings
                from trainer.autotune import search_space
                validate_settings({'env': spec['environment']})
                search_space(spec, {'resolved_token_budget': spec['args'].get('bucket_token_budget') or 16384})
                write_json(run_dir / 'spec.json', spec)
                command = tuning_command(run_dir / 'spec.json', output, state, plan_only=plan)
                launch(run_dir, command)
                state['autotune_run_dir'] = str(run_dir)
                st.rerun()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                st.error(str(exc))
    with st.expander('02 · 调优进度与日志', expanded=bool(state.get('autotune_run_dir'))):
        @st.fragment(run_every=2 if busy else None)
        def live_progress():
            if (run_status(state.get('autotune_run_dir')).get('status') == 'running') != busy:
                # Refresh launch controls and stop the timer when the job finishes.
                st.rerun()
            render_progress(st, state.get('autotune_run_dir'))
        live_progress()
    with st.expander('03 · 查看并应用结果', expanded=True):
        found = profiles(output)
        configured_path = state.get('autotune_profile_path', '').strip()
        if configured_path:
            configured = resolve_path(configured_path)
            if configured.is_file() and configured not in found:
                found.insert(0, configured)
        if found:
            if state.get('autotune_result') not in {str(p) for p in found}:
                state['autotune_result'] = str(found[0])
            selected = st.selectbox('配置 profile（实测 / 手动）', [str(p) for p in found], key='autotune_result')
            try:
                data = read_json(selected)
                if data.get('status') == 'manual':
                    st.info('手动固定配置：未进行短训与独立复测，没有已验收吞吐或峰值显存数据。')
                elif data.get('status') == 'validated':
                    primary, confirmation = data['primary'], data['confirmation']
                    metrics = st.columns(3)
                    metrics[0].metric('最佳有效 tokens/s', f"{primary['useful_tokens_per_second']:,.0f}")
                    metrics[1].metric('独立复测 tokens/s', f"{confirmation['useful_tokens_per_second']:,.0f}")
                    metrics[2].metric('峰值保留显存', f"{primary['peak_reserved_gib']:.2f} GiB" if primary.get('peak_reserved_gib') is not None else '未记录')
                    if primary.get('mfu_estimate_pct') is not None:
                        st.write(f"估算 MFU：{primary['mfu_estimate_pct']:.2f}%")
                else:
                    raise ValueError('profile 既不是实测验收结果，也不是手动固定配置')
                st.json(data['settings'], expanded=False)
                st.download_button('导出 profile', json.dumps(data, ensure_ascii=False, indent=2), file_name='autotune_profile.json')
                if st.button('用于后续预训练', key='btn_autotune_apply', disabled=occupied or busy):
                    state['autotune_profile_path'] = selected
                    state['autotune_use_profile'] = True
                    st.rerun()
                searches = Path(selected).parent / 'search.json'
                if searches.exists():
                    with st.expander('候选结果与质量验收'):
                        rows = read_json(searches)
                        if isinstance(rows, dict):
                            rows = rows.get('trials', rows.get('rows', []))
                        st.dataframe([{key: row.get(key) for key in ('id', 'status', 'useful_tokens_per_second', 'wall_ms_per_step', 'mfu_estimate_pct', 'peak_reserved_gib', 'validation_final', 'quality_passed', 'quality_reason')} for row in rows], width='stretch')
            except (ValueError, KeyError, OSError, TypeError) as exc:
                st.error(f'无法读取调优结果：{exc}')
        else:
            st.info('此缓存目录尚无已验收结果。完成实测后可在这里比较吞吐并应用 profile；已有 CLI 结果可直接填写下方路径。')
        _widget(st, 'text_input', '生产 profile 路径（也可使用 CLI 生成的结果）', key='autotune_profile_path', value='')
        _widget(st, 'checkbox', '预训练使用 profile（实测 / 手动）', key='autotune_use_profile', value=False)
        if state.get('autotune_use_profile'):
            st.info('已选择使用 profile。下一步：回到「训练配置」检查命令；校验通过后点击「开始训练」。')
            st.caption('生产训练仍会检查 GPU / 软件 / 代码 / 模型 / 数据指纹及参数冲突；续训保留已保存的 token budget。参数不兼容时需要重新调优。')
