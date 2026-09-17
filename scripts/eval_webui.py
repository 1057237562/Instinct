"""Unified evaluation UI. Run: python -m streamlit run scripts/eval_webui.py"""
import json
import re
import os
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))
from eval_webui_utils import ROOT, JobManager, build_command, history, tail_log, checkpoint_config
from eval_pass_k import calculate, parse_k

st.set_page_config(page_title='Instinct Eval', page_icon='🧪', layout='wide')


@st.cache_resource
def manager():
    return JobManager()


def choose(label, options, default='', key=None, help=None, container=st, minimum=None):
    """Presets first; expose free entry only when explicitly selected."""
    options = list(dict.fromkeys([default, *options]))
    key = key or 'choice_' + label
    value = container.selectbox(label, [*options, '自定义…'], key=key,
                                format_func=lambda x: '不指定 / 自动' if x == '' else str(x), help=help)
    if value != '自定义…':
        return value
    if isinstance(default, (int, float)):
        return container.number_input('自定义 ' + label, value=default,
                                       min_value=minimum, key=key + '_custom')
    return container.text_input('自定义 ' + label, value=default, key=key + '_custom')


def discover(folder, patterns):
    paths = set()
    for pattern in patterns:
        paths.update(path for path in folder.glob(pattern) if path.is_file())
    return sorted(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
                  for path in paths)


def model_fields(toolcall=False):
    config = {'load_from': 'model', 'save_dir': 'out', 'require_model_config': True}
    files = sorted((ROOT / 'out').glob('*.pth'), key=lambda p: p.stat().st_mtime, reverse=True)
    choices = [str(p.relative_to(ROOT)) for p in files]
    selected = choose('模型权重文件', choices, choices[0] if choices else '', key='out_checkpoint')
    config['checkpoint_path'] = selected
    match = re.fullmatch(r'(.+)_(\d+)(_moe)?', Path(selected).stem)
    config['weight'] = match[1] if match else Path(selected).stem
    saved_path, saved = None, None
    if selected:
        try:
            saved_path, saved = checkpoint_config(selected)
        except (OSError, ValueError) as exc:
            st.error(str(exc))
    config['config_path'] = saved_path
    with st.expander('模型高级设置'):
        use_hf = st.checkbox('使用 Transformers 模型目录')
        if use_hf:
            models = [str(p.parent.relative_to(ROOT)) for pattern in ('*/config.json', 'scripts/*/config.json')
                      for p in ROOT.glob(pattern) if p.parent != ROOT / 'model']
            config['load_from'] = choose('模型目录', models, models[0] if models else '')
            config.pop('checkpoint_path')
            config.pop('config_path')
            config['weight'] = 'full_sft'
        else:
            override = st.checkbox('手动指定模型配置', value=not bool(saved_path), key='override_' + selected)
            if override:
                config['config_path'] = choose('模型 config JSON', discover(ROOT, ['out/*.json', 'checkpoints/*.json']),
                                               saved_path or '', key='config_' + selected)
                if config['config_path']:
                    try:
                        candidate = Path(config['config_path'])
                        saved = json.loads((candidate if candidate.is_absolute() else ROOT / candidate).read_text(encoding='utf-8'))
                    except (OSError, ValueError) as exc:
                        st.error(str(exc))
            if not toolcall:
                weights = sorted({re.sub(r'_\d+(?:_moe)?$', '', p.stem) for p in files})
                config['lora_weight'] = choose('LoRA 权重名称', weights, 'None')
        device = choose('设备', ['自动', 'cuda', 'cuda:0', 'cuda:1', 'cpu'], '自动')
        if device != '自动':
            config['device'] = device
    if not use_hf:
        if saved:
            st.caption(f"自动读取：{saved.get('model_architecture', 'standard')} · "
                       f"{saved.get('num_hidden_layers')} 层 · hidden_size {saved.get('hidden_size')} · "
                       f"MoE {'开启' if saved.get('use_moe') else '关闭'}")
            st.caption('配置来源：' + str(config['config_path']))
        else:
            st.warning('未找到同名训练配置，请在模型高级设置中指定；不会猜测模型结构。')
    return config


def artifacts(run):
    config = run['config']
    output = config.get('output')
    paths = [Path(run['directory']) / 'run.log', Path(run['directory']) / 'run.json']
    if output:
        target = Path(output)
        target = target if target.is_absolute() else ROOT / target
        paths.append(target)
        paths.append(Path(str(target) + '.gz'))
        paths.extend(sorted(target.parent.glob(target.name + '_*')))
        paths.extend(sorted(target.parent.glob(target.stem + '_codegeneration_output*.json')))
    for index, path in enumerate(dict.fromkeys(paths)):
        if not path.is_file():
            continue
        st.caption(str(path))
        if path.name.endswith('_analysis.json'):
            try:
                analysis = json.loads(path.read_text(encoding='utf-8'))
                st.write('失败类型：', analysis['failure_types'])
                with st.expander('查看精简失败分析'):
                    st.json(analysis)
            except (OSError, ValueError, KeyError):
                st.caption('分析文件正在写入。')
        if path.name.endswith('_metrics.json'):
            try:
                report = json.loads(path.read_text(encoding='utf-8'))
                metrics = report.get('metrics', {})
                if metrics:
                    columns = st.columns(len(metrics))
                    for column, (name, value) in zip(columns, metrics.items()):
                        column.metric(name, f'{value:.2%}')
                st.json(report, expanded=False)
            except (ValueError, OSError):
                st.caption('结果文件正在写入，稍后刷新。')
        if path.stat().st_size <= 20 * 1024 * 1024:
            st.download_button('下载 ' + path.name, path.read_bytes(), file_name=path.name,
                               key=f'download_{run["id"]}_{index}')
        else:
            st.caption('文件超过 20 MB，请从上述路径读取。')


st.title('Instinct 评测工作台')
st.caption('统一配置模型与评测任务，查看执行日志和结果。所有相对路径以仓库根目录为基准。')
with st.sidebar:
    benchmark = st.selectbox('评测类型', ['HumanEval', 'LiveCodeBench', 'GSM8K', '推理自动测试', 'ToolCall'])
    st.caption('HumanEval / LiveCodeBench：代码功能评分。GSM8K：数学答案评分。\n\n推理 / ToolCall：内置用例的定性测试。')

configure, monitor, pass_k_tab = st.tabs(['配置评测', '任务与结果', '计算 pass@K'])
with configure:
    config = {'benchmark': benchmark}
    coded = benchmark in ('HumanEval', 'LiveCodeBench', 'GSM8K')
    config['mode'] = st.radio('运行方式', ['generate', 'evaluate', 'all'], horizontal=True,
                               format_func=lambda x: {'generate': '仅生成', 'evaluate': '仅评分', 'all': '生成并评分'}[x]) if coded else 'generate'
    if not coded:
        st.info('运行脚本内置的自动测试用例，结果为对话日志，不产生准确率。ToolCall 使用本地模型。')
    if config['mode'] != 'evaluate':
        config.update(model_fields(benchmark == 'ToolCall'))
    if coded:
        prefix = benchmark.lower()
        if st.session_state.pop('new_output_' + prefix, False):
            st.session_state.pop('output_stamp_' + prefix, None)
            st.session_state.pop(prefix + '_eval_output', None)
        config['k'] = choose('pass@K', ['1', '1,5', '1,5,10', '1,10,100'], '1')
        try:
            samples = max(parse_k(config['k']))
        except ValueError:
            samples = 1
        config.update(num_samples=samples, temperature=0.8 if samples > 1 else 0.0,
                      top_p=0.95, seed=42, limit=0, prompt_style='auto',
                      workers=min(4, os.cpu_count() or 1), timeout=3 if benchmark == 'HumanEval' else 6,
                      max_new_tokens=512, resume=False, problem_file='')
        if config['mode'] != 'evaluate':
            config['batch_size'] = choose('生成 Batch size', [1, 2, 4, 8, 16, 32], 4, minimum=1)
            st.caption(f'自动设置每题 {samples} 个样本；' + ('随机采样，Temperature 0.8。' if samples > 1 else '贪心生成。'))
        stamp_key = 'output_stamp_' + prefix
        if stamp_key not in st.session_state:
            st.session_state[stamp_key] = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        suffix = 'json' if benchmark == 'LiveCodeBench' else 'jsonl'
        default_output = f"eval/{prefix}_{st.session_state[stamp_key]}.{suffix}"
        answers = [v for v in discover(ROOT / 'eval', [f'*{prefix}*.{suffix}'])
                   if not any(marker in Path(v).name for marker in ('_results', '_metrics', '_eval', '_codegeneration'))]
        if config['mode'] == 'evaluate':
            config['output'] = choose('待评分答案文件', answers, answers[0] if answers else '', key=prefix + '_score_output')
        else:
            config['output'] = default_output
        if benchmark == 'LiveCodeBench':
            config['release_version'] = choose('数据版本', ['release_v5', 'release_v6', 'release_latest'], 'release_v6')
            runners = [str(p) for p in (ROOT / 'LiveCodeBench', ROOT.parent / 'LiveCodeBench') if (p / 'lcb_runner').is_dir()]
            config['runner_path'] = runners[0] if runners else ''
        with st.expander('评测高级设置'):
            config['problem_file'] = choose('本地数据文件（留空自动下载）', discover(ROOT / 'dataset', ['*.jsonl', '*.jsonl.gz', '*.json', '*/*.jsonl']), key=prefix + '_data')
            if config['mode'] != 'evaluate':
                config['output'] = choose('答案文件路径', answers, default_output, key=prefix + '_eval_output')
                config['resume'] = st.checkbox('从答案文件续跑')
                config['num_samples'] = choose('每题样本数', [samples, 1, 5, 10, 20, 50, 100, 200], samples, minimum=1, key=f'samples_{samples}')
                config['temperature'] = choose('Temperature', [0.0, 0.2, 0.8, 1.0], 0.8 if config['num_samples'] > 1 else 0.0, minimum=0.0, key=f'temp_{config["num_samples"] > 1}')
                config['max_new_tokens'] = choose('最大生成 tokens', [256, 512, 1024, 2048, 4096, 8192], 512, minimum=1)
                config['prompt_style'] = choose('提示方式', ['auto', 'base', 'chat'], 'auto')
                config['open_thinking'] = st.checkbox('开启思考')
            config['limit'] = choose('题数限制（0 = 全部）', [0, 3, 10, 50, 100], 0, minimum=0)
            if benchmark != 'GSM8K':
                config['workers'] = choose('评分并发数', [1, 2, 4, 8, 16], min(4, os.cpu_count() or 1), minimum=1)
                config['timeout'] = choose('评分超时（秒）', [3, 6, 10, 30, 60], config['timeout'], minimum=1)
            if benchmark == 'LiveCodeBench':
                config['runner_path'] = choose('LiveCodeBench 官方仓库路径', runners, config['runner_path'])
                start = st.date_input('开始日期（可选）', value=None)
                end = st.date_input('结束日期（可选）', value=None)
                config['start_date'] = start.isoformat() if start else ''
                config['end_date'] = end.isoformat() if end else ''
        st.caption('答案文件：' + config['output'])
        if config['mode'] == 'generate':
            st.caption('仅生成答案；需要 pass@K 指标请选择「生成并评分」或「仅评分」。')
        elif benchmark == 'GSM8K':
            st.caption('GSM8K：提取最终 #### 数值，与参考答案比较；无需执行代码。报告 accuracy 和 pass@K。')
        else:
            st.caption('评分会执行生成代码，请在隔离评测环境运行。')
            if config['mode'] == 'all':
                st.caption('HumanEval：每批答案生成后立即并行评分，结束时统一计算 pass@K。' if benchmark == 'HumanEval'
                           else 'LiveCodeBench：官方评分入口要求完整答案文件，生成完成后评分。')
    else:
        config.update(temperature=0.8, top_p=0.95, max_new_tokens=512)
    try:
        command = build_command(config)
        with st.expander('查看执行命令'):
            st.code(subprocess.list2cmdline(command) if sys.platform == 'win32' else shlex.join(command), language='bash')
        valid = True
    except (ValueError, KeyError) as exc:
        st.error(str(exc))
        valid = False
    current = manager().status()
    running = current and current['status'] == 'running'
    if st.button('开始评测', type='primary', disabled=not valid or bool(running)):
        try:
            manager().start(command, config)
            if coded and config['mode'] != 'evaluate' and not config['resume']:
                st.session_state['new_output_' + prefix] = True
            st.success('任务已启动，请切换到「任务与结果」查看。')
        except (OSError, ValueError) as exc:
            st.error(str(exc))

with monitor:
    @st.fragment(run_every=2)
    def show_status():
        current = manager().status()
        if current:
            labels = {'running': '运行中', 'completed': '已完成', 'failed': '失败', 'stopped': '已停止'}
            st.write(f'当前任务：{current["id"]} · {labels[current["status"]]} · 退出码：{current["exit_code"]}')
            if current['status'] == 'running' and st.button('停止评测', key='stop_eval'):
                manager().stop()
                st.rerun()
            path = Path(current['directory']) / 'run.log'
            if path.exists():
                log = tail_log(path)
                rates = re.findall(r'\[Eval batch [^\n]*tokens/s=([\d.]+) time=([\d.]+)s', log)
                if rates:
                    st.metric('最近一批生成速度', f'{float(rates[-1][0]):.1f} tokens/s')
                    st.caption(f'该批耗时 {rates[-1][1]}s；为整批总吞吐，含首轮处理，不是单条回答速度。')
                st.code(log or '等待日志…', language='text', height=400)
            output = current['config'].get('output')
            if output and current['config']['benchmark'] == 'HumanEval' and current['config']['mode'] == 'all':
                progress_path = Path(str(output) + '_progress.json')
                if not progress_path.is_absolute():
                    progress_path = ROOT / progress_path
                try:
                    progress = json.loads(progress_path.read_text(encoding='utf-8'))
                    st.caption(f"已送评 {progress['submitted']} · 已评分 {progress['completed']} · 等待中 {progress['pending']}")
                except (OSError, ValueError, KeyError):
                    pass
        else:
            st.info('暂无当前任务。')
    show_status()
    if st.button('刷新历史与结果'):
        st.rerun()
    runs = history()
    if runs:
        selected = st.selectbox('历史任务', range(len(runs)),
                                format_func=lambda i: f'{runs[i]["id"]} · {runs[i]["config"]["benchmark"]} · {runs[i]["status"]}')
        run = runs[selected]
        st.caption('历史答案按原输出路径读取；需要保留多次结果时，请为每次评测设置不同的答案路径。')
        with st.expander('运行配置'):
            st.json(run)
        with st.expander('历史日志'):
            log_path = Path(run['directory']) / 'run.log'
            if log_path.is_file():
                st.code(tail_log(log_path), language='text', height=350)
        artifacts(run)

with pass_k_tab:
    st.subheader('从已有评分结果计算 pass@K')
    st.caption('无需加载模型或重新执行测试。HumanEval / GSM8K 使用 *_results.jsonl；LiveCodeBench 使用官方 *_eval_all.json。指标仅覆盖文件中的题目。')
    result_files = discover(ROOT / 'eval', ['*_results.jsonl', '*_results.jsonl.gz', '*_eval_all.json', '*/*_results.jsonl', '*/*_eval_all.json'])
    result_path = choose('已评分结果文件', result_files, key='pass_k_source')
    requested_k = choose('需要计算的 K', ['1', '1,5', '1,5,10', '1,2,10', '1,10,100'], '1,5,10', key='pass_k_values')
    if st.button('计算 pass@K', type='primary'):
        try:
            path = Path(result_path)
            report = calculate(path if path.is_absolute() else ROOT / path, parse_k(requested_k))
            st.session_state['pass_k_report'] = report
        except (OSError, ValueError, TypeError) as exc:
            st.session_state.pop('pass_k_report', None)
            st.error(str(exc))
    report = st.session_state.get('pass_k_report')
    if report:
        for name, value in report['metrics'].items():
            st.metric(name, f'{value:.2%}')
        st.write(f"题目数：{report['num_tasks']} · 总样本数：{report['num_samples']} · 每题最少样本：{report['min_samples_per_task']}")
        if report['skipped_k']:
            st.warning(f"样本不足，未计算 K={report['skipped_k']}。请增加每题样本并重新评分。")
        st.download_button('下载 pass@K 指标', json.dumps(report, ensure_ascii=False, indent=2),
                           file_name='pass_at_k_metrics.json', mime='application/json')
        with st.expander('逐题通过数量'):
            st.json(report['per_task'])
