"""Capture independent training-panel snapshots and run them sequentially."""
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def snapshot(state, model, trainer):
    defaults = dict(batch_size=32, max_seq_len=768, accumulation_steps=1,
                    optimizer='adamw', param_dtype='fp32', kv_cache_dtype='fp32',
                    sequence_packing=False, sequence_packing_mode='fixed', seq_bucket=2,
                    bucket_gpu_memory_gb=16.0, bucket_max_seq_len=16384,
                    bucket_large_threshold=8192, packing_batch_size=1000,
                    packing_num_proc=1, bucket_loader_workers=0, use_compile=True,
                    compile_mode='reduce-overhead', use_grad_checkpoint=0,
                    fp8_training='off', fp8_filter='auto', profile='off',
                    profile_warmup=10, profile_interval=100, profile_active_steps=5)
    args = {key: state.get(key, value) for key, value in defaults.items()}
    args.update(dtype=state.get('activation_dtype', 'bfloat16'),
                epochs=state.get(f'epochs_{trainer}', 2),
                learning_rate=state.get(f'learning_rate_{trainer}', 5e-4 if trainer == 'pretrain' else 1e-5),
                data_path=state.get(f'data_path_{trainer}', ''),
                hidden_size=model['hidden_size'], num_hidden_layers=model['num_hidden_layers'],
                use_moe=model['use_moe'],
                use_looped=model.get('model_architecture') == 'looped',
                early_exit=bool(model.get('early_exit_layers') and state.get('early_exit_enabled')))
    if not args['data_path']:
        raise ValueError('请先选择该阶段的数据集')
    return json.loads(json.dumps(dict(name=trainer, trainer=trainer, model=model, args=args)))


def render(st, model, training_active):
    sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
    from trainer.training_pipeline import validate

    with st.expander('训练流水线 · Pretrain → SFT'):
        st.caption('将当前面板分别保存为预训练和 SFT 配置快照。SFT 自动使用流水线预训练权重；各阶段结束后删除分词 / packing 数据集缓存并释放训练进程显存。编译缓存保留。')
        stages = st.session_state.setdefault('pipeline_stages', {})
        selected = st.session_state.get('train_type', 'pretrain')
        if st.button('保存当前配置到流水线', disabled=selected not in ('pretrain', 'full_sft')):
            try:
                stages[selected] = snapshot(st.session_state, model, selected)
                st.session_state.pipeline_stages = stages
                st.success(f'{selected} 配置已保存；后续面板修改不会影响此快照。')
            except ValueError as exc:
                st.error(str(exc))
        for name in ('pretrain', 'full_sft'):
            if name in stages:
                st.write(f"✓ {name} · {stages[name]['args']['epochs']} epochs · {stages[name]['args']['data_path']}")
        plan = {'stages': [dict(stages[name]) for name in ('pretrain', 'full_sft') if name in stages]}
        if 'full_sft' in stages:
            plan['stages'][-1]['input_stage'] = 'pretrain'
        st.json(plan, expanded=False)
        st.download_button('导出流水线配置', json.dumps(plan, ensure_ascii=False, indent=2),
                           file_name='training_pipeline.json', mime='application/json')
        uploaded = st.file_uploader('导入已保存的流水线配置', type=['json'], key='pipeline_upload')
        if st.button('应用导入配置', disabled=uploaded is None):
            try:
                imported = validate(json.loads(uploaded.getvalue()))
                if [s['trainer'] for s in imported['stages']] != ['pretrain', 'full_sft']:
                    raise ValueError('面板流水线需要 Pretrain → SFT 两个阶段')
                st.session_state.pipeline_stages = {s['trainer']: s for s in imported['stages']}
                st.rerun()
            except (ValueError, KeyError) as exc:
                st.error(str(exc))
        proc = st.session_state.get('_pipeline_proc')
        active = proc is not None and proc.poll() is None
        if st.button('启动训练流水线', disabled=active or training_active or len(stages) != 2):
            try:
                validate(plan)
                runs = ROOT / 'trainer' / 'pipeline_runs'
                runs.mkdir(parents=True, exist_ok=True)
                run_dir = runs / uuid.uuid4().hex[:12]
                plan_path = runs / f'{run_dir.name}.json'
                plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
                with (runs / f'{run_dir.name}.log').open('w', encoding='utf-8') as log:
                    st.session_state['_pipeline_proc'] = subprocess.Popen(
                        [sys.executable, '-u', str(ROOT / 'trainer' / 'training_pipeline.py'),
                         '--plan', str(plan_path), '--run-dir', str(run_dir)], cwd=ROOT,
                        env=dict(os.environ, PYTHONUTF8='1'), stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                st.session_state['pipeline_run_dir'] = str(run_dir)
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        if st.button('暂停当前阶段并停止后续阶段', disabled=not active):
            run_dir = Path(st.session_state['pipeline_run_dir'])
            if run_dir.exists():
                (run_dir / 'stop').touch()
        if st.button('刷新流水线进度'):
            st.rerun()
        if st.session_state.get('pipeline_run_dir'):
            run_dir = Path(st.session_state['pipeline_run_dir'])
            status = run_dir / 'status.json'
            if status.exists():
                result = json.loads(status.read_text(encoding='utf-8'))
                st.json(result)
                if result['stages']:
                    log = run_dir / (result['stages'][-1]['name'] + '.log')
                    if log.exists():
                        with log.open('rb') as handle:
                            handle.seek(max(0, log.stat().st_size - 8000))
                            st.code(handle.read().decode('utf-8', errors='replace'))
