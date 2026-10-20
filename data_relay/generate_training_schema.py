"""Export CPU-only relay controls from Config WebUI presets and trainer argparse.

Run after changing those sources; --check verifies the checked-in embedded schema.
No Streamlit, torch, or training-environment imports are needed.
"""
import ast
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.config_templates import load_model_presets, preset_initial_state

TARGET = ROOT / 'data_relay/src/training_schema.json'
STAGES = ['pretrain', 'cpt', 'full_sft', 'lora', 'dpo', 'ppo', 'grpo', 'agent', 'distillation']


def field(key, label, default=None, choices=None, kind=None, **extra):
    return dict(key=key, label=label, default=default,
                type=kind or ('bool' if isinstance(default, bool) else 'int' if isinstance(default, int)
                              else 'float' if isinstance(default, float) else 'text'),
                **({'choices': choices} if choices else {}), **extra)


def presets():
    return {
        name: {key: value for key, value in preset_initial_state(name).items()
               if not key.startswith('_')} | {'preset': name}
        for name in load_model_presets()
    }


def arguments(path):
    result = {}
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != 'add_argument':
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant) or not str(node.args[0].value).startswith('--'):
            continue
        key = node.args[0].value[2:]
        kw = {}
        for k in node.keywords:
            try: kw[k.arg] = ast.literal_eval(k.value)
            except (ValueError, TypeError): pass
        kind = next((k.value.id for k in node.keywords if k.arg == 'type' and isinstance(k.value, ast.Name)), 'text')
        if kind not in ('int', 'float'): kind = 'text'
        if kw.get('action') == 'store_true': kind = 'bool'
        result[key] = field(key, kw.get('help') or key, kw.get('default'), kw.get('choices'), kind)
    return result


def schema():
    f = field
    groups = [dict(title='训练超参数', fields=[
        f('epochs_{stage}', '训练轮数', 2, min=1), f('learning_rate_{stage}', '学习率', 0.0005, min=0),
        f('warmup_ratio_{stage}', 'Warmup 比例', 0.0, min=0, max=1), f('batch_size', 'Batch size', 32, min=1),
        f('accumulation_steps', '梯度累积', 1, min=1), f('max_seq_len', '序列长度（tokens）', 768, min=1),
        f('optimizer', '优化器', 'adamw', ['adamw', 'muon', 'adafactor']),
    ]), dict(title='精度与计算', fields=[
        f('activation_dtype', '激活精度', 'bfloat16', ['bfloat16', 'float16', 'fp32']),
        f('param_dtype', '参数精度', 'fp32', ['fp32', 'bf16', 'fp16']),
        f('muon_mantissa', 'Muon BF16+uint16 补偿权重（需参数 bf16）', False),
        f('muon_momentum_dtype', 'Muon 动量存储精度（补偿权重模式）', 'fp32', ['fp32', 'bf16']),
        f('muon_batch_size', 'Muon 每块矩阵数上限（0 自动）', 0, min=0),
        f('muon_workspace_mb', 'Muon 工作区 MiB（0 自动，16矩阵可选272）', 0.0, min=0),
        f('kv_cache_dtype', 'KV Cache 精度', 'fp32', ['fp32', 'bf16', 'fp16', 'fp8_e4m3', 'fp8_e5m2']),
        f('fp8_training', 'FP8 训练', 'off', ['off', 'tensorwise', 'rowwise', 'rowwise_with_gw_hp']),
        f('fp8_filter', 'FP8 层筛选', 'auto', ['auto', 'eligible']),
        f('use_grad_checkpoint', '梯度检查点（0 关闭）', 0, min=0), f('use_compile', '启用 torch.compile', True),
        f('compile_mode', '编译模式', 'reduce-overhead', ['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs']),
    ]), dict(title='数据加载、Packing 与缓存', fields=[
        f('sequence_packing', '启用序列打包', False), f('sequence_packing_mode', '打包模式', 'fixed', ['fixed', 'bucket']),
        f('seq_bucket', '长度桶倍率', 2, min=1), f('bucket_gpu_memory_gb', '显存预算（GiB）', 16.0, min=0),
        f('bucket_token_budget', '每批 token 预算（0 自动）', 0, min=0), f('bucket_max_seq_len', '桶最大长度', 16384, min=1),
        f('bucket_large_threshold', '大序列阈值', 8192, min=1), f('packing_batch_size', '预处理 batch', 1000, min=1),
        f('packing_num_proc', '预处理进程数（空为平台默认）', None, kind='int', min=1),
        f('bucket_loader_workers', '桶加载进程数', 0, min=-1), f('data_cache_max_gb', '缓存预算（GiB）', 5.0, min=0),
        f('dataset_streaming', '流式加载', 'auto', ['auto', 'on', 'off']), f('streaming_chunk_mb', '流式分块（MiB）', 1024, min=1),
        f('streaming_prefetch_chunks', '预取下一块', True), f('cache_build_mode', '缓存构建', 'inline', ['inline', 'background']),
    ]), dict(title='模型结构', fields=[
        f('model_architecture', '模型主干', 'standard', ['standard', 'linear', 'recurrsive_instinct']),
        f('hidden_size', '隐藏维度', 512, min=1), f('num_hidden_layers', '层数', 32, min=1),
        f('intermediate_size', 'FFN 中间维度（空为自动）', None, kind='int', min=1),
        f('vocab_size', '词表大小', 6400, min=1), f('num_attention_heads', 'Q 头数', 8, min=1),
        f('num_key_value_heads', 'KV 头数', 4, min=1), f('max_position_embeddings', '最大上下文', 32768, min=1),
        f('dropout', 'Dropout', 0.0, min=0, max=1), f('hidden_act', '激活函数', 'silu'),
        f('tie_word_embeddings', '共享输入输出嵌入', False), f('rms_norm_eps', 'RMSNorm epsilon', 1e-6, min=0),
        f('flash_attn', 'Flash Attention', True), f('residual_type', '残差结构', 'standard', ['standard', 'mhc', 'attnres']),
        f('use_moe', '启用 MoE', False), f('num_experts', '专家总数', 4, min=1), f('num_experts_per_tok', 'Top-k', 1, min=1),
        f('moe_intermediate_size', '专家 FFN 维度（空为自动）', None, kind='int', min=1),
        f('norm_topk_prob', '归一化 Top-k 概率', True), f('router_aux_loss_coef', '路由辅助损失', 0.0005, min=0),
        f('moe_router_migration', '继续训练时迁移路由配置', False),
    ]), dict(title='位置编码与高级结构', fields=[
        f('rope_theta', 'RoPE theta', 1e6, min=1), f('rope_scaling_type', 'RoPE 缩放', 'none', ['none', 'yarn', 'longrope']),
        f('factor', 'RoPE factor', 16.0, min=1), f('original_max_position_embeddings', '原始上下文长度', 2048, min=1),
        f('beta_fast', 'YaRN beta fast', 32), f('beta_slow', 'YaRN beta slow', 1), f('attention_factor', 'YaRN 注意力倍率', 1.0),
        f('longrope_short_factors', 'LongRoPE short factors（JSON 数组，空为自动）', ''),
        f('longrope_long_factors', 'LongRoPE long factors（JSON 数组，空为自动）', ''),
        f('longrope_short_attention_factor', 'LongRoPE short attention', 1.0),
        f('longrope_long_attention_factor', 'LongRoPE long attention（0 自动）', 0.0),
        f('longrope_short_retained_tokens', 'LongRoPE short retained tokens', 0, min=0),
        f('longrope_long_retained_tokens', 'LongRoPE long retained tokens', 0, min=0),
        f('hc_mult', 'mHC 扩展倍数', 4, min=1), f('hc_sinkhorn_iters', 'mHC Sinkhorn 次数', 20, min=1), f('hc_eps', 'mHC epsilon', 1e-6),
        f('attnres_variant', 'AttnRes 类型', 'block', ['block', 'full']), f('attnres_block_size', 'AttnRes 块大小（空为自动）', None, kind='int', min=1),
        f('early_exit_enabled', 'Early exit 训练', False), f('early_exit_layers', 'Early exit 层索引（JSON 数组）', [4,5,6,7], kind='json'),
        f('early_exit_loss_weight', 'Early exit loss 权重', 0.3, min=0),
    ]), dict(title='Linear / RecurrsiveInstinct 结构', fields=[
        *[f(k, label, value, min=1) for k, label, value in [
            ('full_attention_interval','完整注意力间隔',4), ('linear_conv_kernel_dim','线性卷积核',4),
            ('prelude_layers','Prelude 层数',2),('recurrent_layers','Recurrent 层数',4),('coda_layers','Coda 层数',2),
            ('loop_iters','循环深度',8),('mean_backprop_depth','平均反传深度',4),('max_recurrence','最大循环深度',32)]],
        *[f(k, label+'（空为自动）', None, kind='int', min=1) for k,label in [('linear_key_head_dim','Linear K head dim'),
            ('linear_value_head_dim','Linear V head dim'),('linear_num_key_heads','Linear K heads'),('linear_num_value_heads','Linear V heads')]],
        f('qk_bias','QK bias',True), f('qk_norm','QK norm',False), f('use_input_injection','输入注入',True),
        f('recurrence_sampling','循环采样','lognormal_poisson',['lognormal_poisson','fixed']),
        f('recurrence_log_normal_sigma','Log-normal sigma',1.0), f('state_init_std','状态初始标准差', math.sqrt(2/5)),
        f('embedding_scale','嵌入缩放（空为自动）',None,kind='float',min=0),
    ]), dict(title='性能分析与调优结果', fields=[
        f('profile','Profiler','off',['off','basic','full']), f('profile_warmup','预热步数',10,min=0),
        f('profile_interval','采样间隔',100,min=1), f('profile_active_steps','采样步数',5,min=1),
        f('autotune_use_profile','使用服务器已验收的调优结果',False), f('autotune_profile_path','服务器 profile.json 路径',''),
    ])]
    common = arguments(ROOT/'trainer/trainer_cli.py')
    # CLI choices are authoritative; UI labels/defaults keep panel semantics.
    for group in groups:
        for item in group['fields']:
            cli_key = {'activation_dtype': 'dtype'}.get(item['key'], item['key'])
            if item['type'] != 'bool' and common.get(cli_key, {}).get('choices'):
                item['choices'] = common[cli_key]['choices']
    reserved = {'save_dir','save_weight','lora_name','pause_file','config_path','from_resume','data_path','from_weight',
                'hidden_size','num_hidden_layers','use_moe','device','use_recurrsive','early_exit','autotune_profile','autotune_trial'}
    mapped = {f['key'].replace('_{stage}', '') for g in groups for f in g['fields']} | {'dtype','num_workers'}
    extra = {}
    for stage in STAGES:
        all_args = common | arguments(ROOT / ('trainer/train_' + ('pretrain' if stage == 'cpt' else stage) + '.py'))
        extra[stage] = {k:v for k,v in all_args.items() if k not in reserved | mapped}
    tree = ast.parse((ROOT/'scripts/config_webui.py').read_text(encoding='utf-8'))
    defaults = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {'_DEFAULT_EPOCHS','_DEFAULT_LEARNING_RATES'}}
    return dict(groups=groups, presets=presets(), arguments=extra, stages=STAGES,
                stage_defaults={s: dict(epochs=defaults['_DEFAULT_EPOCHS'][s], learning_rate=defaults['_DEFAULT_LEARNING_RATES'][s],
                                        warmup_ratio=0.03 if s == 'cpt' else 0.0) for s in STAGES})


if __name__ == '__main__':
    output = json.dumps(schema(), ensure_ascii=False, indent=2) + '\n'
    if '--check' in sys.argv:
        if TARGET.read_text(encoding='utf-8') != output:
            raise SystemExit('Training schema is stale: run python data_relay/generate_training_schema.py')
        print('Training schema matches Config WebUI presets and trainer CLI')
    else:
        TARGET.write_text(output, encoding='utf-8')
