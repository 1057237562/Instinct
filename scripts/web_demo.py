import random
import re
import json
import os
import sys
import gc
import base64
from functools import lru_cache

# Resolve imports from repo root (same pattern as trainer scripts)
__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Persistent Inductor cache shared with the trainers; must run before torch import
import trainer.compile_cache  # noqa: F401

from types import SimpleNamespace

import torch
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, TextIteratorStreamer
from scripts.stream_metrics import TokenRateStreamer, speed_caption, render_updates
from scripts.chat_generation import GenerationTask, stop_generation
from model.model_instinct import InstinctConfig, InstinctForCausalLM
from scripts.web_demo_utils import (
    clear_conversation_state,
    detach_model_state,
    encode_clipboard_text,
    queue_last_response_regeneration,
    resolve_model_config_path,
    render_markdown_stream,
    generation_loading_html,
)

_REPO_LOGO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'images', 'webui_logo.png'))

st.set_page_config(
    page_title="Instinct",
    page_icon=_REPO_LOGO if os.path.isfile(_REPO_LOGO) else None,
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }

        /* 聊天区操作按钮（新对话/重新生成）：胶囊化 + 紧凑，侧边栏按钮不受影响 */
        [data-testid="stMain"] .stButton button {
            border-radius: 999px !important;
            font-size: 13px !important;
            min-height: 30px !important;
            padding: 2px 14px !important;
            transition: background-color .15s ease, border-color .15s ease, color .15s ease !important;
        }
        [data-testid="stMain"] [data-testid="stBaseButton-tertiary"]:hover {
            background-color: rgba(128, 128, 128, .12) !important;
            color: inherit !important;
        }
        /* 破坏性确认（确认清空）：主区唯一的 primary 按钮转红，与加载模型等侧边栏按钮区分 */
        [data-testid="stMain"] [data-testid="stBaseButton-primary"] {
            background-color: #d32f2f !important;
            border: 1px solid #d32f2f !important;
        }
        [data-testid="stMain"] [data-testid="stBaseButton-primary"]:hover {
            background-color: #b71c1c !important;
            border-color: #b71c1c !important;
        }
    </style>
""", unsafe_allow_html=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 多语言文本
LANG_TEXTS = {
    'zh': {
        'settings': '模型设定调整',
        'history_rounds': '历史对话轮次',
        'max_length': '最大输出 Tokens',
        'temperature': '温度',
        'repetition_penalty': '重复性惩罚',
        'repetition_penalty_tip': '1.0 表示关闭；建议 1.05–1.20，过高可能降低回答质量',
        'thinking': '思考',
        'tools': '工具',
        'language': '语言',
        'send': '给 Instinct 发送消息',
        'disclaimer': 'AI 生成内容可能存在错误，请仔细核实',
        'think_tip': '自适应思考，目前多轮对话或Tool Call共存时思考不稳定',
        'tool_select': '工具选择（最多4个）',
        'load_model': '🚀 加载模型',
        'unload_model': '🔄 卸载模型',
        'new_chat': '新对话',
        'new_chat_tip': '清空当前对话并开始新对话',
        'confirm_clear_body': '当前对话将被清空，且无法恢复。',
        'confirm_clear_yes': '确认清空',
        'cancel': '取消',
        'chat_cleared': '已开始新对话',
        'regenerate': '重新生成',
        'regenerate_last': '重新生成最后一条回复',
        'copy_answer': '复制这条回复',
        'copied': '已复制',
        'copy': '复制',
        'loading_model': '正在加载模型，请稍候...',
        'configure_first': '请先配置模型路径，然后点击"加载模型"开始对话',
        'path_changed': '路径已变更，点击加载模型以重新加载',
        'model_loaded': '已加载',
        'load_failed': '加载失败',
        'generation_failed': '生成失败',
        'generation_stopping': '正在停止上一轮生成，请稍后重试。',
        'matched_config': '所选权重匹配的配置',
        'loaded_config': '已加载模型实际使用的配置',
        'config_details': '配置详情',
        'config_missing': '未找到配置文件',
        'config_path_unknown': '本次加载未记录配置路径；下方为模型实际配置',
        'config_error': '无法读取配置',
        'config_fallback': '未找到同名配置，使用默认 config_pretrain.json',
        'logit_lens': '逐层解释',
        'logit_lens_caption': '每个已生成token在各层的Top-1预测',
        'logit_lens_layer': '第{n}层',
        'logit_lens_final': '最终层',
        'logit_lens_aligned': '已对齐采样参数',
        'logit_lens_rank': 'token{n}',
    },
    'en': {
        'settings': 'Model Settings',
        'history_rounds': 'History Rounds',
        'max_length': 'Max Output Tokens',
        'temperature': 'Temperature',
        'repetition_penalty': 'Repetition Penalty',
        'repetition_penalty_tip': '1.0 disables it; 1.05–1.20 is recommended, while high values may reduce quality',
        'thinking': 'Thinking',
        'tools': 'Tools',
        'language': 'Language',
        'send': 'Send a message to Instinct',
        'disclaimer': 'AI-generated content may be inaccurate, please verify',
        'think_tip': 'Adaptive thinking; may be unstable with multi-turn or Tool Call',
        'tool_select': 'Tool Selection (max 4)',
        'load_model': '🚀 Load Model',
        'unload_model': '🔄 Unload Model',
        'new_chat': 'New Chat',
        'new_chat_tip': 'Clear the current conversation and start a new one',
        'confirm_clear_body': 'The current conversation will be deleted and cannot be recovered.',
        'confirm_clear_yes': 'Clear Chat',
        'cancel': 'Cancel',
        'chat_cleared': 'Started a new chat',
        'regenerate': 'Regenerate',
        'regenerate_last': 'Regenerate the last response',
        'copy_answer': 'Copy this response',
        'copied': 'Copied',
        'copy': 'Copy',
        'loading_model': 'Loading model, please wait...',
        'configure_first': 'Please configure model paths and click "Load Model" to start',
        'path_changed': 'Path changed. Click Load Model to reload',
        'model_loaded': 'Loaded',
        'load_failed': 'Load failed',
        'generation_failed': 'Generation failed',
        'generation_stopping': 'Stopping the previous generation; please try again shortly.',
        'matched_config': 'Config matched to selected weight',
        'loaded_config': 'Config used by loaded model',
        'config_details': 'Config details',
        'config_missing': 'No config file found',
        'config_path_unknown': 'Config path was not recorded; showing the actual model config below',
        'config_error': 'Cannot read config',
        'config_fallback': 'No matching config found; using default config_pretrain.json',
        'logit_lens': 'Logit Lens',
        'logit_lens_caption': 'Per-layer Top-1 prediction for each generated token',
        'logit_lens_layer': 'Layer {n}',
        'logit_lens_final': 'final',
        'logit_lens_aligned': 'sampling-aligned',
        'logit_lens_rank': 'token{n}',
    }
}

def get_text(key):
    lang = st.session_state.get('lang', 'en')
    return LANG_TEXTS.get(lang, {}).get(key, LANG_TEXTS['zh'].get(key, key))

# 工具定义
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}, "required": ["min", "max"]}}},
    {"type": "function", "function": {"name": "text_length", "description": "计算文本长度", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位转换", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "获取汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}}, "required": ["text", "target_lang"]}}},
]

TOOL_SHORT_NAMES = {
    'calculate_math': '数学', 'get_current_time': '时间', 'random_number': '随机',
    'text_length': '字数', 'unit_converter': '单位', 'get_current_weather': '天气',
    'get_exchange_rate': '汇率', 'translate_text': '翻译'
}

def execute_tool(tool_name, args):
    import datetime
    try:
        if tool_name == 'calculate_math':
            return {"result": eval(args.get('expression', '0'))}
        elif tool_name == 'get_current_time':
            tz = args.get('timezone', 'Asia/Shanghai')
            return {"result": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        elif tool_name == 'random_number':
            return {"result": random.randint(args.get('min', 0), args.get('max', 100))}
        elif tool_name == 'text_length':
            return {"result": len(args.get('text', ''))}
        elif tool_name == 'unit_converter':
            return {"result": f"{args.get('value', 0)} {args.get('from_unit', '')} = ? {args.get('to_unit', '')}"}
        elif tool_name == 'get_current_weather':
            return {"result": f"{args.get('city', 'Unknown')}: 晴, 7~10°C"}
        elif tool_name == 'get_exchange_rate':
            return {"result": f"1 {args.get('from_currency', 'USD')} = 7.2 {args.get('to_currency', 'CNY')}"}
        elif tool_name == 'translate_text':
            return {"result": f"[翻译结果]: hello world"}
        return {"result": "Unknown tool"}
    except Exception as e:
        return {"error": str(e)}


def _escape_html(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


class LogitLensStreamer(TextIteratorStreamer):
    """TextIteratorStreamer 的增强版：额外把实际生成的 token id 记录进共享列表。

    generate() 开头 put 的是完整 prompt（skip_prompt=True 时被丢弃，且不记录）；
    之后每次 put 是一块新生成的 token，逐 id 追加到 gen_ids，作为逐层解释的列头。
    """

    def __init__(self, tokenizer, gen_ids, skip_prompt=True, skip_special_tokens=True, on_block=None):
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
        self.gen_ids = gen_ids
        self.on_block = on_block

    def put(self, value):
        if len(value.shape) > 1:
            value = value[0]
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        if self.on_block:
            self.on_block(value)
        self.gen_ids.extend(value.tolist())
        super().put(value)


def setup_logit_lens(model, tokenizer, temperature=None, top_p=None,
                     top_k_sampling=None, repetition_penalty=1.0,
                     prompt_token_ids=None):
    """把"逐层解释"挂在真实生成过程上，构建 层 × 已生成token 的 Top-1 表。

    布局（区别于标准 Logit Lens）：
      * 列 = 实际生成的每个 token（token1, token2, ...，按生成顺序，列头即该步真正采样的 token）
      * 行 = Transformer 各层
      * 单元 (第L层, 第k列) = 第 k 个 token 被采样前，第 L 层预测概率最大的 Top-1 token（含概率）
    —— 因此列的维度是"实际生成了多少个 token"，每个单元只展示 Top-1 数据。

    实现：Instinct 的 generate() 会把额外 kwargs 原样透传给 forward，而 forward 支持
    layer_callback 钩子（每层算完即回调 normed 状态）。借助它即可在不复写生成循环的
    前提下，逐步采集每一生成步的各层 Top-1。最终层额外做与 generate() 相同的采样对齐
    （温度→重复性惩罚→top-k→top-p），使末行展示的正是采样器实际面对的分布；
    中间层保持原始 softmax。

    返回 SimpleNamespace，暴露：
      * streamer : 可替代 TextIteratorStreamer 的流式对象（记录实际生成的 token id）
      * layer_cb : 传给 model.generate(layer_callback=...) 的逐层回调
      * render() : 把最新表格渲染进占位符（长序列自动降频，final=True 强制渲染最终版）
    """
    n_layers = getattr(model.config, 'num_hidden_layers', None) or 0
    ph = st.empty()
    steps = []          # 每个生成步一个 dict: {'tops': [(token_text, prob, token_id) × n_layers]}
    gen_ids = []        # 实际生成的 token id（由 streamer 线程填充）
    prompt_token_ids = list(prompt_token_ids or [])
    last_rendered = [0]  # 节流：已渲染到的列数
    broken = [False]    # 采集异常后静默降级（生成线程内禁止抛异常，否则流式对话会卡死）

    def decode_token(tid):
        # skip_special_tokens=False：保留 <think>/<|im_end|> 等特殊 token 的可读名称
        text = tokenizer.decode([tid], skip_special_tokens=False)
        if not text or not text.strip():
            text = '<%d>' % tid
        return text

    def cell_color(p):
        # 白 -> 蓝，概率越大越饱和
        r = int(255 - (255 - 33) * p)
        g = int(255 - (255 - 118) * p)
        b = int(255 - (255 - 255) * p)
        return '#%02x%02x%02x' % (r, g, b)

    pending_logits = []

    def record_layer(layer_idx, lg):
        if broken[0]:
            return
        try:
            # 一次 forward 只开一个新列：首轮处理完整 prompt（取最后位置，预测第1个生成 token），
            # 之后每步只输入 1 个新 token，对应预测下一个 token。
            if layer_idx == 1:
                steps.append({'tops': [None] * n_layers})
            # 采样对齐只作用于最终层（该层决定实际输出）；中间层保持原始 softmax 展示预测轨迹
            if n_layers > 0 and layer_idx == n_layers:
                if temperature is not None and temperature > 0:
                    lg = lg / temperature
                if repetition_penalty != 1.0:
                    seen = torch.tensor(
                        prompt_token_ids + gen_ids,
                        dtype=torch.long,
                        device=lg.device,
                    ).unique()
                    score = lg[seen]
                    lg[seen] = torch.where(
                        score > 0,
                        score / repetition_penalty,
                        score * repetition_penalty,
                    )
                if top_k_sampling is not None and top_k_sampling > 0:
                    k = min(top_k_sampling, lg.shape[-1])
                    lg = torch.where(lg < torch.topk(lg, k).values[-1], torch.full_like(lg, float('-inf')), lg)
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(lg, descending=True)
                    mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                    mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                    lg = torch.where(mask.scatter(-1, sorted_indices, mask), torch.full_like(lg, float('-inf')), lg)
            probs = torch.softmax(lg, dim=-1)
            p, tid = probs.max(dim=-1)
            p, tid = p.item(), tid.item()
            steps[-1]['tops'][layer_idx - 1] = (decode_token(tid), p, tid)
            del lg
        except Exception:
            broken[0] = True

    def layer_cb(layer_idx, normed_h):
        if not broken[0]:
            pending_logits.append((layer_idx, model.lm_head(normed_h[:, -1:, :])[0, 0].detach()))

    def flush_layers(tokens):
        if not pending_logits:
            return
        base = len(gen_ids)
        try:
            # One transfer for all layer diagnostics collected in this block.
            logits = torch.stack([item[1] for item in pending_logits]).float().cpu()
            token_values = tokens.tolist()
            step_index = -1
            for (layer_idx, _), lg in zip(pending_logits, logits):
                if layer_idx == 1:
                    if step_index >= 0 and step_index < len(token_values):
                        gen_ids.append(token_values[step_index])
                    step_index += 1
                if step_index >= len(token_values):
                    break
                record_layer(layer_idx, lg)
        except Exception:
            broken[0] = True
        finally:
            del gen_ids[base:]
            pending_logits.clear()

    def render(final=False):
        if broken[0]:
            return
        n = min(len(gen_ids), len(steps))
        if n == 0:
            return
        # 长序列降频：超过 256 列后每 8 个 token 才刷新一次，避免大表逐 token 重渲染拖慢对话
        if not final:
            throttle = 1 if n <= 256 else 8
            if n - last_rendered[0] < throttle:
                return
        last_rendered[0] = n
        actual = [decode_token(t) for t in gen_ids[:n]]

        rank_label = get_text('logit_lens_rank') or 'token{n}'
        thead = ('<tr><th style="border: 1px solid #ddd; padding: 4px 8px; text-align: left;"></th>' + ''.join(
            '<th style="border: 1px solid #ddd; padding: 4px 6px; text-align: center; min-width: 44px;">'
            '<div style="font-size: 10px; opacity: .6;">%s</div>'
            '<div style="font-size: 13px;">%s</div></th>'
            % (_escape_html(rank_label.format(n=i + 1)), _escape_html(actual[i]))
            for i in range(n)) + '</tr>')

        tbody = ''
        for li in range(n_layers):
            is_final = li == n_layers - 1
            layer_label = (get_text('logit_lens_layer') or '').format(n=li + 1)
            final_marker = ' (%s)' % get_text('logit_lens_final') if is_final else ''
            label_style = 'border: 1px solid #ddd; padding: 4px 8px; white-space: nowrap; font-weight: 600;'
            if is_final:
                label_style = label_style.replace('font-weight: 600;', 'font-weight: 700; background-color: #eef4fb;')
            tds = '<td style="%s">%s%s</td>' % (label_style, _escape_html(layer_label), _escape_html(final_marker))
            for i in range(n):
                text, p, tid = steps[i]['tops'][li]
                # 绿色描边：该层 Top-1 与"实际采样出的 token"一致（展示模型在第几层就已"决定"了这个 token）
                if is_final:
                    inner_style = 'font-size: 12px; font-weight: 700;'
                    match_style = ' border: 2px solid #4caf50;' if tid == gen_ids[i] else ''
                else:
                    inner_style = 'font-size: 11px; opacity: .8;'
                    match_style = ' border: 1px solid #4caf50;' if tid == gen_ids[i] else ''
                tds += ('<td style="background-color: %s; border: 1px solid #ddd; padding: 4px 6px; text-align: center;%s">'
                        '<div>%s</div><div style="%s">%.1f%%</div></td>') % (
                            cell_color(p), match_style, _escape_html(text), inner_style, p * 100)
            tbody += '<tr>%s</tr>' % tds

        summary = '%s · %s' % (get_text('logit_lens'), get_text('logit_lens_caption'))
        if temperature is not None:
            summary += ' (%s)' % get_text('logit_lens_aligned')
        html = ('<details style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;">'
                '<summary style="cursor: pointer; color: #888;">%s</summary>'
                '<div style="margin-top: 8px; overflow-x: auto;">'
                '<table style="border-collapse: collapse; font-size: 12px;">%s%s</table>'
                '</div></details>') % (_escape_html(summary), thead, tbody)
        ph.markdown(html, unsafe_allow_html=True)

    return SimpleNamespace(streamer=LogitLensStreamer(tokenizer, gen_ids, on_block=flush_layers), layer_cb=layer_cb, render=render)


def render_model_config(config_path, *, config=None):
    """Show the resolved file and effective architecture without loading weights."""
    if config_path:
        st.sidebar.code(os.path.abspath(config_path), language=None)
    elif config is not None:
        st.sidebar.caption(get_text('config_path_unknown'))
    else:
        st.sidebar.warning(get_text('config_missing'))
        return
    try:
        if config is None:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = InstinctConfig(**json.load(f))
    except (OSError, ValueError, TypeError) as exc:
        st.sidebar.error(f"{get_text('config_error')}: {exc}")
        return
    architecture = 'MoE' if config.use_moe else 'Dense'
    summary = (f"{architecture} · layers={config.num_hidden_layers} · "
               f"hidden={config.hidden_size} · "
               f"Q/KV={config.num_attention_heads}/{config.num_key_value_heads}")
    if config.use_moe:
        summary += f" · experts={config.num_experts} · top-{config.num_experts_per_tok}"
    st.sidebar.caption(summary)
    with st.sidebar.expander(get_text('config_details')):
        st.json(config.to_dict())


def load_model_tokenizer(config_path, tokenizer_path, weight_path=None):
    """Load a model owned by the current Streamlit session.

    Do not cache this function with ``st.cache_resource``: that cache keeps a
    hidden global reference after the session's unload button is pressed and
    therefore prevents CUDA memory from being released.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg_dict = json.load(f)
    config = InstinctConfig(**cfg_dict)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = InstinctForCausalLM(config)
    if weight_path:
        state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    from model.inference_runtime import optimize_inference, select_inference_dtype, warmup_decode
    model = model.to(dtype=select_inference_dtype(device)).eval().to(device)
    # 'auto' keeps model loading interactive: compiling the whole trunk costs
    # tens of seconds (minutes for the 32-layer MoE), while the decode step is
    # already captured as a CUDA graph. 'full' buys ~30% more decode throughput
    # and a faster prefill; set INSTINCT_INFERENCE_COMPILE=full for it.
    compile_mode = os.environ.get('INSTINCT_INFERENCE_COMPILE', 'auto')
    model = optimize_inference(model, compile_mode)
    # Auto mode intentionally skips the expensive full-trunk compile, but the
    # two fused kernels and the reusable decode CUDA graph are cheap enough to
    # prepare here.  This moves the one-time ~2 s cost out of the first answer.
    if compile_mode == 'auto' and device.startswith('cuda'):
        warmup_decode(model)
    return model, tokenizer


def unload_model():
    """Release this session's model and return cached CUDA blocks to the driver."""
    if not stop_generation(st.session_state):
        st.sidebar.warning(get_text('generation_stopping'))
        return
    model, tokenizer = detach_model_state(st.session_state)

    # Drop entries created by older versions that cached the model globally.
    # The current loader is intentionally session-owned and creates no entries.
    st.cache_resource.clear()

    # Moving the parameters off CUDA releases their allocations immediately.
    # Keep cleanup best-effort so a partially loaded/broken model can still be
    # detached from the session.
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_chat_messages():
    """Start a clean conversation while keeping the loaded model and settings."""
    clear_conversation_state(st.session_state)


def init_chat_messages():
    if "messages" in st.session_state:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                render_markdown_stream(st.empty(), message['content'], streaming=False)
            else:
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{message["content"]}</div></div>',
                    unsafe_allow_html=True)

    else:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    return st.session_state.messages

def regenerate_answer():
    st.session_state.pop("confirm_clear_chat", None)
    if queue_last_response_regeneration(st.session_state):
        st.rerun()


def render_regenerate_button(message_index):
    """Offer a fresh generation for the final completed assistant response."""
    if st.button(
        get_text('regenerate'),
        icon=":material/restart_alt:",
        type="tertiary",
        key=f"regenerate_response_{message_index}",
        help=get_text('regenerate_last'),
    ):
        regenerate_answer()


def render_copy_button(answer_text, message_index):
    """Render a clipboard pill that visually matches the native regenerate button.

    Streamlit has no server-side clipboard API, so the click is handled inside
    an iframe (same payload/fallback logic as before); only the chrome changes.
    """
    payload = encode_clipboard_text(answer_text)
    tooltip = json.dumps(get_text('copy_answer'), ensure_ascii=False)
    copied_tooltip = json.dumps(get_text('copied'), ensure_ascii=False)
    idle_label = get_text('copy')
    copied_label = get_text('copied')
    idle_label_js = json.dumps(idle_label, ensure_ascii=False)
    copied_label_js = json.dumps(copied_label, ensure_ascii=False)

    def text_width(text):
        # 13px 字号下的近似宽度：CJK 全角 13px，拉丁字符 8px
        return sum(13 if ord(char) > 127 else 8 for char in text)

    # 宽度需同时容纳 idle 与 copied 两种文案（点击后文案会临时变长）
    width = 52 + max(text_width(idle_label), text_width(copied_label))

    st.iframe(
        f"""
        <style>
            html, body {{ margin: 0; padding: 0; overflow: hidden; background: transparent; }}
            button {{
                all: unset; box-sizing: border-box; width: 100%; height: 30px; margin-top: 1px;
                display: flex; align-items: center; justify-content: center; gap: 6px;
                border-radius: 999px; color: #808080;
                font: 500 13px/1 "Source Sans 3", "Segoe UI", system-ui, sans-serif;
                cursor: pointer; transition: background-color .15s ease, color .15s ease;
            }}
            button:hover {{ background-color: rgba(128, 128, 128, .12); color: #4d4d4d; }}
            button.copied {{ color: #43a047; background-color: rgba(67, 160, 71, .08); }}
            button svg {{ width: 14px; height: 14px; flex: none; }}
            button .check {{ display: none; }}
            button.copied .copy {{ display: none; }}
            button.copied .check {{ display: block; }}
        </style>
        <button id="copy-{message_index}" type="button">
            <svg class="copy" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                 stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <rect x="9" y="9" width="13" height="13" rx="2"/>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
            </svg>
            <svg class="check" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"
                 stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M20 6 9 17l-5-5"/>
            </svg>
            <span class="label">{_escape_html(idle_label)}</span>
        </button>
        <script>
            const button = document.getElementById("copy-{message_index}");
            const labelEl = button.querySelector(".label");
            const payload = "{payload}";
            const tooltip = {tooltip};
            const copiedTooltip = {copied_tooltip};
            const idleLabel = {idle_label_js};
            const copiedLabel = {copied_label_js};
            button.title = tooltip;
            button.setAttribute("aria-label", tooltip);

            function decodePayload(value) {{
                const bytes = Uint8Array.from(atob(value), c => c.charCodeAt(0));
                return new TextDecoder("utf-8").decode(bytes);
            }}

            async function copyText(text) {{
                if (navigator.clipboard && window.isSecureContext) {{
                    try {{
                        await navigator.clipboard.writeText(text);
                        return true;
                    }} catch (error) {{}}
                }}
                const area = document.createElement("textarea");
                area.value = text;
                area.style.position = "fixed";
                area.style.opacity = "0";
                document.body.appendChild(area);
                area.focus();
                area.select();
                const copied = document.execCommand("copy");
                area.remove();
                return copied;
            }}

            button.addEventListener("click", async () => {{
                if (!await copyText(decodePayload(payload))) return;
                button.classList.add("copied");
                labelEl.textContent = copiedLabel;
                button.title = copiedTooltip;
                setTimeout(() => {{
                    button.classList.remove("copied");
                    labelEl.textContent = idleLabel;
                    button.title = tooltip;
                }}, 1200);
            }});
        </script>
        """,
        width=width,
        height=32,
        tab_index=0,
    )


def render_answer_actions(answer_text, message_index, is_last):
    """Show the copy control on every answer; the last answer can be regenerated."""
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        render_copy_button(answer_text, message_index)
        if is_last:
            render_regenerate_button(message_index)


def render_chat_toolbar():
    """Top-right chat actions. New chat clears irreversibly, so confirm inline first."""
    messages = st.session_state.get("messages", [])
    confirming = bool(messages) and st.session_state.get("confirm_clear_chat", False)

    with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        if confirming:
            st.caption(get_text('confirm_clear_body'))
            if st.button(get_text('cancel'), icon=":material/close:", type="tertiary"):
                st.session_state.pop("confirm_clear_chat", None)
                st.rerun()
            if st.button(get_text('confirm_clear_yes'), icon=":material/delete_sweep:", type="primary"):
                st.session_state.pop("confirm_clear_chat", None)
                clear_chat_messages()
                st.toast(get_text('chat_cleared'), icon=":material/check_circle:")
                st.rerun()
        elif st.button(get_text('new_chat'), icon=":material/add_comment:",
                       help=get_text('new_chat_tip'), disabled=not messages):
            if st.session_state.get("messages"):
                st.session_state.confirm_clear_chat = True
            else:
                clear_chat_messages()
            st.rerun()


# 模型路径配置
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))

# A Streamlit rerun can interrupt the consumer while its producer thread is
# still using CUDA weights/cache. Finish that thread before any model controls.
if not stop_generation(st.session_state):
    st.warning(get_text('generation_stopping'))
    st.stop()

st.sidebar.markdown("### Model Paths")

default_tokenizer = os.path.join(repo_root, "model")
if "tokenizer_path" not in st.session_state:
    st.session_state.tokenizer_path = default_tokenizer if os.path.isdir(default_tokenizer) else ""

default_weight = os.path.join(repo_root, "out", "pretrain_768.pth")
if "weight_path" not in st.session_state:
    st.session_state.weight_path = default_weight if os.path.exists(default_weight) else ""

tokenizer_path = st.sidebar.text_input("tokenizer dir", value=st.session_state.tokenizer_path, key="tok_path")

# Scan for available .pth weight files
def _scan_weights():
    entries = {}
    for scan_dir, label in [("../out", "out"), ("../checkpoints", "checkpoints")]:
        scan_path = os.path.join(script_dir, scan_dir)
        if not os.path.isdir(scan_path):
            continue
        for f in sorted(os.listdir(scan_path)):
            if f.endswith(".pth") and "_resume" not in f:
                full = os.path.join(scan_path, f)
                size_mb = os.path.getsize(full) / (1024 * 1024)
                entries[f"{label}/{f}  ({size_mb:.0f} MB)"] = full
    return entries

weight_options = _scan_weights()
weight_labels = list(weight_options.keys())

if weight_labels:
    CUSTOM_TOKEN = "📂 Custom path..."
    weight_labels.append(CUSTOM_TOKEN)
    current = st.session_state.weight_path
    default_idx = 0
    for i, (label, path) in enumerate(weight_options.items()):
        if os.path.normpath(path) == os.path.normpath(current):
            default_idx = i
            break

    selected = st.sidebar.selectbox("weight .pth", weight_labels, index=default_idx,
                                    key="wt_select", help="Detected weight files in out/ and checkpoints/")
    if selected == CUSTOM_TOKEN:
        weight_path = st.sidebar.text_input("Custom path", value=current, key="wt_custom")
    else:
        weight_path = weight_options.get(selected, current)
        if "_wt_custom" in st.session_state:
            del st.session_state._wt_custom
else:
    weight_path = st.sidebar.text_input("weight .pth", value=st.session_state.weight_path, key="wt_path")

config_path = resolve_model_config_path(weight_path, repo_root)

st.sidebar.caption(get_text('matched_config'))
render_model_config(config_path)
if config_path and os.path.normcase(os.path.abspath(config_path)) == os.path.normcase(
        os.path.join(repo_root, 'trainer', 'config_pretrain.json')):
    st.sidebar.info(get_text('config_fallback'))

st.session_state.config_path = config_path
st.session_state.tokenizer_path = tokenizer_path
st.session_state.weight_path = weight_path

ready = os.path.exists(config_path) and os.path.isdir(tokenizer_path)
slogan = "Instinct Chat"

if not st.session_state.get('model_loaded', False):
    if ready:
        if st.sidebar.button(get_text('load_model'), width="stretch", type="primary"):
            with st.spinner(get_text('loading_model')):
                try:
                    model, tokenizer = load_model_tokenizer(
                        config_path, tokenizer_path,
                        weight_path if os.path.exists(weight_path) else None
                    )
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model_loaded = True
                    st.session_state.loaded_weight_path = weight_path
                    st.session_state.loaded_config_path = config_path
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"{get_text('load_failed')}: {e}")
    else:
        st.sidebar.warning("⚠️ Please configure valid config & tokenizer paths first")
else:
    loaded_name = os.path.basename(st.session_state.get('loaded_weight_path', ''))
    st.sidebar.success(f"✅ {get_text('model_loaded')}: {loaded_name}")
    st.sidebar.caption(get_text('loaded_config'))
    render_model_config(
        st.session_state.get('loaded_config_path', ''),
        config=st.session_state.model.config,
    )
    if st.session_state.get('loaded_weight_path', '') != weight_path:
        st.sidebar.warning(get_text('path_changed'))
    if st.sidebar.button(get_text('unload_model'), width="stretch"):
        unload_model()
        st.rerun()

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

# 语言选择
lang_options = {'中文': 'zh', 'English': 'en'}
current_lang = st.session_state.get('lang', 'en')
lang_index = 0 if current_lang == 'zh' else 1
lang_label = st.sidebar.radio('Language / 语言', list(lang_options.keys()), index=lang_index, horizontal=True)
if lang_options[lang_label] != current_lang:
    st.session_state.lang = lang_options[lang_label]
    st.rerun()

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

# 参数设置
st.session_state.history_chat_num = st.sidebar.slider(get_text('history_rounds'), 0, 8, 0, step=2)
st.session_state.max_new_tokens = st.sidebar.slider(get_text('max_length'), 128, 16384, 2048, step=128)
st.session_state.temperature = st.sidebar.slider(get_text('temperature'), 0.6, 1.2, 0.90, step=0.01)
st.session_state.repetition_penalty = st.sidebar.slider(
    get_text('repetition_penalty'), 1.0, 2.0, 1.0, step=0.01,
    help=get_text('repetition_penalty_tip'),
)

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

# 功能开关
st.session_state.enable_thinking = st.sidebar.checkbox(get_text('thinking'), value=False, help=get_text('think_tip'))
st.session_state.enable_logit_lens = st.sidebar.checkbox(get_text('logit_lens'), value=False)
st.session_state.selected_tools = []
with st.sidebar.expander(get_text('tools')):
    st.caption(get_text('tool_select'))
    selected_count = sum(1 for tool in TOOLS if st.session_state.get(f"tool_{tool['function']['name']}", False))
    for tool in TOOLS:
        name = tool['function']['name']
        short_name = TOOL_SHORT_NAMES.get(name, name)
        checked = st.checkbox(short_name, key=f"tool_{name}", disabled=(selected_count >= 4 and not st.session_state.get(f"tool_{name}", False)))
        if checked and len(st.session_state.selected_tools) < 4:
            st.session_state.selected_tools.append(name)

@lru_cache(maxsize=1)
def webui_logo_uri():
    """本地 logo 以 data URI 内嵌（离线可用）；缺失时回退到远程图片。"""
    logo_path = os.path.join(repo_root, "images", "webui_logo.png")
    if os.path.isfile(logo_path):
        with open(logo_path, 'rb') as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
    return "https://raw.githubusercontent.com/1057237562/Instinct/main/images/logo2.png"


image_url = webui_logo_uri()

st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    f'<img src="{image_url}" style="width: 40px; height: 40px; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,.18);"> '
    f'<span style="font-size: 26px; margin-left: 10px;">{slogan}</span>'
    '</div>'
    f'<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">{get_text("disclaimer")}</span>'
    '</div>',
    unsafe_allow_html=True
)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def render_chat_generation(model, generation_kwargs, placeholder, speed_slot,
                           loading_slot, *, prefix='', lens=None):
    streamer = generation_kwargs['streamer']
    task = GenerationTask(model, generation_kwargs)
    st.session_state.generation_task = task
    task.start()
    answer = ''
    try:
        for updated_answer, final_update in render_updates(streamer):
            speed_slot.caption(speed_caption(streamer.snapshot()))
            render_markdown_stream(
                placeholder, prefix + updated_answer, previous=prefix + answer,
                thinking=st.session_state.get('enable_thinking', False),
                streaming=not final_update,
            )
            answer = updated_answer
            if lens is not None:
                lens.render()
        task.raise_if_failed()
        return answer
    except Exception as exc:
        loading_slot.empty()
        st.error(f"{get_text('generation_failed')}: {exc}")
        return None
    finally:
        task.stop()
        if not task.thread.is_alive() and st.session_state.get('generation_task') is task:
            st.session_state.pop('generation_task', None)


def main():
    if not st.session_state.get('model_loaded', False):
        if not ready:
            st.warning("Please set valid config.json and tokenizer dir paths in the sidebar, then click Load Model.")
        else:
            st.info(get_text('configure_first'))
        return

    model = st.session_state.model
    tokenizer = st.session_state.tokenizer

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    messages = st.session_state.messages

    # 工具栏占据顶部位置，但在运行结束时才渲染：「新对话」的禁用状态需要
    # 反映本轮新增的回复，而不是生成开始前的会话快照
    toolbar_slot = st.empty()

    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            render_markdown_stream(st.empty(), message['content'],
                                   thinking=message.get('thinking_enabled', False), streaming=False)
            if message.get('generation_stats'):
                st.caption(speed_caption(message['generation_stats']))
            render_answer_actions(message["content"], i, i == len(messages) - 1)
        else:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{message["content"]}</div></div>',
                unsafe_allow_html=True)

    prompt = st.chat_input(key="input", placeholder=get_text('send'))

    if st.session_state.pop('regenerate', False):
        prompt = st.session_state.pop('last_user_message', None)
        st.session_state.pop('regenerate_index', None)

    if prompt:
        # 用户直接发消息：放弃未完成的清空确认
        st.session_state.pop('confirm_clear_chat', None)
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{prompt}</div></div>',
            unsafe_allow_html=True)
        logit_lens_slot = st.container() if st.session_state.get('enable_logit_lens', False) else None
        messages.append({"role": "user", "content": prompt})
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        placeholder = st.empty()
        loading_slot = st.empty()
        with loading_slot.container():
            st.html(generation_loading_html())

        random_seed = random.randint(0, 2 ** 32 - 1)
        setup_seed(random_seed)

        tools = [t for t in TOOLS if t['function']['name'] in st.session_state.get('selected_tools', [])] or None
        sys_prompt = [] if tools else [{"role": "system", "content": "你是Instinct，一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题。"}]
        st.session_state.chat_messages = sys_prompt + st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):]
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if st.session_state.get('enable_thinking', False):
            template_kwargs["open_thinking"] = True
        if tools:
            template_kwargs["tools"] = tools
        new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)

        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        lens = None
        if logit_lens_slot is not None:
            with logit_lens_slot:
                lens = setup_logit_lens(model, tokenizer,
                                        temperature=st.session_state.get('temperature', 0.9),
                                        top_p=0.85, top_k_sampling=50,
                                        repetition_penalty=st.session_state.get('repetition_penalty', 1.0),
                                        prompt_token_ids=inputs.input_ids[0].tolist())

        streamer = lens.streamer if lens is not None else TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        streamer = TokenRateStreamer(streamer)
        speed_slot = st.empty()
        generation_stats = []
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "max_new_tokens": st.session_state.max_new_tokens,
            "num_return_sequences": 1,
            "do_sample": True,
            "attention_mask": inputs.attention_mask,
            # Native generate uses attention_mask for padding and EOS to stop.
            # Do not pass unused HF metadata to the Transformer decode path.
            "eos_token_id": tokenizer.eos_token_id,
            "temperature": st.session_state.temperature,
            "repetition_penalty": st.session_state.repetition_penalty,
            "top_p": 0.85,
            "streamer": streamer,
            "stream_chunk_size": 16,
        }
        if lens is not None:
            generation_kwargs["layer_callback"] = lens.layer_cb

        render_markdown_stream(placeholder, '', thinking=st.session_state.get('enable_thinking', False))
        answer = render_chat_generation(
            model, generation_kwargs, placeholder, speed_slot, loading_slot, lens=lens)
        if answer is None:
            return
        if lens is not None:
            lens.render(final=True)
            # 工具调用多轮复用 generation_kwargs：摘掉钩子，避免后续轮次继续污染已冻结的逐层解释表
            generation_kwargs.pop("layer_callback", None)

        full_answer = answer
        generation_stats.append(streamer.snapshot())
        speed_slot.caption(speed_caption(generation_stats[-1]))
        for _ in range(16):
            tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', answer, re.DOTALL)
            if not tool_calls:
                break
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            tool_results = []
            for tc_str in tool_calls:
                try:
                    tc = json.loads(tc_str.strip())
                    result = execute_tool(tc.get('name', ''), tc.get('arguments', {}))
                    st.session_state.chat_messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
                    tool_results.append(f"**ToolCalled · {tc.get('name', '')}**\n\n```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```")
                except:
                    pass
            full_answer += "\n" + "\n".join(tool_results) + "\n"
            render_markdown_stream(placeholder, full_answer, streaming=False)
            new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            streamer = TokenRateStreamer(streamer)
            generation_kwargs["input_ids"] = inputs.input_ids
            generation_kwargs["attention_mask"] = inputs.attention_mask
            generation_kwargs["max_new_tokens"] = st.session_state.max_new_tokens
            generation_kwargs["streamer"] = streamer
            answer = render_chat_generation(
                model, generation_kwargs, placeholder, speed_slot, loading_slot,
                prefix=full_answer)
            if answer is None:
                return
            full_answer += answer
            generation_stats.append(streamer.snapshot())
        answer = full_answer
        loading_slot.empty()
        total_tokens = sum(item['tokens'] for item in generation_stats)
        total_seconds = sum(item['seconds'] for item in generation_stats)
        stats = {'tokens': total_tokens, 'seconds': total_seconds,
                 'tokens_per_second': total_tokens / total_seconds if total_seconds else 0}
        speed_slot.caption(speed_caption(stats))

        messages.append({"role": "assistant", "content": answer, "generation_stats": stats,
                         "thinking_enabled": st.session_state.get('enable_thinking', False)})
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        render_answer_actions(answer, len(messages) - 1, True)

    with toolbar_slot.container():
        render_chat_toolbar()


if __name__ == "__main__":
    main()
