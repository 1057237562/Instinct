import random
import re
import json
import os
import sys

# Resolve imports from repo root (same pattern as trainer scripts)
__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from threading import Thread
from types import SimpleNamespace

import torch
import numpy as np
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from model.model_instinct import InstinctConfig, InstinctForCausalLM
from scripts.web_demo_utils import (
    clear_conversation_state,
    queue_last_response_regeneration,
    resolve_model_config_path,
)

st.set_page_config(page_title="Instinct", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        /* 添加操作按钮样式（仅作用于主聊天区的操作按钮，避免影响侧边栏按钮） */
        [data-testid="stMain"] .stButton button {
            border-radius: 50% !important;  /* 改为圆形 */
            width: 32px !important;         /* 固定宽度 */
            height: 32px !important;        /* 固定高度 */
            padding: 0 !important;          /* 移除内边距 */
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;         /* 更柔和的颜色 */
            margin: 5px 10px 5px 0 !important;  /* 调整按钮间距 */
        }
        [data-testid="stMain"] .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }
        
        /* 重置按钮基础样式（仅作用于主聊天区） */
        [data-testid="stMain"] .stButton > button {
            all: unset !important;  /* 重置所有默认样式 */
            box-sizing: border-box !important;
            border-radius: 50% !important;
            width: 18px !important;
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            max-width: 18px !important;
            max-height: 18px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #888 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            margin: 0 2px !important;  /* 调整这里的 margin 值 */
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
        'new_chat': '✨ 新对话',
        'regenerate_last': '重新生成最后一条回复',
        'loading_model': '正在加载模型，请稍候...',
        'configure_first': '请先配置模型路径，然后点击"加载模型"开始对话',
        'path_changed': '路径已变更，点击加载模型以重新加载',
        'model_loaded': '已加载',
        'load_failed': '加载失败',
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
        'new_chat': '✨ New Chat',
        'regenerate_last': 'Regenerate the last response',
        'loading_model': 'Loading model, please wait...',
        'configure_first': 'Please configure model paths and click "Load Model" to start',
        'path_changed': 'Path changed. Click Load Model to reload',
        'model_loaded': 'Loaded',
        'load_failed': 'Load failed',
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


def process_assistant_content(content, is_streaming=False):
    # 处理tool_call标签，格式化显示
    if '<tool_call>' in content:
        def format_tool_call(match):
            try:
                tc = json.loads(match.group(1))
                name = tc.get('name', 'unknown')
                args = tc.get('arguments', {})
                return f'<div style="background: rgba(80, 110, 150, 0.20); border: 1px solid rgba(140, 170, 210, 0.30); padding: 10px 12px; border-radius: 12px; margin: 6px 0;"><div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalling</div><div><b>{name}</b>: {json.dumps(args, ensure_ascii=False)}</div></div>'
            except:
                return match.group(0)
        content = re.sub(r'<tool_call>(.*?)</tool_call>', format_tool_call, content, flags=re.DOTALL)
    
    # 流式生成且开启思考时，一开始就放到折叠里
    if is_streaming and st.session_state.get('enable_thinking', False) and '</think>' not in content and '<think>' not in content:
        m = re.search(r'(\n\n(?:我是|您好|你好)[^\n]*)', content)
        if m and m.start(1) > 5:
            i = m.start(1)
            think_part = content[:i]
            answer_part = content[i:]
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_part.strip()}</div></details>{answer_part}'
        elif len(content) > 5:
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">思考中...</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; display: flex; flex-direction: column-reverse;"><div style="margin-bottom: auto;">{content.strip().replace(chr(10), "<br>")}</div></div></details>'

    if '<think>' in content and '</think>' in content:
        def format_think(match):
            think_content = match.group(2)
            if think_content.replace('\n', '').strip():  # 不是全换行
                return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_content.strip()}</div></details>'
            return ''
        content = re.sub(r'(<think>)(.*?)(</think>)', format_think, content, flags=re.DOTALL)

    if '<think>' in content and '</think>' not in content:
        def format_think_in_progress(match):
            tc = match.group(1)
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">思考中...</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; display: flex; flex-direction: column-reverse;"><div style="margin-bottom: auto;">{tc.strip().replace(chr(10), "<br>")}</div></div></details>'
        content = re.sub(r'<think>(.*?)$', format_think_in_progress, content, flags=re.DOTALL)

    if '<think>' not in content and '</think>' in content:
        def format_think_no_start(match):
            think_content = match.group(1)
            if think_content.replace('\n', '').strip():
                return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_content.strip()}</div></details>'
            return ''
        content = re.sub(r'(.*?)</think>', format_think_no_start, content, flags=re.DOTALL)

    return content


def _escape_html(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


class LogitLensStreamer(TextIteratorStreamer):
    """TextIteratorStreamer 的增强版：额外把实际生成的 token id 记录进共享列表。

    generate() 开头 put 的是完整 prompt（skip_prompt=True 时被丢弃，且不记录）；
    之后每次 put 恰好是 1 个新生成的 token，逐 id 追加到 gen_ids，作为逐层解释的列头。
    """

    def __init__(self, tokenizer, gen_ids, skip_prompt=True, skip_special_tokens=True):
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
        self.gen_ids = gen_ids

    def put(self, value):
        if len(value.shape) > 1:
            value = value[0]
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        super().put(value)
        self.gen_ids.extend(value.tolist())


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

    def layer_cb(layer_idx, normed_h):
        if broken[0]:
            return
        try:
            # 一次 forward 只开一个新列：首轮处理完整 prompt（取最后位置，预测第1个生成 token），
            # 之后每步只输入 1 个新 token，对应预测下一个 token。
            if layer_idx == 1:
                steps.append({'tops': [None] * n_layers})
            lg = model.lm_head(normed_h[:, -1:, :]).float().cpu()[0, 0]
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
            del lg, normed_h  # 只保留标量 (text, prob, id)，避免显存累积
        except Exception:
            broken[0] = True

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

    return SimpleNamespace(streamer=LogitLensStreamer(tokenizer, gen_ids), layer_cb=layer_cb, render=render)


@st.cache_resource
def load_model_tokenizer(config_path, tokenizer_path, weight_path=None):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg_dict = json.load(f)
    config = InstinctConfig(**cfg_dict)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = InstinctForCausalLM(config)
    if weight_path:
        state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    model = model.half().eval().to(device)
    return model, tokenizer


def clear_chat_messages():
    """Start a clean conversation while keeping the loaded model and settings."""
    clear_conversation_state(st.session_state)


def init_chat_messages():
    if "messages" in st.session_state:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{message["content"]}</div></div>',
                    unsafe_allow_html=True)

    else:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    return st.session_state.messages

def regenerate_answer():
    if queue_last_response_regeneration(st.session_state):
        st.rerun()


def render_regenerate_button(message_index):
    """Render the action only for the final completed assistant response."""
    if st.button(
        "↻",
        key=f"regenerate_response_{message_index}",
        help=get_text('regenerate_last'),
    ):
        regenerate_answer()


# 模型路径配置
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))

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
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"{get_text('load_failed')}: {e}")
    else:
        st.sidebar.warning("⚠️ Please configure valid config & tokenizer paths first")
else:
    loaded_name = os.path.basename(st.session_state.get('loaded_weight_path', ''))
    st.sidebar.success(f"✅ {get_text('model_loaded')}: {loaded_name}")
    if st.session_state.get('loaded_weight_path', '') != weight_path:
        st.sidebar.warning(get_text('path_changed'))
    if st.sidebar.button(get_text('unload_model'), width="stretch"):
        keys_to_clear = ['model', 'tokenizer', 'model_loaded', 'loaded_weight_path',
                         'messages', 'chat_messages']
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()
    if st.sidebar.button(get_text('new_chat'), width="stretch"):
        clear_chat_messages()
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

image_url = "https://raw.githubusercontent.com/1057237562/Instinct/main/images/logo2.png"

st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    f'<img src="{image_url}" style="width: 40px; height: 40px; "> '
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

    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
            if i == len(messages) - 1:
                render_regenerate_button(i)
        else:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{message["content"]}</div></div>',
                unsafe_allow_html=True)

    prompt = st.chat_input(key="input", placeholder=get_text('send'))

    if st.session_state.pop('regenerate', False):
        prompt = st.session_state.pop('last_user_message', None)
        st.session_state.pop('regenerate_index', None)

    if prompt:
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{prompt}</div></div>',
            unsafe_allow_html=True)
        logit_lens_slot = st.container() if st.session_state.get('enable_logit_lens', False) else None
        messages.append({"role": "user", "content": prompt})
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        placeholder = st.empty()

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
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "max_new_tokens": st.session_state.max_new_tokens,
            "num_return_sequences": 1,
            "do_sample": True,
            "attention_mask": inputs.attention_mask,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "temperature": st.session_state.temperature,
            "repetition_penalty": st.session_state.repetition_penalty,
            "top_p": 0.85,
            "streamer": streamer,
        }
        if lens is not None:
            generation_kwargs["layer_callback"] = lens.layer_cb

        Thread(target=model.generate, kwargs=generation_kwargs).start()

        answer = ""
        for new_text in streamer:
            answer += new_text
            placeholder.markdown(process_assistant_content(answer, is_streaming=True), unsafe_allow_html=True)
            if lens is not None:
                lens.render()
        if lens is not None:
            lens.render(final=True)
            # 工具调用多轮复用 generation_kwargs：摘掉钩子，避免后续轮次继续污染已冻结的逐层解释表
            generation_kwargs.pop("layer_callback", None)

        full_answer = answer
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
                    tool_results.append(f'<div style="background: rgba(90, 130, 110, 0.20); border: 1px solid rgba(150, 200, 170, 0.30); padding: 10px 12px; border-radius: 12px; margin: 6px 0;"><div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalled</div><div><b>{tc.get("name", "")}</b>: {json.dumps(result, ensure_ascii=False)}</div></div>')
                except:
                    pass
            full_answer += "\n" + "\n".join(tool_results) + "\n"
            placeholder.markdown(process_assistant_content(full_answer, is_streaming=True), unsafe_allow_html=True)
            new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs["input_ids"] = inputs.input_ids
            generation_kwargs["attention_mask"] = inputs.attention_mask
            generation_kwargs["max_new_tokens"] = st.session_state.max_new_tokens
            generation_kwargs["streamer"] = streamer
            Thread(target=model.generate, kwargs=generation_kwargs).start()
            answer = ""
            for new_text in streamer:
                answer += new_text
                placeholder.markdown(process_assistant_content(full_answer + answer, is_streaming=True), unsafe_allow_html=True)
            full_answer += answer
        answer = full_answer

        messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        render_regenerate_button(len(messages) - 1)


if __name__ == "__main__":
    main()
