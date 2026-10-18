import argparse
import json
import re
import os
import sys
import uuid

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trainer.compile_cache  # noqa: F401 -- configure Inductor before torch
import datasets  # noqa: F401 -- Windows DLL import order
import time
import torch
import warnings
import uvicorn

from threading import Lock, Thread
from queue import Queue
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, PrivateAttr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_instinct import InstinctConfig, InstinctForCausalLM
from model.model_lora import apply_lora, load_lora
from model.inference_runtime import optimize_inference, select_inference_dtype, warmup_decode
from scripts.openai_responses import create_responses_router
from scripts.stream_metrics import GenerationMetricsStreamer

warnings.filterwarnings('ignore')

app = FastAPI()
# Optimized generation reuses the model's static cache and captured graph.
# They belong to one active producer, including across JSON and SSE requests.
generation_lock = Lock()


def init_model(args):
    target_device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        if args.config_path:
            with open(args.config_path, 'r', encoding='utf-8') as config_file:
                config_kwargs = json.load(config_file)
        else:
            config_kwargs = {
                'hidden_size': args.hidden_size,
                'num_hidden_layers': args.num_hidden_layers,
                'max_position_embeddings': args.max_seq_len,
                'use_moe': bool(args.use_moe),
                'inference_rope_scaling': args.inference_rope_scaling,
            }
        model = InstinctForCausalLM(InstinctConfig(**config_kwargs))
        moe_suffix = '_moe' if model.config.use_moe else ''
        ckp = args.weight_path or f'../{args.save_dir}/{args.weight}_{model.config.hidden_size}{moe_suffix}.pth'
        state = torch.load(ckp, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'../{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    print(f'Instinct模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    dtype_name = getattr(args, 'dtype', 'auto')
    dtypes = {'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16,
              'fp16': torch.float16, 'float16': torch.float16, 'fp32': torch.float32}
    dtype = select_inference_dtype(target_device) if dtype_name == 'auto' else dtypes[dtype_name]
    model = model.to(dtype=dtype).eval().to(target_device)
    compile_mode = getattr(args, 'inference_compile', None) or os.environ.get('INSTINCT_INFERENCE_COMPILE', 'auto')
    if compile_mode not in {'off', 'auto', 'full', 'reduce-overhead'}:
        raise ValueError(f'Unsupported inference compile mode: {compile_mode}')
    model = optimize_inference(model, compile_mode)
    if compile_mode == 'auto' and target_device.startswith('cuda') and getattr(model, '_static_cache_ok', False):
        warmup_decode(model)
    print(f'[API] device={target_device}, dtype={dtype}, inference_compile={compile_mode}, '
          f'static_cache={getattr(model, "_static_cache_ok", False)}', flush=True)
    return model, tokenizer


class ChatRequest(BaseModel):
    _request_started: float | None = PrivateAttr(default=None)
    model: str
    messages: list
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.92, gt=0, le=1)
    max_tokens: int = Field(default=8192, ge=1)
    stream: bool = False
    tools: list = Field(default_factory=list)
    open_thinking: bool = False
    chat_template_kwargs: dict | None = None
    
    def get_open_thinking(self) -> bool:
        """兼容多种方式开启 thinking"""
        if self.open_thinking:
            return True
        if self.chat_template_kwargs:
            return self.chat_template_kwargs.get('open_thinking', False) or \
                   self.chat_template_kwargs.get('enable_thinking', False)
        return False


class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


def parse_response(text):
    reasoning_content = None
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    elif '</think>' in text:
        parts = text.split('</think>', 1)
        reasoning_content = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ''
    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        try:
            call = json.loads(m.strip())
            tool_calls.append({"id": f"call_{int(time.time())}_{i}", "type": "function", "function": {"name": call.get("name", ""), "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)}})
        except Exception:
            pass
    if tool_calls:
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    return text.strip(), reasoning_content, tool_calls or None


class StreamTextParser:
    """Hold partial delimiters so thinking/tool markup never becomes text."""
    def __init__(self, open_thinking=False):
        self.state = "reasoning" if open_thinking else "content"
        self.pending = ""

    def feed(self, text, final=False):
        self.pending += text
        deltas = []
        while self.pending:
            markers = {
                "reasoning": {"<think>": "reasoning", "</think>": "content"},
                "content": {"<think>": "reasoning", "<tool_call>": "tool"},
                "tool": {"</tool_call>": "content"},
            }[self.state]
            matches = [(self.pending.find(marker), marker) for marker in markers if marker in self.pending]
            if matches:
                end, marker = min(matches)
                rest = self.pending[end + len(marker):]
            else:
                held = 0
                if not final:
                    for marker in markers:
                        for length in range(1, min(len(marker), len(self.pending) + 1)):
                            if self.pending.endswith(marker[:length]):
                                held = max(held, length)
                end = len(self.pending) - held
                rest = self.pending[end:]
            value = self.pending[:end]
            if value and self.state != "tool":
                deltas.append({"reasoning_content" if self.state == "reasoning" else "content": value})
            self.pending = rest
            if not matches:
                break
            self.state = markers[marker]
        return deltas


def generation_finish_reason(generated_ids, prompt_length, max_tokens):
    if generated_ids.shape[1] - prompt_length < max_tokens:
        return "stop"
    eos = tokenizer.eos_token_id
    eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
    return "stop" if generated_ids[0, -1].item() in eos_ids else "length"


def generate_stream_response(messages, temperature, top_p, max_tokens, tools=None, open_thinking=False, request_started=None):
    try:
        metrics = GenerationMetricsStreamer(request_started=request_started)
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools or None, open_thinking=open_thinking)
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        queue = Queue()
        metrics.streamer = CustomStreamer(tokenizer, queue)
        usage = {}
        generation_info = {}

        def _generate():
            try:
                with generation_lock, torch.inference_mode():
                    metrics.start_generation()
                    generated_ids = model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else 1.0,
                        top_p=top_p,
                        attention_mask=inputs.attention_mask,
                        eos_token_id=tokenizer.eos_token_id,
                        streamer=metrics,
                    )
                    metrics.finish_generation()
                usage.update(prompt_tokens=inputs.input_ids.shape[1],
                             completion_tokens=generated_ids.shape[1] - inputs.input_ids.shape[1],
                             total_tokens=generated_ids.shape[1])
                generation_info["finish_reason"] = generation_finish_reason(generated_ids, inputs.input_ids.shape[1], max_tokens)
            except Exception as e:
                queue.put({"error": str(e)})
                queue.put(None)

        worker = Thread(target=_generate)
        worker.start()

        full_text = ""
        parser = StreamTextParser(open_thinking)

        while True:
            text = queue.get()
            if text is None:
                break
            if isinstance(text, dict):
                yield json.dumps(text, ensure_ascii=False)
                continue
            full_text += text

            for delta in parser.feed(text):
                yield json.dumps({"choices": [{"delta": delta}]}, ensure_ascii=False)

        worker.join()
        for delta in parser.feed("", final=True):
            yield json.dumps({"choices": [{"delta": delta}]}, ensure_ascii=False)
        _, _, tool_calls = parse_response(full_text)
        if tool_calls:
            yield json.dumps({"choices": [{"delta": {"tool_calls": tool_calls}}]}, ensure_ascii=False)
        yield json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls" if tool_calls else generation_info.get("finish_reason", "stop")}], "usage": usage,
                         "performance": metrics.snapshot(usage["completion_tokens"])}, ensure_ascii=False)

    except Exception as e:
        yield json.dumps({"error": str(e)})


def generate_sse_response(request, request_started=None):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def event(choices, **extra):
        return "data: " + json.dumps({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": choices,
            **extra,
        }, ensure_ascii=False) + "\n\n"

    yield event([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])
    for chunk in generate_stream_response(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        tools=request.tools,
        open_thinking=request.get_open_thinking(),
        request_started=request_started,
    ):
        payload = json.loads(chunk)
        if "error" in payload:
            yield f"data: {chunk}\n\n"
            break
        choices = payload["choices"]
        for index, choice in enumerate(choices):
            choice.setdefault("index", index)
            choice.setdefault("finish_reason", None)
            for tool_index, call in enumerate(choice.get("delta", {}).get("tool_calls", [])):
                call.setdefault("index", tool_index)
        extra = {key: payload[key] for key in ("usage", "performance") if key in payload}
        yield event(choices, **extra)
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        started = request._request_started if request._request_started is not None else time.perf_counter()
        if request.stream:
            return StreamingResponse(
                generate_sse_response(request, request_started=started),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        else:
            metrics = GenerationMetricsStreamer(request_started=started)
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=request.tools or None,
                open_thinking=request.get_open_thinking()
            )
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            with generation_lock, torch.inference_mode():
                metrics.start_generation()
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=request.max_tokens,
                    do_sample=request.temperature > 0,
                    attention_mask=inputs["attention_mask"],
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=request.top_p,
                    temperature=request.temperature if request.temperature > 0 else 1.0,
                    streamer=metrics,
                )
                metrics.finish_generation()
                answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            content, reasoning_content, tool_calls = parse_response(answer)
            message = {"role": "assistant", "content": content}
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            if tool_calls:
                message["tool_calls"] = tool_calls
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "performance": metrics.snapshot(generated_ids.shape[1] - inputs["input_ids"].shape[1]),
                "usage": {
                    "prompt_tokens": inputs["input_ids"].shape[1],
                    "completion_tokens": generated_ids.shape[1] - inputs["input_ids"].shape[1],
                    "total_tokens": generated_ids.shape[1],
                },
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else generation_finish_reason(generated_ids, inputs["input_ids"].shape[1], request.max_tokens)
                    }
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(create_responses_router(
    ChatRequest, chat_completions, lambda **kwargs: generate_stream_response(**kwargs),
))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server for Instinct")
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, dpo, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--weight_path', default=None, type=str, help="原生 .pth 权重文件路径，优先于 --weight/--save_dir")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=8192, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--config_path', default=None, type=str, help="原生权重对应的模型配置 JSON（LongRoPE 必须通过此项加载缩放向量）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用默认 YaRN；LongRoPE 请使用 --config_path")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--dtype', default='auto', choices=['auto', 'bf16', 'bfloat16', 'fp16', 'float16', 'fp32'], help="推理权重精度；auto 与 Chat WebUI 一致，在 Ampere+ CUDA 上使用 BF16")
    parser.add_argument('--inference_compile', default=None, choices=['off', 'auto', 'full', 'reduce-overhead'], help="推理优化模式；默认读取 INSTINCT_INFERENCE_COMPILE，未设置时为 auto")
    parser.add_argument('--host', default='0.0.0.0', type=str, help="API 监听地址")
    parser.add_argument('--port', default=8998, type=int, help="API 监听端口")
    args = parser.parse_args()
    device = args.device
    model, tokenizer = init_model(args)
    uvicorn.run(app, host=args.host, port=args.port)
