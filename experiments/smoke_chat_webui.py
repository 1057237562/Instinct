"""Smoke-test the Chat WebUI's default load + generation path, end to end.

Loads exactly like scripts/web_demo.py (bf16, optimize_inference('auto')), then
runs one streaming turn the way the WebUI does, so a break in the static cache,
the decode capture or the streamer shows up here rather than in the UI.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trainer.compile_cache  # noqa: F401 -- before torch, as web_demo does
import datasets  # noqa: F401 -- Windows DLL order
import torch
from transformers import AutoTokenizer

from model.model_instinct import InstinctConfig, InstinctForCausalLM
from model.inference_runtime import optimize_inference, select_inference_dtype, warmup_decode

REPO = Path(__file__).resolve().parents[1]
device = 'cuda'

with open(REPO / 'checkpoints' / 'pretrain_20260919_174858_512_moe.json', encoding='utf-8') as fh:
    config = InstinctConfig(**json.load(fh))
tokenizer = AutoTokenizer.from_pretrained(REPO / 'model', trust_remote_code=True)
model = InstinctForCausalLM(config)
state = torch.load(REPO / 'out' / 'pretrain_20260919_174858_512_moe.pth', map_location='cpu', weights_only=True)
model.load_state_dict(state, strict=False)
del state
model = model.to(dtype=select_inference_dtype(device)).eval().to(device)
compile_mode = os.environ.get('INSTINCT_INFERENCE_COMPILE', 'auto')
model = optimize_inference(model, compile_mode)
if compile_mode == 'auto':
    warmup_decode(model)

messages = [{"role": "system", "content": "你是Instinct，一个乐于助人的AI助手。"},
            {"role": "user", "content": "用一句话介绍你自己。"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
print(f'[smoke] prompt tokens: {inputs.input_ids.shape[1]}')

from transformers import TextIteratorStreamer
from threading import Thread
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
kwargs = dict(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
              max_new_tokens=32, do_sample=True, temperature=0.9, top_p=0.85,
              repetition_penalty=1.1, streamer=streamer, stream_chunk_size=16,
              eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
start = time.perf_counter()
Thread(target=model.generate, kwargs=kwargs).start()
chunks = [chunk for chunk in streamer]
print(f'[smoke] {time.perf_counter() - start:.2f}s, {len(chunks)} chunks: {"".join(chunks)!r}')

# A second turn with a different prompt length: the chat path must not pay a
# compile, and the sequence must keep growing sensibly.
messages.append({"role": "assistant", "content": "".join(chunks)})
messages.append({"role": "user", "content": "再举一个例子。"})
prompt2 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs2 = tokenizer(prompt2, return_tensors='pt', truncation=True).to(device)
streamer2 = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
kwargs.update(input_ids=inputs2.input_ids, attention_mask=inputs2.attention_mask, streamer=streamer2)
start = time.perf_counter()
Thread(target=model.generate, kwargs=kwargs).start()
second = ''.join(streamer2)
print(f'[smoke] turn 2 ({inputs2.input_ids.shape[1]} prompt tokens) '
      f'{time.perf_counter() - start:.2f}s: {second!r}')
print('[smoke] PASS' if chunks and second else '[smoke] FAIL: empty output')
