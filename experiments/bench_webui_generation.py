"""Measure native Chat WebUI generation, including its HF-compatible kwargs.

Run from the repo root with PYTHONUTF8=1. --ignore-eos forces a full-length
answer for a conservative throughput check (default: respect model EOS).
Reports both end-to-end wall time and the UI's decode metric; no timing reset
or shorter max_new_tokens allowance is used to meet the throughput target.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from threading import Thread

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import trainer.compile_cache  # noqa: E402,F401
import datasets  # noqa: E402,F401 -- Windows DLL import order
import torch
from transformers import AutoTokenizer, TextIteratorStreamer

from model.inference_runtime import optimize_inference, select_inference_dtype, warmup_decode
from model.model_instinct import InstinctConfig, InstinctForCausalLM
from scripts.stream_metrics import TokenRateStreamer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='checkpoints/pretrain_20260919_174858_512_moe.json')
    parser.add_argument('--weight', default='out/pretrain_20260919_174858_512_moe.pth')
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--turns', type=int, default=2)
    parser.add_argument('--ignore-eos', action='store_true')
    args = parser.parse_args()
    with open(ROOT / args.config, encoding='utf-8') as handle:
        config = InstinctConfig(**json.load(handle))
    model = InstinctForCausalLM(config)
    weights = torch.load(ROOT / args.weight, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=False)
    del weights
    model = model.to(dtype=select_inference_dtype('cuda')).eval().cuda()
    optimize_inference(model, 'auto')
    warmup_decode(model)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', trust_remote_code=True)
    prompt = tokenizer.apply_chat_template([
        {'role': 'system', 'content': '你是Instinct，一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题。'},
        {'role': 'user', 'content': '请详细解释快速排序算法，并给出 Python 示例。'},
    ], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to('cuda')
    print(f'prompt_tokens={inputs.input_ids.shape[1]} max_new_tokens={args.max_new_tokens} '
          f'ignore_eos={args.ignore_eos}', flush=True)
    for turn in range(args.turns):
        torch.manual_seed(42 + turn)
        delegate = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True,
                                        timeout=60)
        streamer = TokenRateStreamer(delegate)
        result, errors = [], []

        def generate():
            try:
                result.append(model.generate(
                    input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                    max_new_tokens=args.max_new_tokens, num_return_sequences=1,
                    do_sample=True, temperature=0.9, repetition_penalty=1.1,
                    top_p=0.85, streamer=streamer, stream_chunk_size=16,
                    # Keep this compatibility argument: omitting it masked the
                    # original WebUI graph-eligibility regression in benchmarks.
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=None if args.ignore_eos else tokenizer.eos_token_id,
                ))
            except BaseException as exc:
                errors.append(exc)
                streamer.end()

        torch.cuda.synchronize()
        started = time.perf_counter()
        worker = Thread(target=generate)
        worker.start()
        try:
            for _ in streamer:
                pass
        finally:
            worker.join(timeout=60)
        if errors:
            raise errors[0]
        if worker.is_alive():
            raise RuntimeError('generation thread did not finish')
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        count = result[0].shape[1] - inputs.input_ids.shape[1]
        if args.ignore_eos:
            assert count == args.max_new_tokens
        stats = streamer.snapshot()
        print(json.dumps(dict(turn=turn + 1, tokens=count, seconds=round(elapsed, 3),
                              end_to_end_tps=round(count / elapsed, 1),
                              decode_tps=round(stats['tokens_per_second'], 1),
                              cache_capacities=sorted(model._decode_states))), flush=True)


if __name__ == '__main__':
    main()
