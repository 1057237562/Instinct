"""Torch-free CUDA parity and optional checkpoint throughput checks.

Run with training paused. Reports belong under experiments/results or tmp.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np


def bf16(values):
    bits = np.asarray(values, np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).view(np.float32)


def run(args):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'scripts'))
    sys.path.insert(0, str(Path(args.runtime_root).resolve()))
    from chat_native_worker import enable_fast_decode, inference_config, stream_tokens
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    from model.dense_cublas.checkpoint import metadata, load
    from model.dense_cublas.rollout import sample

    report = dict(kernel_cases=[], tiny_models=[], checkpoint_runs=[])
    with TrainingBackend() as backend:
        engine = Engine(backend, fp8=False)
        cfg = dict(hidden_size=64, num_hidden_layers=2, vocab_size=128, intermediate_size=128,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256)
        model = Model(engine, cfg, param_dtype='bf16', activation_dtype='bf16', kv_cache_dtype='bf16').freeze()
        assert enable_fast_decode(model)
        fast = sys.modules['_instinct_native_chat_decode']
        rng = np.random.default_rng(19)
        with engine.no_grad():
            for dim, heads, kv_heads in ((32, 4, 2), (96, 16, 4), (128, 4, 1), (256, 2, 2)):
                for length in (1, 31, 127, 128, 129, 257, 4095):
                    mark = backend.mark()
                    cap = 4096 if length > 256 else 256
                    q = bf16(rng.normal(size=(1, 1, heads, dim)))
                    k = bf16(rng.normal(size=(1, cap, kv_heads, dim)))
                    v = bf16(rng.normal(size=k.shape))
                    k[:, length:] = np.nan
                    v[:, length:] = np.nan  # Any unmasked capacity reads fail.
                    result = fast.fused_decode(engine, engine.array(q, 'bf16'),
                                               engine.array(k, 'bf16'), engine.array(v, 'bf16'), length)
                    actual = engine.numpy(result)[0, 0]
                    expected = []
                    for head in range(heads):
                        kh = head // (heads // kv_heads)
                        score = k[0, :length, kh] @ q[0, 0, head] / np.sqrt(dim)
                        prob = np.exp(score - score.max())
                        prob = bf16(prob / prob.sum())
                        expected.append(bf16(prob @ v[0, :length, kh]))
                    error = float(np.max(np.abs(actual - expected)))
                    np.testing.assert_allclose(actual, expected, atol=.02, rtol=.02)
                    report['kernel_cases'].append(dict(dim=dim, heads=heads, kv_heads=kv_heads,
                                                        length=length, max_abs=error))
                    backend.release_since(mark)

        def collect(current, tokens, count, forced=None):
            logits, timestamps, output = [], [], []
            def choose(values, *sampling):
                logits.append(values.copy())
                if forced is not None:
                    return int(forced[len(logits) - 1]), 0.
                return sample(values, *sampling)
            request = dict(tokens=tokens, seed=42, max_new_tokens=count, eos_token_id=None,
                           temperature=0., top_k=0, top_p=1., repetition_penalty=1.)
            start = time.perf_counter()
            for token in stream_tokens(current, request, threading.Event(), choose):
                timestamps.append(time.perf_counter() - start)
                output.append(token)
            return dict(tokens=output, ttft=timestamps[0], total=timestamps[-1],
                        decode_tps=(count - 1) / (timestamps[-1] - timestamps[0]),
                        compiled=len(backend._cache)), np.stack(logits)

        for longrope in (False, True):
            mark = backend.mark()
            config = dict(cfg)
            if longrope:
                config['rope_scaling'] = dict(rope_type='longrope', factor=32.,
                    original_max_position_embeddings=8, short_factor=[1.] * 8, long_factor=[2.] * 8)
            tiny = Model(engine, config, param_dtype='bf16', activation_dtype='bf16', kv_cache_dtype='bf16').freeze()
            old, reference = collect(tiny, [1, 2, 3, 4, 5, 6], 10)
            assert enable_fast_decode(tiny)
            new, actual = collect(tiny, [1, 2, 3, 4, 5, 6], 10, old['tokens'])
            np.testing.assert_allclose(actual, reference, atol=.025, rtol=.03)
            # Repeat with a different prompt, then cancel and start another turn.
            collect(tiny, [4, 3, 2], 4)
            cancelled = threading.Event()
            iterator = stream_tokens(tiny, dict(tokens=[1, 2], seed=1, max_new_tokens=12,
                eos_token_id=None, temperature=0., top_k=0, top_p=1., repetition_penalty=1.), cancelled, sample)
            next(iterator)
            cancelled.set()
            assert list(iterator) == []
            collect(tiny, [1, 2, 3], 4)
            report['tiny_models'].append(dict(longrope=longrope, max_abs=float(np.max(np.abs(actual-reference)))))
            backend.release_since(mark)

        if args.checkpoint:
            meta = metadata(args.checkpoint)
            current = Model(engine, inference_config(meta['config']), param_dtype='bf16',
                            activation_dtype='bf16', kv_cache_dtype='bf16').freeze()
            load(args.checkpoint, current)
            prompt = [1, 1602, 36, 1343] * 8
            report['model'] = dict(checkpoint=args.checkpoint, config=meta['config'],
                                  dtype='bf16', batch=1, prompt_tokens=len(prompt), new_tokens=16)
            old, reference = collect(current, prompt, 16)
            old_warm, _ = collect(current, prompt, 16)
            assert enable_fast_decode(current)
            new, actual = collect(current, prompt, 16, old['tokens'])
            new_warm, _ = collect(current, prompt, 16)
            np.testing.assert_allclose(actual, reference, atol=.15, rtol=.04)
            report['logits_max_abs'] = float(np.max(np.abs(actual-reference)))
            report['logits_mean_abs'] = float(np.mean(np.abs(actual-reference)))
            report['greedy_agreement'] = float(np.mean(actual.argmax(-1) == reference.argmax(-1)))
            report['checkpoint_runs'] = [dict(mode=mode, **item) for mode, item in
                [('reference_first', old), ('reference_warm', old_warm), ('fast_first', new), ('fast_warm', new_warm)]]
            # New decode positions must not compile more kernels in the same capacity bucket.
            before = len(backend._cache)
            collect(current, prompt, 32)
            assert len(backend._cache) == before
    assert 'torch' not in sys.modules, 'native inference must remain Torch-free'
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--output')
    run(parser.parse_args())
