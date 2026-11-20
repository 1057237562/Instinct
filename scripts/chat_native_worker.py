"""Torch-free cuBLAS chat worker; JSON lines on stdout, diagnostics on stderr."""
import contextlib
import json
import importlib.util
import os
from pathlib import Path
import queue
import sys
import threading

import numpy as np


def enable_fast_decode(model):
    if os.environ.get('INSTINCT_NATIVE_CHAT_FAST', '1') == '0':
        return False
    name = '_instinct_native_chat_decode'
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / 'model/native_chat_decode.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].install(model)


def inference_config(config):
    """Keep topology/RoPE, replacing training-only precision policy with BF16."""
    config = dict(config)
    for key in ('native_fp8_recipe', 'native_muon_gram_recurrence',
                'native_mxfp8_gradient_dtype', 'native_mxfp8_small_projections',
                'native_mxfp8_cache_ffn', 'native_mxfp8_fast_accum'):
        config.pop(key, None)
    config['use_grad_checkpoint'] = 0
    return config


def stream_tokens(model, request, cancelled, sample):
    """Native cached decode, including the runtime's LongRoPE cache transition."""
    prompt = request['tokens']
    limit = model.config['max_position_embeddings']
    if not prompt or len(prompt) >= limit:
        raise ValueError(f'Prompt must contain 1..{limit - 1} tokens; shorten chat history.')
    if any(token < 0 or token >= model.config['vocab_size'] for token in prompt):
        raise ValueError('Tokenizer vocabulary does not match the checkpoint')
    eos = request.get('eos_token_id')
    eos = set(eos if isinstance(eos, list) else [eos])
    rng = np.random.default_rng(request['seed'])
    e = model.e
    requested = min(limit, len(prompt) + max(0, request['max_new_tokens']))
    initial = min(requested, len(prompt) + 256)
    model._chat_cache_capacity = min(limit, ((initial + 255) // 256) * 256)
    mark = e.backend.mark()
    cache, generated, current = None, [], list(prompt)
    fast = sys.modules.get('_instinct_native_chat_decode') if hasattr(model, '_chat_original_attention') else None
    graph = None
    try:
        with e.no_grad():
            for _ in range(min(request['max_new_tokens'], limit - len(prompt))):
                if cancelled.is_set():
                    break
                if cache is not None and getattr(model, 'short_rope', None) is not None:
                    original = model.config['rope_scaling']['original_max_position_embeddings']
                    if not cache.get('rope_long', False) and cache['position'] + len(current) > original:
                        if graph is not None:
                            graph.close()
                            graph = None
                        cache, current = None, [*prompt, *generated]
                        e.backend.release_since(mark)
                if fast is not None and cache is not None:
                    capacity = cache['0']['keys'].shape[1]
                    if cache['position'] >= capacity:
                        if graph is not None:
                            graph.close()
                            graph = None
                        model._chat_cache_capacity = min(limit, max(capacity * 2, cache['position'] + 1))
                        fast.grow_cache(model, cache, model._chat_cache_capacity)
                if fast is not None and cache is not None and os.environ.get('INSTINCT_NATIVE_CHAT_GRAPH', '1') != '0':
                    if graph is None:
                        graph = fast.DecodeGraph(model, cache, current[0])
                    values = graph.step(current[0], cache['position'])
                    cache['position'] += 1
                else:
                    output = model(np.asarray([current], np.int64), cache=cache, use_cache=True)
                    cache = output.cache
                    values = output.logits if output.logits.shape[1] == 1 else output.logits[:, -1, :]
                logits = e.numpy(values).reshape(-1)
                token, _ = sample(logits, rng, [*prompt, *generated],
                                  request['temperature'], request['top_k'], request['top_p'],
                                  request['repetition_penalty'])
                generated.append(token)
                e.backend.release_since(mark, keep=model.cache_tensors(cache))
                yield token
                if token in eos:
                    break
                current = [token]
    finally:
        if graph is not None:
            graph.close()
        e.backend.release_since(mark)


def main():
    runtime_root, checkpoint = sys.argv[1:3]
    sys.path.insert(0, runtime_root)
    protocol = sys.stdout

    def emit(event, **values):
        protocol.write(json.dumps(dict(event=event, **values), ensure_ascii=False) + '\n')
        protocol.flush()

    commands = queue.Queue()

    def read_commands():
        active = threading.Event()
        try:
            for line in sys.stdin:
                command = json.loads(line)
                if command['op'] == 'cancel':
                    active.set()
                else:
                    if command['op'] == 'generate':
                        active = threading.Event()
                        command['_cancelled'] = active
                    commands.put(command)
        finally:
            active.set()
            commands.put({'op': 'close'})

    threading.Thread(target=read_commands, daemon=True).start()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from model.dense_cublas.memory import TrainingBackend
            from model.dense_cublas.autograd import Engine
            from model.dense_cublas.model import Model
            from model.dense_cublas.checkpoint import metadata, load
            from model.dense_cublas.rollout import sample

            meta = metadata(checkpoint)
            if meta.get('format') != 'instinct-cublas-v1':
                raise ValueError('Unsupported native checkpoint format')
            # The arena subclass supplies mark/release for KV-cache ownership;
            # no trainer, backward graph or optimizer is constructed.
            with TrainingBackend() as backend:
                model = Model(Engine(backend, fp8=False), inference_config(meta['config']),
                              param_dtype='bf16', activation_dtype='bf16', kv_cache_dtype='bf16')
                load(checkpoint, model)  # No optimizer: only model/ members are read.
                model.freeze()
                optimized = enable_fast_decode(model)
                emit('ready', config=model.config, backend='cuBLASLt', precision='bf16', fast_decode=optimized,
                     cuda_graph=optimized and os.environ.get('INSTINCT_NATIVE_CHAT_GRAPH', '1') != '0')
                while True:
                    request = commands.get()
                    if request['op'] == 'close':
                        break
                    cancelled = request.pop('_cancelled')
                    try:
                        for token in stream_tokens(model, request, cancelled, sample):
                            emit('token', token=token)
                    except Exception as exc:
                        emit('error', message=f'{type(exc).__name__}: {exc}')
                    finally:
                        emit('done')
    except Exception as exc:
        emit('error', message=f'{type(exc).__name__}: {exc}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
