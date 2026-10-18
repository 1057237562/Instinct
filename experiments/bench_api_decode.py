"""Interleaved API A/B: legacy FP16/eager versus Chat WebUI BF16/auto.

Uses best.pth for both models and actual ASGI JSON/SSE endpoints. Disables EOS
for fixed output lengths. Run from repo root with PYTHONUTF8=1, preferably
while other GPU workloads are idle. Results measure server processing, without
network transport. Both models are loaded at once for interleaved comparison.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import trainer.compile_cache  # noqa: F401
import datasets  # noqa: F401 -- before torch on Windows
import torch
from fastapi.testclient import TestClient
from scripts import serve_openai_api as api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--output', default='experiments/results/api_decode_ab.json')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    if args.tokens < 2 or args.rounds < 1:
        parser.error('tokens must be >= 2 and rounds >= 1')
    api.device = 'cuda'
    loader = dict(load_from=str(ROOT / 'model'), config_path=str(ROOT / 'out/best.json'),
                  weight_path=str(ROOT / 'out/best.pth'), device=api.device, lora_weight='None')
    models = {}
    for name, dtype, mode in [('legacy', 'fp16', 'off'), ('webui', 'auto', 'auto')]:
        models[name], tokenizer = api.init_model(SimpleNamespace(**loader, dtype=dtype, inference_compile=mode))
        # Keep both paths generating exactly the requested number of tokens.
        tokenizer.eos_token_id = None
        api.tokenizer = tokenizer
    prompt = '请用中文详细解释二分查找算法的原理、边界条件与复杂度，并给出实现步骤。'

    def request(client, name, protocol, stream):
        api.model = models[name]
        body = {'model': 'instinct', 'temperature': 0, 'stream': stream}
        if protocol == 'chat':
            body.update(messages=[{'role': 'user', 'content': prompt}], max_tokens=args.tokens)
            response = client.post('/v1/chat/completions', json=body)
            result = json.loads(response.text.strip().split('\n\n')[-2][6:]) if stream else response.json()
        else:
            body.update(input=prompt, max_output_tokens=args.tokens, store=False)
            response = client.post('/v1/responses', json=body)
            result = json.loads(response.text.strip().split('\n\n')[-1].split('\ndata: ', 1)[1])['response'] if stream else response.json()
        if response.status_code != 200 or 'performance' not in result:
            raise RuntimeError(response.text[:1000])
        stats = result['performance']
        assert stats['output_tokens'] == args.tokens
        return stats

    samples = {name: {} for name in models}
    with TestClient(api.app) as client:
        for name in models:
            request(client, name, 'chat', False)  # Settle first-use kernels and prompt cache.
        for protocol in ('chat', 'responses'):
            for stream in (False, True):
                key = f'{protocol}_{"sse" if stream else "json"}'
                for name in models:
                    samples[name][key] = []
                for _ in range(args.rounds):
                    for name in models:
                        stats = request(client, name, protocol, stream)
                        samples[name][key].append(stats)
                        print(f'{name:6s} {key:14s} {stats["tokens_per_second"]:7.1f} tok/s '
                              f'TPOT={stats["tpot_ms"]:.2f} ms', flush=True)
    medians = {name: {key: statistics.median(row['tokens_per_second'] for row in rows)
                      for key, rows in paths.items()} for name, paths in samples.items()}
    graph_counts = {name: sum(state.graph is not None for state in model._decode_states.values())
                    for name, model in models.items()}
    report = {'gpu': torch.cuda.get_device_name(), 'config': loader['config_path'], 'weight': loader['weight_path'],
              'tokens': args.tokens, 'rounds': args.rounds, 'median_tokens_per_second': medians,
              'captured_decode_graphs': graph_counts, 'samples': samples}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'median_tokens_per_second': medians, 'captured_decode_graphs': graph_counts}, indent=2))
    print(f'Report: {output}')


if __name__ == '__main__':
    main()
