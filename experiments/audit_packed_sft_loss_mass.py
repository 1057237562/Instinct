"""Audit what the SFT cross-entropy actually supervises, straight from the Arrow cache.

Reads the packed SFT cache that the native trainer consumes
(``native_packed_16k``), rebuilds the labels exactly like
``model/dense_cublas/sft_data.py`` (``ids where loss_mask, else -100`` with the
first token of every packed document cleared), then reports:

* supervised share of input tokens and the assistant-only label proof,
* predict-token mass of ``<think>`` bodies vs the visible answer,
* the per-turn thinking-length distribution and which bucket owns the token mass
  (the CE denominator is a plain token mean, so mass == gradient weight),
* the opening tokens of a think body (what the model is taught to say first),
* language mix of thinking vs answer,
* how many packed documents exceed the 4096-token native RoPE context.

Usage (CPU only, no torch):

    python experiments/audit_packed_sft_loss_mass.py \
        --manifest dataset/instinctv2_sft_concise_reasoning_635m_v4/native_packed_16k/manifest.json \
        --blocks-per-shard 300 --json tmp/v4_packed_loss_mass.json
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
from transformers import AutoTokenizer

THINK_OPEN, THINK_CLOSE = 25, 26


def cjk_share(text):
    if not text:
        return 0.0
    return sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff') / len(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--manifest', default=(
        'dataset/instinctv2_sft_concise_reasoning_635m_v4/native_packed_16k/manifest.json'))
    parser.add_argument('--tokenizer', default='model')
    parser.add_argument('--blocks-per-shard', type=int, default=300)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--json', default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    bos = tok(f'{tok.bos_token}assistant\n', add_special_tokens=False).input_ids
    rng = random.Random(args.seed)

    total_tokens = total_supervised = 0
    runs = bad_starts = 0
    think = answer = tags = 0
    lengths = []
    openers = Counter()
    think_cjk, answer_cjk = [], []
    docs = long_docs = 0
    blocks = blocks_long = 0
    sampled_tokens = tokens_long_blocks = 0

    for item in manifest['files']:
        path = (manifest_path.parent / item['name']).resolve()
        with pa.memory_map(str(path), 'r') as mapped:
            table = ipc.open_stream(mapped).read_all()
        total_tokens += pc.sum(table['valid_tokens']).as_py()
        total_supervised += pc.sum(table['train_tokens']).as_py()
        for _ in range(args.blocks_per_shard):
            row = table.slice(rng.randrange(len(table)), 1).to_pylist()[0]
            ids, mask, valid = row['input_ids'], row['loss_mask'], row['valid_tokens']
            block_lengths = row['segment_lengths']
            docs += len(block_lengths)
            long_docs += sum(1 for x in block_lengths if x > 4096)
            blocks += 1
            sampled_tokens += valid
            if max(block_lengths) > 4096:
                blocks_long += 1
                tokens_long_blocks += valid
            i = 0
            while i < valid:
                if not mask[i]:
                    i += 1
                    continue
                start = i
                while i < valid and mask[i]:
                    i += 1
                end = i
                runs += 1
                if start < len(bos) or ids[start - len(bos):start] != bos:
                    bad_starts += 1
                body = ids[start:end]
                if (THINK_OPEN in body and THINK_CLOSE in body
                        and body.index(THINK_OPEN) < body.index(THINK_CLOSE)):
                    opening, closing = body.index(THINK_OPEN), body.index(THINK_CLOSE)
                    n_think = closing - opening - 1
                    think += n_think
                    tags += 2
                    answer += len(body) - closing - 1
                    lengths.append(n_think)
                    openers[tuple(body[opening + 2:opening + 2 + 3])] += 1
                    if n_think:
                        think_cjk.append(cjk_share(tok.decode(body[opening + 1:closing])))
                        answer_cjk.append(cjk_share(tok.decode(body[closing + 1:])))
                else:
                    answer += len(body)
                    lengths.append(0)

    supervised = think + answer + tags
    with_block = sum(1 for x in lengths if x)
    ordered = sorted(x for x in lengths if x)
    buckets = [(1, 4), (5, 64), (65, 256), (257, 768), (769, 1536),
               (1537, 2560), (2561, 4096), (4097, 10 ** 9)]

    print('cache %d tokens, %d supervised (%.2f%%); sampled %d blocks, %d turns, %d bad starts'
          % (total_tokens, total_supervised, 100.0 * total_supervised / total_tokens,
             blocks, runs, bad_starts))
    print('sampled supervised: think %d (%.1f%%), tags %d, answer %d (%.1f%%)'
          % (think, 100.0 * think / supervised, tags, answer, 100.0 * answer / supervised))
    print('turns with a think block %d (%.1f%%), without %d'
          % (with_block, 100.0 * with_block / runs, runs - with_block))
    print('\nthink tokens | turns | % of turns | think tokens | % of thinking tokens')
    for low, high in buckets:
        n = sum(1 for x in ordered if low <= x <= high)
        mass = sum(x for x in ordered if low <= x <= high)
        print('%5d-%-5d %6d %7.2f%% %12d %8.2f%%'
              % (low, high, n, 100.0 * n / runs, mass, 100.0 * mass / max(think, 1)))
    print('think tokens/turn: p50 %d p75 %d p90 %d p99 %d max %d'
          % (ordered[len(ordered) // 2], ordered[3 * len(ordered) // 4],
             ordered[9 * len(ordered) // 10], ordered[99 * len(ordered) // 100], ordered[-1]))
    for k in (1, 5, 10, 20):
        take = max(1, len(ordered) * k // 100)
        print('top %2d%% longest thinking turns hold %.1f%% of thinking tokens'
              % (k, 100.0 * sum(ordered[-take:]) / max(think, 1)))
    print('\nmost common first tokens after <think>:')
    for ids, count in openers.most_common(8):
        print('  %-28r %d' % (tok.decode(list(ids)), count))
    print('think CJK share mean %.2f; answers mean %.2f'
          % (sum(think_cjk) / max(len(think_cjk), 1), sum(answer_cjk) / max(len(answer_cjk), 1)))
    print('packed documents >4096 tokens %d/%d (%.2f%%); blocks whose longest document >4096 '
          '%d/%d (%.2f%%), holding %.2f%% of sampled tokens'
          % (long_docs, docs, 100.0 * long_docs / docs, blocks_long, blocks,
             100.0 * blocks_long / blocks,
             100.0 * tokens_long_blocks / max(sampled_tokens, 1)))

    if args.json:
        Path(args.json).write_text(json.dumps({
            'tokens': total_tokens, 'supervised': total_supervised, 'turns': runs,
            'think_tokens': think, 'answer_tokens': answer, 'with_block': with_block,
            'p50': ordered[len(ordered) // 2], 'p90': ordered[9 * len(ordered) // 10],
            'p99': ordered[99 * len(ordered) // 100], 'max': ordered[-1],
            'buckets': [[lo, hi, sum(1 for x in ordered if lo <= x <= hi),
                         sum(x for x in ordered if lo <= x <= hi)] for lo, hi in buckets],
            'documents_over_4096': long_docs, 'documents': docs,
            'blocks_over_4096': blocks_long, 'blocks': blocks,
            'sampled_tokens': sampled_tokens,
            'tokens_in_blocks_over_4096': tokens_long_blocks,
        }, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print('wrote', args.json)


if __name__ == '__main__':
    main()
