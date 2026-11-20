"""Audit a compiled InstinctV2 SFT release and exercise the trainer loader."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
from pathlib import Path
import random
import re
import subprocess

import datasets  # noqa: F401, before torch on Windows
from transformers import AutoTokenizer

from build_instinctv2_sft import ROOT, digest, fingerprint, normalize_text, rows, verify_dialogue, verify_replay
from scripts.data_builder.audit_identity_contamination import run as identity_audit
from scripts.data_builder.filter_anomaly_candidates import run as triage
from scripts.data_loader.lm_dataset import SFTDataset


def verify_split(directory, split, metadata, compiler):
    jsonl = directory / f'{split}_identity_clean.jsonl'
    parquet = jsonl.with_suffix('.parquet')
    provenance = directory / f'{split}.provenance.jsonl.gz'
    if digest(jsonl) != metadata['sha256'] or digest(provenance) != metadata['provenance_sha256']:
        raise ValueError(f'{split}: hash mismatch')
    result = subprocess.run([str(compiler), 'verify', str(jsonl), str(parquet), '--format', 'sft',
                             '--rows', str(metadata['rows'])], check=True, capture_output=True, text=True)
    print(result.stdout.strip(), flush=True)
    prompts, groups, checks = set(), set(), {}
    counts = {'rows': 0, 'tool_episodes': 0, 'trace_rows': 0, 'replay_rows': 0,
              'input_tokens': 0, 'supervised_tokens': 0}
    source_rows, source_tokens = Counter(), Counter()
    max_tokens = 0
    for index, (row, meta) in enumerate(zip(rows(jsonl), rows(provenance), strict=True), 1):
        turns = row['conversations']
        verify_dialogue(turns)
        prompt = fingerprint(normalize_text(next(t['content'] for t in turns if t['role'] == 'user')))
        if index != meta['row'] or prompt != meta['first_user_sha256'] or prompt in prompts:
            raise ValueError(f'{split}:{index}: invalid alignment or duplicate prompt')
        if any(re.search(r'instinct\s*v\s*1', t['content'], re.I) for t in turns):
            raise ValueError(f'{split}:{index}: legacy identity')
        if meta['category'] == 'identity' and 'InstinctV2' not in turns[-1]['content']:
            raise ValueError(f'{split}:{index}: incorrect identity target')
        if meta.get('path') == 'dataset/sft_t2t_mini.jsonl' and meta['tool_calls']:
            verify_replay(turns)
            counts['replay_rows'] += 1
        if meta['tool_calls']:
            checks.setdefault('tool', index - 1)
            counts['tool_episodes'] += 1
        if meta['thinking']:
            checks.setdefault(f"think/{meta['category']}", index - 1)
            counts['trace_rows'] += 1
        checks.setdefault(meta['category'], index - 1)
        if meta['tokens'] > max_tokens:
            max_tokens = meta['tokens']
            checks['longest'] = index - 1
        prompts.add(prompt)
        groups.add((meta['category'], meta['group']))
        counts['rows'] += 1
        source_rows[meta['category']] += 1
        source_tokens[meta['category']] += meta['tokens']
        counts['input_tokens'] += meta['tokens']
        counts['supervised_tokens'] += meta['supervised_tokens']
    if (counts['rows'] != metadata['rows'] or counts['input_tokens'] != metadata['tokens']
            or max_tokens != metadata['max_tokens']
            or counts['supervised_tokens'] != metadata['supervised_tokens']
            or dict(source_rows) != metadata['source_rows'] or dict(source_tokens) != metadata['source_tokens']):
        raise ValueError(f'{split}: row or token counts differ from the provenance')
    counts.update(max_tokens=max_tokens, jsonl_sha256=digest(jsonl), parquet_sha256=digest(parquet), parquet_bytes=parquet.stat().st_size,
                  provenance_sha256=digest(provenance), full_parquet_comparison=True)
    return counts, prompts, groups, checks


def trainer_smoke(directory, split, checks, tokenizer, max_length):
    indices = sorted(set(checks.values()))
    verified = {}
    for suffix in ('jsonl', 'parquet'):
        path = directory / f'{split}_identity_clean.{suffix}'
        sample = SFTDataset(str(path), tokenizer, max_length=max_length, packing=False, sample_indices=indices)
        for name, index in checks.items():
            random.seed(1729 + index)
            tokens, labels = sample[indices.index(index)]
            active = labels.ne(-100)
            if len(tokens) != max_length or not active.any() or not tokens[active].equal(labels[active]):
                raise ValueError(f'{split}/{suffix}/{name}: invalid trainer labels')
            target = tokenizer.decode(labels[active].tolist(), skip_special_tokens=False)
            if name == 'longest' and not target.rstrip().endswith(tokenizer.eos_token):
                raise ValueError(f'{split}/{suffix}: longest sample lost its final assistant end token')
            if name == 'tool' and ('<tool_call>' not in target or '<tool_response>' in target):
                raise ValueError('tool call target or tool response masking is wrong')
            if name.startswith('think/') and not re.search(r'<think>\s*\S.*?</think>', target, re.S):
                raise ValueError('thinking target was lost in the actual SFT loader')
            if name == 'identity' and 'InstinctV2' not in target:
                raise ValueError('identity target was lost in the actual SFT loader')
        verified[suffix] = len(indices)
    return verified


def verified_audit(auditor, inputs, directory, name, files_key, parameters):
    candidate_path = directory / f'identity.{name}.jsonl'
    report_path = directory / f'identity.{name}.report.json'
    if report_path.exists():
        report = json.loads(report_path.read_bytes())
    else:
        report = auditor(inputs, candidate_path, report_path, *parameters)
    actual = {Path(item['path']).resolve(): item['sha256'] for item in report[files_key]}
    expected = {path.resolve(): digest(path) for path in inputs}
    if actual != expected or digest(candidate_path) != report['output_sha256']:
        raise ValueError(f'identity {name} report does not describe the current files')
    return report.get('counts', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'dataset/instinctv2_sft_v1')
    args = parser.parse_args()
    directory = args.directory
    build = json.loads((directory / 'build.report.json').read_bytes())
    compiler = ROOT / 'dataset_compiler/target/release/dataset_compiler.exe'
    if not compiler.exists():
        compiler = compiler.with_suffix('')
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {split: pool.submit(verify_split, directory, split, info, compiler)
                for split, info in build['splits'].items()}
        validation, vp, vg, vc = jobs['validation'].result()
        train, tp, tg, tc = jobs['train'].result()
    if vp & tp or vg & tg:
        raise ValueError('train/validation share a prompt or provenance group')

    inputs = [directory / f'{split}_identity_clean.{ext}'
              for split in ('train', 'validation') for ext in ('jsonl', 'parquet')]
    with ProcessPoolExecutor(max_workers=2) as pool:
        triaged = pool.submit(verified_audit, triage, inputs, directory, 'triage', 'input_files',
                              ('identity', None, False))
        audited = pool.submit(verified_audit, identity_audit, inputs, directory, 'audit', 'inputs',
                              (None, False, 'review'))
        triage_result, audit_result = triaged.result(), audited.result()
    expected = 2 * (train['rows'] + validation['rows'])
    for counts in (triage_result, audit_result):
        if counts['rows_scanned'] != expected or any(counts.get(key, 0) for key in (
                'invalid_json', 'contaminated_rows', 'review_rows', 'candidate_rows', 'candidates_written')):
            raise ValueError(f'final identity audit requires review: {counts}')
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
    max_length = build['training_max_seq_len']
    if max(train['max_tokens'], validation['max_tokens']) > max_length:
        raise ValueError('An input exceeds the declared training context')
    train['trainer_loader_samples'] = trainer_smoke(directory, 'train', tc, tokenizer, max_length)
    validation['trainer_loader_samples'] = trainer_smoke(directory, 'validation', vc, tokenizer, max_length)
    report = {'status': 'verified', 'identity': 'InstinctV2', 'train': train, 'validation': validation,
              'prompt_overlap': 0, 'provenance_group_overlap': 0, 'identity_scan_rows_both_formats': expected,
              'build_report_sha256': digest(directory / 'build.report.json')}
    with (directory / 'verification.report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
