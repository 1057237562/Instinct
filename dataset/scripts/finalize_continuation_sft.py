"""Add math remediation, group validation splits, and balance continuation tokens."""
import ast
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import orjson
from datasets import load_dataset  # noqa: F401; before torch-backed tokenizer imports
from transformers import AutoTokenizer

from dataset.scripts.build_quality_python_sft import ROOT, OUT, MAX_TOKENS, SEED, norm, fingerprint, static_check
sys.path.append(str(ROOT))


def arithmetic(node):
    if isinstance(node, ast.Expression): return arithmetic(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float): return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return arithmetic(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
    if isinstance(node, ast.BinOp):
        a, b = arithmetic(node.left), arithmetic(node.right)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.Div): return a / b
        if isinstance(node.op, ast.FloorDiv): return Fraction(a // b)
        if isinstance(node.op, ast.Mod): return a % b
        if isinstance(node.op, ast.Pow) and b.denominator == 1 and abs(b) <= 6: return a ** int(b)
    raise ValueError('Unsupported arithmetic')


def main():
    initial = OUT / 'initial_build_report.json'
    if not initial.exists(): initial.write_bytes((OUT / 'build_report.json').read_bytes())
    report = json.loads(initial.read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
    def tokens(messages):
        # Use the same tools-aware renderer as the trainer, without random augmentation.
        from dataset.lm_dataset import _create_chat_prompt
        return len(tokenizer.backend_tokenizer.encode(_create_chat_prompt(tokenizer, messages), add_special_tokens=False).ids)
    records = []
    chats = {split: [orjson.loads(line)['conversations'] for line in (OUT / f'continue_sft_{split}.jsonl').open('rb')]
             for split in ('train', 'validation')}
    with gzip.open(OUT / 'provenance_and_tests.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            row['conversations'] = chats[row['split']][row['row']]
            records.append(row)
    test_grams = set()
    for row in map(orjson.loads, (ROOT / 'dataset/gsm8k/test.jsonl').open('rb')):
        words = norm(row['question']).split()
        test_grams.update(tuple(words[i:i+13]) for i in range(len(words)-12))
    def gsm_overlap(text):
        words = norm(text).split()
        return any(tuple(words[i:i+13]) in test_grams for i in range(len(words)-12))
    stats = Counter(); code = []; replay = []
    for row in records:
        text = '\n'.join(str(m.get('content', '')) for m in row['conversations'])
        if gsm_overlap(text): stats['gsm_test_overlap_removed'] += 1; continue
        if row['source'] != 'prior_sft_replay':
            reason, _ = static_check(row['conversations'][1]['content'][10:-4])
            if reason: stats['extra_static_' + reason] += 1; continue
            group = re.sub(r'_[IC]$', '', str(row['source_id'])) if row['source'] == 'kodcode' else row['source_id']
            row['group_id'] = row['source'] + ':' + str(group)
            row['split'] = 'validation' if int(hashlib.sha256(row['group_id'].encode()).hexdigest()[:8], 16) % 20 == 0 else 'train'
            code.append(row)
        else:
            row['tokens'] = tokens(row['conversations'])
            if row['tokens'] <= MAX_TOKENS: replay.append(row)
            else: stats['replay_tools_render_overlength'] += 1
    raw_math = [orjson.loads(line) for line in (OUT / 'raw/gsm8k/train.jsonl').open('rb')]
    wanted = {fingerprint(row['question']) for row in raw_math}
    old_math = set()
    for line in (ROOT / 'dataset/sft_t2t_mini.jsonl').open('rb'):
        for m in orjson.loads(line)['conversations']:
            if m.get('role') == 'user':
                h = fingerprint(str(m.get('content', '')))
                if h in wanted: old_math.add(h)
    print(f'GSM8K train exact prior-SFT prompt matches: {len(old_math)}', flush=True)
    math_rows = []
    for index, row in enumerate(raw_math):
        question, answer = row['question'], row['answer']
        if gsm_overlap(question): stats['math_test_overlap'] += 1; continue
        checked = 0; wrong = False
        for expression in re.findall(r'<<([^<>]+)>>', answer):
            if '=' not in expression: continue
            left, right = expression.rsplit('=', 1)
            try:
                value = arithmetic(ast.parse(left.replace(',', ''), mode='eval'))
                expected = Fraction(right.replace(',', '').strip())
                if abs(value - expected) > Fraction(1, 100): wrong = True; break
                checked += 1
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
                continue
        if wrong or checked < 2:
            stats['math_inconsistent_or_fewer_than_two_checked_steps'] += 1; continue
        if not re.search(r'(?m)^####\s*-?[\d,]+(?:\.\d+)?\s*$', answer):
            stats['math_answer_format'] += 1; continue
        # Remove calculator markup only; retain the human-written solution and result.
        clean = re.sub(r'<<[^<>]*>>', '', answer)
        messages = [{'role': 'user', 'content': question + '\n\nGive a concise calculation and end with #### <number>.'},
                    {'role': 'assistant', 'content': clean}]
        n = tokens(messages)
        if n > MAX_TOKENS: stats['math_overlength'] += 1; continue
        ph = fingerprint(question)
        math_rows.append({'source': 'gsm8k_train', 'source_id': str(index), 'group_id': 'gsm8k:' + ph,
                          'prompt_hash': ph, 'conversations': messages, 'tokens': n, 'license': 'MIT',
                          'checked_arithmetic_steps': checked, 'prior_sft_exact_prompt_match': ph in old_math,
                          'split': 'validation' if ph not in old_math and int(ph[:8],16) % 20 == 0 else 'train',
                          'verification': 'human-written upstream; local restricted arithmetic checks, not full semantic proof'})
    math_train = [r for r in math_rows if r['split'] == 'train']
    rng = random.Random(SEED); rng.shuffle(code); rng.shuffle(replay); rng.shuffle(math_train)
    # No repetition needed: cap code to the available math/replay token budgets.
    code_budget = min(sum(r['tokens'] for r in code if r['split']=='train'),
                      3 * sum(r['tokens'] for r in math_train), 3 * sum(r['tokens'] for r in replay))
    selected_code = []; code_tokens = 0
    for row in code:
        if row['split'] == 'train' and code_tokens + row['tokens'] <= code_budget:
            selected_code.append(row); code_tokens += row['tokens']
    def take_budget(rows, budget):
        selected = []; count = 0
        for row in rows:
            if count + row['tokens'] <= budget:
                selected.append(row); count += row['tokens']
        return selected
    selected_math = take_budget(math_train, code_tokens / 3)
    used_prompts = {fingerprint(m['content']) for r in math_rows + code for m in r['conversations'] if m['role']=='user'}
    replay = [r for r in replay if not any(fingerprint(m.get('content','')) in used_prompts or fingerprint(m.get('content','')) in wanted for m in r['conversations'] if m['role']=='user')]
    selected_replay = take_budget(replay, code_tokens / 3)
    mixed = selected_code + selected_math + selected_replay; rng.shuffle(mixed)
    validation = [r for r in code + math_rows if r['split']=='validation']; rng.shuffle(validation)
    def write(name, rows):
        with (OUT / name).open('wb') as f:
            for row in rows: f.write(orjson.dumps({'conversations':row['conversations']})+b'\n')
    for source in ('kodcode','bigcode','gsm8k_train'):
        all_rows = math_rows if source=='gsm8k_train' else code
        for split in ('train','validation'):
            write(f'{source}_{split}.jsonl', [r for r in all_rows if r['source']==source and r['split']==split])
    write('continue_sft_train.jsonl', mixed); write('continue_sft_validation.jsonl', validation)
    with gzip.open(OUT / 'provenance_and_tests.jsonl.gz','wt',encoding='utf-8') as f:
        for split, rows in [('train',mixed),('validation',validation)]:
            for i,row in enumerate(rows): f.write(json.dumps({**{k:v for k,v in row.items() if k!='conversations'},'split':split,'row':i},ensure_ascii=False)+'\n')
    # Preserve provenance/test pairs for the complete curated source sets too.
    with gzip.open(OUT / 'all_curated_sources_metadata.jsonl.gz','wt',encoding='utf-8') as f:
        for row in code + math_rows: f.write(json.dumps({k:v for k,v in row.items() if k!='conversations'},ensure_ascii=False)+'\n')
    source_tokens = {s:sum(r['tokens'] for r in mixed if r['source']==s) for s in {r['source'] for r in mixed}}
    report['final_continuation'] = {'target_token_mix':'60% new Python, 20% math, 20% prior general/tool replay; experimental starting ratio',
        'rows':len(mixed),'validation_rows':len(validation),'source_rows':dict(Counter(r['source'] for r in mixed)),
        'source_tokens':source_tokens,'source_token_fraction':{s:n/sum(source_tokens.values()) for s,n in source_tokens.items()},
        'additional_filters':dict(stats),'gsm8k_prior_sft_exact_prompt_matches':len(old_math),
        'math_curated_rows':len(math_rows),'split':'KodCode grouped by original question ID ignoring _I/_C; GSM8K normalized prompt; no old-SFT math in validation',
        'GSM8K_diagnostic':'1319 strict format errors; 157 conservative explicit/boxed numerical outputs contained 2 matches; not a replacement benchmark score'}
    report['files'] = {}
    for path in OUT.glob('*.jsonl'):
        h=hashlib.sha256(); count=0
        for line in path.open('rb'): h.update(line); orjson.loads(line); count+=1
        report['files'][path.name]={'rows':count,'bytes':path.stat().st_size,'sha256':h.hexdigest()}
    report['limitations'].append('The original 75/25 counters above describe the initial Python-only build; final_continuation is authoritative after GSM8K diagnosis.')
    (OUT / 'build_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['final_continuation'],ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
