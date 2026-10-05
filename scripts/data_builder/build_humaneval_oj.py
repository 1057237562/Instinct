"""Convert HumanEval tasks into OJ-style (stdin/stdout) problems with executable verification.

Reads the local official HumanEval snapshot, extracts (args, expected) pairs from the
asserts in each task's `check()` function, plans a plain-text line-based I/O format from
the signature (classic OJ style: one line per scalar, N-prefixed arrays, one line per
matrix row - never JSON), generates a reference program whose parsing/printing code is
derived from the same plan, and EXECUTES it on every case. The stored expected output is
the rendered result of the canonical solution, cross-checked against the assert literal;
only verified pairs are kept.

I/O plan (per task, derived from type annotations with value-based fallback):
  int/float      -> one line with the number
  bool           -> one line, 1 (true) or 0 (false)
  str            -> one line, the raw string
  list of numbers-> line "N", then one line with N numbers separated by spaces
  list of str    -> line "N", then N lines with one string each
  matrix         -> line "N" (rows), then N lines of space-separated numbers
  return         -> bool as True/False; numbers/str via str(); number lists
                    space-separated on one line; string lists one per line; matrices
                    one row per line; None prints nothing.
Output is compared byte for byte against the reference program's stdout.

Outputs (dataset/humaneval_oj/):
  humaneval_oj_problems.jsonl  one row per task: statement + verified hidden test cases
  humaneval_oj_sft.jsonl       {"conversations": [user=statement, assistant=program]}
  build.report.json            provenance, per-rule counts, identity scan, SHA-256

Contamination note: HumanEval is this project's held-out functional benchmark. Using
humaneval_oj_sft.jsonl for training burns it as an eval. Kept separate from the
v1moe SFT mixes until that trade-off is explicitly accepted.
"""
import ast
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "dataset" / "humaneval" / "HumanEval.jsonl.gz"
OUT_DIR = ROOT / "dataset" / "humaneval_oj"
TIMEOUT_SECONDS = 15
MAX_SAMPLES_IN_STATEMENT = 2

IDENTITY_KEYWORDS = (
    "qwen", "通义千问", "千问", "alibaba cloud", "阿里云", "阿里巴巴", "通义万相", "通义听悟",
)


def load_tasks():
    with gzip.open(SOURCE, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def is_foldable(node):
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(is_foldable(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(is_foldable(k) and is_foldable(v) for k, v in zip(node.keys, node.values))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return is_foldable(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        return is_foldable(node.left) and is_foldable(node.right)
    return False


def fold(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [fold(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {fold(k): fold(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp):
        return -fold(node.operand) if isinstance(node.op, ast.USub) else +fold(node.operand)
    if isinstance(node, ast.BinOp):
        left, right = fold(node.left), fold(node.right)
        return {"Add": lambda: left + right, "Sub": lambda: left - right,
                "Mult": lambda: left * right, "Div": lambda: left / right}[type(node.op).__name__]()
    raise ValueError(f"unfoldable node {ast.dump(node)[:80]}")


_MISSING = object()


def extract_cases(test_source):
    """Pull (args, expected, is_tolerance) out of check() asserts. Returns cases, rejects."""
    cases, rejects = [], []
    tree = ast.parse(test_source)
    check = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check")

    def candidate_args(call):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "candidate" and not call.keywords
                and all(is_foldable(a) for a in call.args)):
            return [fold(a) for a in call.args]
        return None

    for stmt in check.body:
        if not isinstance(stmt, ast.Assert):
            continue
        test = stmt.test
        parsed = False
        # Pattern: abs(candidate(...) - expr) < eps  (tolerance compare)
        if (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Lt)
                and isinstance(test.left, ast.Call) and isinstance(test.left.func, ast.Name)
                and test.left.func.id == "abs" and test.left.args
                and isinstance(test.comparators[0], ast.Constant)):
            inner = test.left.args[0]
            if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Sub):
                for a, b in ((inner.left, inner.right), (inner.right, inner.left)):
                    args = candidate_args(a)
                    if args is not None and is_foldable(b):
                        cases.append((args, fold(b), True))
                        parsed = True
                        break
        if not parsed and (isinstance(test, ast.Compare) and len(test.ops) == 1
                           and isinstance(test.ops[0], ast.Eq)):
            # Pattern: candidate(args) == expr   or   expr == candidate(args)
            left, right = test.left, test.comparators[0]
            args = candidate_args(left)
            if args is not None:
                expected = fold(right) if is_foldable(right) else _MISSING
            else:
                args = candidate_args(right)
                expected = fold(left) if (args is not None and is_foldable(left)) else _MISSING
            if args is not None and expected is not _MISSING:
                cases.append((args, expected, False))
                parsed = True
        if not parsed:
            rejects.append(ast.unparse(stmt)[:120])
    return cases, rejects


def signature_info(prompt):
    """Return (param_names, param_annotations, description) from the prompt's def."""
    tree = ast.parse(prompt)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    names = [a.arg for a in fn.args.args]
    annotations = [ast.unparse(a.annotation) if a.annotation else None for a in fn.args.args]
    docstring = ast.get_docstring(fn) or ""
    lines = []
    skipping = False
    for line in docstring.splitlines():
        if ">>>" in line:
            skipping = True
            continue
        if skipping:
            if line.strip() == "" or not line.startswith((" " * 8, "\t")):
                skipping = False
            else:
                continue
        lines.append(line)
    return names, annotations, "\n".join(lines).strip()


def _value_kind(value):
    """Infer an input kind from a folded literal value (annotation fallback)."""
    if isinstance(value, bool):
        return 'bool'
    if isinstance(value, int):
        return 'int'
    if isinstance(value, float):
        return 'float'
    if isinstance(value, str):
        return 'str'
    if isinstance(value, (list, tuple)):
        if not value:
            return None  # unknowable without annotation
        inner = {_value_kind(v) for v in value}
        if inner <= {'int'}:
            return 'list_int'
        if inner <= {'int', 'float'}:
            return 'list_float'
        if inner <= {'str'}:
            return 'list_str'
        if all(isinstance(v, (list, tuple)) for v in value):
            row_kinds = {_value_kind(v) for v in value}
            if row_kinds == {'list_int'}:
                return 'matrix_int'
            if row_kinds <= {'list_int', 'list_float'}:
                return 'matrix_float'
    return None  # dict, mixed, nested strings: unplannable


def _annotation_kind(annotation):
    if not annotation:
        return None
    text = annotation.replace(' ', '').replace('typing.', '')
    if text in ('bool',):
        return 'bool'
    if text in ('int',):
        return 'int'
    if text in ('float',):
        return 'float'
    if text in ('str',):
        return 'str'
    if text in ('List[int]', 'list[int]'):
        return 'list_int'
    if text in ('List[float]', 'list[float]'):
        return 'list_float'
    if text in ('List[str]', 'list[str]'):
        return 'list_str'
    if text in ('List[List[int]]', 'list[list[int]]'):
        return 'matrix_int'
    if text in ('List[List[float]]', 'list[list[float]]'):
        return 'matrix_float'
    return None


def _merge_kinds(a, b):
    """Union two inferred kinds; int widens to float, conflict returns None."""
    if a is None:
        return b
    if b is None:
        return a
    if a == b:
        return a
    pair = {a, b}
    if pair == {'int', 'float'}:
        return 'float'
    if pair == {'list_int', 'list_float'}:
        return 'list_float'
    if pair == {'matrix_int', 'matrix_float'}:
        return 'matrix_float'
    return None


def _contains_newline(value):
    if isinstance(value, str):
        return '\n' in value
    if isinstance(value, (list, tuple)):
        return any(_contains_newline(v) for v in value)
    return False


def _contains_none(value):
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_none(v) for v in value)
    return False


def plan_io(param_names, annotations, cases):
    """Plan input kinds per parameter and the output kind. Returns plan or (None, reason).

    A merge conflict is terminal: None from _merge_kinds means "unplannable" and must
    not be overwritten by later cases (None also means "no info" there, so conflicts
    are checked explicitly instead of being carried through the accumulator).
    """
    param_kinds = []
    for index, name in enumerate(param_names):
        kind = _annotation_kind(annotations[index] if index < len(annotations) else None)
        for args, _, _ in cases:
            if index < len(args):
                if _contains_newline(args[index]):
                    return None, f'param {name}: string with newline'
                vk = _value_kind(args[index])
                if vk is None:
                    continue
                if kind is None:
                    kind = vk
                else:
                    kind = _merge_kinds(kind, vk)
                    if kind is None:
                        return None, f'param {name}: conflicting value types'
        if kind is None:
            return None, f'param {name}: unplannable (dict/empty/nested/mixed)'
        param_kinds.append(kind)
    out_kind = None
    for _, expected, _ in cases:
        kind = _value_kind(expected) if expected is not None else 'none'
        if out_kind is None:
            out_kind = kind
        elif kind is not None:
            out_kind = _merge_kinds(out_kind, kind)
            if out_kind is None:
                return None, 'return value unplannable'
    if out_kind is None:
        return None, 'return value unplannable (empty everywhere)'
    return {'params': [{'name': n, 'kind': k} for n, k in zip(param_names, param_kinds)],
            'output': out_kind}, None


def _num_literal(value):
    """Render a number so the literal's type survives (ints stay ints for isinstance checks)."""
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return repr(float(value))


def render_input(args, plan):
    """Render one test case's stdin. Raises ValueError when a value violates the plan."""
    lines = []
    for value, spec in zip(args, plan['params']):
        kind = spec['kind']
        if kind == 'int':
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'{spec["name"]} expected int, got {type(value).__name__}')
            lines.append(str(value))
        elif kind == 'float':
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'{spec["name"]} expected number, got {type(value).__name__}')
            lines.append(_num_literal(value))
        elif kind == 'bool':
            lines.append('1' if value else '0')
        elif kind == 'str':
            if not isinstance(value, str) or '\n' in value:
                raise ValueError(f'{spec["name"]} expected single-line string')
            lines.append(value)
        elif kind in ('list_int', 'list_float'):
            if not isinstance(value, (list, tuple)):
                raise ValueError(f'{spec["name"]} expected a list')
            if kind == 'list_int' and any(isinstance(v, bool) or not isinstance(v, int) for v in value):
                raise ValueError(f'{spec["name"]} expected all ints')
            lines.append(str(len(value)))
            lines.append(' '.join(_num_literal(v) for v in value))
        elif kind == 'list_str':
            if (not isinstance(value, (list, tuple))
                    or any(not isinstance(v, str) or '\n' in v for v in value)):
                raise ValueError(f'{spec["name"]} expected single-line strings')
            lines.append(str(len(value)))
            lines.extend(value)
        elif kind in ('matrix_int', 'matrix_float'):
            if not isinstance(value, (list, tuple)):
                raise ValueError(f'{spec["name"]} expected a matrix')
            lines.append(str(len(value)))
            for row in value:
                if not isinstance(row, (list, tuple)):
                    raise ValueError(f'{spec["name"]} expected matrix rows')
                lines.append(' '.join(_num_literal(v) for v in row))
    return '\n'.join(lines) + '\n'


def render_output(result, out_kind):
    if out_kind == 'none':
        return ''
    if isinstance(result, tuple):
        result = list(result)
    if out_kind == 'bool':
        return 'True' if result else 'False'
    if out_kind in ('int', 'float', 'str'):
        return str(result)
    if out_kind in ('list_int', 'list_float'):
        return ' '.join(str(v) for v in result)
    if out_kind == 'list_str':
        return '\n'.join(result)
    if out_kind in ('matrix_int', 'matrix_float'):
        return '\n'.join(' '.join(str(v) for v in row) for row in result)
    raise ValueError(f'unrenderable output kind {out_kind}')


def parse_back(stdout, out_kind):
    """Parse rendered stdout back into a Python value to cross-check the assert."""
    text = stdout.rstrip('\n')
    if out_kind == 'none':
        return None if text == '' else _MISSING
    if out_kind == 'bool':
        return {'True': True, 'False': False}.get(text, _MISSING)
    try:
        if out_kind == 'int':
            return int(text)
        if out_kind == 'float':
            return float(text)
        if out_kind == 'list_int':
            return [int(t) for t in text.split()] if text else []
        if out_kind == 'list_float':
            return [float(t) for t in text.split()] if text else []
        if out_kind in ('matrix_int', 'matrix_float'):
            cast = int if out_kind == 'matrix_int' else float
            return [[cast(t) for t in line.split()] for line in text.split('\n')] if text else []
    except ValueError:
        return _MISSING
    if out_kind == 'str':
        return text
    if out_kind == 'list_str':
        return text.split('\n') if text else []
    return _MISSING


_PARSE_CODE = {
    'int': "    {name} = _num(next_line())",
    'float': "    {name} = _num(next_line())",
    'bool': "    {name} = next_line().strip() == '1'",
    'str': "    {name} = next_line()",
    'list_int': "    n = int(next_line())\n    {name} = [_num(t) for t in next_line().split()]",
    'list_float': "    n = int(next_line())\n    {name} = [_num(t) for t in next_line().split()]",
    'list_str': "    n = int(next_line())\n    {name} = [next_line() for _ in range(n)]",
    'matrix_int': "    n = int(next_line())\n    {name} = [[_num(t) for t in next_line().split()] for _ in range(n)]",
    'matrix_float': "    n = int(next_line())\n    {name} = [[_num(t) for t in next_line().split()] for _ in range(n)]",
}

_PRINT_CODE = {
    'none': "    pass",
    'bool': "    print('True' if result else 'False')",
    'int': "    print(result)",
    'float': "    print(result)",
    'str': "    print(result)",
    'list_int': "    print(' '.join(str(v) for v in result))",
    'list_float': "    print(' '.join(str(v) for v in result))",
    'list_str': "    print('\\n'.join(result))",
    'matrix_int': "    print('\\n'.join(' '.join(str(v) for v in row) for row in result))",
    'matrix_float': "    print('\\n'.join(' '.join(str(v) for v in row) for row in result))",
}

_INPUT_DOC = {
    'int': "`{name}` — a single integer on its own line",
    'float': "`{name}` — a single real number on its own line",
    'bool': "`{name}` — 1 (true) or 0 (false) on its own line",
    'str': "`{name}` — one line; the whole line (including spaces) is the string",
    'list_int': "first line: N, the number of elements of `{name}`; second line: N integers separated by spaces (an empty line if N = 0)",
    'list_float': "first line: N, the number of elements of `{name}`; second line: N real numbers separated by spaces (an empty line if N = 0)",
    'list_str': "first line: N, the number of elements of `{name}`; then N lines, each containing one string",
    'matrix_int': "first line: N, the number of rows of `{name}`; then N lines, each with space-separated integers",
    'matrix_float': "first line: N, the number of rows of `{name}`; then N lines, each with space-separated real numbers",
}

_OUTPUT_DOC = {
    'none': "print nothing (no output)",
    'bool': "a single line: True or False",
    'int': "a single line with the integer",
    'float': "a single line with the real number (Python str format, e.g. 6.0)",
    'str': "a single line containing the string",
    'list_int': "a single line with the integers separated by spaces (empty output for an empty list)",
    'list_float': "a single line with the real numbers separated by spaces, Python str format (empty output for an empty list)",
    'list_str': "one line per string, in order (no output for an empty list)",
    'matrix_int': "one line per row, integers separated by spaces (no output for an empty matrix)",
    'matrix_float': "one line per row, real numbers separated by spaces, Python str format (no output for an empty matrix)",
}


def reference_program(task, plan):
    """Program executed for verification AND used as the SFT assistant target."""
    prompt = task["prompt"]
    if not prompt.endswith("\n"):
        prompt += "\n"
    parse_lines = '\n'.join(_PARSE_CODE[spec['kind']].format(name=spec['name'])
                            for spec in plan['params'])
    program = (
        "import sys\nimport re\n\n\n"
        "def _num(t):\n"
        "    # Keep the literal's numeric type: canonical solutions may isinstance-check ints.\n"
        "    t = t.strip()\n"
        "    return int(t) if re.fullmatch(r'-?\\d+', t) else float(t)\n\n\n"
        + prompt
        + task["canonical_solution"]
        + "\n\n\ndef main():\n"
        "    lines = sys.stdin.read().split('\\n')\n"
        "    pos = 0\n"
        "    def next_line():\n"
        "        nonlocal pos\n"
        "        line = lines[pos]\n"
        "        pos += 1\n"
        "        return line\n"
        + parse_lines + '\n'
        "    result = " + task['entry_point'] + "("
        + ", ".join(spec['name'] for spec in plan['params']) + ")\n"
        + _PRINT_CODE[plan['output']] + '\n'
        "\n\nif __name__ == \"__main__\":\n    main()\n"
    )
    return program


def values_match(result, expected, is_tolerance):
    if isinstance(result, bool) or isinstance(expected, bool):
        return isinstance(result, bool) and isinstance(expected, bool) and result == expected
    if isinstance(result, float) and isinstance(expected, (int, float)):
        # Non-tolerance asserts keep HumanEval's own `==` semantics; only abs()-style
        # asserts widen to isclose.
        if is_tolerance:
            return math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-6)
        return result == expected
    if isinstance(result, (int, float)) and isinstance(expected, (int, float)):
        return result == expected
    if isinstance(result, list) and isinstance(expected, list):
        return len(result) == len(expected) and all(
            values_match(r, e, is_tolerance) for r, e in zip(result, expected))
    if isinstance(result, dict) and isinstance(expected, dict):
        return result.keys() == expected.keys() and all(
            values_match(result[k], expected[k], is_tolerance) for k in result)
    if isinstance(result, str) and isinstance(expected, str):
        return result == expected
    if result is None and expected is None:
        return True
    return False


def normalize(value):
    if isinstance(value, tuple):
        return [normalize(v) for v in value]
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    return value


def run_program(program, stdin_text):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(program)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, "-I", path],
            input=stdin_text, capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS, encoding="utf-8",
            cwd=str(tempfile.gettempdir()),
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'OMP_NUM_THREADS': '1'},
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    finally:
        Path(path).unlink(missing_ok=True)
    if proc.returncode != 0:
        return None, proc.stderr.strip().splitlines()[-1][:200] if proc.stderr.strip() else "nonzero exit"
    return proc.stdout, None


def render_statement(task_id, entry_point, description, plan, cases):
    input_doc = '\n'.join(f"- {doc}" for doc in
                          (_INPUT_DOC[spec['kind']].format(name=spec['name']) for spec in plan['params']))
    sample_blocks = []
    for case in cases[:MAX_SAMPLES_IN_STATEMENT]:
        sample_blocks.append(
            "### Sample Input\n```\n" + case['input'].rstrip('\n') + "\n```\n"
            "### Sample Output\n```\n" + case['output'] + "\n```")
    samples = "\n\n".join(sample_blocks)
    description = description or f"Implement `{entry_point}` and print its result following the I/O format below."
    return (
        f"## {task_id}: {entry_point}\n\n"
        f"{description}\n\n"
        "### Input Format\n\n"
        "Arguments appear on stdin in the order below, one section per argument "
        "(classic OJ style, plain text).\n\n"
        f"{input_doc}\n\n"
        "### Output Format\n\n"
        f"{_OUTPUT_DOC[plan['output']]}. The output is compared byte for byte.\n\n"
        + samples + "\n"
    )


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    tasks = load_tasks()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    problems_path = OUT_DIR / "humaneval_oj_problems.jsonl"
    sft_path = OUT_DIR / "humaneval_oj_sft.jsonl"

    report = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256_of(SOURCE),
        "license": "HumanEval is MIT-licensed (OpenAI); local official snapshot",
        "protocol": {"stdin": "line-based plain text, one section per argument, N-prefixed arrays",
                      "stdout": "plain text rendered by the I/O plan, exact byte compare"},
        "tasks_total": len(tasks),
        "tasks_converted": 0,
        "tasks_no_verified_case": 0,
        "asserts_total": 0,
        "asserts_unparsed": 0,
        "cases_extracted": 0,
        "cases_verified": 0,
        "cases_failed": 0,
        "skip_reasons": {},
        "identity_keyword_hits": [],
        "unparsed_examples": [],
        "task_details": {},
    }

    def note_skip(task_id, reason):
        report["skip_reasons"][reason] = report["skip_reasons"].get(reason, 0) + 1
        report["task_details"][task_id] = {"status": "skipped", "reason": reason}

    problems_out, sft_out = [], []
    for task in tasks:
        task_id = task["task_id"]
        entry_point = task["entry_point"]
        cases, rejects = extract_cases(task["test"])
        param_names, annotations, description = signature_info(task["prompt"])
        report["asserts_total"] += len(cases) + len(rejects)
        report["asserts_unparsed"] += len(rejects)
        report["cases_extracted"] += len(cases)
        if rejects and len(report["unparsed_examples"]) < 10:
            report["unparsed_examples"].append({"task_id": task_id, "assert": rejects[0]})
        if not cases:
            note_skip(task_id, "no parseable assert")
            continue
        plan, reason = plan_io(param_names, annotations, cases)
        if plan is None:
            note_skip(task_id, reason)
            continue
        program = reference_program(task, plan)

        verified = []
        none_dropped = 0
        for args, expected, is_tolerance in cases:
            if _contains_none(args) or _contains_none(expected):
                # Plain-text protocols have no natural encoding for None; drop the case.
                none_dropped += 1
                continue
            try:
                stdin_text = render_input(args, plan)
                rendered = render_output(expected, plan['output'])
            except (ValueError, TypeError) as exc:
                report["cases_failed"] += 1
                report["task_details"].setdefault(task_id, {"status": "ok"})['render_error'] = str(exc)
                continue
            stdout, error = run_program(program, stdin_text)
            if stdout is None:
                report["cases_failed"] += 1
                continue
            parsed = parse_back(stdout, plan['output'])
            # Exact asserts must also match the rendered literal byte for byte; tolerance
            # asserts (abs(...) < eps) trust the reference program's actual stdout.
            byte_ok = is_tolerance or stdout.rstrip('\n') == rendered.rstrip('\n')
            if (byte_ok and parsed is not _MISSING
                    and values_match(normalize(parsed), normalize(expected), is_tolerance)):
                verified.append({"input": stdin_text, "output": stdout.rstrip('\n')})
                report["cases_verified"] += 1
            else:
                report["cases_failed"] += 1

        if not verified:
            report["tasks_no_verified_case"] += 1
            note_skip(task_id, "no verified case")
            continue

        statement = render_statement(task_id, entry_point, description, plan, verified)
        problems_out.append({
            "task_id": task_id, "entry_point": entry_point, "statement": statement,
            "io_plan": plan, "num_test_cases": len(verified), "test_cases": verified,
            "reference_program": program,
        })
        sft_out.append({
            "task_id": task_id,
            "conversations": [
                {"role": "user", "content": statement},
                {"role": "assistant", "content": "```python\n" + program + "```"},
            ],
        })
        report["tasks_converted"] += 1
        report["cases_none_dropped"] = report.get("cases_none_dropped", 0) + none_dropped
        report["task_details"][task_id] = {
            "status": "ok", "asserts": len(cases) + len(rejects),
            "verified_cases": len(verified), "failed_cases": len(cases) - len(verified) - none_dropped,
            "none_cases_dropped": none_dropped,
            "io_plan": plan,
        }

    for artifact in problems_out + sft_out:
        blob = json.dumps(artifact, ensure_ascii=False)
        for keyword in IDENTITY_KEYWORDS:
            if re.search(keyword, blob, re.IGNORECASE):
                report["identity_keyword_hits"].append({"task_id": artifact["task_id"], "keyword": keyword})

    problems_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in problems_out), encoding="utf-8")
    sft_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sft_out), encoding="utf-8")
    report["output_sha256"] = {
        problems_path.name: sha256_of(problems_path),
        sft_path.name: sha256_of(sft_path),
    }
    (OUT_DIR / "build.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"tasks: {report['tasks_total']} converted: {report['tasks_converted']} "
          f"skipped: {report['tasks_no_verified_case']}")
    print(f"skip reasons: {report['skip_reasons']}")
    print(f"asserts: {report['asserts_total']} unparsed: {report['asserts_unparsed']} "
          f"cases verified: {report['cases_verified']} failed: {report['cases_failed']}")
    print(f"identity keyword hits: {len(report['identity_keyword_hits'])}")
    print(f"outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
